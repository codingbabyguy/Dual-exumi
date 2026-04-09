#!/usr/bin/env python3
"""
Visualize trajectories from LeRobot v3 dataset converted by:
scripts/convert_session_to_lerobot_dp.py

Input expected:
- <dataset_root>/meta/info.json
- <dataset_root>/meta/episodes/chunk-*/file-*.parquet
- <dataset_root>/data/chunk-*/file-*.parquet

This script does NOT read raw demos/* files.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

def _require_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:
        raise RuntimeError(
            "This script needs pandas to read LeRobot parquet files. "
            "Please install pandas in your running environment first."
        ) from exc


def _load_visualize_helpers():
    try:
        from visualize_pose import save_animated_video, save_static_image, save_trajectory_stats

        return save_animated_video, save_static_image, save_trajectory_stats
    except Exception as exc:
        raise RuntimeError(
            "Cannot import visualize helpers from visualize_pose.py. "
            "Please ensure matplotlib and related dependencies are installed."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trajectory from converted LeRobot v3 dataset (not raw demos)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="LeRobot v3 dataset root (output_dir from convert_session_to_lerobot_dp.py).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Episode selector, e.g. '0,2,5-8'. Empty means all episodes.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Only process first N selected episodes. 0 means no limit.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="state",
        choices=("state", "action"),
        help="Use observation.state or action as trajectory source.",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default="observation.state",
        help="Parquet column key for state vector.",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="action",
        help="Parquet column key for action vector.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Default: <dataset_root>/reports/trajectory_viz_lerobot",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="mp4",
        choices=("mp4", "gif", "both", "none"),
        help="Animation output format.",
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip animation generation.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=0.0,
        help="Animation fps. <=0 means use dataset fps from meta/info.json.",
    )
    return parser.parse_args()


def _parse_episode_selector(selector: str, available: list[int]) -> list[int]:
    if not selector.strip():
        return sorted(available)

    selected: set[int] = set()
    for token in selector.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", maxsplit=1)
            if len(pieces) != 2:
                continue
            start = int(pieces[0].strip())
            end = int(pieces[1].strip())
            if end < start:
                start, end = end, start
            for ep in range(start, end + 1):
                selected.add(ep)
        else:
            selected.add(int(part))
    available_set = set(available)
    return sorted([ep for ep in selected if ep in available_set])


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_episodes_df(dataset_root: Path):
    pd = _require_pandas()
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Cannot find episode metadata parquet under: {dataset_root / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(p) for p in paths], axis=0, ignore_index=True)
    if "episode_index" not in df.columns:
        df["episode_index"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(by="episode_index", kind="stable").reset_index(drop=True)
    return df


def _to_int(value, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def rot6d_to_rotmats(rot6d: np.ndarray) -> np.ndarray:
    """Convert Nx6 rot6d to Nx3x3 rotation matrices using Gram-Schmidt."""
    r6 = np.asarray(rot6d, dtype=np.float64)
    if r6.ndim != 2 or r6.shape[1] != 6:
        raise ValueError(f"rot6d must be Nx6, got {r6.shape}")

    a1 = r6[:, 0:3]
    a2 = r6[:, 3:6]

    b1 = _normalize(a1)
    proj = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = _normalize(a2 - proj * b1)
    b3 = np.cross(b1, b2)

    rot = np.stack([b1, b2, b3], axis=2)
    return rot


def compute_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid_pts = np.median(points, axis=0)
    return mid_pts - max_range, mid_pts + max_range


def _read_episode_rows(
    dataset_root: Path,
    episode_row,
    state_key: str,
    action_key: str,
    data_cache: dict[Path, Any],
):
    pd = _require_pandas()
    episode_index = int(episode_row["episode_index"])
    chunk_idx = _to_int(episode_row.get("data/chunk_index"), default=-1)
    file_idx = _to_int(episode_row.get("data/file_index"), default=-1)

    candidate_paths: list[Path] = []
    if chunk_idx >= 0 and file_idx >= 0:
        candidate_paths.append(dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet")
    if not candidate_paths:
        candidate_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))

    read_cols = [c for c in ["index", "episode_index", "frame_index", state_key, action_key] if c]
    for path in candidate_paths:
        if not path.is_file():
            continue
        if path not in data_cache:
            data_cache[path] = pd.read_parquet(path, columns=read_cols)
        df = data_cache[path]
        if "episode_index" in df.columns:
            selected = df[df["episode_index"] == episode_index].copy()
        else:
            selected = pd.DataFrame()
        if len(selected) > 0:
            return selected

    # Fallback to global index range filter if episode_index column is missing or filtered rows are empty.
    ep_from = _to_int(episode_row.get("dataset_from_index"), default=-1)
    ep_to = _to_int(episode_row.get("dataset_to_index"), default=-1)
    if ep_from >= 0 and ep_to > ep_from:
        all_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        chunks: list[pd.DataFrame] = []
        for path in all_paths:
            if path not in data_cache:
                data_cache[path] = pd.read_parquet(path, columns=read_cols)
            df = data_cache[path]
            if "index" not in df.columns:
                continue
            sub = df[(df["index"] >= ep_from) & (df["index"] < ep_to)]
            if len(sub) > 0:
                chunks.append(sub.copy())
        if chunks:
            return pd.concat(chunks, axis=0, ignore_index=True)

    raise ValueError(f"Cannot find rows for episode {episode_index}")


def _stack_column_as_array(df, key: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Missing column: {key}")
    arrs: list[np.ndarray] = []
    for obj in df[key].tolist():
        a = np.asarray(obj, dtype=np.float64).reshape(-1)
        arrs.append(a)
    if not arrs:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(arrs, axis=0)


def _save_state_action_diff_report(
    out_dir: Path,
    action_names: list[str],
    states: np.ndarray,
    actions: np.ndarray,
) -> dict:
    if len(states) == 0 or len(actions) == 0:
        report = {"num_frames": 0}
    else:
        common = min(len(states), len(actions))
        diff = actions[:common] - states[:common]
        abs_diff = np.abs(diff)
        per_dim = {}
        dim = min(diff.shape[1], len(action_names))
        for i in range(dim):
            per_dim[action_names[i]] = {
                "max_abs": float(np.max(abs_diff[:, i])),
                "mean_abs": float(np.mean(abs_diff[:, i])),
            }
        report = {
            "num_frames": int(common),
            "overall_max_abs": float(np.max(abs_diff)),
            "overall_mean_abs": float(np.mean(abs_diff)),
            "per_dim": per_dim,
        }
    out_path = out_dir / "state_action_diff.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    args = parse_args()
    save_animated_video, save_static_image, save_trajectory_stats = _load_visualize_helpers()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    info = _load_json(info_path)
    features = info.get("features", {})
    action_names = list(features.get("action", {}).get("names", []))
    fps = float(info.get("fps", 30))
    if args.video_fps > 0:
        fps = float(args.video_fps)

    out_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (dataset_root / "reports" / "trajectory_viz_lerobot")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    episodes_df = _load_episodes_df(dataset_root)
    available = [int(x) for x in episodes_df["episode_index"].tolist()]
    selected = _parse_episode_selector(args.episodes, available)
    if args.max_episodes > 0:
        selected = selected[: int(args.max_episodes)]

    print("\n" + "=" * 78)
    print("  LeRobot v3 trajectory visualizer (for converted dataset)")
    print("=" * 78)
    print(f"[CONFIG] dataset_root={dataset_root}")
    print(f"[CONFIG] output_root ={out_root}")
    print(f"[CONFIG] source={args.source} state_key={args.state_key} action_key={args.action_key}")
    print(f"[CONFIG] fps={fps:.3f} output_format={args.output_format} skip_video={int(args.skip_video)}")
    print(f"[INFO] total episodes in dataset={len(available)}")
    print(f"[INFO] selected episodes={selected}")

    if not selected:
        raise ValueError("No episodes selected.")

    data_cache: dict[Path, Any] = {}
    done = 0
    skipped: list[tuple[int, str]] = []
    summary_rows: list[dict] = []

    for episode_index in selected:
        row = episodes_df[episodes_df["episode_index"] == episode_index]
        if len(row) == 0:
            skipped.append((episode_index, "episode not found in metadata"))
            continue
        ep_row = row.iloc[0]
        ep_out = out_root / f"episode_{episode_index:03d}"
        ep_out.mkdir(parents=True, exist_ok=True)

        try:
            ep_df = _read_episode_rows(
                dataset_root=dataset_root,
                episode_row=ep_row,
                state_key=args.state_key,
                action_key=args.action_key,
                data_cache=data_cache,
            )
            sort_key = "frame_index" if "frame_index" in ep_df.columns else ("index" if "index" in ep_df.columns else "")
            if sort_key:
                ep_df = ep_df.sort_values(by=sort_key, kind="stable").reset_index(drop=True)

            states = _stack_column_as_array(ep_df, args.state_key)
            actions = _stack_column_as_array(ep_df, args.action_key)
            source_arr = states if args.source == "state" else actions
            if len(source_arr) == 0:
                raise ValueError("empty trajectory")
            if source_arr.shape[1] < 9:
                raise ValueError(f"{args.source} shape invalid: {source_arr.shape}, expected at least 9 dims")

            pts = source_arr[:, 0:3]
            rotmats = rot6d_to_rotmats(source_arr[:, 3:9])
            bounds = compute_bounds(pts)

            print("\n" + "-" * 78)
            print(
                f"[EP {episode_index:03d}] frames={len(source_arr)} "
                f"source={args.source} pos_range=({np.ptp(pts[:,0]):.4f}, {np.ptp(pts[:,1]):.4f}, {np.ptp(pts[:,2]):.4f})"
            )

            save_trajectory_stats(pts, str(ep_out), use_absolute=False)
            save_static_image(pts, bounds, str(ep_out), use_absolute=False)

            if (not args.skip_video) and args.output_format != "none":
                if args.output_format in ("mp4", "both"):
                    save_animated_video(
                        pts,
                        rotmats,
                        bounds,
                        str(ep_out),
                        use_absolute=False,
                        output_format="mp4",
                        video_fps=fps,
                        output_basename=f"trajectory_animation_{args.source}",
                    )
                if args.output_format in ("gif", "both"):
                    save_animated_video(
                        pts,
                        rotmats,
                        bounds,
                        str(ep_out),
                        use_absolute=False,
                        output_format="gif",
                        output_basename=f"trajectory_animation_{args.source}",
                    )

            diff_report = _save_state_action_diff_report(ep_out, action_names, states, actions)
            print(
                f"[EP {episode_index:03d}] state-action diff: "
                f"max_abs={diff_report.get('overall_max_abs', 0.0):.6e} "
                f"mean_abs={diff_report.get('overall_mean_abs', 0.0):.6e}"
            )

            summary_rows.append(
                {
                    "episode_index": int(episode_index),
                    "num_frames": int(len(source_arr)),
                    "source": args.source,
                    "state_action_overall_max_abs": float(diff_report.get("overall_max_abs", 0.0)),
                    "state_action_overall_mean_abs": float(diff_report.get("overall_mean_abs", 0.0)),
                    "output_dir": str(ep_out),
                }
            )
            done += 1
        except Exception as exc:
            skipped.append((episode_index, str(exc)))

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(out_root),
        "source": args.source,
        "processed": done,
        "selected": len(selected),
        "fps": fps,
        "episodes": summary_rows,
        "skipped": [{"episode_index": int(ep), "reason": reason} for ep, reason in skipped],
    }
    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"[DONE] processed={done}/{len(selected)}")
    print(f"[DONE] summary={summary_path}")
    if skipped:
        print("[SKIPPED]")
        for ep, reason in skipped:
            print(f"  - episode_{ep:03d}: {reason}")
    print("=" * 78)


if __name__ == "__main__":
    main()
