#!/usr/bin/env python3
"""
Convert an exUMI session into a LeRobot v3 dataset with a single, explicit
coordinate contract:

- `aligned_arcap_poses.json` is assumed to already be expressed in the policy
  manual frame.
- `observation.state` and `action` are both stored as raw 10D manual-frame
  vectors: [x, y, z, rot6d(6), gripper_norm].
- Training/inference normalization is delegated to LeRobot's
  `meta/stats.json` and processor pipeline.

The script also writes visual diagnostics so each stage can be checked quickly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - visualization is best effort
    plt = None


THIS_FILE = Path(__file__).resolve()
DUAL_EXUMI_ROOT = THIS_FILE.parents[1]
WORKSPACE_ROOT = THIS_FILE.parents[3]
MY_LEROBOT_SRC = WORKSPACE_ROOT / "my_lerobot" / "src"

for path in (DUAL_EXUMI_ROOT, MY_LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from umi.common.timecode_util import mp4_get_start_datetime
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ACTION_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
]


@dataclass
class EpisodeMeta:
    episode_name: str
    video_path: Path
    aligned_pose_path: Path
    start_timestamp: float
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one exUMI session into LeRobot v3 manual-frame data")
    parser.add_argument("--session_dir", type=str, required=True, help="Session directory, e.g. data/foo/batch_1")
    parser.add_argument("--output_dir", type=str, required=True, help="Output LeRobot v3 root directory")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot repo_id written to metadata. Defaults to local/<output_dir_name>.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task string stored in dataset metadata. Defaults to the session folder name.",
    )
    parser.add_argument("--robot_type", type=str, default="realman_manual_frame", help="robot_type metadata value")
    parser.add_argument("--image_height", type=int, default=224, help="Processed image height")
    parser.add_argument("--image_width", type=int, default=224, help="Processed image width")
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=1.0,
        help="Center crop ratio before resize. 1.0 keeps full frame.",
    )
    parser.add_argument(
        "--report_samples",
        type=int,
        default=6,
        help="How many processed frames to keep for the montage report.",
    )
    parser.add_argument(
        "--allow_legacy_frame",
        action="store_true",
        help="Allow non-manual_relative_frame aligned poses (not recommended).",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If set, remove existing output_dir before creating dataset.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_segment_start_timestamp(video_path: Path) -> float:
    segment_meta_path = video_path.parent / "segment_meta.json"
    if segment_meta_path.is_file():
        return float(load_json(segment_meta_path)["start_timestamp"])
    return float(mp4_get_start_datetime(str(video_path)).timestamp())


def discover_episodes(session_dir: Path) -> list[EpisodeMeta]:
    demos_dir = session_dir / "demos"
    if not demos_dir.is_dir():
        raise FileNotFoundError(f"Missing demos directory: {demos_dir}")

    metas: list[EpisodeMeta] = []
    for video_path in sorted(demos_dir.glob("*/raw_video.mp4")):
        aligned_pose_path = video_path.parent / "aligned_arcap_poses.json"
        if not aligned_pose_path.is_file():
            continue
        with av.open(str(video_path), "r") as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
        metas.append(
            EpisodeMeta(
                episode_name=video_path.parent.name,
                video_path=video_path,
                aligned_pose_path=aligned_pose_path,
                start_timestamp=load_segment_start_timestamp(video_path),
                fps=fps,
            )
        )
    if not metas:
        raise FileNotFoundError("No aligned demos found. Please run AR_03 first.")
    return metas


def _resolve_aligned_pose_coordinate_frame(aligned_obj: dict, aligned_pose_path: Path) -> str:
    frame = aligned_obj.get("coordinate_frame")
    if frame is None and isinstance(aligned_obj.get("metadata"), dict):
        frame = aligned_obj["metadata"].get("coordinate_frame")
    if frame is None:
        summary_path = aligned_pose_path.parent / "aligned_pose_summary.json"
        if summary_path.is_file():
            try:
                frame = load_json(summary_path).get("coordinate_frame")
            except Exception:
                frame = None
    if frame is None:
        return "manual_relative_frame"
    return str(frame).strip()


def load_aligned_episode(meta: EpisodeMeta) -> tuple[np.ndarray, np.ndarray, str]:
    obj = load_json(meta.aligned_pose_path)
    pose = np.asarray(obj.get("pose"), dtype=np.float64)
    width = np.asarray(obj.get("width"), dtype=np.float64).reshape(-1)
    coordinate_frame = _resolve_aligned_pose_coordinate_frame(obj, meta.aligned_pose_path)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Invalid pose shape in {meta.aligned_pose_path}: {pose.shape}")
    if len(width) != len(pose):
        raise ValueError(
            f"Pose/width length mismatch in {meta.aligned_pose_path}: pose={len(pose)} width={len(width)}"
        )
    return pose, width, coordinate_frame


def find_calibration_file(session_dir: Path) -> Path | None:
    candidates = [
        session_dir / "calibration_params.npz",
        session_dir.parent / "calibration_params.npz",
        session_dir.parent.parent / "calibration_params.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_calibration(session_dir: Path) -> dict | None:
    calib_path = find_calibration_file(session_dir)
    if calib_path is None:
        return None
    with np.load(str(calib_path)) as obj:
        return {
            "path": str(calib_path),
            "manual_origin": np.asarray(obj["manual_origin"], dtype=np.float64).tolist(),
            "manual_rotation": np.asarray(obj["manual_rotation"], dtype=np.float64).tolist(),
        }


def prepare_output_dir(path: Path, overwrite_output: bool) -> None:
    if path.exists():
        if overwrite_output:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Please remove it, choose a new --output_dir, or pass --overwrite_output."
            )


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-12:
        raise ValueError("Encountered near-zero quaternion")
    rot_m = R.from_quat(quat_xyzw / norm).as_matrix()
    return np.concatenate([rot_m[:, 0], rot_m[:, 1]], axis=0).astype(np.float32)


def preprocess_rgb_frame(frame_rgb: np.ndarray, output_hw: tuple[int, int], crop_ratio: float) -> np.ndarray:
    out_h, out_w = output_hw
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H,W,3), got {frame.shape}")

    in_h, in_w = frame.shape[:2]
    crop_ratio = float(crop_ratio)
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError(f"crop_ratio must be in (0,1], got {crop_ratio}")

    if crop_ratio < 1.0:
        target_ratio = out_w / float(out_h)
        crop_h = max(1, min(int(round(in_h * crop_ratio)), in_h))
        crop_w = max(1, min(int(round(crop_h * target_ratio)), in_w))
        if crop_w > in_w:
            crop_w = in_w
            crop_h = max(1, min(int(round(crop_w / target_ratio)), in_h))
        x0 = max((in_w - crop_w) // 2, 0)
        y0 = max((in_h - crop_h) // 2, 0)
        frame = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        interp = cv2.INTER_LINEAR if (crop_w < out_w or crop_h < out_h) else cv2.INTER_AREA
        return cv2.resize(frame, (out_w, out_h), interpolation=interp)

    scale = max(out_w / float(in_w), out_h / float(in_h))
    resize_w = max(int(np.ceil(in_w * scale)), out_w)
    resize_h = max(int(np.ceil(in_h * scale)), out_h)
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    frame = cv2.resize(frame, (resize_w, resize_h), interpolation=interp)
    x0 = max((resize_w - out_w) // 2, 0)
    y0 = max((resize_h - out_h) // 2, 0)
    frame = frame[y0 : y0 + out_h, x0 : x0 + out_w]
    if frame.shape[0] != out_h or frame.shape[1] != out_w:
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.uint8, copy=False)


def build_dataset_features(image_hw: tuple[int, int]) -> dict:
    image_h, image_w = image_hw
    return {
        "observation.image": {
            "dtype": "image",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": ACTION_NAMES,
        },
    }


def flatten_stats_range(stats: dict, key: str) -> tuple[list[float], list[float]]:
    min_vals = np.asarray(stats[key]["min"], dtype=np.float64).reshape(-1)
    max_vals = np.asarray(stats[key]["max"], dtype=np.float64).reshape(-1)
    return min_vals.tolist(), max_vals.tolist()


def save_state_action_plot(report_dir: Path, episode_name: str, states: np.ndarray, actions: np.ndarray) -> None:
    if plt is None or len(states) == 0:
        return
    t = np.arange(len(states), dtype=np.float64)
    fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
    axes = axes.flatten()
    for idx, name in enumerate(ACTION_NAMES):
        ax = axes[idx]
        ax.plot(t, states[:, idx], label="state", linewidth=1.5)
        ax.plot(t, actions[:, idx], label="action", linewidth=1.1, alpha=0.85)
        ax.set_title(name)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Episode {episode_name}: observation.state vs action", fontsize=14)
    fig.tight_layout()
    fig.savefig(report_dir / "episode_000_state_action.png", dpi=160)
    plt.close(fig)


def save_frame_montage(report_dir: Path, frames: Iterable[np.ndarray]) -> None:
    if plt is None:
        return
    frames = list(frames)
    if not frames:
        return
    cols = min(3, len(frames))
    rows = int(np.ceil(len(frames) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax, frame_idx in zip(axes_arr, range(len(frames)), strict=False):
        ax.imshow(frames[frame_idx])
        ax.set_title(f"processed frame {frame_idx}")
        ax.axis("off")
    for ax in axes_arr[len(frames) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(report_dir / "episode_000_frames.png", dpi=160)
    plt.close(fig)


def build_deployment_template(
    output_dir: Path,
    repo_id: str,
    calibration: dict | None,
    stats: dict,
    image_shape: tuple[int, int, int],
) -> dict:
    action_min, action_max = flatten_stats_range(stats, "action")
    state_min, state_max = flatten_stats_range(stats, "observation.state")
    return {
        "coordinate_frame": "manual_relative_frame",
        "dataset": {
            "repo_id": repo_id,
            "root": str(output_dir),
            "meta_info_path": str(output_dir / "meta" / "info.json"),
            "meta_stats_path": str(output_dir / "meta" / "stats.json"),
        },
        "action_schema": {
            "names": ACTION_NAMES,
            "coordinate_frame": "manual_relative_frame",
            "position_unit": "meter",
            "rotation_representation": "rot6d",
            "gripper_unit": "normalized",
        },
        "safety": {
            "enable_policy_workspace_clip": True,
            "action_bounds": {
                name: [float(action_min[i]), float(action_max[i])] for i, name in enumerate(ACTION_NAMES)
            },
            "workspace_bounds": {
                axis: [float(state_min[i]), float(state_max[i])]
                for i, axis in enumerate(["x", "y", "z"])
            },
        },
        "robot_adapter": {
            "config": {
                "policy_frame": "manual_relative_frame",
                "manual_origin": calibration["manual_origin"] if calibration else None,
                "manual_rotation": calibration["manual_rotation"] if calibration else None,
                "image_shape": list(image_shape),
                "use_sdk_pose_transform": True,
                "lock_work_tool_frame": True,
                "frame_lock_require_expected_names": True,
                "expected_work_frame_names": [],
                "expected_tool_frame_names": [],
                "workspace_clip_in_adapter": False,
                "runtime_joint_guard": {
                    "enabled": True,
                    "warn_only": False,
                    "joint_limit_margin_deg": 8.0,
                    "max_joint_step_deg": 6.0,
                    "enable_self_collision_check": False,
                    "enable_singularity_check": True,
                    "require_algo_checks": False,
                    "fail_on_ik_error": True,
                },
            }
        },
        "calibration": calibration,
    }


def main() -> None:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_id = args.repo_id or f"local/{output_dir.name}"
    task_name = args.task or session_dir.name
    image_hw = (int(args.image_height), int(args.image_width))
    prepare_output_dir(output_dir, overwrite_output=bool(args.overwrite_output))

    print(f"[INFO] session_dir={session_dir}")
    print(f"[INFO] output_dir ={output_dir}")
    print(f"[INFO] repo_id    ={repo_id}")
    print(f"[INFO] task       ={task_name}")

    episodes = discover_episodes(session_dir)
    common_fps = episodes[0].fps
    for ep in episodes[1:]:
        if abs(ep.fps - common_fps) > 1e-6:
            raise ValueError(f"Inconsistent fps across episodes: {episodes[0].fps} vs {ep.fps} ({ep.episode_name})")

    width_values: list[np.ndarray] = []
    pose_cache: dict[str, np.ndarray] = {}
    width_cache: dict[str, np.ndarray] = {}
    frame_cache: dict[str, str] = {}
    for ep in episodes:
        pose_arr, width_arr, coordinate_frame = load_aligned_episode(ep)
        if (coordinate_frame != "manual_relative_frame") and (not args.allow_legacy_frame):
            raise ValueError(
                f"{ep.aligned_pose_path} is in {coordinate_frame!r}, expected manual_relative_frame. "
                "Run AR_03 without --legacy_flexiv_transform, or pass --allow_legacy_frame if intentional."
            )
        pose_cache[ep.episode_name] = pose_arr
        width_cache[ep.episode_name] = width_arr
        frame_cache[ep.episode_name] = coordinate_frame
        width_values.append(width_arr)
    all_width = np.concatenate(width_values, axis=0)
    width_min = float(np.min(all_width))
    width_max = float(np.max(all_width))
    if abs(width_max - width_min) < 1e-12:
        width_max = width_min + 1.0

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(common_fps)),
        features=build_dataset_features(image_hw),
        root=output_dir,
        robot_type=args.robot_type,
        use_videos=False,
    )

    debug_frames: list[np.ndarray] = []
    first_episode_states: list[np.ndarray] = []
    first_episode_actions: list[np.ndarray] = []
    total_frames = 0

    for episode_index, ep in enumerate(episodes):
        pose_arr = pose_cache[ep.episode_name]
        width_arr = width_cache[ep.episode_name]
        coordinate_frame = frame_cache[ep.episode_name]
        decoded_frames = 0
        print(
            f"[EP {episode_index:03d}] {ep.episode_name} | frames={len(pose_arr)} | "
            f"fps={ep.fps:.6f} | frame={coordinate_frame}"
        )

        with av.open(str(ep.video_path), "r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_idx, frame in enumerate(container.decode(stream)):
                decoded_frames += 1
                if frame_idx >= len(pose_arr):
                    raise ValueError(f"Video has more frames than aligned poses: {ep.video_path}")

                img_rgb = frame.to_ndarray(format="rgb24")
                img_proc = preprocess_rgb_frame(img_rgb, image_hw, crop_ratio=args.crop_ratio)

                pose = pose_arr[frame_idx]
                xyz = pose[:3].astype(np.float32)
                quat = pose[3:7]
                rot6d = quat_xyzw_to_rot6d(quat)
                grip = np.array([(width_arr[frame_idx] - width_min) / (width_max - width_min)], dtype=np.float32)
                grip = np.clip(grip, 0.0, 1.0)

                state = np.concatenate([xyz, rot6d, grip], axis=0).astype(np.float32)
                action = state.copy()

                dataset.add_frame(
                    {
                        "observation.image": img_proc,
                        "observation.state": state,
                        "action": action,
                        "task": task_name,
                    }
                )

                if episode_index == 0:
                    first_episode_states.append(state.copy())
                    first_episode_actions.append(action.copy())
                    if len(debug_frames) < int(args.report_samples):
                        debug_frames.append(img_proc.copy())

        if decoded_frames != len(pose_arr):
            raise ValueError(
                f"Video/pose length mismatch in {ep.video_path.parent}: decoded={decoded_frames}, pose={len(pose_arr)}"
            )

        dataset.save_episode()
        total_frames += len(pose_arr)

    dataset.finalize()

    info_path = output_dir / "meta" / "info.json"
    stats_path = output_dir / "meta" / "stats.json"
    info = load_json(info_path)
    stats = load_json(stats_path)
    calibration = load_calibration(session_dir)

    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    coordinate_frames_seen = sorted(set(frame_cache.values()))

    if first_episode_states:
        save_state_action_plot(
            report_dir,
            episode_name=episodes[0].episode_name,
            states=np.stack(first_episode_states, axis=0),
            actions=np.stack(first_episode_actions, axis=0),
        )
    save_frame_montage(report_dir, debug_frames)

    summary = {
        "repo_id": repo_id,
        "coordinate_frame": "manual_relative_frame",
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "task": task_name,
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "fps": float(common_fps),
        "aligned_pose_coordinate_frames": coordinate_frames_seen,
        "image_shape": [int(args.image_height), int(args.image_width), 3],
        "gripper_width_raw_min": width_min,
        "gripper_width_raw_max": width_max,
        "calibration": calibration,
    }
    with (report_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    deployment_template = build_deployment_template(
        output_dir=output_dir,
        repo_id=repo_id,
        calibration=calibration,
        stats=stats,
        image_shape=(int(args.image_height), int(args.image_width), 3),
    )
    with (report_dir / "deployment_manual_frame_template.json").open("w", encoding="utf-8") as f:
        json.dump(deployment_template, f, indent=2, ensure_ascii=False)

    contract = {
        "coordinate_frame": "manual_relative_frame",
        "manual_frame_definition": {
            "position": "p_manual = R_manual^T (p_base - o_manual)",
            "rotation": "R_manual_obj = R_manual^T R_base_obj",
        },
        "inference_inverse_mapping": {
            "position": "p_base = o_manual + R_manual p_manual",
            "rotation": "R_base_obj = R_manual R_manual_obj",
        },
        "action_semantics": "absolute_manual_pose_target_10d",
        "state_semantics": "absolute_manual_pose_state_10d",
        "notes": [
            "action equals state during conversion (behavior cloning target).",
            "manual_origin/manual_rotation must come from the same collection calibration.",
            "Do not enable startup anchor remapping unless training also used startup-relative frame.",
        ],
    }
    with (report_dir / "manual_frame_contract.json").open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)

    print(f"[DONE] dataset root : {output_dir}")
    print(f"[DONE] info.json    : {info_path}")
    print(f"[DONE] stats.json   : {stats_path}")
    print(f"[DONE] report dir   : {report_dir}")
    print(f"[DONE] image shape  : {info['features']['observation.image']['shape']}")
    print(f"[DONE] state shape  : {info['features']['observation.state']['shape']}")
    print(f"[DONE] action shape : {info['features']['action']['shape']}")


if __name__ == "__main__":
    main()
