#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rm65_offline_replay.config import load_config, transform_from_cfg
from rm65_offline_replay.io_pose import find_aligned_pose_jsons, load_aligned_pose_json
from rm65_offline_replay.math3d import (
    compose,
    matrix_to_pose7,
    pose7_to_matrix,
    pose_error_components,
    transform_inverse,
)
from rm65_offline_replay.pin_solver import PinocchioIKBatchSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search mapping params to reduce limit hits / branch flicker for offline replay."
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML config.")
    parser.add_argument("--output_dir", type=str, required=True, help="Optimization output directory.")
    parser.add_argument("--session_dir", type=str, default=None, help="Session dir with demos/demo_*/aligned_arcap_poses.json")
    parser.add_argument("--demo_json", type=str, nargs="*", default=None, help="Optional explicit demo json list.")
    parser.add_argument("--max_trials", type=int, default=120, help="Total sampled candidates (including baseline).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Subsample frames for speed.")
    parser.add_argument("--max_demos", type=int, default=2, help="Use up to N demos during search.")

    parser.add_argument("--search_b_xyz_m", type=float, default=0.08, help="Search half-range for T_B_from_pose_frame xyz.")
    parser.add_argument("--search_b_rpy_deg", type=float, default=20.0, help="Search half-range for T_B_from_pose_frame rpy.")
    parser.add_argument("--search_pose_tcp_xyz_m", type=float, default=0.10, help="Search half-range for T_pose_to_tcp xyz.")
    parser.add_argument("--search_pose_tcp_rpy_deg", type=float, default=30.0, help="Search half-range for T_pose_to_tcp rpy.")
    parser.add_argument("--search_home_deg", type=float, default=15.0, help="Search half-range for home_q_deg.")
    parser.add_argument("--optimize_home_q", action="store_true", help="Enable search over home_q_deg.")
    parser.add_argument(
        "--presearch_scalar_steps",
        type=int,
        default=9,
        help="Grid steps for each scalar stage of T_B_from_pose_frame presearch.",
    )
    parser.add_argument(
        "--presearch_rp_steps",
        type=int,
        default=5,
        help="Grid steps per axis for roll/pitch presearch stage.",
    )
    parser.add_argument(
        "--presearch_fine_scale",
        type=float,
        default=0.35,
        help="Fine-stage span scale after coarse T_B_from_pose_frame presearch.",
    )
    parser.add_argument(
        "--presearch_frame_window",
        type=int,
        default=12,
        help="First-N-frame window used to judge comfortable starting workspace.",
    )
    parser.add_argument(
        "--auto_continue",
        action="store_true",
        help="Skip interactive yes-confirmation after base-frame presearch.",
    )
    return parser.parse_args()


def _ensure_rpy_node(node: dict) -> tuple[np.ndarray, np.ndarray]:
    T = transform_from_cfg(node)
    xyz = T[:3, 3].copy()
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    return xyz, rpy


def _set_xyz_rpy(node: dict, xyz: np.ndarray, rpy_rad: np.ndarray) -> dict:
    return {
        "xyz": [float(x) for x in xyz.tolist()],
        "rpy_rad": [float(x) for x in rpy_rad.tolist()],
    }


def _normalize_joint_index(idx_raw: Any, nq: int) -> int | None:
    try:
        idx = int(idx_raw)
    except (TypeError, ValueError):
        return None
    if idx < 0:
        idx += int(nq)
    if idx < 0 or idx >= int(nq):
        return None
    return int(idx)


def build_targets(
    pose_arr: np.ndarray,
    cfg: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames_cfg = cfg["frames"]
    input_pose_represents = str(cfg["robot"].get("input_pose_represents", "tcp")).strip().lower()
    if input_pose_represents not in {"tcp", "flange"}:
        raise ValueError(
            f"robot.input_pose_represents must be tcp|flange, got {input_pose_represents}"
        )
    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    if solve_frame not in {"flange", "tcp"}:
        raise ValueError(f"robot.solve_frame must be flange|tcp, got {solve_frame}")

    T_B_from_pose = transform_from_cfg(frames_cfg["T_B_from_pose_frame"])
    T_pose_to_tcp = transform_from_cfg(frames_cfg["T_pose_to_tcp"])
    T_flange_to_tcp = transform_from_cfg(frames_cfg["T_flange_to_tcp"])
    T_pose_to_flange = transform_inverse(T_flange_to_tcp)

    target_solve_T: list[np.ndarray] = []
    target_tcp_T: list[np.ndarray] = []
    for i in range(pose_arr.shape[0]):
        T_pose = pose7_to_matrix(pose_arr[i])
        if input_pose_represents == "flange":
            T_B_flange = compose(T_B_from_pose, T_pose)
            T_B_tcp = compose(T_B_flange, T_flange_to_tcp)
        else:
            T_B_tcp = compose(compose(T_B_from_pose, T_pose), T_pose_to_tcp)
            T_B_flange = compose(T_B_tcp, T_pose_to_flange)
        if solve_frame == "flange":
            T_B_solve = T_B_flange
        else:
            T_B_solve = T_B_tcp
        target_solve_T.append(T_B_solve)
        target_tcp_T.append(T_B_tcp)
    return target_solve_T, target_tcp_T


def summarize_one_demo(
    solver: PinocchioIKBatchSolver,
    cfg: dict,
    pose_arr: np.ndarray,
    ts: np.ndarray,
    frame_stride: int,
    focus_window: int,
) -> dict[str, Any]:
    if frame_stride > 1:
        pose_arr = pose_arr[::frame_stride]
        ts = ts[::frame_stride]

    target_solve_T, target_tcp_T = build_targets(pose_arr, cfg)
    ik_out = solver.solve_sequence(target_solve_T, cfg)

    q = ik_out["q_selected"]
    success = ik_out["success"]
    limit_margin = ik_out["limit_margin_rad"]
    sigma_min = ik_out["sigma_min"]
    branch_count = int(ik_out["branch_count"])
    joint_step_max = float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0
    home_q = np.asarray(ik_out.get("home_q_rad", np.zeros((solver.nq,), dtype=np.float64)), dtype=np.float64)
    q_dist_home = np.linalg.norm(q - home_q[None, :], axis=1) if len(q) > 0 else np.zeros((0,), dtype=np.float64)
    early_n = max(1, min(int(focus_window), int(len(q)))) if len(q) > 0 else 0
    early_slice = slice(0, early_n)

    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    T_flange_to_tcp = transform_from_cfg(cfg["frames"]["T_flange_to_tcp"])

    achieved_solve_T = ik_out["achieved_T"]
    achieved_tcp_T: list[np.ndarray] = []
    for T_solve in achieved_solve_T:
        achieved_tcp_T.append(compose(T_solve, T_flange_to_tcp) if solve_frame == "flange" else T_solve)

    pos_err = np.zeros((len(ts),), dtype=np.float64)
    rot_err = np.zeros((len(ts),), dtype=np.float64)
    for i in range(len(ts)):
        p, r = pose_error_components(target_tcp_T[i], achieved_tcp_T[i])
        pos_err[i] = p
        rot_err[i] = r

    near_limit_th = float(cfg["report"]["near_limit_threshold_rad"])
    near_sing_th = float(cfg["report"]["near_singular_threshold"])
    elbow_sign_violation = np.asarray(ik_out.get("elbow_sign_violation", np.zeros_like(success)), dtype=np.float64)
    elbow_halfspace_violation = np.asarray(
        ik_out.get("elbow_halfspace_violation", np.zeros_like(success)),
        dtype=np.float64,
    )
    joint3_strict_violation = np.asarray(
        ik_out.get("joint3_strict_violation", np.zeros_like(success)),
        dtype=np.float64,
    )
    wrist_flip_flags = np.asarray(ik_out.get("wrist_flip_flags", np.zeros_like(success)), dtype=np.float64)
    branch_flags = np.asarray(ik_out.get("branch_flags", np.zeros_like(success)), dtype=np.float64)

    joint3_idx = _normalize_joint_index(cfg.get("selection", {}).get("joint3_index", 2), solver.nq)
    if joint3_idx is None or len(q) == 0:
        joint3_margin = np.zeros((len(q),), dtype=np.float64)
    else:
        joint3_margin = np.minimum(
            q[:, joint3_idx] - solver.lower[joint3_idx],
            solver.upper[joint3_idx] - q[:, joint3_idx],
        )

    out = {
        "n_frames": int(len(ts)),
        "success_rate": float(np.mean(success)) if len(success) > 0 else 0.0,
        "branch_switch_count": branch_count,
        "joint_step_max_rad": joint_step_max,
        "limit_margin_min": float(np.min(limit_margin)),
        "limit_margin_mean": float(np.mean(limit_margin)),
        "near_limit_ratio": float(np.mean(limit_margin < near_limit_th)),
        "sigma_min_min": float(np.min(sigma_min)),
        "near_singular_ratio": float(np.mean(sigma_min < near_sing_th)),
        "pos_err_mean": float(np.mean(pos_err)),
        "pos_err_max": float(np.max(pos_err)),
        "rot_err_deg_mean": float(np.rad2deg(np.mean(rot_err))),
        "rot_err_deg_max": float(np.rad2deg(np.max(rot_err))),
        "post_stabilize_fix_count": int(ik_out.get("post_stabilize_fix_count", 0)),
        "wrist_flip_count": int(ik_out.get("wrist_flip_count", 0)),
        "joint_pref_violation_ratio": float(ik_out.get("joint_pref_violation_ratio", 0.0)),
        "elbow_sign_violation_ratio": float(ik_out.get("elbow_sign_violation_ratio", 0.0)),
        "elbow_halfspace_violation_ratio": float(ik_out.get("elbow_halfspace_violation_ratio", 0.0)),
        "joint3_strict_violation_ratio": float(ik_out.get("joint3_strict_violation_ratio", 0.0)),
        "first_frame_home_dist_rad": float(q_dist_home[0]) if len(q_dist_home) > 0 else 0.0,
        "early_home_mean_rad": float(np.mean(q_dist_home[early_slice])) if early_n > 0 else 0.0,
        "first_frame_limit_margin": float(limit_margin[0]) if len(limit_margin) > 0 else 0.0,
        "first_frame_sigma_min": float(sigma_min[0]) if len(sigma_min) > 0 else 0.0,
        "joint3_margin_min": float(np.min(joint3_margin)) if len(joint3_margin) > 0 else 0.0,
        "early_joint3_margin_min": float(np.min(joint3_margin[early_slice])) if early_n > 0 else 0.0,
        "early_elbow_sign_violation_ratio": float(np.mean(elbow_sign_violation[early_slice])) if early_n > 0 else 0.0,
        "early_elbow_halfspace_violation_ratio": float(np.mean(elbow_halfspace_violation[early_slice])) if early_n > 0 else 0.0,
        "early_joint3_strict_violation_ratio": float(np.mean(joint3_strict_violation[early_slice])) if early_n > 0 else 0.0,
        "early_wrist_flip_count": float(np.sum(wrist_flip_flags[early_slice])) if early_n > 0 else 0.0,
        "early_branch_count": float(np.sum(branch_flags[early_slice])) if early_n > 0 else 0.0,
    }
    return out


def score_metrics(m: dict[str, Any]) -> float:
    # Lower is better.
    # Priority: avoid limit/switch/flicker first, then improve geometric consistency.
    score = 0.0
    score += 8.0 * float(m["near_limit_ratio"])
    score += 2.5 * min(float(m["branch_switch_count"]) / 5.0, 1.0)
    score += 2.0 * min(float(m["joint_step_max_rad"]) / 0.35, 2.0)
    score += 4.0 * (1.0 - float(m["success_rate"]))
    score += 6.0 * min(float(m["pos_err_mean"]) / 0.03, 3.0)
    score += 1.2 * min(float(m.get("wrist_flip_count", 0.0)) / 4.0, 2.0)
    score += 2.0 * float(m.get("joint_pref_violation_ratio", 0.0))
    score += 1.0 * float(m.get("elbow_sign_violation_ratio", 0.0))
    score += 2.5 * float(m.get("elbow_halfspace_violation_ratio", 0.0))
    score += 3.0 * float(m.get("joint3_strict_violation_ratio", 0.0))
    # bonus penalty if still fully hugging limit
    if float(m["limit_margin_min"]) < 0.005:
        score += 2.0
    return float(score)


def score_workspace_metrics(m: dict[str, Any]) -> float:
    # Lower is better. Prioritize "comfortable working region" before later fine tuning.
    score = 0.0
    score += 4.5 * min(float(m.get("first_frame_home_dist_rad", 0.0)) / 0.60, 3.0)
    score += 2.5 * min(float(m.get("early_home_mean_rad", 0.0)) / 0.75, 3.0)

    early_joint3_margin = float(m.get("early_joint3_margin_min", 0.0))
    all_joint3_margin = float(m.get("joint3_margin_min", 0.0))
    score += 4.0 * max(0.0, (0.35 - early_joint3_margin) / 0.35)
    score += 2.0 * max(0.0, (0.25 - all_joint3_margin) / 0.25)

    score += 2.0 * max(0.0, (0.06 - float(m.get("first_frame_limit_margin", 0.0))) / 0.06)
    score += 2.0 * max(0.0, (0.025 - float(m.get("first_frame_sigma_min", 0.0))) / 0.025)

    score += 2.0 * float(m.get("early_elbow_sign_violation_ratio", 0.0))
    score += 3.5 * float(m.get("early_elbow_halfspace_violation_ratio", 0.0))
    score += 2.0 * float(m.get("early_joint3_strict_violation_ratio", 0.0))
    score += 1.5 * min(float(m.get("early_wrist_flip_count", 0.0)) / 2.0, 3.0)
    score += 1.0 * min(float(m.get("early_branch_count", 0.0)) / 2.0, 3.0)

    score += 5.0 * float(m.get("near_limit_ratio", 0.0))
    score += 5.0 * float(m.get("near_singular_ratio", 0.0))
    score += 3.0 * (1.0 - float(m.get("success_rate", 0.0)))
    score += 2.0 * min(float(m.get("pos_err_mean", 0.0)) / 0.03, 3.0)
    return float(score)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "success_rate",
        "branch_switch_count",
        "joint_step_max_rad",
        "limit_margin_min",
        "limit_margin_mean",
        "near_limit_ratio",
        "sigma_min_min",
        "near_singular_ratio",
        "pos_err_mean",
        "pos_err_max",
        "rot_err_deg_mean",
        "rot_err_deg_max",
        "post_stabilize_fix_count",
        "wrist_flip_count",
        "joint_pref_violation_ratio",
        "elbow_sign_violation_ratio",
        "elbow_halfspace_violation_ratio",
        "joint3_strict_violation_ratio",
        "first_frame_home_dist_rad",
        "early_home_mean_rad",
        "first_frame_limit_margin",
        "first_frame_sigma_min",
        "joint3_margin_min",
        "early_joint3_margin_min",
        "early_elbow_sign_violation_ratio",
        "early_elbow_halfspace_violation_ratio",
        "early_joint3_strict_violation_ratio",
        "early_wrist_flip_count",
        "early_branch_count",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows], dtype=np.float64)
        if k in {"branch_switch_count", "post_stabilize_fix_count", "wrist_flip_count", "early_wrist_flip_count", "early_branch_count"}:
            out[k] = float(np.sum(vals))
        elif k in {"limit_margin_min", "sigma_min_min", "joint3_margin_min", "early_joint3_margin_min", "first_frame_limit_margin", "first_frame_sigma_min"}:
            out[k] = float(np.min(vals))
        elif k in {"joint_step_max_rad", "pos_err_max", "rot_err_deg_max"}:
            out[k] = float(np.max(vals))
        else:
            out[k] = float(np.mean(vals))
    out["score"] = score_metrics(out)
    out["workspace_score"] = score_workspace_metrics(out)
    return out


def dump_yaml(path: Path, obj: dict) -> None:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML required: pip install pyyaml") from e
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def evaluate_cfg_on_demos(
    solver: PinocchioIKBatchSolver,
    cfg: dict,
    demos: list[dict[str, Any]],
    frame_stride: int,
    focus_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in demos:
        m = summarize_one_demo(
            solver=solver,
            cfg=cfg,
            pose_arr=d["pose"],
            ts=d["timestamp"],
            frame_stride=frame_stride,
            focus_window=focus_window,
        )
        m["demo_name"] = d["demo_name"]
        rows.append(m)
    agg = aggregate_metrics(rows)
    return rows, agg


def _with_base_frame(cfg: dict, xyz: np.ndarray, rpy: np.ndarray) -> dict:
    out = copy.deepcopy(cfg)
    out["frames"]["T_B_from_pose_frame"] = _set_xyz_rpy(out["frames"]["T_B_from_pose_frame"], xyz, rpy)
    return out


def _record_trial(
    history: list[dict[str, Any]],
    phase: str,
    trial_idx: int,
    cfg_trial: dict,
    agg: dict[str, Any],
    rows: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    rec = {
        "phase": phase,
        "trial": int(trial_idx),
        "metrics": agg,
        "per_demo": rows,
        "frames": {
            "T_B_from_pose_frame": cfg_trial["frames"]["T_B_from_pose_frame"],
            "T_pose_to_tcp": cfg_trial["frames"]["T_pose_to_tcp"],
        },
        "home_q_deg": cfg_trial["selection"].get("home_q_deg", None),
    }
    if extra:
        rec.update(extra)
    history.append(rec)


def run_base_frame_presearch(
    solver: PinocchioIKBatchSolver,
    cfg_start: dict,
    demos: list[dict[str, Any]],
    frame_stride: int,
    focus_window: int,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, Any], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    trial_idx = 0

    rows, best_metrics = evaluate_cfg_on_demos(
        solver=solver,
        cfg=cfg_start,
        demos=demos,
        frame_stride=frame_stride,
        focus_window=focus_window,
    )
    best_cfg = copy.deepcopy(cfg_start)
    _record_trial(
        history,
        phase="presearch_baseline",
        trial_idx=trial_idx,
        cfg_trial=best_cfg,
        agg=best_metrics,
        rows=rows,
        extra={"stage": "baseline"},
    )
    trial_idx += 1
    print(
        f"[PRESEARCH] baseline workspace_score={best_metrics['workspace_score']:.4f} "
        f"first_home={best_metrics['first_frame_home_dist_rad']:.3f} "
        f"joint3_margin={best_metrics['joint3_margin_min']:.3f} "
        f"near_limit={best_metrics['near_limit_ratio']:.3f} "
        f"near_sing={best_metrics['near_singular_ratio']:.3f}",
        flush=True,
    )

    scalar_steps = max(3, int(args.presearch_scalar_steps))
    rp_steps = max(3, int(args.presearch_rp_steps))
    fine_scale = max(0.05, min(float(args.presearch_fine_scale), 1.0))
    axis_order = [
        ("yaw", "rpy", 2, float(np.deg2rad(args.search_b_rpy_deg))),
        ("x", "xyz", 0, float(args.search_b_xyz_m)),
        ("z", "xyz", 2, float(args.search_b_xyz_m)),
        ("y", "xyz", 1, float(args.search_b_xyz_m)),
    ]

    for axis_name, kind, idx, full_span in axis_order:
        for pass_name, span_scale in [("coarse", 1.0), ("fine", fine_scale)]:
            center_xyz, center_rpy = _ensure_rpy_node(best_cfg["frames"]["T_B_from_pose_frame"])
            offsets = np.linspace(-full_span * span_scale, full_span * span_scale, scalar_steps, dtype=np.float64)
            pass_best_cfg = best_cfg
            pass_best_metrics = best_metrics
            improved = False
            print(
                f"[PRESEARCH] stage={axis_name} pass={pass_name} center={center_xyz[idx] if kind == 'xyz' else center_rpy[idx]:.4f} span={full_span * span_scale:.4f}",
                flush=True,
            )
            for offset in offsets:
                xyz = center_xyz.copy()
                rpy = center_rpy.copy()
                if kind == "xyz":
                    xyz[idx] = center_xyz[idx] + float(offset)
                else:
                    rpy[idx] = center_rpy[idx] + float(offset)
                cfg_trial = _with_base_frame(best_cfg, xyz=xyz, rpy=rpy)
                rows, agg = evaluate_cfg_on_demos(
                    solver=solver,
                    cfg=cfg_trial,
                    demos=demos,
                    frame_stride=frame_stride,
                    focus_window=focus_window,
                )
                _record_trial(
                    history,
                    phase="presearch_axis",
                    trial_idx=trial_idx,
                    cfg_trial=cfg_trial,
                    agg=agg,
                    rows=rows,
                    extra={
                        "stage": axis_name,
                        "pass": pass_name,
                        "kind": kind,
                        "index": int(idx),
                        "offset": float(offset),
                    },
                )
                trial_idx += 1
                if float(agg["workspace_score"]) < float(pass_best_metrics["workspace_score"]):
                    pass_best_cfg = cfg_trial
                    pass_best_metrics = agg
                    improved = True
            best_cfg = pass_best_cfg
            best_metrics = pass_best_metrics
            print(
                f"[PRESEARCH] stage={axis_name} pass={pass_name} best workspace_score={best_metrics['workspace_score']:.4f} "
                f"first_home={best_metrics['first_frame_home_dist_rad']:.3f} "
                f"joint3_margin={best_metrics['joint3_margin_min']:.3f} "
                f"wrist_early={best_metrics['early_wrist_flip_count']:.1f} "
                f"near_sing={best_metrics['near_singular_ratio']:.3f} "
                f"improved={int(improved)}",
                flush=True,
            )

    rp_span = float(np.deg2rad(args.search_b_rpy_deg))
    for pass_name, span_scale in [("coarse", 1.0), ("fine", fine_scale)]:
        center_xyz, center_rpy = _ensure_rpy_node(best_cfg["frames"]["T_B_from_pose_frame"])
        roll_offsets = np.linspace(-rp_span * span_scale, rp_span * span_scale, rp_steps, dtype=np.float64)
        pitch_offsets = np.linspace(-rp_span * span_scale, rp_span * span_scale, rp_steps, dtype=np.float64)
        pass_best_cfg = best_cfg
        pass_best_metrics = best_metrics
        improved = False
        print(
            f"[PRESEARCH] stage=roll_pitch pass={pass_name} span={rp_span * span_scale:.4f}",
            flush=True,
        )
        for d_roll in roll_offsets:
            for d_pitch in pitch_offsets:
                xyz = center_xyz.copy()
                rpy = center_rpy.copy()
                rpy[0] = center_rpy[0] + float(d_roll)
                rpy[1] = center_rpy[1] + float(d_pitch)
                cfg_trial = _with_base_frame(best_cfg, xyz=xyz, rpy=rpy)
                rows, agg = evaluate_cfg_on_demos(
                    solver=solver,
                    cfg=cfg_trial,
                    demos=demos,
                    frame_stride=frame_stride,
                    focus_window=focus_window,
                )
                _record_trial(
                    history,
                    phase="presearch_roll_pitch",
                    trial_idx=trial_idx,
                    cfg_trial=cfg_trial,
                    agg=agg,
                    rows=rows,
                    extra={
                        "stage": "roll_pitch",
                        "pass": pass_name,
                        "roll_offset_rad": float(d_roll),
                        "pitch_offset_rad": float(d_pitch),
                    },
                )
                trial_idx += 1
                if float(agg["workspace_score"]) < float(pass_best_metrics["workspace_score"]):
                    pass_best_cfg = cfg_trial
                    pass_best_metrics = agg
                    improved = True
        best_cfg = pass_best_cfg
        best_metrics = pass_best_metrics
        print(
            f"[PRESEARCH] stage=roll_pitch pass={pass_name} best workspace_score={best_metrics['workspace_score']:.4f} "
            f"first_home={best_metrics['first_frame_home_dist_rad']:.3f} "
            f"joint3_margin={best_metrics['joint3_margin_min']:.3f} "
            f"near_limit={best_metrics['near_limit_ratio']:.3f} "
            f"improved={int(improved)}",
            flush=True,
        )

    return best_cfg, best_metrics, history


def confirm_continue_after_presearch(auto_continue: bool) -> bool:
    if auto_continue:
        print("[CONFIRM] auto_continue enabled, continue to refinement.", flush=True)
        return True
    try:
        answer = input("[CONFIRM] Type `yes` to continue with downstream search: ").strip().lower()
    except EOFError as e:
        raise RuntimeError(
            "Interactive confirmation required after presearch. Run in a tty or pass --auto_continue."
        ) from e
    if answer != "yes":
        print("[STOP] Confirmation not received. Exiting after T_B_from_pose_frame presearch.", flush=True)
        return False
    return True


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] output_dir={out_dir}", flush=True)

    cfg_base = load_config(args.config)
    print(f"[INIT] loaded config={cfg_base.get('config_path', args.config)}", flush=True)
    urdf_path_path = Path(cfg_base["robot"]["urdf_path"]).expanduser().resolve()
    if not urdf_path_path.is_file():
        raise FileNotFoundError(
            f"robot.urdf_path must be a URDF file, got: {urdf_path_path}"
        )
    urdf_path = str(urdf_path_path)
    print(f"[INIT] urdf_path={urdf_path}", flush=True)
    ee_frame_name = str(cfg_base["robot"]["ee_frame_name"])
    input_pose_represents = str(cfg_base["robot"].get("input_pose_represents", "tcp")).strip().lower()
    solver = PinocchioIKBatchSolver(urdf_path=urdf_path, ee_frame_name=ee_frame_name)
    print(f"[INIT] pinocchio model loaded, nq={solver.nq}, ee_frame={ee_frame_name}", flush=True)
    print(f"[INIT] input_pose_represents={input_pose_represents} solve_frame={cfg_base['robot']['solve_frame']}", flush=True)

    demo_paths: list[Path]
    if args.demo_json:
        demo_paths = [Path(x).expanduser().resolve() for x in args.demo_json]
    elif args.session_dir:
        demo_paths = find_aligned_pose_jsons(args.session_dir)
    else:
        raise ValueError("Either --session_dir or --demo_json must be provided.")

    if len(demo_paths) == 0:
        raise RuntimeError("No demos found.")
    demo_paths = demo_paths[: max(1, int(args.max_demos))]
    print(f"[DATA] using {len(demo_paths)} demo(s) for search:", flush=True)
    for p in demo_paths:
        print(f"[DATA] - {p}", flush=True)

    demos = [load_aligned_pose_json(p) for p in demo_paths]
    for d in demos:
        print(
            f"[DATA] demo={d['demo_name']} frames={d['pose'].shape[0]} coordinate_frame={d.get('coordinate_frame','unknown')}",
            flush=True,
        )

    search_frame_stride = max(1, int(args.frame_stride))
    focus_window = max(1, int(args.presearch_frame_window))
    rng = np.random.default_rng(int(args.seed))
    print(
        f"[SEARCH] seed={args.seed} max_trials={args.max_trials} frame_stride={search_frame_stride} max_demos={args.max_demos}",
        flush=True,
    )
    print(
        f"[PRESEARCH] order=yaw -> x -> z -> y -> roll/pitch focus_window={focus_window}",
        flush=True,
    )

    pre_cfg, pre_metrics, pre_history = run_base_frame_presearch(
        solver=solver,
        cfg_start=cfg_base,
        demos=demos,
        frame_stride=search_frame_stride,
        focus_window=focus_window,
        args=args,
    )
    pre_cfg_to_save = copy.deepcopy(pre_cfg)
    pre_cfg_to_save.pop("config_path", None)
    pre_cfg_path = out_dir / "presearch_best_config.yaml"
    dump_yaml(pre_cfg_path, pre_cfg_to_save)
    pre_metrics_path = out_dir / "presearch_best_metrics.json"
    with open(pre_metrics_path, "w", encoding="utf-8") as f:
        json.dump(pre_metrics, f, indent=2, ensure_ascii=False)
    pre_history_path = out_dir / "presearch_history.json"
    with open(pre_history_path, "w", encoding="utf-8") as f:
        json.dump(pre_history, f, indent=2, ensure_ascii=False)

    best_base_frame = pre_cfg["frames"]["T_B_from_pose_frame"]
    print("[PRESEARCH DONE] Suggested T_B_from_pose_frame:", flush=True)
    print(json.dumps(best_base_frame, indent=2, ensure_ascii=False), flush=True)
    print(
        f"[PRESEARCH DONE] workspace_score={pre_metrics['workspace_score']:.4f} "
        f"first_home={pre_metrics['first_frame_home_dist_rad']:.3f} "
        f"joint3_margin={pre_metrics['joint3_margin_min']:.3f} "
        f"near_limit={pre_metrics['near_limit_ratio']:.3f} "
        f"near_sing={pre_metrics['near_singular_ratio']:.3f}",
        flush=True,
    )
    print(f"[PRESEARCH DONE] config saved: {pre_cfg_path}", flush=True)
    print(f"[PRESEARCH DONE] metrics saved: {pre_metrics_path}", flush=True)
    print(f"[PRESEARCH DONE] history saved: {pre_history_path}", flush=True)

    if not confirm_continue_after_presearch(auto_continue=bool(args.auto_continue)):
        return

    cfg_seed = copy.deepcopy(pre_cfg)

    # Base parameter vector for post-confirm refinement
    b_xyz0, b_rpy0 = _ensure_rpy_node(cfg_seed["frames"]["T_B_from_pose_frame"])
    p_xyz0, p_rpy0 = _ensure_rpy_node(cfg_seed["frames"]["T_pose_to_tcp"])
    home0 = np.asarray(cfg_seed["selection"].get("home_q_deg", [0.0] * solver.nq), dtype=np.float64)
    if home0.shape[0] != solver.nq:
        home0 = np.zeros((solver.nq,), dtype=np.float64)

    history: list[dict[str, Any]] = list(pre_history)
    best_cfg = copy.deepcopy(cfg_seed)
    _, best_metrics = evaluate_cfg_on_demos(
        solver=solver,
        cfg=best_cfg,
        demos=demos,
        frame_stride=search_frame_stride,
        focus_window=focus_window,
    )
    t0 = time.time()

    def sample_candidate(trial_idx: int) -> dict:
        cfg = copy.deepcopy(cfg_seed)
        if trial_idx == 0:
            # baseline after presearch confirmation
            return cfg

        # 2-stage random search: first half broad, second half fine around current best
        frac = trial_idx / max(1.0, float(args.max_trials - 1))
        fine_scale = 1.0 if frac < 0.5 else 0.35

        if best_metrics is not None and frac >= 0.5:
            b_xyz_c, b_rpy_c = _ensure_rpy_node(best_cfg["frames"]["T_B_from_pose_frame"])
            p_xyz_c, p_rpy_c = _ensure_rpy_node(best_cfg["frames"]["T_pose_to_tcp"])
            home_c = np.asarray(best_cfg["selection"].get("home_q_deg", home0.tolist()), dtype=np.float64)
        else:
            b_xyz_c, b_rpy_c = b_xyz0, b_rpy0
            p_xyz_c, p_rpy_c = p_xyz0, p_rpy0
            home_c = home0

        b_xyz = b_xyz_c + rng.uniform(
            -args.search_b_xyz_m * fine_scale, args.search_b_xyz_m * fine_scale, size=3
        )
        b_rpy = b_rpy_c + np.deg2rad(
            rng.uniform(-args.search_b_rpy_deg * fine_scale, args.search_b_rpy_deg * fine_scale, size=3)
        )
        p_xyz = p_xyz_c + rng.uniform(
            -args.search_pose_tcp_xyz_m * fine_scale, args.search_pose_tcp_xyz_m * fine_scale, size=3
        )
        p_rpy = p_rpy_c + np.deg2rad(
            rng.uniform(
                -args.search_pose_tcp_rpy_deg * fine_scale,
                args.search_pose_tcp_rpy_deg * fine_scale,
                size=3,
            )
        )

        cfg["frames"]["T_B_from_pose_frame"] = _set_xyz_rpy(cfg["frames"]["T_B_from_pose_frame"], b_xyz, b_rpy)
        if input_pose_represents != "flange":
            cfg["frames"]["T_pose_to_tcp"] = _set_xyz_rpy(cfg["frames"]["T_pose_to_tcp"], p_xyz, p_rpy)

        if args.optimize_home_q:
            home = home_c + rng.uniform(
                -args.search_home_deg * fine_scale, args.search_home_deg * fine_scale, size=solver.nq
            )
            cfg["selection"]["home_q_deg"] = [float(x) for x in home.tolist()]

        return cfg

    n_trials = int(args.max_trials)
    for trial in range(n_trials):
        trial_t0 = time.time()
        if trial == 0:
            print(f"[TRIAL {trial+1}/{n_trials}] refinement baseline evaluation...", flush=True)
        elif trial % 10 == 0:
            elapsed = time.time() - t0
            avg = elapsed / max(1, trial)
            remain = avg * (n_trials - trial)
            print(
                f"[TRIAL {trial+1}/{n_trials}] running... elapsed={elapsed:.1f}s eta={remain:.1f}s",
                flush=True,
            )
        cfg_trial = sample_candidate(trial)
        rows, agg = evaluate_cfg_on_demos(
            solver=solver,
            cfg=cfg_trial,
            demos=demos,
            frame_stride=search_frame_stride,
            focus_window=focus_window,
        )
        _record_trial(
            history,
            phase="refine",
            trial_idx=trial,
            cfg_trial=cfg_trial,
            agg=agg,
            rows=rows,
        )
        trial_dt = time.time() - trial_t0

        improved = best_metrics is None or float(agg["score"]) < float(best_metrics["score"])
        if improved:
            best_metrics = agg
            best_cfg = cfg_trial
            print(
                f"[BEST] trial={trial} score={agg['score']:.4f} "
                f"workspace={agg['workspace_score']:.4f} "
                f"near_limit={agg['near_limit_ratio']:.3f} "
                f"branch={agg['branch_switch_count']:.1f} "
                f"wrist_flip={agg['wrist_flip_count']:.1f} "
                f"elbow_hs={agg['elbow_halfspace_violation_ratio']:.3f} "
                f"step_max={agg['joint_step_max_rad']:.3f} "
                f"pos_err={agg['pos_err_mean']:.4f} "
                f"trial_time={trial_dt:.2f}s"
            )
        elif trial < 3:
            # Show a few early non-best trials to confirm progress.
            print(
                f"[TRIAL {trial+1}/{n_trials}] score={agg['score']:.4f} "
                f"workspace={agg['workspace_score']:.4f} "
                f"near_limit={agg['near_limit_ratio']:.3f} "
                f"branch={agg['branch_switch_count']:.1f} "
                f"wrist_flip={agg['wrist_flip_count']:.1f} "
                f"step_max={agg['joint_step_max_rad']:.3f} "
                f"trial_time={trial_dt:.2f}s",
                flush=True,
            )

    if best_metrics is None:
        raise RuntimeError("No valid trial result.")

    history_path = out_dir / "search_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    best_metrics_path = out_dir / "best_metrics.json"
    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    best_cfg_path = out_dir / "best_config.yaml"
    best_cfg_to_save = copy.deepcopy(best_cfg)
    best_cfg_to_save.pop("config_path", None)
    dump_yaml(best_cfg_path, best_cfg_to_save)

    total_dt = time.time() - t0
    print(f"[DONE] best config: {best_cfg_path}", flush=True)
    print(f"[DONE] best metrics: {best_metrics_path}", flush=True)
    print(f"[DONE] history: {history_path}", flush=True)
    print(
        f"[DONE] total_time={total_dt:.1f}s best_score={best_metrics['score']:.4f} "
        f"workspace_score={best_metrics['workspace_score']:.4f} "
        f"near_limit={best_metrics['near_limit_ratio']:.3f} "
        f"branch={best_metrics['branch_switch_count']:.1f} "
        f"wrist_flip={best_metrics.get('wrist_flip_count', 0.0):.1f} "
        f"elbow_hs={best_metrics.get('elbow_halfspace_violation_ratio', 0.0):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
