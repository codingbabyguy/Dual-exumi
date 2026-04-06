from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def summarize_metrics(
    success: np.ndarray,
    limit_margin_rad: np.ndarray,
    sigma_min: np.ndarray,
    branch_count: int,
    pos_err_m: np.ndarray,
    rot_err_rad: np.ndarray,
    near_limit_threshold_rad: float,
    near_singular_threshold: float,
) -> dict:
    n = int(success.shape[0])
    success_rate = float(np.mean(success)) if n > 0 else 0.0
    near_limit = limit_margin_rad < near_limit_threshold_rad
    near_singular = sigma_min < near_singular_threshold
    return {
        "n_frames": n,
        "success_rate": success_rate,
        "failed_frames": int(np.sum(1 - success)),
        "branch_switch_count": int(branch_count),
        "limit_margin_rad_min": float(np.min(limit_margin_rad)),
        "limit_margin_rad_mean": float(np.mean(limit_margin_rad)),
        "near_limit_threshold_rad": float(near_limit_threshold_rad),
        "near_limit_ratio": float(np.mean(near_limit)),
        "sigma_min_min": float(np.min(sigma_min)),
        "sigma_min_mean": float(np.mean(sigma_min)),
        "near_singular_threshold": float(near_singular_threshold),
        "near_singular_ratio": float(np.mean(near_singular)),
        "pos_err_m_max": float(np.max(pos_err_m)),
        "pos_err_m_mean": float(np.mean(pos_err_m)),
        "rot_err_deg_max": float(np.rad2deg(np.max(rot_err_rad))),
        "rot_err_deg_mean": float(np.rad2deg(np.mean(rot_err_rad))),
    }


def save_demo_outputs(
    out_dir: str | Path,
    demo_name: str,
    summary: dict,
    timestamps: np.ndarray,
    target_tcp_pose7: np.ndarray,
    achieved_tcp_pose7: np.ndarray,
    q_selected: np.ndarray,
    success: np.ndarray,
    limit_margin_rad: np.ndarray,
    sigma_min: np.ndarray,
    pos_err_m: np.ndarray,
    rot_err_rad: np.ndarray,
    branch_flags: np.ndarray,
    local_cost: np.ndarray,
    joint_names: list[str],
) -> dict:
    out_dir = Path(out_dir).expanduser().resolve()
    demo_out = out_dir / demo_name
    demo_out.mkdir(parents=True, exist_ok=True)

    summary_path = demo_out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = demo_out / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "timestamp",
                "success",
                "limit_margin_rad",
                "sigma_min",
                "pos_err_m",
                "rot_err_deg",
                "branch_jump",
                "local_cost",
            ]
        )
        for i in range(len(timestamps)):
            writer.writerow(
                [
                    i,
                    float(timestamps[i]),
                    int(success[i]),
                    float(limit_margin_rad[i]),
                    float(sigma_min[i]),
                    float(pos_err_m[i]),
                    float(np.rad2deg(rot_err_rad[i])),
                    int(branch_flags[i]),
                    float(local_cost[i]),
                ]
            )

    npz_path = demo_out / "selected_trajectory.npz"
    np.savez_compressed(
        npz_path,
        timestamps=timestamps.astype(np.float64),
        target_tcp_pose7=target_tcp_pose7.astype(np.float64),
        achieved_tcp_pose7=achieved_tcp_pose7.astype(np.float64),
        q_selected=q_selected.astype(np.float64),
        success=success.astype(np.int32),
        limit_margin_rad=limit_margin_rad.astype(np.float64),
        sigma_min=sigma_min.astype(np.float64),
        pos_err_m=pos_err_m.astype(np.float64),
        rot_err_rad=rot_err_rad.astype(np.float64),
        branch_flags=branch_flags.astype(np.int32),
        local_cost=local_cost.astype(np.float64),
        joint_names=np.asarray(joint_names),
    )

    return {
        "demo_out": str(demo_out),
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
    }


def save_global_summary(out_dir: str | Path, rows: list[dict]) -> str:
    out_dir = Path(out_dir).expanduser().resolve()
    path = out_dir / "global_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return str(path)

