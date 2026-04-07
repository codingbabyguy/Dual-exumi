#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

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
from rm65_offline_replay.report import save_demo_outputs, save_global_summary, summarize_metrics
from rm65_offline_replay.rm_baseline import run_rm_baseline_ik


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RM65 offline replay validation: pose -> IK -> sequence selection -> report."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config path.",
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        default=None,
        help="Session directory that contains demos/demo_*/aligned_arcap_poses.json.",
    )
    parser.add_argument(
        "--demo_json",
        type=str,
        nargs="*",
        default=None,
        help="Optional direct list of aligned_arcap_poses.json paths.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root for reports.",
    )
    return parser.parse_args()


def build_targets(
    pose_arr: np.ndarray,
    cfg: dict,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Convert input pose frame into robot-base targets.

    Returns:
    - target_solve_T: list of target transforms for IK frame (flange or tcp)
    - target_tcp_T: list of target transforms for TCP frame
    - target_tcp_pose7: (N,7)
    """
    frames_cfg = cfg["frames"]
    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    if solve_frame not in {"flange", "tcp"}:
        raise ValueError(f"robot.solve_frame must be flange|tcp, got {solve_frame}")

    T_B_from_pose = transform_from_cfg(frames_cfg["T_B_from_pose_frame"])
    T_pose_to_tcp = transform_from_cfg(frames_cfg["T_pose_to_tcp"])
    T_flange_to_tcp = transform_from_cfg(frames_cfg["T_flange_to_tcp"])
    T_tcp_to_flange = transform_inverse(T_flange_to_tcp)

    target_solve_T: list[np.ndarray] = []
    target_tcp_T: list[np.ndarray] = []

    for i in range(pose_arr.shape[0]):
        T_pose = pose7_to_matrix(pose_arr[i])
        T_B_tcp = compose(compose(T_B_from_pose, T_pose), T_pose_to_tcp)
        if solve_frame == "flange":
            T_B_solve = compose(T_B_tcp, T_tcp_to_flange)
        else:
            T_B_solve = T_B_tcp
        target_solve_T.append(T_B_solve)
        target_tcp_T.append(T_B_tcp)

    target_tcp_pose7 = np.stack([matrix_to_pose7(T) for T in target_tcp_T], axis=0)
    return target_solve_T, target_tcp_T, target_tcp_pose7


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_paths: list[Path] = []
    if args.demo_json:
        demo_paths = [Path(x).expanduser().resolve() for x in args.demo_json]
    elif args.session_dir:
        demo_paths = find_aligned_pose_jsons(args.session_dir)
    else:
        raise ValueError("Either --session_dir or --demo_json must be provided.")

    if len(demo_paths) == 0:
        raise RuntimeError("No aligned_arcap_poses.json found.")

    urdf_path_path = Path(cfg["robot"]["urdf_path"]).expanduser().resolve()
    if not urdf_path_path.is_file():
        raise FileNotFoundError(
            f"robot.urdf_path must be a URDF file, got: {urdf_path_path}"
        )
    urdf_path = str(urdf_path_path)
    ee_frame_name = str(cfg["robot"]["ee_frame_name"])
    solver = PinocchioIKBatchSolver(urdf_path=urdf_path, ee_frame_name=ee_frame_name)

    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    T_flange_to_tcp = transform_from_cfg(cfg["frames"]["T_flange_to_tcp"])

    global_rows: list[dict] = []
    for demo_json in demo_paths:
        data = load_aligned_pose_json(demo_json)
        pose_arr = data["pose"]
        ts = data["timestamp"]
        demo_name = data["demo_name"]

        target_solve_T, target_tcp_T, target_tcp_pose7 = build_targets(pose_arr, cfg)
        ik_out = solver.solve_sequence(target_solve_T, cfg)

        q_selected = ik_out["q_selected"]
        q_selected_raw = ik_out.get("q_selected_raw", None)
        success = ik_out["success"]
        limit_margin_rad = ik_out["limit_margin_rad"]
        sigma_min = ik_out["sigma_min"]
        branch_flags = ik_out["branch_flags"]
        branch_count = ik_out["branch_count"]
        local_cost = ik_out["local_cost"]
        achieved_solve_T = ik_out["achieved_T"]

        achieved_tcp_T: list[np.ndarray] = []
        for T_solve in achieved_solve_T:
            if solve_frame == "flange":
                achieved_tcp_T.append(compose(T_solve, T_flange_to_tcp))
            else:
                achieved_tcp_T.append(T_solve)

        achieved_tcp_pose7 = np.stack([matrix_to_pose7(T) for T in achieved_tcp_T], axis=0)

        pos_err_m = np.zeros((len(ts),), dtype=np.float64)
        rot_err_rad = np.zeros((len(ts),), dtype=np.float64)
        for i in range(len(ts)):
            p_err, r_err = pose_error_components(target_tcp_T[i], achieved_tcp_T[i])
            pos_err_m[i] = p_err
            rot_err_rad[i] = r_err

        report_cfg = cfg["report"]
        summary = summarize_metrics(
            success=success,
            limit_margin_rad=limit_margin_rad,
            sigma_min=sigma_min,
            branch_count=branch_count,
            pos_err_m=pos_err_m,
            rot_err_rad=rot_err_rad,
            near_limit_threshold_rad=float(report_cfg["near_limit_threshold_rad"]),
            near_singular_threshold=float(report_cfg["near_singular_threshold"]),
        )
        summary.update(
            {
                "demo_name": demo_name,
                "demo_json": str(demo_json),
                "coordinate_frame": data.get("coordinate_frame", "unknown"),
                "urdf_path": urdf_path,
                "ee_frame_name": ee_frame_name,
                "solve_frame": solve_frame,
            }
        )
        if q_selected.shape[0] > 1:
            dq = np.diff(q_selected, axis=0)
            summary["joint_step_max_rad"] = float(np.max(np.abs(dq)))
            summary["joint_step_mean_rad"] = float(np.mean(np.linalg.norm(dq, axis=1)))
        else:
            summary["joint_step_max_rad"] = 0.0
            summary["joint_step_mean_rad"] = 0.0
        summary["post_stabilize_fix_count"] = int(ik_out.get("post_stabilize_fix_count", 0))
        summary["branch_switch_count_raw"] = int(ik_out.get("branch_count_raw", branch_count))
        summary["home_q_rad"] = np.asarray(ik_out.get("home_q_rad", []), dtype=np.float64).tolist()

        rm_baseline = run_rm_baseline_ik(
            cfg=cfg,
            target_T_list=target_solve_T,
            q_init=q_selected[0],
        )
        if rm_baseline is not None:
            summary["rm_baseline"] = {
                k: v for k, v in rm_baseline.items() if k != "q_track"
            }

        out_paths = save_demo_outputs(
            out_dir=output_dir,
            demo_name=demo_name,
            summary=summary,
            timestamps=ts,
            target_tcp_pose7=target_tcp_pose7,
            achieved_tcp_pose7=achieved_tcp_pose7,
            q_selected=q_selected,
            success=success,
            limit_margin_rad=limit_margin_rad,
            sigma_min=sigma_min,
            pos_err_m=pos_err_m,
            rot_err_rad=rot_err_rad,
            branch_flags=branch_flags,
            local_cost=local_cost,
            joint_names=ik_out["joint_names"],
            q_selected_raw=q_selected_raw,
        )
        summary["artifacts"] = out_paths

        print(
            f"[DONE] {demo_name}: success_rate={summary['success_rate']:.3f}, "
            f"branch={summary['branch_switch_count']}, "
            f"limit_min={summary['limit_margin_rad_min']:.4f} rad, "
            f"sigma_min={summary['sigma_min_min']:.5f}"
        )
        global_rows.append(summary)

    global_path = save_global_summary(output_dir, global_rows)
    print(f"[GLOBAL] summary saved: {global_path}")
    print(f"[GLOBAL] demos processed: {len(global_rows)}")


if __name__ == "__main__":
    main()
