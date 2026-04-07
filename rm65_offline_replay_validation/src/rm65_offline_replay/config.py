from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .math3d import pose7_to_matrix, xyz_rpy_to_matrix


DEFAULT_CONFIG = {
    "robot": {
        "urdf_path": "",
        "ee_frame_name": "Link6",
        "solve_frame": "flange",  # flange | tcp
        "input_pose_represents": "tcp",  # tcp | flange
    },
    "frames": {
        "T_B_from_pose_frame": {
            "xyz": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
        "T_pose_to_tcp": {
            "xyz": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
        # RM-65 flange -> tool default from rm_65_flange.urdf.xacro
        "T_flange_to_tcp": {
            "xyz": [0.0, 0.0, 0.03103],
            "rpy_rad": [0.0, 0.0, 3.14],
        },
    },
    "ik": {
        "n_random_seeds": 6,
        "random_seed": 42,
        "max_nfev": 80,
        "pos_tol_m": 0.005,
        "rot_tol_deg": 5.0,
        "dedup_joint_tol_rad": 0.02,
        "max_candidates_per_frame": 10,
    },
    "selection": {
        "w_local_pos": 1.0,
        "w_local_rot": 0.4,
        "w_local_limit": 0.01,
        "w_local_sing": 0.01,
        "w_local_center": 0.05,
        "w_home": 0.10,
        "w_start_home": 0.40,
        "start_home_window": 30,
        "home_q_deg": [0.0, -35.0, 75.0, 0.0, 50.0, 0.0],
        "w_transition_smooth": 0.3,
        "w_transition_l2": 0.3,
        "w_transition_linf": 0.4,
        "branch_jump_rad": 0.6,
        "branch_penalty": 2.0,
        "hard_max_step_rad": 0.35,
        "hard_step_penalty": 300.0,
        "post_stabilize_passes": 2,
        "post_stabilize_dq_scale": 0.9,
        "failed_candidate_penalty": 20.0,
        # Shape prior (all optional, weight=0 means disabled)
        "w_shape_joint_range": 0.0,
        # list length=nq, each item can be [min_deg, max_deg] or null
        "joint_preferred_ranges_deg": None,
        "w_elbow_sign": 0.0,
        "elbow_joint_index": 2,
        "elbow_preferred_sign": 1.0,
        "elbow_sign_deadband_rad": 0.05,
        "w_elbow_halfspace": 0.0,
        "elbow_frame_name": "Link3",
        "elbow_halfspace_normal_xyz": [0.0, 0.0, 1.0],
        "elbow_halfspace_offset_m": 0.0,
        "elbow_halfspace_preferred_sign": 1.0,
        "elbow_halfspace_scale_m": 0.10,
        "w_wrist_flip": 0.0,
        # Supports negative index, e.g. [-2, -1] means last two joints.
        "wrist_joint_indices": [-2, -1],
        "wrist_flip_step_threshold_rad": 0.8,
        "wrist_flip_sign_epsilon_rad": 0.15,
    },
    "report": {
        "near_limit_threshold_rad": 0.08,
        "near_singular_threshold": 0.03,
    },
    "rm_baseline": {
        "enable": False,
        "rm_api_python_dir": "",
        "arm_model": "RM_65",
        "force_type": "RM_B",
    },
}


def _deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_update(DEFAULT_CONFIG, user_cfg)
    cfg["config_path"] = str(config_path)
    return cfg


def transform_from_cfg(node: dict) -> np.ndarray:
    """
    Build 4x4 transform from config node.

    Supports:
    - {"pose7": [x,y,z,qx,qy,qz,qw]}
    - {"xyz": [...], "rpy_rad": [...]}
    - {"xyz": [...], "rpy_deg": [...]}
    """
    if "pose7" in node:
        return pose7_to_matrix(np.asarray(node["pose7"], dtype=np.float64))
    xyz = node.get("xyz", [0.0, 0.0, 0.0])
    if "rpy_rad" in node:
        rpy = node["rpy_rad"]
    elif "rpy_deg" in node:
        rpy = np.deg2rad(np.asarray(node["rpy_deg"], dtype=np.float64)).tolist()
    else:
        rpy = [0.0, 0.0, 0.0]
    return xyz_rpy_to_matrix(xyz, rpy)
