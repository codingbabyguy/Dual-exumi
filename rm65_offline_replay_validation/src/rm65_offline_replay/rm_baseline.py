from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def _resolve_rm_model_enums(api_mod, arm_model: str, force_type: str):
    arm_map = {
        "RM_65": "RM_MODEL_RM_65_E",
        "RM_75": "RM_MODEL_RM_75_E",
        "RML_63": "RM_MODEL_RM_63_II_E",
        "ECO_65": "RM_MODEL_ECO_65_E",
    }
    force_map = {
        "RM_B": "RM_MODEL_RM_B_E",
    }
    arm_key = arm_map.get(arm_model, arm_map["RM_65"])
    force_key = force_map.get(force_type, force_map["RM_B"])
    arm_enum = getattr(api_mod.rm_robot_arm_model_e, arm_key)
    force_enum = getattr(api_mod.rm_force_type_e, force_key)
    return arm_enum, force_enum


def run_rm_baseline_ik(
    cfg: dict,
    target_T_list: list[np.ndarray],
    q_init: np.ndarray,
) -> dict | None:
    """
    Optional baseline using RM_API2 Algo IK.
    This runs without real robot connection, but requires RM SDK runtime.
    """
    rm_cfg = cfg.get("rm_baseline", {})
    if not bool(rm_cfg.get("enable", False)):
        return None

    rm_api_python_dir = str(rm_cfg.get("rm_api_python_dir", "")).strip()
    if not rm_api_python_dir:
        return {"enabled": True, "available": False, "reason": "rm_api_python_dir is empty"}
    rm_api_python_dir = str(Path(rm_api_python_dir).expanduser().resolve())
    if rm_api_python_dir not in sys.path:
        sys.path.append(rm_api_python_dir)

    try:
        from Robotic_Arm import rm_robot_interface as api_mod
    except Exception as e:
        return {
            "enabled": True,
            "available": False,
            "reason": f"import RM_API2 failed: {type(e).__name__}: {e}",
        }

    try:
        arm_enum, force_enum = _resolve_rm_model_enums(
            api_mod=api_mod,
            arm_model=str(rm_cfg.get("arm_model", "RM_65")),
            force_type=str(rm_cfg.get("force_type", "RM_B")),
        )
        algo = api_mod.Algo(arm_enum, force_enum)
    except Exception as e:
        return {
            "enabled": True,
            "available": False,
            "reason": f"init Algo failed: {type(e).__name__}: {e}",
        }

    q_ref = np.asarray(q_init, dtype=np.float64).copy()
    ok_flags = []
    pos_err = []
    rot_err = []
    q_track = []

    for T in target_T_list:
        pose = np.zeros(6, dtype=np.float64)
        pose[:3] = T[:3, 3]
        pose[3:] = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
        try:
            params = api_mod.rm_inverse_kinematics_params_t(q_ref.tolist(), pose.tolist(), 1)
            ret, q_out = algo.rm_algo_inverse_kinematics(params)
            q_out = np.asarray(q_out, dtype=np.float64)
            q_track.append(q_out)
            ok = int(ret) == 0
            ok_flags.append(1 if ok else 0)
            q_ref = q_out
            if ok:
                # optional FK check in SDK, fallback to zeros if unavailable
                fk_pose = algo.rm_algo_forward_kinematics(q_out.tolist(), 1)
                fk_pose = np.asarray(fk_pose, dtype=np.float64)
                dp = float(np.linalg.norm(fk_pose[:3] - pose[:3]))
                dr = float(np.linalg.norm(fk_pose[3:] - pose[3:]))
            else:
                dp = np.nan
                dr = np.nan
            pos_err.append(dp)
            rot_err.append(dr)
        except Exception:
            ok_flags.append(0)
            pos_err.append(np.nan)
            rot_err.append(np.nan)
            q_track.append(q_ref.copy())

    return {
        "enabled": True,
        "available": True,
        "success_rate": float(np.mean(ok_flags)) if ok_flags else 0.0,
        "failed_frames": int(np.sum(1 - np.asarray(ok_flags, dtype=np.int32))),
        "pos_err_mean_m": float(np.nanmean(pos_err)) if len(pos_err) > 0 else np.nan,
        "rot_err_mean_rad": float(np.nanmean(rot_err)) if len(rot_err) > 0 else np.nan,
        "q_track": np.asarray(q_track, dtype=np.float64),
    }

