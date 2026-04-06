from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qx,qy,qz,qw] to 4x4 transform."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    if pose7.shape != (7,):
        raise ValueError(f"pose7 shape must be (7,), got {pose7.shape}")
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = pose7[:3]
    out[:3, :3] = R.from_quat(pose7[3:]).as_matrix()
    return out


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform to [x,y,z,qx,qy,qz,qw]."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T shape must be (4,4), got {T.shape}")
    out = np.zeros(7, dtype=np.float64)
    out[:3] = T[:3, 3]
    out[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return out


def xyz_rpy_to_matrix(xyz: list[float], rpy_rad: list[float]) -> np.ndarray:
    """Build 4x4 transform from translation and XYZ-fixed rpy (rad)."""
    if len(xyz) != 3 or len(rpy_rad) != 3:
        raise ValueError("xyz and rpy_rad must both have length 3")
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(rpy_rad, dtype=np.float64)).as_matrix()
    return T


def transform_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T shape must be (4,4), got {T.shape}")
    R_part = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_part.T
    out[:3, 3] = -(R_part.T @ t)
    return out


def pose_error_components(T_target: np.ndarray, T_actual: np.ndarray) -> tuple[float, float]:
    """Return translation error (m) and rotation error (rad)."""
    T_target = np.asarray(T_target, dtype=np.float64)
    T_actual = np.asarray(T_actual, dtype=np.float64)
    t_err = float(np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3]))
    R_err = T_actual[:3, :3].T @ T_target[:3, :3]
    r_err = float(np.linalg.norm(R.from_matrix(R_err).as_rotvec()))
    return t_err, r_err


def compose(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose transforms."""
    return np.asarray(T_a_b, dtype=np.float64) @ np.asarray(T_b_c, dtype=np.float64)

