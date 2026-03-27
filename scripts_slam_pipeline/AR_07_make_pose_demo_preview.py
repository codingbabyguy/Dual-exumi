"""
Generate a single multimodal preview MP4 for each aligned demo.

Output grid (2x3):
- Top-left: aligned GoPro video (raw_video.mp4)
- Top-middle: 3D VR pose trajectory visualization
- Top-right: gripper angle/width timeline
- Bottom-left: tactile_left.mp4
- Bottom-middle: tactile_right.mp4
- Bottom-right: status/info panel

Usage:
python scripts_slam_pipeline/AR_07_make_pose_demo_preview.py \
    --input_dir data/<session>/batch_x/demos
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
from typing import Optional, Tuple

import click
import cv2
import numpy as np


def find_demo_dirs(input_dir: pathlib.Path):
    return sorted([x.parent for x in input_dir.glob("demo_*/raw_video.mp4")])


def load_aligned_arrays(aligned_json_path: pathlib.Path):
    data = json.load(open(str(aligned_json_path), "r"))
    if "pose" not in data:
        raise KeyError(f"Missing key 'pose' in {aligned_json_path}")
    pose = np.asarray(data["pose"], dtype=float)
    if pose.ndim != 2 or pose.shape[1] < 7:
        raise ValueError(f"Invalid pose shape {pose.shape} in {aligned_json_path}")
    if "width" in data:
        width = np.asarray(data["width"], dtype=float)
    elif "angle" in data:
        width = np.asarray(data["angle"], dtype=float)
    else:
        width = np.zeros((pose.shape[0],), dtype=float)
    if width.ndim != 1:
        width = width.reshape(-1)
    n = min(len(pose), len(width))
    if n <= 0:
        raise ValueError(f"Empty aligned sequence in {aligned_json_path}")
    return pose[:n], width[:n]


def compute_projection_bounds(points_xyz: np.ndarray, margin_ratio: float = 0.08):
    min_xyz = np.min(points_xyz, axis=0)
    max_xyz = np.max(points_xyz, axis=0)
    span = max_xyz - min_xyz
    span[span < 1e-8] = 1e-8

    max_span = float(np.max(span))
    center = (min_xyz + max_xyz) * 0.5
    half = 0.5 * max_span * (1.0 + margin_ratio * 2.0)

    min_b = center - half
    max_b = center + half
    return min_b, max_b


def quat_to_rotmat_xyzw(q: np.ndarray):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def project_3d_to_2d(points_xyz: np.ndarray, canvas_w: int, canvas_h: int, min_b: np.ndarray, max_b: np.ndarray):
    # Fixed view matrix for stable 3D-looking projection.
    az = np.deg2rad(-55.0)
    el = np.deg2rad(25.0)
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=float,
    )
    view = rx @ rz

    center = 0.5 * (min_b + max_b)
    scale = float(np.max(max_b - min_b))
    if scale < 1e-8:
        scale = 1.0
    pts = (points_xyz - center[None, :]) / scale
    pv = (view @ pts.T).T

    pv_xy = pv[:, :2]
    min_xy = np.min(pv_xy, axis=0)
    max_xy = np.max(pv_xy, axis=0)
    span = max_xy - min_xy
    span[span < 1e-8] = 1e-8
    norm = (pv_xy - min_xy[None, :]) / span[None, :]
    x = np.clip((norm[:, 0] * (canvas_w - 1)).astype(np.int32), 0, canvas_w - 1)
    y = np.clip(((1.0 - norm[:, 1]) * (canvas_h - 1)).astype(np.int32), 0, canvas_h - 1)
    return np.stack([x, y], axis=1), view, center, scale


def draw_pose_panel(points_xyz: np.ndarray, pose_quat: np.ndarray, frame_idx: int, panel_w: int, panel_h: int):
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    for gx in range(1, 4):
        x = int(gx * panel_w / 4)
        cv2.line(panel, (x, 0), (x, panel_h - 1), (225, 225, 225), 1)
    for gy in range(1, 4):
        y = int(gy * panel_h / 4)
        cv2.line(panel, (0, y), (panel_w - 1, y), (225, 225, 225), 1)

    min_b, max_b = compute_projection_bounds(points_xyz)
    pix, view, center, scale = project_3d_to_2d(points_xyz, panel_w, panel_h, min_b, max_b)

    if len(pix) >= 2:
        cv2.polylines(panel, [pix.reshape(-1, 1, 2)], False, (180, 180, 180), 1, cv2.LINE_AA)

    upto = min(max(frame_idx + 1, 1), len(pix))
    past = pix[:upto]
    if len(past) >= 2:
        cv2.polylines(panel, [past.reshape(-1, 1, 2)], False, (55, 155, 55), 2, cv2.LINE_AA)

    cur = pix[upto - 1]
    cv2.circle(panel, (int(cur[0]), int(cur[1])), 5, (50, 50, 220), -1, cv2.LINE_AA)

    # Draw local orientation axes at current pose.
    q = pose_quat[upto - 1]
    if np.all(np.isfinite(q)):
        nq = float(np.linalg.norm(q))
        if nq > 1e-8:
            q = q / nq
            rot = quat_to_rotmat_xyzw(q)
            pos = points_xyz[upto - 1]
            axis_len = 0.08 * scale
            basis = np.eye(3, dtype=float) * axis_len
            ends = pos[None, :] + (rot @ basis).T

            pts = np.vstack([pos[None, :], ends])
            pts = (pts - center[None, :]) / scale
            pts_v = (view @ pts.T).T[:, :2]
            mn = np.min(pts_v, axis=0)
            mx = np.max(pts_v, axis=0)
            sp = mx - mn
            sp[sp < 1e-8] = 1e-8
            uv = (pts_v - mn[None, :]) / sp[None, :]
            uvx = np.clip((uv[:, 0] * (panel_w - 1)).astype(np.int32), 0, panel_w - 1)
            uvy = np.clip(((1.0 - uv[:, 1]) * (panel_h - 1)).astype(np.int32), 0, panel_h - 1)
            p0 = (int(uvx[0]), int(uvy[0]))
            colors = [(40, 40, 220), (40, 180, 40), (220, 120, 40)]
            for k in range(3):
                p1 = (int(uvx[k + 1]), int(uvy[k + 1]))
                cv2.line(panel, p0, p1, colors[k], 2, cv2.LINE_AA)

    cv2.putText(panel, "VR Pose 3D Trajectory", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx}", (12, panel_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)

    return panel


def draw_scalar_panel(values: np.ndarray, frame_idx: int, panel_w: int, panel_h: int, title: str):
    panel = np.full((panel_h, panel_w, 3), 250, dtype=np.uint8)

    x0, y0 = 56, 24
    x1, y1 = panel_w - 20, panel_h - 36
    cv2.rectangle(panel, (x0, y0), (x1, y1), (215, 215, 215), 1)

    if len(values) <= 1:
        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
        return panel

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    xs = np.linspace(x0, x1, len(values)).astype(np.int32)
    yn = (values - vmin) / (vmax - vmin)
    ys = (y1 - yn * (y1 - y0)).astype(np.int32)
    curve = np.stack([xs, ys], axis=1)

    if len(curve) >= 2:
        cv2.polylines(panel, [curve.reshape(-1, 1, 2)], False, (180, 180, 180), 1, cv2.LINE_AA)

    upto = min(max(frame_idx + 1, 1), len(values))
    cur_curve = curve[:upto]
    if len(cur_curve) >= 2:
        cv2.polylines(panel, [cur_curve.reshape(-1, 1, 2)], False, (200, 90, 40), 2, cv2.LINE_AA)
    cur = cur_curve[-1]
    cv2.circle(panel, (int(cur[0]), int(cur[1])), 5, (40, 40, 220), -1, cv2.LINE_AA)

    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, f"min={vmin:.4f} max={vmax:.4f}", (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    return panel


def read_frame_or_placeholder(cap: Optional[cv2.VideoCapture], panel_w: int, panel_h: int, text: str):
    panel = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
    if cap is not None:
        ok, frame = cap.read()
        if ok and frame is not None:
            if frame.shape[1] != panel_w or frame.shape[0] != panel_h:
                frame = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
            return True, frame
    cv2.putText(panel, text, (18, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    return False, panel


def open_video_if_exists(path: pathlib.Path):
    if not path.is_file():
        return None, 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, n


def render_demo(demo_dir: pathlib.Path, output_name: str):
    gopro_path = demo_dir.joinpath("raw_video.mp4")
    tactile_left_path = demo_dir.joinpath("tactile_left.mp4")
    tactile_right_path = demo_dir.joinpath("tactile_right.mp4")
    aligned_json = demo_dir.joinpath("aligned_arcap_poses.json")

    if not aligned_json.is_file() or not gopro_path.is_file():
        return False, "missing aligned_arcap_poses.json or raw_video.mp4"

    pose_7d, width = load_aligned_arrays(aligned_json)
    points_xyz = pose_7d[:, :3]
    pose_quat = pose_7d[:, 3:7]
    n_pose = len(pose_7d)

    cap_gopro, n_gopro = open_video_if_exists(gopro_path)
    if cap_gopro is None:
        return False, f"cannot open gopro: {gopro_path}"
    cap_left, n_left = open_video_if_exists(tactile_left_path)
    cap_right, n_right = open_video_if_exists(tactile_right_path)

    fps = cap_gopro.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    video_w = int(cap_gopro.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap_gopro.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_w <= 0 or video_h <= 0:
        cap_gopro.release()
        if cap_left is not None:
            cap_left.release()
        if cap_right is not None:
            cap_right.release()
        return False, f"invalid video shape: {video_w}x{video_h}"

    cell_w = video_w
    cell_h = video_h
    out_w = cell_w * 3
    out_h = cell_h * 2

    out_path = demo_dir.joinpath(output_name)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (out_w, out_h),
    )

    counts = [n_pose, n_gopro]
    if cap_left is not None and n_left > 0:
        counts.append(n_left)
    if cap_right is not None and n_right > 0:
        counts.append(n_right)
    n_total = int(min(counts)) if counts else 0

    if n_total <= 0:
        cap_gopro.release()
        if cap_left is not None:
            cap_left.release()
        if cap_right is not None:
            cap_right.release()
        writer.release()
        return False, "no overlapping frames to render"

    n_written = 0
    for i in range(n_total):
        _, gopro_frame = read_frame_or_placeholder(cap_gopro, cell_w, cell_h, "No GoPro")
        _, left_frame = read_frame_or_placeholder(cap_left, cell_w, cell_h, "No tactile_left")
        _, right_frame = read_frame_or_placeholder(cap_right, cell_w, cell_h, "No tactile_right")

        pose_panel = draw_pose_panel(points_xyz, pose_quat, i, cell_w, cell_h)
        angle_panel = draw_scalar_panel(width, i, cell_w, cell_h, "Gripper Angle/Width")

        info_panel = np.full((cell_h, cell_w, 3), 20, dtype=np.uint8)
        cv2.putText(info_panel, "Aligned Multimodal Preview", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(info_panel, f"demo: {demo_dir.name}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(info_panel, f"frame: {i+1}/{n_total}", (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(info_panel, f"fps: {fps:.2f}", (18, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(info_panel, "Panels:", (18, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(info_panel, "TL GoPro | TM 3D Pose | TR Angle", (18, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(info_panel, "BL tactile_left | BM tactile_right", (18, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        row_top = np.hstack([gopro_frame, pose_panel, angle_panel])
        row_bottom = np.hstack([left_frame, right_frame, info_panel])
        merged = np.vstack([row_top, row_bottom])

        writer.write(merged)
        n_written += 1

    cap_gopro.release()
    if cap_left is not None:
        cap_left.release()
    if cap_right is not None:
        cap_right.release()
    writer.release()

    if n_written <= 0:
        return False, "no frames written"

    return True, f"saved={out_path}, frames={n_written}"


@click.command()
@click.option("-i", "--input_dir", required=True, help="demos directory path")
@click.option("--output_name", default="multimodal_preview.mp4", show_default=True, help="output mp4 filename per demo")
def main(input_dir, output_name):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    demo_dirs = find_demo_dirs(input_dir)
    print(f"Found {len(demo_dirs)} demo dirs")

    if len(demo_dirs) == 0:
        print("No demo dirs found.")
        return

    ok_count = 0
    for demo_dir in demo_dirs:
        ok, msg = render_demo(demo_dir, output_name=output_name)
        tag = "OK" if ok else "SKIP"
        print(f"[{tag}] {demo_dir.name}: {msg}")
        if ok:
            ok_count += 1

    print(f"Done. generated={ok_count}/{len(demo_dirs)}")


if __name__ == "__main__":
    main()
