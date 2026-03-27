"""
Generate side-by-side-in-time preview videos for aligned demos.

Output video layout:
- Top half: pose trajectory rendering (2D projection)
- Bottom half: aligned demo video frames

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

import click
import cv2
import numpy as np


def find_demo_dirs(input_dir: pathlib.Path):
    return sorted([x.parent for x in input_dir.glob("demo_*/raw_video.mp4")])


def load_pose_array(aligned_json_path: pathlib.Path):
    data = json.load(open(str(aligned_json_path), "r"))
    if "pose" not in data:
        raise KeyError(f"Missing key 'pose' in {aligned_json_path}")
    pose = np.asarray(data["pose"], dtype=float)
    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"Invalid pose shape {pose.shape} in {aligned_json_path}")
    return pose[:, :3]


def compute_projection_bounds(points_xy: np.ndarray, margin_ratio: float = 0.08):
    min_xy = np.min(points_xy, axis=0)
    max_xy = np.max(points_xy, axis=0)
    span = max_xy - min_xy
    span[span < 1e-8] = 1e-8

    # Keep isotropic scale to avoid distortion.
    max_span = float(np.max(span))
    cx, cy = (min_xy + max_xy) * 0.5
    half = 0.5 * max_span * (1.0 + margin_ratio * 2.0)

    return np.array([cx - half, cy - half]), np.array([cx + half, cy + half])


def world_to_canvas(points_xy: np.ndarray, canvas_w: int, canvas_h: int, min_xy: np.ndarray, max_xy: np.ndarray):
    span = max_xy - min_xy
    span[span < 1e-8] = 1e-8

    norm = (points_xy - min_xy[None, :]) / span[None, :]
    x = np.clip((norm[:, 0] * (canvas_w - 1)).astype(np.int32), 0, canvas_w - 1)
    y = np.clip(((1.0 - norm[:, 1]) * (canvas_h - 1)).astype(np.int32), 0, canvas_h - 1)
    return np.stack([x, y], axis=1)


def draw_top_panel(points_xy: np.ndarray, frame_idx: int, top_w: int, top_h: int):
    panel = np.full((top_h, top_w, 3), 245, dtype=np.uint8)

    # Grid lines.
    for gx in range(1, 4):
        x = int(gx * top_w / 4)
        cv2.line(panel, (x, 0), (x, top_h - 1), (225, 225, 225), 1)
    for gy in range(1, 4):
        y = int(gy * top_h / 4)
        cv2.line(panel, (0, y), (top_w - 1, y), (225, 225, 225), 1)

    min_xy, max_xy = compute_projection_bounds(points_xy)
    pix = world_to_canvas(points_xy, top_w, top_h, min_xy, max_xy)

    # Full trajectory in light gray.
    if len(pix) >= 2:
        cv2.polylines(panel, [pix.reshape(-1, 1, 2)], False, (180, 180, 180), 1, cv2.LINE_AA)

    # Past trajectory in green.
    upto = min(max(frame_idx + 1, 1), len(pix))
    past = pix[:upto]
    if len(past) >= 2:
        cv2.polylines(panel, [past.reshape(-1, 1, 2)], False, (55, 155, 55), 2, cv2.LINE_AA)

    # Current point in red.
    cur = pix[upto - 1]
    cv2.circle(panel, (int(cur[0]), int(cur[1])), 5, (50, 50, 220), -1, cv2.LINE_AA)

    cv2.putText(panel, "Aligned Pose Trajectory (XY)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx}", (12, top_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)

    return panel


def render_demo(demo_dir: pathlib.Path, output_name: str, top_ratio: float):
    video_path = demo_dir.joinpath("raw_video.mp4")
    aligned_json = demo_dir.joinpath("aligned_arcap_poses.json")

    if not aligned_json.is_file() or not video_path.is_file():
        return False, "missing raw_video.mp4 or aligned_arcap_poses.json"

    points_xyz = load_pose_array(aligned_json)
    n_pose = len(points_xyz)
    if n_pose <= 0:
        return False, "empty pose sequence"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"cannot open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_w <= 0 or video_h <= 0:
        cap.release()
        return False, f"invalid video shape: {video_w}x{video_h}"

    top_h = max(120, int(video_h * top_ratio))
    bot_h = video_h
    out_w = video_w
    out_h = top_h + bot_h

    out_path = demo_dir.joinpath(output_name)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (out_w, out_h),
    )

    # Use the min length to guarantee temporal alignment.
    n_written = 0
    for i in range(n_pose):
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[0] != bot_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, bot_h), interpolation=cv2.INTER_LINEAR)

        top_panel = draw_top_panel(points_xyz[:, :2], i, out_w, top_h)
        merged = np.vstack([top_panel, frame])
        writer.write(merged)
        n_written += 1

    cap.release()
    writer.release()

    if n_written <= 0:
        return False, "no frames written"

    return True, f"saved={out_path}, frames={n_written}"


@click.command()
@click.option("-i", "--input_dir", required=True, help="demos directory path")
@click.option("--output_name", default="pose_demo_preview.mp4", show_default=True, help="output mp4 filename per demo")
@click.option("--top_ratio", default=1.0, type=float, show_default=True, help="top panel height = top_ratio * video_height")
def main(input_dir, output_name, top_ratio):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    demo_dirs = find_demo_dirs(input_dir)
    print(f"Found {len(demo_dirs)} demo dirs")

    if len(demo_dirs) == 0:
        print("No demo dirs found.")
        return

    ok_count = 0
    for demo_dir in demo_dirs:
        ok, msg = render_demo(demo_dir, output_name=output_name, top_ratio=top_ratio)
        tag = "OK" if ok else "SKIP"
        print(f"[{tag}] {demo_dir.name}: {msg}")
        if ok:
            ok_count += 1

    print(f"Done. generated={ok_count}/{len(demo_dirs)}")


if __name__ == "__main__":
    main()
