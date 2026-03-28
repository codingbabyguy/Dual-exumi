"""
Visualize aligned ARCap poses from demos/*/aligned_arcap_poses.json.

This script reuses the same plotting helpers from visualize_pose.py so the
visual style and outputs stay consistent.

Usage:
  1) Edit DEMOS_FOLDER below.
  2) Run: python3 visualize_aligned_poses.py
"""

import json
import os
import glob
import numpy as np
import cv2
import av

# Reuse the existing visualization style/functions.
from visualize_pose import (
    quats_to_rotmats,
    save_static_image,
    save_animated_video,
    save_trajectory_stats,
)


# ================= visualization config =================
DEMOS_FOLDER = "/home/icrlab/tactile_work_Wy/data/simple-5.3/batch_2/demos"
USE_ABSOLUTE = False
OUTPUT_FORMAT = "mp4"  # "mp4", "gif", "both"
SKIP_VIDEO = False
MAKE_COMPARISON = True
TARGET_SUMMARY_SECONDS = 60
# ========================================================


def compute_bounds(points):
    """Match visualize_pose bound logic to keep the same visual effect."""
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid_pts = np.median(points, axis=0)
    return (mid_pts - max_range, mid_pts + max_range)


def load_aligned_pose_json(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    poses = data.get("pose", [])
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Invalid pose array shape in {json_path}: {arr.shape}")

    pts = arr[:, :3]
    quats = arr[:, 3:7]
    rotmats = quats_to_rotmats(quats)
    return pts, rotmats


def get_video_info(video_path):
    with av.open(video_path, "r") as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        n_frames = int(stream.frames)
        width = int(stream.width)
        height = int(stream.height)
    return fps, n_frames, width, height


def _read_next_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _resize_to_height(frame, target_h):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def build_side_by_side_video(raw_video_path, pose_video_path, output_path, fps):
    cap_raw = cv2.VideoCapture(raw_video_path)
    cap_pose = cv2.VideoCapture(pose_video_path)
    if not cap_raw.isOpened() or not cap_pose.isOpened():
        if cap_raw.isOpened():
            cap_raw.release()
        if cap_pose.isOpened():
            cap_pose.release()
        raise OSError("Failed to open raw/pose video for side-by-side merge")

    raw_h = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_w = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    pose_h = int(cap_pose.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_w = int(cap_pose.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = max(raw_h, pose_h)

    out_raw_w = int(round(raw_w * target_h / max(1, raw_h)))
    out_pose_w = int(round(pose_w * target_h / max(1, pose_h)))
    out_w = out_raw_w + out_pose_w
    out_h = target_h

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap_raw.release()
        cap_pose.release()
        raise OSError(f"Cannot create writer: {output_path}")

    written = 0
    while True:
        f_raw = _read_next_frame(cap_raw)
        f_pose = _read_next_frame(cap_pose)
        if f_raw is None or f_pose is None:
            break

        f_raw = _resize_to_height(f_raw, target_h)
        f_pose = _resize_to_height(f_pose, target_h)
        merged = np.hstack([f_raw, f_pose])

        cv2.putText(
            merged,
            "GoPro raw_video",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            merged,
            "Aligned Pose Animation",
            (out_raw_w + 20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(merged)
        written += 1

    cap_raw.release()
    cap_pose.release()
    writer.release()
    return written


def build_summary_video(comparison_videos, output_path, target_seconds=60):
    if not comparison_videos:
        return False, "no comparison clips"

    cap0 = cv2.VideoCapture(comparison_videos[0])
    if not cap0.isOpened():
        return False, "cannot open first comparison clip"
    out_fps = float(cap0.get(cv2.CAP_PROP_FPS))
    out_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        return False, f"cannot create writer: {output_path}"

    max_frames = int(round(target_seconds * out_fps))
    written = 0
    for clip in comparison_videos:
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            continue
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        cap.release()
        if written >= max_frames:
            break

    writer.release()
    if written == 0:
        return False, "summary has 0 frame"
    return True, f"written {written} frames @ {out_fps:.3f}fps"


def process_demo(demo_dir):
    aligned_json = os.path.join(demo_dir, "aligned_arcap_poses.json")
    raw_video = os.path.join(demo_dir, "raw_video.mp4")
    if not os.path.isfile(aligned_json):
        return False, "missing aligned_arcap_poses.json"
    if not os.path.isfile(raw_video):
        return False, "missing raw_video.mp4"

    pts, rotmats = load_aligned_pose_json(aligned_json)
    if len(pts) == 0:
        return False, "empty pose list"

    video_fps, video_frames, _, _ = get_video_info(raw_video)

    bounds = compute_bounds(pts)

    print("\n" + "=" * 70)
    print(f"[DEMO] {demo_dir}")
    print(f"[INFO] Pose count: {len(pts)}")
    print(f"[INFO] GoPro fps: {video_fps:.6f}, frames: {video_frames}")

    save_trajectory_stats(pts, demo_dir, use_absolute=USE_ABSOLUTE)
    save_static_image(pts, bounds, demo_dir, use_absolute=USE_ABSOLUTE)

    pose_mp4_path = None
    if not SKIP_VIDEO:
        if OUTPUT_FORMAT in ("mp4", "both"):
            pose_mp4_path = save_animated_video(
                pts,
                rotmats,
                bounds,
                demo_dir,
                use_absolute=USE_ABSOLUTE,
                output_format="mp4",
                video_fps=video_fps,
                output_basename="trajectory_animation_aligned",
            )
        if OUTPUT_FORMAT in ("gif", "both"):
            save_animated_video(
                pts,
                rotmats,
                bounds,
                demo_dir,
                use_absolute=USE_ABSOLUTE,
                output_format="gif",
                output_basename="trajectory_animation_aligned",
            )

    comparison_path = None
    if MAKE_COMPARISON:
        if pose_mp4_path is None:
            pose_mp4_path = os.path.join(demo_dir, "trajectory_animation_aligned.mp4")
        if os.path.isfile(pose_mp4_path):
            comparison_path = os.path.join(demo_dir, "comparison_raw_vs_pose.mp4")
            written = build_side_by_side_video(raw_video, pose_mp4_path, comparison_path, video_fps)
            print(f"[INFO] comparison clip saved ({written} frames): {comparison_path}")
        else:
            print("[WARN] pose mp4 not found, skip comparison merge")

    return True, {"comparison_path": comparison_path}


def main():
    print("\n" + "=" * 70)
    print("  Visualize aligned_arcap_poses from demos")
    print("=" * 70)
    print(f"[CONFIG] demos folder: {DEMOS_FOLDER}")
    print(f"[CONFIG] use absolute: {USE_ABSOLUTE}")
    print(f"[CONFIG] output format: {OUTPUT_FORMAT}")
    print(f"[CONFIG] skip video: {SKIP_VIDEO}")

    if not os.path.isdir(DEMOS_FOLDER):
        print(f"[ERROR] demos folder not found: {DEMOS_FOLDER}")
        return

    demo_dirs = sorted([p for p in glob.glob(os.path.join(DEMOS_FOLDER, "demo_*")) if os.path.isdir(p)])
    print(f"[INFO] Found {len(demo_dirs)} demo dirs")

    ok_count = 0
    skipped = []
    comparison_videos = []
    for demo_dir in demo_dirs:
        try:
            ok, msg = process_demo(demo_dir)
            if ok:
                ok_count += 1
                if isinstance(msg, dict) and msg.get("comparison_path"):
                    comparison_videos.append(msg["comparison_path"])
            else:
                skipped.append((demo_dir, msg))
        except Exception as e:
            skipped.append((demo_dir, str(e)))

    if MAKE_COMPARISON:
        summary_path = os.path.join(DEMOS_FOLDER, "comparison_summary_around_1min.mp4")
        ok, info = build_summary_video(
            comparison_videos=comparison_videos,
            output_path=summary_path,
            target_seconds=TARGET_SUMMARY_SECONDS,
        )
        if ok:
            print(f"[INFO] summary comparison saved: {summary_path} ({info})")
        else:
            print(f"[WARN] summary comparison not generated: {info}")

    print("\n" + "=" * 70)
    print(f"[DONE] Processed {ok_count}/{len(demo_dirs)} demos")
    if skipped:
        print("[SKIPPED]")
        for demo_dir, reason in skipped:
            print(f"  - {demo_dir}: {reason}")
    print("=" * 70)


if __name__ == "__main__":
    main()
