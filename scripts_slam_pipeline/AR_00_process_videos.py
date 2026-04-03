"""
AR_00_process_videos.py - 视频预处理模块

功能说明:
- 该脚本是exUMI数据处理流水线的第一步，负责预处理采集的原始视频文件
- 主要任务：将raw_videos中的MP4文件重新组织到demos文件夹中，并按相机序列号和开始时间命名
- 为后续的SLAM流程准备标准化的视频数据格式

使用方式:
    python scripts_slam_pipeline/AR_00_process_videos.py <session_dir>

数据流向:
    raw_videos/*.mp4 → 提取元数据 → 按规则重命名 → demos/demo_<camera>_<timestamp>/raw_video.mp4

注意事项:
- 假设视频文件位于 <session_dir>/raw_videos 目录下
- 需要触觉数据文件夹 tactile_* 存在
- 使用ExifTool读取视频元数据，需要安装exiftool
"""
# %%
import sys
import os

# 设置项目根目录和Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
import json
from datetime import datetime

import cv2
from exiftool import ExifToolHelper

from umi.common.timecode_util import mp4_get_start_datetime

from scripts_slam_pipeline.utils.misc import get_single_path


def _sanitize_marks(raw_marks):
    if not isinstance(raw_marks, list):
        return []
    out = []
    for item in raw_marks:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _sanitize_intervals(raw_intervals):
    if not isinstance(raw_intervals, list):
        return []
    out = []
    for item in raw_intervals:
        if not isinstance(item, dict):
            continue
        try:
            start_ts = float(item["start_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        end_raw = item.get("end_timestamp", None)
        if end_raw is None:
            end_ts = None
        else:
            try:
                end_ts = float(end_raw)
            except (TypeError, ValueError):
                continue
        if end_ts is not None and end_ts <= start_ts:
            continue
        out.append(
            {
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
            }
        )
    out = sorted(out, key=lambda x: x["start_timestamp"])
    for idx, item in enumerate(out):
        item["segment_index"] = idx
        item["is_calibration"] = bool(idx == 0)
    return out


def _build_intervals_from_start_end(starts, ends):
    if not starts or not ends:
        return []
    intervals = []
    end_idx = 0
    for start_ts in starts:
        while end_idx < len(ends) and ends[end_idx] <= start_ts:
            end_idx += 1
        if end_idx >= len(ends):
            break
        end_ts = ends[end_idx]
        end_idx += 1
        intervals.append(
            {
                "start_timestamp": float(start_ts),
                "end_timestamp": float(end_ts),
            }
        )
    return intervals


def _build_intervals_from_legacy_starts(starts):
    if not starts:
        return []
    intervals = []
    for idx, start_ts in enumerate(starts):
        end_ts = starts[idx + 1] if idx + 1 < len(starts) else None
        intervals.append(
            {
                "start_timestamp": float(start_ts),
                "end_timestamp": float(end_ts) if end_ts is not None else None,
            }
        )
    return intervals


def _load_capture_intervals(session_path: pathlib.Path):
    marks_path = session_path.joinpath("capture_marks.json")
    if not marks_path.is_file():
        return [], None, None
    with open(marks_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    has_new_schema = (
        "trajectory_intervals" in payload
        or "trajectory_end_timestamps" in payload
    )

    intervals = _sanitize_intervals(payload.get("trajectory_intervals", []))
    source = "trajectory_intervals"
    if not intervals:
        start_marks = _sanitize_marks(payload.get("trajectory_start_timestamps", []))
        end_marks = _sanitize_marks(payload.get("trajectory_end_timestamps", []))
        intervals = _sanitize_intervals(_build_intervals_from_start_end(start_marks, end_marks))
        source = "trajectory_start+end_timestamps"
    if not intervals and not has_new_schema:
        start_marks = _sanitize_marks(payload.get("trajectory_start_timestamps", []))
        intervals = _sanitize_intervals(_build_intervals_from_legacy_starts(start_marks))
        source = "legacy_trajectory_start_timestamps"

    return intervals, marks_path, source


def _load_latency_value(latency_json_path):
    if latency_json_path is None:
        return None
    path = pathlib.Path(os.path.expanduser(latency_json_path)).absolute()
    if not path.is_file():
        raise FileNotFoundError(f"Latency json not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if "mean" not in data:
        raise KeyError(f"Missing 'mean' in latency json: {path}")
    return float(data["mean"])


def _time_to_str(ts_unix: float):
    return datetime.fromtimestamp(ts_unix).strftime(r"%Y.%m.%d_%H.%M.%S.%f")


def _segment_video_by_intervals(mp4_path, output_dir, cam_serial, start_ts, intervals, session_path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {mp4_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or n_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(
            f"Invalid video info fps={fps}, n_frames={n_frames}, size=({width},{height}) for {mp4_path}"
        )

    end_ts = start_ts + n_frames / fps
    valid_intervals = []
    for item in intervals:
        seg_start_raw = item["start_timestamp"]
        seg_end_raw = item.get("end_timestamp")
        seg_start = max(start_ts, seg_start_raw)
        seg_end_cap = end_ts if seg_end_raw is None else min(end_ts, seg_end_raw)
        if seg_end_cap <= seg_start:
            continue
        valid_intervals.append(
            {
                "segment_index": int(item.get("segment_index", 0)),
                "is_calibration": bool(item.get("is_calibration", False)),
                "start_timestamp": float(seg_start),
                "end_timestamp": float(seg_end_cap),
            }
        )

    if len(valid_intervals) == 0:
        cap.release()
        return []

    frame_intervals = []
    for item in valid_intervals:
        idx = item["segment_index"]
        seg_start = item["start_timestamp"]
        seg_end = item["end_timestamp"]
        start_frame = max(0, int((seg_start - start_ts) * fps))
        end_frame = min(n_frames, int((seg_end - start_ts) * fps))
        if end_frame - start_frame < 2:
            continue
        frame_intervals.append((item, idx, seg_start, seg_end, start_frame, end_frame))

    created = []
    for item, idx, seg_start, seg_end, start_frame, end_frame in frame_intervals:
        out_dname = f"demo_{cam_serial}_{_time_to_str(seg_start)}"
        this_out_dir = output_dir.joinpath(out_dname)
        if this_out_dir.exists():
            this_out_dir = output_dir.joinpath(f"{out_dname}_seg{idx:03d}")
        this_out_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = this_out_dir.joinpath("raw_video.mp4")
        writer = cv2.VideoWriter(
            str(out_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise OSError(f"Cannot create writer: {out_video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written = 0
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1
            frame_idx += 1
        writer.release()

        if written < 2:
            if out_video_path.exists():
                out_video_path.unlink()
            continue

        segment_meta = {
            "source_video": str(mp4_path.relative_to(session_path)),
            "segment_index": int(idx),
            "is_calibration": bool(item["is_calibration"]),
            "is_usable_episode": bool(not item["is_calibration"]),
            "start_timestamp": float(seg_start),
            "end_timestamp": float(seg_end),
            "source_start_frame": int(start_frame),
            "source_end_frame": int(start_frame + written),
            "written_frames": int(written),
            "fps": float(fps),
        }
        with open(this_out_dir.joinpath("segment_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(segment_meta, fp, indent=2)

        created.append(this_out_dir)

    cap.release()
    return created



# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
@click.option('--latency_json', type=str, default=None, help='latency_of_arcap.json path for projecting marks to video time')
def main(session_dir, latency_json):
    """
    视频预处理主函数
    
    参数:
        session_dir: 一个或多个会话目录路径，每个目录对应一个数据批次
        
    处理流程:
        1. 检查输入目录结构
        2. 查找所有MP4视频文件
        3. 提取视频元数据（相机序列号、开始时间）
        4. 按规则重命名和组织视频文件
        5. 创建符号链接保持原始位置访问
    """
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        
        # 硬编码的目录结构 - 符合exUMI数据采集标准
        input_dir = session.joinpath('raw_videos')      # 原始视频输入目录
        output_dir = session.joinpath('demos')          # 处理后的输出目录
        tactile_dir = get_single_path(session.glob('tactile_*'))  # 触觉数据目录

        intervals, marks_path, interval_source = _load_capture_intervals(session)
        if marks_path is not None:
            calib_count = sum(1 for item in intervals if item["is_calibration"])
            episode_count = max(0, len(intervals) - calib_count)
            print(
                f"Loaded {len(intervals)} capture intervals from {marks_path} "
                f"(source={interval_source}, calibration={calib_count}, usable_episode={episode_count})"
            )
        else:
            print("No capture_marks.json found, fallback to original AR_00 behavior.")

        latency_value = _load_latency_value(latency_json)
        if latency_value is not None and intervals:
            print(
                f"Loaded latency mean={latency_value:.6f}. "
                "Using projected video intervals = pose_interval - latency."
            )
            projected_intervals = []
            for item in intervals:
                pose_start = item["start_timestamp"]
                pose_end = item.get("end_timestamp")
                video_start = pose_start - latency_value
                video_end = None if pose_end is None else (pose_end - latency_value)
                projected_item = dict(item)
                projected_item["start_timestamp"] = video_start
                projected_item["end_timestamp"] = video_end
                projected_intervals.append(projected_item)
                if video_end is None:
                    print(
                        f"  interval#{item['segment_index']}: "
                        f"pose=({pose_start:.6f}, None) -> video=({video_start:.6f}, None)"
                    )
                else:
                    print(
                        f"  interval#{item['segment_index']}: "
                        f"pose=({pose_start:.6f}, {pose_end:.6f}) -> "
                        f"video=({video_start:.6f}, {video_end:.6f})"
                    )
            intervals = projected_intervals

        
        # 检查raw_videos目录是否存在
        if not input_dir.is_dir():
            raise FileNotFoundError(f"{input_dir.name} subdir don't exits")
            
        # 在input_dir及其所有子目录中查找MP4视频文件
        # 支持大小写不同的扩展名(.MP4和.mp4)
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        # 使用ExifTool读取视频元数据
        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                # 跳过已处理的符号链接文件
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                # 提取视频的开始时间和相机序列号
                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']

                # 如果存在采集打点，则按打点切分连续长视频
                is_special = (
                    mp4_path.name.startswith('mapping')
                    or mp4_path.name.startswith('gripper_cal')
                    or mp4_path.parent.name.startswith('gripper_cal')
                )
                if intervals and not is_special:
                    created = _segment_video_by_intervals(
                        mp4_path=mp4_path,
                        output_dir=output_dir,
                        cam_serial=cam_serial,
                        start_ts=start_date.timestamp(),
                        intervals=intervals,
                        session_path=session,
                    )
                    if created:
                        print(
                            f"Segmented {mp4_path.name} into {len(created)} demos using capture intervals."
                        )
                        continue
                    print(
                        f"No valid segment generated from intervals for {mp4_path.name}, fallback to original move."
                    )
                
                # 生成标准化的输出目录名格式: demo_<相机序列号>_<时间戳>
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # 特殊文件夹处理 - 映射和夹爪校准视频
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # 创建输出目录
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # 移动视频文件到新位置并重命名为标准名称
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # 创建符号链接回到原始位置，保持向后兼容性
                # 由于Python 3.12之前没有relative_to的walk_up参数，手动计算相对路径
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
