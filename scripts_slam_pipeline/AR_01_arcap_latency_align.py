"""
AR_01_arcap_latency_align.py - ARCap时间延迟对齐模块

功能说明:
- 该脚本是exUMI数据处理流水线的关键步骤，负责解决VR姿态数据和视频数据的时间延迟问题
- 使用标定视频（含ArUco标记）进行时间对齐，确保多模态数据的时间同步
- 实现两种对齐算法：转向点检测算法和互相关算法

使用方式:
    python scripts_slam_pipeline/AR_01_arcap_latency_align.py <session_dir> -c <calibration_dir>

核心算法:
1. 算法1（转向点检测）：检测运动轨迹中的转向点，通过匹配转向点计算时间偏移
2. 算法2（互相关）：使用最小二乘法优化，找到使两个轨迹误差最小的时间偏移

输出文件:
- latency_of_arcap.json: 时间延迟参数文件
- latency_trajectory_*.pdf: 轨迹对齐可视化图表

注意事项:
- 需要标定视频位于 <session_dir>/latency_calibration 目录
- 依赖ArUco标记检测和相机内参标定文件
"""

import sys
import os

# 设置项目根目录和Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import av
from scipy.ndimage import gaussian_filter

from umi.common.timecode_util import mp4_get_start_datetime


from scripts_slam_pipeline.utils.constants import ARUCO_ID
from scripts_slam_pipeline.utils.misc import (
    get_single_path, custom_minimize, 
    plot_trajectories, plot_long_horizon_trajectory
)
from scripts_slam_pipeline.utils.data_loading import load_proprio_interp


def find_calibration_video(session_dir: pathlib.Path):
    """Find calibration video path with priority: raw_videos -> latency_calibration -> demos(segment_meta)."""
    raw_dir = session_dir.joinpath('raw_videos')
    if raw_dir.is_dir():
        mp4_candidates = list(raw_dir.glob('**/*.MP4')) + list(raw_dir.glob('**/*.mp4'))
        if len(mp4_candidates) > 0:
            if len(mp4_candidates) == 1:
                return mp4_candidates[0], "raw_videos"
            dated = []
            for p in mp4_candidates:
                try:
                    dated.append((mp4_get_start_datetime(str(p)).timestamp(), p))
                except Exception:
                    dated.append((float('inf'), p))
            dated.sort(key=lambda x: x[0])
            return dated[0][1], "raw_videos"

    latency_calib_dir = session_dir.joinpath('latency_calibration')
    if latency_calib_dir.is_dir():
        mp4_candidates = list(latency_calib_dir.glob('**/*.MP4')) + list(latency_calib_dir.glob('**/*.mp4'))
        if len(mp4_candidates) > 0:
            return get_single_path(mp4_candidates), "latency_calibration"

    demos_dir = session_dir.joinpath('demos')
    meta_candidates = []
    if demos_dir.is_dir():
        for meta_path in demos_dir.glob('*/segment_meta.json'):
            try:
                with open(meta_path, 'r') as fp:
                    meta = json.load(fp)
            except Exception:
                continue
            if not bool(meta.get('is_calibration', False)):
                continue
            video_path = meta_path.parent.joinpath('raw_video.mp4')
            if not video_path.is_file():
                continue
            meta_candidates.append((float(meta.get('start_timestamp', 0.0)), video_path))

    if len(meta_candidates) > 0:
        meta_candidates.sort(key=lambda x: x[0])
        return meta_candidates[0][1], "demos_segment_meta"

    raise FileNotFoundError(
        f"No calibration video found in raw_videos, {latency_calib_dir}, or no is_calibration segment in {demos_dir}"
    )


def get_video_start_timestamp(calibration_mp4: pathlib.Path):
    """Resolve absolute start timestamp for calibration video."""
    meta_path = calibration_mp4.parent.joinpath('segment_meta.json')
    if meta_path.is_file():
        with open(meta_path, 'r') as fp:
            meta = json.load(fp)
        if 'start_timestamp' in meta:
            return float(meta['start_timestamp']), 'segment_meta'

    try:
        return mp4_get_start_datetime(str(calibration_mp4)).timestamp(), 'mp4_timecode'
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve start timestamp for {calibration_mp4}. "
            f"Need MP4 timecode or segment_meta.json with start_timestamp. Original error: {e}"
        )


def get_video_duration_sec(calibration_mp4: pathlib.Path):
    with av.open(str(calibration_mp4), 'r') as container:
        stream = container.streams.video[0]
        if stream.frames and stream.average_rate:
            return float(stream.frames / stream.average_rate)
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    raise RuntimeError(f"Cannot determine video duration for {calibration_mp4}")


def detect_turning_points(data):
    """
    检测运动轨迹中的转向点（方向变化点）
    
    算法原理:
    1. 去除轨迹开始和结束的稳定区域（运动幅度小的部分）
    2. 对轨迹进行高斯滤波去除噪声
    3. 检测相邻三点之间的方向变化
    4. 返回转向点的时间戳、方向变化和位置
    
    参数:
        data: 轨迹数据列表，每个元素为 (timestamp, position) 元组
        
    返回:
        turning_points: 转向点列表，每个元素为 (timestamp, direction_change, position)
        - direction_change: 方向变化量，+2表示从左转右，-2表示从右转左
    """
    # 提取位置和时间戳数据
    position = [d[1] for d in data]
    timestamp = [d[0] for d in data]

    # 去除轨迹开始和结束的稳定区域
    # 稳定区域定义为运动幅度小于总幅度10%的区域
    max_diff = np.max(position) - np.min(position)
    stable_diff = 0.1 * max_diff
    
    # 从左侧开始找到第一个非稳定区域
    left = 0
    window_max, window_min = position[0], position[0]
    while window_max - window_min < stable_diff:
        left += 1
        window_max = max(window_max, position[left])
        window_min = min(window_min, position[left])

    # 从右侧开始找到最后一个非稳定区域
    right = -1
    window_max, window_min = position[-1], position[-1]
    while window_max - window_min < stable_diff:
        right -= 1
        window_max = max(window_max, position[right])
        window_min = min(window_min, position[right])
    
    # 截取有效运动区域
    position = position[left:right+1]
    timestamp = timestamp[left:right+1]

    # 使用高斯滤波去除轨迹噪声
    # 参数sigma=1表示滤波强度，可根据数据调整
    position = gaussian_filter(position, sigma=1)

    # 检测转向点
    turning_points = []
    for i in range(1, len(position) - 1):
        # 计算相邻点之间的变化量
        dt1 = position[i] - position[i - 1]  # 前一段变化
        dt2 = position[i + 1] - position[i]   # 后一段变化
        
        # 检查方向变化：前后变化量的符号不同
        if np.sign(dt1) != np.sign(dt2):
            # 计算方向变化量：+2表示从左转右，-2表示从右转左
            direction_change = np.sign(dt2) - np.sign(dt1)
            turning_points.append((timestamp[i], direction_change, position[i]))
    
    return turning_points



# %%
@click.command()
@click.argument('session_dir')
@click.option('-c', '--calibration_dir', required=True, help='标定文件目录路径')
@click.option('--calibration_axis', type=str, default="y", help='用于对齐的坐标轴(x/y/z)')
@click.option('--init_offset', type=float, default=0, help='初始时间偏移估计值(秒)')
@click.option('--calib_max_rel_sec', type=float, default=8.0, show_default=True,
              help='仅使用视频前N秒内检测到的ArUco点做标定')
def main(session_dir, calibration_dir, calibration_axis, init_offset, calib_max_rel_sec):
    """
    时间延迟对齐主函数
    
    参数:
        session_dir: 数据会话目录路径
        calibration_dir: 标定文件目录，包含相机内参和ArUco配置
        calibration_axis: 用于对齐的坐标轴，默认为x轴
        init_offset: 初始时间偏移估计，用于优化搜索范围
        
    处理流程:
        1. 加载标定文件和会话数据
        2. 检测ArUco标记轨迹
        3. 加载VR姿态数据插值器
        4. 使用两种算法进行时间对齐
        5. 保存对齐结果和可视化图表
    """
    
    # 坐标轴索引映射：将字符串坐标轴转换为数值索引
    calibration_axis_index = {'x': 0, 'y': 1, 'z': 2}[calibration_axis]
    
    def get_axis_value(interp, value):
        """
        从姿态插值器中获取指定坐标轴的值
        
        参数:
            interp: 姿态数据插值器
            value: 时间戳
            
        返回:
            指定坐标轴的位置值，取负号以匹配视觉坐标系
        """
        try:
            pose = interp(value)
            return -pose[calibration_axis_index]  # 取负号匹配视觉坐标系
        except ValueError:
            return None

    # 路径处理
    calibration_dir = pathlib.Path(calibration_dir)
    session_dir = pathlib.Path(__file__).parent.joinpath(session_dir).absolute()
    latency_calib_dir = session_dir.joinpath('latency_calibration')
    latency_calib_dir.mkdir(parents=True, exist_ok=True)

    # 查找标定视频文件：优先 latency_calibration，其次 demos 中标记为 is_calibration 的首段
    calibration_mp4, calibration_source = find_calibration_video(session_dir)
    print(f"使用标定视频: {calibration_mp4} (source={calibration_source})")


    # 步骤1: 加载VR姿态轨迹插值器
    print("加载VR姿态轨迹插值器")
    # load_proprio_interp函数加载VR姿态数据并创建时间插值器
    # latency=0.0表示不应用时间延迟，extend_boundary=10表示扩展边界10秒
    traj_interp, _ = load_proprio_interp(session_dir, latency=0.0, extend_boundary=10)
    
    # 步骤2: 检测ArUco标记
    print("检测ArUco标记")
    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco_newversion.py')
    assert script_path.is_file(), f"ArUco检测脚本不存在: {script_path}"
    
    # 检查标定文件存在性
    camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calibration_dir.joinpath('aruco_config.yaml')
    assert camera_intrinsics.is_file(), f"相机内参文件不存在: {camera_intrinsics}"
    assert aruco_config.is_file(), f"ArUco配置文件不存在: {aruco_config}"
    
    # 运行ArUco检测（如果结果文件不存在，或缓存来自不同视频）
    aruco_out_dir = latency_calib_dir.joinpath('tag_detection.pkl')
    aruco_meta_path = latency_calib_dir.joinpath('tag_detection_meta.json')
    current_video_stat = calibration_mp4.stat()
    aruco_need_run = not aruco_out_dir.is_file()
    if not aruco_need_run and aruco_meta_path.is_file():
        try:
            with open(aruco_meta_path, 'r') as fp:
                cached_meta = json.load(fp)
            cached_input = cached_meta.get('input_video', '')
            cached_size = int(cached_meta.get('input_size', -1))
            cached_mtime_ns = int(cached_meta.get('input_mtime_ns', -1))
            aruco_need_run = (
                cached_input != str(calibration_mp4)
                or cached_size != int(current_video_stat.st_size)
                or cached_mtime_ns != int(current_video_stat.st_mtime_ns)
            )
        except Exception:
            aruco_need_run = True
    elif not aruco_need_run and not aruco_meta_path.is_file():
        aruco_need_run = True

    if aruco_need_run:
        cmd = [
            'python', str(script_path),
            '--input', str(calibration_mp4),        # 输入标定视频
            '--output', str(aruco_out_dir),         # 输出检测结果
            '--intrinsics_json', str(camera_intrinsics), # 相机内参文件
            '--aruco_yaml', str(aruco_config),      # ArUco配置文件
            '--num_workers', '1'                    # 单线程处理
        ]
        print(f"执行ArUco检测命令: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        assert result.returncode == 0, "ArUco检测失败"
        with open(aruco_meta_path, 'w') as fp:
            json.dump({
                'input_video': str(calibration_mp4),
                'input_size': int(current_video_stat.st_size),
                'input_mtime_ns': int(current_video_stat.st_mtime_ns),
            }, fp)
    else:
        print(f"tag_detection.pkl已存在，跳过ArUco检测: {calibration_mp4}")


    print("align visual and trajectory")

    # get aruco trajectory in video-relative time first
    video_start_time, video_start_source = get_video_start_timestamp(calibration_mp4)
    video_duration = get_video_duration_sec(calibration_mp4)

    aruco_pickle_path = latency_calib_dir.joinpath('tag_detection.pkl')
    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)
        aruco_trajectory_rel = []
        for frame in aruco_pkl:
            if ARUCO_ID in frame['tag_dict']:
                x = frame['tag_dict'][ARUCO_ID]['tvec'][0]    # x-axis
                t_rel = float(frame['time'])
                aruco_trajectory_rel.append((t_rel, float(x)))

    raw_detect_count = len(aruco_trajectory_rel)
    aruco_trajectory_rel = [
        (t_rel, x) for t_rel, x in aruco_trajectory_rel
        if 0.0 <= t_rel <= float(calib_max_rel_sec)
    ]

    print(
        f"[ALIGN] ArUco点筛选: 原始={raw_detect_count}, "
        f"前{calib_max_rel_sec:.3f}s内保留={len(aruco_trajectory_rel)}"
    )

    if len(aruco_trajectory_rel) == 0:
        raise RuntimeError(
            f"No ArUco points found within first {calib_max_rel_sec:.3f}s in {calibration_mp4}. "
            f"Raw detected points={raw_detect_count}."
        )

    if len(aruco_trajectory_rel) < 30:
        raise RuntimeError(
            f"Too few ArUco points within first {calib_max_rel_sec:.3f}s: {len(aruco_trajectory_rel)}. "
            "Please increase --calib_max_rel_sec or improve marker visibility."
        )

    aruco_rel_start = min(t for t, _ in aruco_trajectory_rel)
    aruco_rel_end = max(t for t, _ in aruco_trajectory_rel)
    aruco_abs_start = video_start_time + aruco_rel_start
    aruco_abs_end = video_start_time + aruco_rel_end
    start_pct = 100.0 * aruco_rel_start / video_duration
    end_pct = 100.0 * aruco_rel_end / video_duration

    print("[ALIGN] 可用于对齐的ArUco时间段:")
    print(f"[ALIGN]  绝对时间: {aruco_abs_start:.6f} ~ {aruco_abs_end:.6f}")
    print(f"[ALIGN]  视频内时间: {aruco_rel_start:.3f}s ~ {aruco_rel_end:.3f}s (总长 {video_duration:.3f}s)")
    print(f"[ALIGN]  位于整段视频前 {start_pct:.2f}% ~ {end_pct:.2f}%")

    alignment_window = {
        'calibration_video': str(calibration_mp4),
        'calib_max_rel_sec': float(calib_max_rel_sec),
        'aruco_detected_points_raw': int(raw_detect_count),
        'aruco_points_used': int(len(aruco_trajectory_rel)),
        'video_start_time_unix': float(video_start_time),
        'video_start_source': video_start_source,
        'video_duration_sec': float(video_duration),
        'aruco_abs_start_unix': float(aruco_abs_start),
        'aruco_abs_end_unix': float(aruco_abs_end),
        'aruco_rel_start_sec': float(aruco_rel_start),
        'aruco_rel_end_sec': float(aruco_rel_end),
        'aruco_window_start_pct': float(start_pct),
        'aruco_window_end_pct': float(end_pct),
    }
    with open(latency_calib_dir.joinpath('alignment_window.json'), 'w') as fp:
        json.dump(alignment_window, fp, indent=2)

    # ---- Global matching in VR timeline (do not assume absolute clock sync) ----
    aruco_trajectory_rel = sorted(aruco_trajectory_rel, key=lambda x: x[0])
    tv_min = aruco_trajectory_rel[0][0]
    tv_max = aruco_trajectory_rel[-1][0]

    # Search t_pose = t_video_rel + b across the full valid VR timeline.
    left_bound = float(traj_interp.left_max - tv_max)
    right_bound = float(traj_interp.right_min - tv_min)
    if left_bound >= right_bound:
        raise RuntimeError(
            f"Invalid offset search range: [{left_bound}, {right_bound}] with tv=[{tv_min}, {tv_max}]"
        )

    print("[ALIGN] 在整段VR轨迹上搜索最优时间映射: t_pose = t_video_rel + b")
    print(f"[ALIGN] 搜索区间 b: {left_bound:.6f} ~ {right_bound:.6f}")

    def collect_pairs(offset_b):
        pairs = []
        for t_rel, x_aruco in aruco_trajectory_rel:
            t_pose = t_rel + offset_b
            v_pose = get_axis_value(traj_interp, t_pose)
            if v_pose is None:
                continue
            if not np.isfinite(v_pose):
                continue
            pairs.append((t_rel, t_pose, x_aruco, float(v_pose)))
        return pairs

    def pairs_mse(pairs):
        if len(pairs) < 30:
            return np.inf
        aruco_pos = np.array([p[2] for p in pairs], dtype=np.float64)
        pose_pos = np.array([p[3] for p in pairs], dtype=np.float64)
        astd = float(np.std(aruco_pos))
        pstd = float(np.std(pose_pos))
        if astd < 1e-8 or pstd < 1e-8:
            return np.inf
        aruco_pos = (aruco_pos - np.mean(aruco_pos)) / astd
        pose_pos = (pose_pos - np.mean(pose_pos)) / pstd
        return float(np.mean((aruco_pos - pose_pos) ** 2))

    # coarse search for robust initialization
    coarse_count = 400
    coarse_grid = np.linspace(left_bound, right_bound, coarse_count)
    coarse_scores = []
    best_b = None
    best_score = np.inf
    best_pairs = []
    for b in coarse_grid:
        pairs = collect_pairs(float(b))
        score = pairs_mse(pairs)
        coarse_scores.append(score)
        if score < best_score:
            best_score = score
            best_b = float(b)
            best_pairs = pairs

    if best_b is None or not np.isfinite(best_score):
        raise RuntimeError("Failed to find any valid overlap between ArUco trajectory and VR trajectory.")

    coarse_step = (right_bound - left_bound) / max(1, coarse_count - 1)
    refine_left = max(left_bound, best_b - 5.0 * coarse_step)
    refine_right = min(right_bound, best_b + 5.0 * coarse_step)

    def mse_error(x):
        pairs = collect_pairs(float(x[0]))
        return pairs_mse(pairs)

    res = custom_minimize(mse_error, 0.0, bounds=[(refine_left, refine_right)])
    best_b = float(res.x[0])
    best_pairs = collect_pairs(best_b)
    best_score = pairs_mse(best_pairs)

    if len(best_pairs) < 30 or not np.isfinite(best_score):
        raise RuntimeError(
            "Global matching did not find enough valid samples after refinement. "
            f"pairs={len(best_pairs)}, score={best_score}"
        )

    matched_pose_start = min(p[1] for p in best_pairs)
    matched_pose_end = max(p[1] for p in best_pairs)
    matched_rel_start = min(p[0] for p in best_pairs)
    matched_rel_end = max(p[0] for p in best_pairs)

    print(f"[ALIGN] 最优映射 b={best_b:.6f}, mse={best_score:.6f}, 样本数={len(best_pairs)}")
    print(f"[ALIGN] 匹配到的VR时间段: {matched_pose_start:.6f} ~ {matched_pose_end:.6f}")
    print(f"[ALIGN] 匹配到的视频相对时间段: {matched_rel_start:.3f}s ~ {matched_rel_end:.3f}s")

    # latency used by AR_00 projection: video_abs = pose - latency
    latency_mean = float(best_b - video_start_time)

    latency_of_arcap_result = {
        "mean": latency_mean,
        "mean_2": latency_mean,
        "error": float(best_score),
        "mapping_b_pose_from_video_rel": float(best_b),
        "matched_pairs": int(len(best_pairs)),
        "matched_pose_start": float(matched_pose_start),
        "matched_pose_end": float(matched_pose_end),
        "matched_video_rel_start": float(matched_rel_start),
        "matched_video_rel_end": float(matched_rel_end),
    }

    with open(str(latency_calib_dir.joinpath('latency_of_arcap.json')), "w") as fp:
        json.dump(latency_of_arcap_result, fp, indent=2)

    alignment_window.update({
        'matched_pose_start_unix': float(matched_pose_start),
        'matched_pose_end_unix': float(matched_pose_end),
        'matched_video_rel_start_sec': float(matched_rel_start),
        'matched_video_rel_end_sec': float(matched_rel_end),
        'mapping_b_pose_from_video_rel': float(best_b),
        'latency_mean_pose_minus_video': float(latency_mean),
        'mse_error': float(best_score),
        'matched_pairs': int(len(best_pairs)),
    })
    with open(latency_calib_dir.joinpath('alignment_window.json'), 'w') as fp:
        json.dump(alignment_window, fp, indent=2)

    # plot matched trajectories on pose timeline
    aruco_for_plot = [(p[1], p[2]) for p in best_pairs]
    arcap_for_plot = [(p[1], p[3]) for p in best_pairs]
    plot_trajectories(
        aruco_for_plot,
        arcap_for_plot,
        [],
        [],
        latency_calib_dir.joinpath('latency_trajectory_final_global_match.pdf')
    )


    
    




## %%
if __name__ == "__main__":
    main()
