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
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import Counter

from umi.common.timecode_util import mp4_get_start_datetime


from scripts_slam_pipeline.utils.constants import ARUCO_ID
from scripts_slam_pipeline.utils.misc import (
    get_single_path, custom_minimize, 
    plot_trajectories, plot_long_horizon_trajectory
)
from scripts_slam_pipeline.utils.data_loading import load_proprio_interp


def log(level, message):
    print(f"[ARCAP_ALIGN][{level}] {message}")


def log_section(title):
    print(f"\n[ARCAP_ALIGN][STEP] {'=' * 12} {title} {'=' * 12}")


def summarize_trajectory(name, trajectory):
    if not trajectory:
        log("WARN", f"{name} 轨迹为空")
        return
    ts = np.array([t for t, _ in trajectory], dtype=float)
    vals = np.array([v for _, v in trajectory], dtype=float)
    log(
        "INFO",
        (
            f"{name}: 点数={len(trajectory)}, 时间范围={ts.min():.6f}->{ts.max():.6f}, "
            f"时长={ts.max()-ts.min():.3f}s, 数值范围={vals.min():.6f}->{vals.max():.6f}, "
            f"均值={vals.mean():.6f}, 标准差={vals.std():.6f}"
        )
    )


def save_time_overlap_plot(save_path, aruco_start, aruco_end, proprio_start, proprio_end, init_offset):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.8))
    ax.hlines(2.0, aruco_start, aruco_end, linewidth=8, color="#4c72b0", label="ArUco raw")
    ax.hlines(
        1.2,
        aruco_start + init_offset,
        aruco_end + init_offset,
        linewidth=8,
        color="#55a868",
        label=f"ArUco + init_offset({init_offset:.2f})",
    )
    ax.hlines(0.4, proprio_start, proprio_end, linewidth=8, color="#dd8452", label="ARCap proprio")
    ax.set_yticks([0.4, 1.2, 2.0])
    ax.set_yticklabels(["ARCap", "ArUco+offset", "ArUco"])
    ax.set_xlabel("timestamp (s)")
    ax.set_title("Time Range Overlap Diagnosis")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)


def save_offset_diagnosis_plot(save_path, offsets, valid_counts, mses, corrs, init_offset):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(offsets, valid_counts, color="#4c72b0")
    axes[0].axhline(3, color="black", linestyle="--", linewidth=1, label="min valid=3")
    axes[0].axvline(init_offset, color="#55a868", linestyle="--", linewidth=1, label="init_offset")
    axes[0].set_ylabel("valid_count")
    axes[0].set_title("Offset Diagnosis")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(offsets, mses, color="#dd8452")
    axes[1].axvline(init_offset, color="#55a868", linestyle="--", linewidth=1)
    axes[1].set_ylabel("mse")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(offsets, corrs, color="#8172b3")
    axes[2].axvline(init_offset, color="#55a868", linestyle="--", linewidth=1)
    axes[2].set_xlabel("offset (s)")
    axes[2].set_ylabel("corr")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)


def save_vr_sampling_process_plot(
    save_path,
    aruco_timepoints,
    traj_interp,
    get_axis_value,
    init_offset,
    candidate_offsets,
):
    aruco_timepoints = np.array(aruco_timepoints, dtype=float)
    if aruco_timepoints.size == 0:
        return

    offsets = np.array(candidate_offsets, dtype=float)
    validity = np.zeros((len(offsets), len(aruco_timepoints)), dtype=int)
    sampled_values = np.full((len(offsets), len(aruco_timepoints)), np.nan, dtype=float)

    for i, off in enumerate(offsets):
        for j, t in enumerate(aruco_timepoints):
            v = get_axis_value(traj_interp, t + off)
            if v is not None and np.isfinite(v):
                validity[i, j] = 1
                sampled_values[i, j] = float(v)

    init_idx = int(np.argmin(np.abs(offsets - init_offset)))
    q_init = aruco_timepoints + offsets[init_idx]
    valid_init = validity[init_idx].astype(bool)

    left_min = getattr(traj_interp, "left_min", np.nan)
    left_max = getattr(traj_interp, "left_max", np.nan)
    right_min = getattr(traj_interp, "right_min", np.nan)
    right_max = getattr(traj_interp, "right_max", np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)

    # 子图1: 时间范围与插值边界
    axes[0].hlines(2.0, aruco_timepoints[0], aruco_timepoints[-1], color="#4c72b0", linewidth=8, label="ArUco raw")
    axes[0].hlines(1.5, q_init[0], q_init[-1], color="#55a868", linewidth=8, label=f"ArUco + init_offset({offsets[init_idx]:.2f})")
    if np.isfinite(left_min) and np.isfinite(left_max):
        axes[0].hlines(1.0, left_min, left_max, color="#9ecae1", linewidth=10, label="VR left extension")
    if np.isfinite(left_max) and np.isfinite(right_min):
        axes[0].hlines(0.7, left_max, right_min, color="#dd8452", linewidth=10, label="VR strict interp range")
    if np.isfinite(right_min) and np.isfinite(right_max):
        axes[0].hlines(0.4, right_min, right_max, color="#fdd0a2", linewidth=10, label="VR right extension")
    axes[0].set_yticks([0.4, 0.7, 1.0, 1.5, 2.0])
    axes[0].set_yticklabels(["VR right ext", "VR interp", "VR left ext", "Aruco+offset", "Aruco"])
    axes[0].set_title("VR Parsing/Sampling Time Windows")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    # 子图2: 不同offset下每个采样点的有效性
    extent = [0, len(aruco_timepoints), offsets[0] - 0.5, offsets[-1] + 0.5]
    axes[1].imshow(validity, aspect="auto", cmap="RdYlGn", interpolation="nearest", extent=extent, origin="lower")
    axes[1].set_ylabel("offset (s)")
    axes[1].set_title("Sample Validity Matrix (green=valid, red=invalid)")
    axes[1].grid(False)

    # 子图3: init_offset下采样值与有效性
    rel_t = aruco_timepoints - aruco_timepoints[0]
    axes[2].plot(rel_t, sampled_values[init_idx], color="#dd8452", label="VR sampled value")
    invalid_rel_t = rel_t[~valid_init]
    if invalid_rel_t.size > 0:
        axes[2].scatter(invalid_rel_t, np.zeros_like(invalid_rel_t), s=12, color="red", alpha=0.7, label="invalid samples")
    axes[2].set_xlabel("relative time from first ArUco sample (s)")
    axes[2].set_ylabel("sampled VR value")
    axes[2].set_title("VR Sampled Values at init_offset")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)


def eval_offset_diagnostics(timepoints, aruco_vals, traj_interp, get_axis_value, offset):
    arcap_samples = [get_axis_value(traj_interp, t + offset) for t in timepoints]
    valid_mask = np.array([v is not None and np.isfinite(v) for v in arcap_samples], dtype=bool)
    valid_count = int(np.sum(valid_mask))

    result = {
        "offset": float(offset),
        "valid_count": valid_count,
        "mse": 1e6,
        "corr": np.nan,
        "reason": "ok",
    }

    if valid_count < 3:
        result["reason"] = f"valid_count_too_small({valid_count})"
        return result

    arcap_pos = np.array([v for v, m in zip(arcap_samples, valid_mask) if m], dtype=float)
    aruco_pos = np.array([v for v, m in zip(aruco_vals, valid_mask) if m], dtype=float)

    arcap_std = np.std(arcap_pos)
    aruco_std = np.std(aruco_pos)
    if arcap_std <= 1e-12 or aruco_std <= 1e-12:
        result["reason"] = f"std_too_small(arcap={arcap_std:.3e}, aruco={aruco_std:.3e})"
        return result

    arcap_norm = (arcap_pos - np.mean(arcap_pos)) / arcap_std
    aruco_norm = (aruco_pos - np.mean(aruco_pos)) / aruco_std
    result["mse"] = float(np.mean((aruco_norm - arcap_norm) ** 2))
    result["corr"] = float(np.corrcoef(aruco_norm, arcap_norm)[0, 1])
    return result


def log_turning_points_list(name, turning_points, max_items=30):
    log("INFO", f"{name}: 转向点总数={len(turning_points)}")
    if len(turning_points) == 0:
        return
    show_items = turning_points[:max_items]
    for i, (t, d, v) in enumerate(show_items):
        log("INFO", f"{name}[{i:02d}] t={t:.6f}, dir={int(d)}, pos={v:.6f}")
    if len(turning_points) > max_items:
        log("INFO", f"{name}: ... 其余 {len(turning_points)-max_items} 个点省略")


def log_turning_points_pair_diff(name, turn_aruco, turn_arcap, max_items=30):
    n = min(len(turn_aruco), len(turn_arcap), max_items)
    if n == 0:
        log("WARN", f"{name}: 无可比较转向点")
        return
    for i in range(n):
        ta, da, va = turn_aruco[i]
        tb, db, vb = turn_arcap[i]
        log(
            "INFO",
            (
                f"{name}[{i:02d}] dt={tb-ta:+.6f}s, ddir={int(db-da):+d}, "
                f"aruco(t={ta:.6f},dir={int(da)},v={va:.6f}) vs "
                f"arcap(t={tb:.6f},dir={int(db)},v={vb:.6f})"
            ),
        )


def save_turning_points_pair_plot(save_path, turn_aruco, turn_arcap, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle(title)

    idx_a = np.arange(len(turn_aruco))
    idx_b = np.arange(len(turn_arcap))

    if len(turn_aruco) > 0:
        t_a = np.array([x[0] for x in turn_aruco], dtype=float)
        d_a = np.array([x[1] for x in turn_aruco], dtype=float)
        axes[0].plot(idx_a, t_a, marker="o", color="#4c72b0", label="ArUco turn time")
        axes[0].set_ylabel("turn timestamp")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(idx_a, d_a, marker="o", color="#4c72b0", label="ArUco turn dir")

    if len(turn_arcap) > 0:
        t_b = np.array([x[0] for x in turn_arcap], dtype=float)
        d_b = np.array([x[1] for x in turn_arcap], dtype=float)
        axes[0].plot(idx_b, t_b, marker="x", color="#dd8452", label="ARCap turn time")
        axes[1].plot(idx_b, d_b, marker="x", color="#dd8452", label="ARCap turn dir")
        axes[0].legend(loc="best")

    axes[1].set_xlabel("turning-point index")
    axes[1].set_ylabel("direction change")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)


def save_zscore_shape_plot(save_path, aruco_time, aruco_raw, arcap_raw, title):
    aruco_time = np.array(aruco_time, dtype=float)
    aruco_raw = np.array(aruco_raw, dtype=float)
    arcap_raw = np.array(arcap_raw, dtype=float)

    aruco_std = np.std(aruco_raw)
    arcap_std = np.std(arcap_raw)

    if aruco_std <= 1e-12 or arcap_std <= 1e-12:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
        ax.text(
            0.5,
            0.5,
            f"std too small for z-score\naruco_std={aruco_std:.3e}, arcap_std={arcap_std:.3e}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(save_path))
        plt.close(fig)
        return

    aruco_z = (aruco_raw - np.mean(aruco_raw)) / aruco_std
    arcap_z = (arcap_raw - np.mean(arcap_raw)) / arcap_std
    corr = np.corrcoef(aruco_z, arcap_z)[0, 1]

    t_rel = aruco_time - aruco_time[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{title} | corr={corr:.4f}")

    axes[0].plot(t_rel, aruco_z, color="#4c72b0", label="ArUco z-score")
    axes[0].plot(t_rel, arcap_z, color="#dd8452", label="ARCap z-score")
    axes[0].set_ylabel("z-score")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(t_rel, aruco_z - arcap_z, color="#55a868", label="z-score diff")
    axes[1].set_xlabel("relative time (s)")
    axes[1].set_ylabel("z_a - z_b")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)


def detect_turning_points(data, name="trajectory"):
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
    # 过滤插值越界导致的无效点
    valid_data = [
        (t, p) for t, p in data
        if p is not None and np.isfinite(p)
    ]
    if len(valid_data) < 3:
        log("WARN", f"{name}: 有效点不足3个，无法检测转向点 (valid={len(valid_data)})")
        return []

    # 提取位置和时间戳数据
    timestamp = [d[0] for d in valid_data]
    position = np.array([d[1] for d in valid_data], dtype=float)

    # 去除轨迹开始和结束的稳定区域
    # 稳定区域定义为运动幅度小于总幅度10%的区域
    max_diff = np.max(position) - np.min(position)
    stable_diff = 0.1 * max_diff
    
    if max_diff <= 1e-12:
        log("WARN", f"{name}: 轨迹变化幅度过小(max_diff={max_diff:.6e})，无法检测转向点")
        return []

    # 从左侧开始找到第一个非稳定区域
    left = 0
    window_max, window_min = position[0], position[0]
    while left < len(position) - 1 and window_max - window_min < stable_diff:
        left += 1
        window_max = max(window_max, position[left])
        window_min = min(window_min, position[left])

    # 从右侧开始找到最后一个非稳定区域
    right = len(position) - 1
    window_max, window_min = position[right], position[right]
    while right > 0 and window_max - window_min < stable_diff:
        right -= 1
        window_max = max(window_max, position[right])
        window_min = min(window_min, position[right])
    
    # 截取有效运动区域
    position = position[left:right+1]
    timestamp = timestamp[left:right+1]

    if len(position) < 3:
        log("WARN", f"{name}: 去除稳定区后点数不足3个，无法检测转向点")
        return []

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

            log("INFO", f"{name}: 检测到转向点 {len(turning_points)} 个")
    
    return turning_points



# %%
@click.command()
@click.argument('session_dir')
@click.option('-c', '--calibration_dir', required=True, help='标定文件目录路径')
@click.option('--calibration_axis', type=str, default="x", help='用于对齐的坐标轴(x/y/z)')
@click.option('--init_offset', type=float, default=0, help='初始时间偏移估计值(秒)')
def main(session_dir, calibration_dir, calibration_axis, init_offset):
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
    log_section("参数与路径初始化")
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if calibration_axis not in axis_map:
        raise ValueError(f"无效 calibration_axis={calibration_axis}，仅支持 x/y/z")
    calibration_axis_index = axis_map[calibration_axis]
    
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
    log("INFO", f"session_dir={session_dir}")
    log("INFO", f"calibration_dir={calibration_dir}")
    log("INFO", f"latency_calibration_dir={latency_calib_dir}")
    log("INFO", f"calibration_axis={calibration_axis} (index={calibration_axis_index}), init_offset={init_offset}")

    # 检查目录存在性
    if not latency_calib_dir.is_dir():
        raise FileNotFoundError(f"标定目录不存在: {latency_calib_dir}")

    # 查找标定视频文件
    mp4_candidates = list(latency_calib_dir.glob('**/*.MP4')) + list(latency_calib_dir.glob('**/*.mp4'))
    log("INFO", f"发现标定视频候选数量: {len(mp4_candidates)}")
    try:
        calibration_mp4 = get_single_path(mp4_candidates)
    except Exception as exc:
        log("ERROR", f"无法确定唯一标定视频，请检查latency_calibration目录: {exc}")
        raise
    log("INFO", f"使用标定视频: {calibration_mp4}")


    # 步骤1: 加载VR姿态轨迹插值器
    log_section("加载VR姿态轨迹插值器")
    # load_proprio_interp函数加载VR姿态数据并创建时间插值器
    # latency=0.0表示不应用时间延迟，extend_boundary=10表示扩展边界10秒
    traj_interp, _ = load_proprio_interp(session_dir, latency=0.0, extend_boundary=10)
    proprio_start_ts = traj_interp.left_max
    proprio_end_ts = traj_interp.right_min
    log("INFO", f"VR轨迹有效时间范围: {proprio_start_ts:.6f} -> {proprio_end_ts:.6f} (时长={proprio_end_ts-proprio_start_ts:.3f}s)")
    
    # 步骤2: 检测ArUco标记
    log_section("检测ArUco标记")
    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco_newversion.py')
    if not script_path.is_file():
        raise FileNotFoundError(f"ArUco检测脚本不存在: {script_path}")
    
    # 检查标定文件存在性
    camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calibration_dir.joinpath('aruco_config.yaml')
    if not camera_intrinsics.is_file():
        raise FileNotFoundError(f"相机内参文件不存在: {camera_intrinsics}")
    if not aruco_config.is_file():
        raise FileNotFoundError(f"ArUco配置文件不存在: {aruco_config}")
    log("INFO", f"检测脚本: {script_path}")
    log("INFO", f"相机内参: {camera_intrinsics}")
    log("INFO", f"ArUco配置: {aruco_config}")
    
    # 运行ArUco检测（如果结果文件不存在）
    aruco_out_dir = latency_calib_dir.joinpath('tag_detection.pkl')
    if not aruco_out_dir.is_file():
        cmd = [
            'python', script_path,
            '--input', str(calibration_mp4),        # 输入标定视频
            '--output', str(aruco_out_dir),         # 输出检测结果
            '--intrinsics_json', camera_intrinsics, # 相机内参文件
            '--aruco_yaml', str(aruco_config),      # ArUco配置文件
            '--num_workers', '1'                    # 单线程处理
        ]
        cmd = [str(item) for item in cmd]
        log("INFO", f"执行ArUco检测命令: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        log("INFO", f"ArUco检测结束: returncode={result.returncode}, 耗时={elapsed:.2f}s")
        if result.returncode != 0:
            raise RuntimeError("ArUco检测失败，请检查上方detect_aruco_newversion.py输出日志")
    else:
        log("INFO", f"tag_detection.pkl已存在，跳过ArUco检测: {aruco_out_dir}")


    log_section("视觉轨迹与VR轨迹对齐")

    # get aruco trajectory
    video_start_time = mp4_get_start_datetime(str(calibration_mp4)).timestamp()

    aruco_pickle_path = latency_calib_dir.joinpath('tag_detection.pkl')
    if not aruco_pickle_path.is_file():
        raise FileNotFoundError(f"Aruco检测结果不存在: {aruco_pickle_path}")

    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)
        if not isinstance(aruco_pkl, (list, tuple)):
            raise ValueError(f"Aruco检测结果格式异常，期望list/tuple，实际={type(aruco_pkl)}")

        aruco_trajectory = []
        frame_total = 0
        frame_has_tag = 0
        for frame in aruco_pkl:
            frame_total += 1
            if ARUCO_ID in frame['tag_dict']:
                frame_has_tag += 1
                x = frame['tag_dict'][ARUCO_ID]['tvec'][0]    # x-axis
                aruco_trajectory.append((frame['time']+video_start_time, x))

    log("INFO", f"Aruco结果帧数: total={frame_total}, 含目标tag={frame_has_tag}, 轨迹点数={len(aruco_trajectory)}")
    if len(aruco_trajectory) < 3:
        raise ValueError(f"Aruco轨迹点不足(<3)，无法进行时间对齐。当前点数={len(aruco_trajectory)}")

    summarize_trajectory("aruco_trajectory", aruco_trajectory)

    aruco_timepoints = sorted([t for t, _ in aruco_trajectory])
    aruco_start_ts = aruco_timepoints[0]
    aruco_end_ts = aruco_timepoints[-1]

    query_start_ts = aruco_start_ts + init_offset
    query_end_ts = aruco_end_ts + init_offset

    overlap_start = max(proprio_start_ts, query_start_ts)
    overlap_end = min(proprio_end_ts, query_end_ts)
    overlap_sec = max(0.0, overlap_end - overlap_start)

    log("INFO", f"[ALIGN DEBUG] proprio_valid_ts: {proprio_start_ts:.6f} -> {proprio_end_ts:.6f}")
    log("INFO", f"[ALIGN DEBUG] aruco_detected_ts: {aruco_start_ts:.6f} -> {aruco_end_ts:.6f}")
    log("INFO", f"[ALIGN DEBUG] query_ts_with_init_offset: {query_start_ts:.6f} -> {query_end_ts:.6f} (init_offset={init_offset})")
    log("INFO", f"[ALIGN DEBUG] overlap_ts: {overlap_start:.6f} -> {overlap_end:.6f} (overlap_sec={overlap_sec:.3f})")
    log("INFO", f"[ALIGN DEBUG] aruco_sample_head: {aruco_timepoints[:5]}")
    log("INFO", f"[ALIGN DEBUG] aruco_sample_tail: {aruco_timepoints[-5:]}")

    # 诊断图1.5: VR解析与采样过程可视化
    vr_sampling_plot_path = latency_calib_dir.joinpath('latency_diag_vr_sampling_process.pdf')
    vr_diag_offsets = [init_offset - 2.0, init_offset - 1.0, init_offset, init_offset + 1.0, init_offset + 2.0]
    save_vr_sampling_process_plot(
        vr_sampling_plot_path,
        aruco_timepoints,
        traj_interp,
        get_axis_value,
        init_offset,
        vr_diag_offsets,
    )
    log("INFO", f"已保存VR解析与采样过程诊断图: {vr_sampling_plot_path}")
    for off in vr_diag_offsets:
        vals = [get_axis_value(traj_interp, t + off) for t in aruco_timepoints]
        valid_count = int(np.sum([v is not None and np.isfinite(v) for v in vals]))
        log("INFO", f"[VR SAMPLE] offset={off:+.2f}, valid_count={valid_count}/{len(aruco_timepoints)}")

    # 诊断图1: 时间范围重叠
    overlap_plot_path = latency_calib_dir.joinpath('latency_diag_time_overlap.pdf')
    save_time_overlap_plot(
        overlap_plot_path,
        aruco_start_ts,
        aruco_end_ts,
        proprio_start_ts,
        proprio_end_ts,
        init_offset,
    )
    log("INFO", f"已保存时间重叠诊断图: {overlap_plot_path}")

    if overlap_sec <= 0:
        raise ValueError(
            "Aruco与VR时间区间无重叠，请检查init_offset、相机时间戳或姿态数据有效区间"
        )

    # 诊断扫描: 不影响算法，仅用于解释为什么匹配不到
    diag_offsets = np.arange(-6.0 + init_offset, 6.0 + init_offset + 0.025, 0.05)
    aruco_vals = np.array([v for _, v in aruco_trajectory], dtype=float)
    diag_stats = []
    for i, off in enumerate(diag_offsets):
        stat = eval_offset_diagnostics(aruco_timepoints, aruco_vals, traj_interp, get_axis_value, off)
        diag_stats.append(stat)
        if i % 30 == 0 or i == len(diag_offsets) - 1:
            log(
                "INFO",
                (
                    f"[DIAG SCAN] {i+1}/{len(diag_offsets)} offset={off:.2f}, "
                    f"valid={stat['valid_count']}/{len(aruco_timepoints)}, "
                    f"mse={stat['mse']:.6f}, corr={stat['corr']}, reason={stat['reason']}"
                ),
            )

    fail_counter = Counter([d["reason"] for d in diag_stats])
    for reason, count in fail_counter.most_common():
        log("INFO", f"[DIAG SUMMARY] {reason}: {count} offsets")

    valid_counts = np.array([d["valid_count"] for d in diag_stats], dtype=float)
    mses = np.array([d["mse"] for d in diag_stats], dtype=float)
    corrs = np.array([d["corr"] for d in diag_stats], dtype=float)
    finite_mask = np.isfinite(mses) & (mses < 1e6)
    if np.any(finite_mask):
        best_diag_idx = int(np.argmin(mses))
        log(
            "INFO",
            (
                f"[DIAG BEST] offset={diag_stats[best_diag_idx]['offset']:.6f}, "
                f"mse={diag_stats[best_diag_idx]['mse']:.6f}, "
                f"corr={diag_stats[best_diag_idx]['corr']:.6f}, "
                f"valid={diag_stats[best_diag_idx]['valid_count']}"
            ),
        )
    else:
        log("WARN", "[DIAG BEST] 扫描区间内无任何有效MSE点（均为惩罚值1e6）")

    diag_plot_path = latency_calib_dir.joinpath('latency_diag_offset_scan.pdf')
    save_offset_diagnosis_plot(
        diag_plot_path,
        diag_offsets,
        valid_counts,
        mses,
        corrs,
        init_offset,
    )
    log("INFO", f"已保存offset扫描诊断图: {diag_plot_path}")

    
    # get arcap trajectory for plotting
    extend_range = 15
    timepoints = aruco_timepoints
    time_extend_before = np.linspace(timepoints[0]-extend_range, timepoints[0], 100).tolist()
    time_extend_after = np.linspace(timepoints[-1], timepoints[-1]+extend_range, 100).tolist()
    timepoints = time_extend_before + timepoints + time_extend_after

    arcap_trajectory_for_plotting = [
        (t, get_axis_value(traj_interp, t+init_offset)) 
        for t in timepoints
    ]
    valid_for_plot = sum(v is not None and np.isfinite(v) for _, v in arcap_trajectory_for_plotting)
    log("INFO", f"长时域绘图轨迹采样点={len(arcap_trajectory_for_plotting)}, 有效点={valid_for_plot}")
    plot_long_horizon_trajectory(
        aruco_trajectory, arcap_trajectory_for_plotting,
        title=f"Offset {init_offset:.2f} (move arcap {'left' if init_offset > 0 else 'right'})",
        save_dir=latency_calib_dir.joinpath(f'latency_trajectory_offset{init_offset}_long.pdf')
    )
    log("INFO", f"已保存长时域可视化: {latency_calib_dir.joinpath(f'latency_trajectory_offset{init_offset}_long.pdf')}")
    


    ###### algorithm 1: align by turning points
    log_section("算法1: 转向点检测")
    turn_point_aruco = detect_turning_points(aruco_trajectory, name="aruco_trajectory")
    turn_point_arcap = None
    log("INFO", f"算法1初始: aruco转向点数量={len(turn_point_aruco)}")

    for arcap_time_offset in [0.0, 1.0, -1.0, 2.0, -2.0]:
        arcap_time_offset += init_offset
        log("INFO", f"算法1尝试offset={arcap_time_offset:.3f}")

        # get arcap trajectory
        arcap_trajectory = []
        timepoints = [t+arcap_time_offset for t, _ in aruco_trajectory]
        timepoints = sorted(timepoints)
        for t in timepoints:
            arcap_trajectory.append( ( t, get_axis_value(traj_interp, t) ) )

        arcap_trajectory_valid = [
            (t, v) for t, v in arcap_trajectory
            if v is not None and np.isfinite(v)
        ]
        log(
            "INFO",
            f"算法1@offset={arcap_time_offset:.2f}: valid_count={len(arcap_trajectory_valid)}/{len(arcap_trajectory)}"
        )
        if len(arcap_trajectory_valid) < 3:
            # 即使失败也保存一张“转向点对比图”，便于回看为何算法1无法匹配
            turn_pair_plot_path = latency_calib_dir.joinpath(
                f'latency_turning_points_compare_offset{arcap_time_offset:.2f}.pdf'
            )
            save_turning_points_pair_plot(
                turn_pair_plot_path,
                turn_point_aruco,
                [],
                title=(
                    f"Turning-Points Compare offset={arcap_time_offset:.2f} | "
                    f"FAILED: valid_count={len(arcap_trajectory_valid)}"
                ),
            )
            log("INFO", f"已保存算法1转向点对比图(失败态): {turn_pair_plot_path}")

            log("WARN", f"Offset {arcap_time_offset:.2f}: 有效ARCap轨迹点不足，跳过")
            turn_point_arcap = None
            continue

        summarize_trajectory(f"arcap_trajectory_valid(offset={arcap_time_offset:.2f})", arcap_trajectory_valid)

        turn_point_arcap = detect_turning_points(arcap_trajectory_valid, name=f"arcap(offset={arcap_time_offset:.2f})")
        log(
            "INFO",
            f"算法1@offset={arcap_time_offset:.2f}: aruco_turn={len(turn_point_aruco)}, arcap_turn={len(turn_point_arcap)}"
        )

        # 算法1诊断输出: 转向点列表与逐项差异
        log_turning_points_list("algo1_aruco_turn", turn_point_aruco, max_items=20)
        log_turning_points_list(f"algo1_arcap_turn@{arcap_time_offset:.2f}", turn_point_arcap, max_items=20)
        log_turning_points_pair_diff(
            f"algo1_turn_diff@{arcap_time_offset:.2f}",
            turn_point_aruco,
            turn_point_arcap,
            max_items=20,
        )

        turn_pair_plot_path = latency_calib_dir.joinpath(
            f'latency_turning_points_compare_offset{arcap_time_offset:.2f}.pdf'
        )
        save_turning_points_pair_plot(
            turn_pair_plot_path,
            turn_point_aruco,
            turn_point_arcap,
            title=f"Turning-Points Compare offset={arcap_time_offset:.2f}",
        )
        log("INFO", f"已保存算法1转向点对比图: {turn_pair_plot_path}")

        # pickle.dump({
        #     "aruco": aruco_trajectory,
        #     "arcap": arcap_trajectory,
        # }, open(str(latency_calib_dir.joinpath(f'latency_data_{arcap_time_offset}.pkl')), 'wb'))
        
        plot_trajectories(aruco_trajectory, arcap_trajectory_valid, 
                        [], [],
                        #turn_point_aruco, turn_point_arcap, 
                        latency_calib_dir.joinpath(f'latency_trajectory_offset{arcap_time_offset}.pdf'))
        log("INFO", f"已保存算法1中间图: {latency_calib_dir.joinpath(f'latency_trajectory_offset{arcap_time_offset}.pdf')}")

        if len(turn_point_aruco) == len(turn_point_arcap):
            log("INFO", f"算法1找到候选匹配: aruco={len(turn_point_aruco)} vs arcap={len(turn_point_arcap)}")
            break
        else:
            log("WARN", f"算法1转向点数量不匹配: aruco={len(turn_point_aruco)} vs arcap={len(turn_point_arcap)}")
            turn_point_arcap = None


    latency_of_arcap_result = {}
        
    if turn_point_arcap is not None:
        # algorithm 1 is good
        latency_of_arcap = []
        for (t1, d1, _), (t2, d2, _) in zip(turn_point_aruco, turn_point_arcap):
            if(d1 != d2):
                log("WARN", f"算法1方向不匹配: aruco_direction={d1}, arcap_direction={d2}")
                break
            latency_of_arcap.append(t2 - t1)
        else:
            if np.std(latency_of_arcap) < 0.1:
                # everything is good
                log("INFO", f"算法1延迟估计: mean={np.mean(latency_of_arcap):.6f}, std={np.std(latency_of_arcap):.6f}")

                latency_of_arcap_result = {
                    "mean": np.mean(latency_of_arcap),
                    "std": np.std(latency_of_arcap),
                    "data": latency_of_arcap,
                }
                
                lat = latency_of_arcap_result["mean"]
                calibrated_arcap_trajectory = []
                for t, v in arcap_trajectory:
                    calibrated_arcap_trajectory.append((t-lat, v))
                plot_trajectories(aruco_trajectory, calibrated_arcap_trajectory, 
                                [], [], 
                                latency_calib_dir.joinpath(f'latency_trajectory_final_algo_1.pdf'))
                log("INFO", f"已保存算法1最终图: {latency_calib_dir.joinpath('latency_trajectory_final_algo_1.pdf')}")
            else:
                log("WARN", f"算法1延迟离散过大，放弃结果: std={np.std(latency_of_arcap):.6f}")
    else:
        log("WARN", "算法1未能得到有效结果，将依赖算法2")
                
    

    #### algorithm 2: align by cross correlation

    log_section("算法2: 最小二乘误差优化")

    aruco_trajectory = sorted(aruco_trajectory, key=lambda x: x[0])
    timepoints = [t for t, v in aruco_trajectory]
    aruco_pos_raw = np.array([v for t, v in aruco_trajectory], dtype=float)

    eval_counter = {"count": 0}


    def mse_error(x):
        eval_counter["count"] += 1
        arcap_samples = [get_axis_value(traj_interp, t + x[0]) for t in timepoints]
        valid_mask = np.array(
            [v is not None and np.isfinite(v) for v in arcap_samples],
            dtype=bool
        )
        if np.sum(valid_mask) < 3:
            if eval_counter["count"] <= 5:
                log("WARN", f"mse_error@offset={x[0]:.6f}: 有效匹配点不足3个")
            return 1e6

        arcap_pos = np.array([v for v, m in zip(arcap_samples, valid_mask) if m], dtype=float)
        aruco_pos = aruco_pos_raw[valid_mask]

        arcap_std = np.std(arcap_pos)
        aruco_std = np.std(aruco_pos)
        if arcap_std <= 1e-12 or aruco_std <= 1e-12:
            if eval_counter["count"] <= 5:
                log("WARN", f"mse_error@offset={x[0]:.6f}: 标准差过小 arcap_std={arcap_std:.3e}, aruco_std={aruco_std:.3e}")
            return 1e6

        arcap_pos = (arcap_pos - np.mean(arcap_pos)) / arcap_std
        aruco_pos = (aruco_pos - np.mean(aruco_pos)) / aruco_std
        return np.mean((aruco_pos - arcap_pos)**2)

    left_offset_bound  = -1.0 + init_offset
    right_offset_bound = 1.0 + init_offset
    epsilon = 0.01
    shift_iter = 0
    while True:
        shift_iter += 1
        log("INFO", f"算法2第{shift_iter}轮搜索区间: [{left_offset_bound:.3f}, {right_offset_bound:.3f}]")
        res = custom_minimize(mse_error, 0.0, bounds=[(left_offset_bound, right_offset_bound)])
        if res.x - left_offset_bound < epsilon:
            log("WARN", f"最优值触及左边界(res={res.x[0]:.6f})，搜索区间整体左移1秒")
            left_offset_bound -= 1.0
            right_offset_bound -= 1.0
        elif right_offset_bound - res.x < epsilon:
            log("WARN", f"最优值触及右边界(res={res.x[0]:.6f})，搜索区间整体右移1秒")
            left_offset_bound += 1.0
            right_offset_bound += 1.0
        else:
            break

    final_error = mse_error(res.x)
    final_diag = eval_offset_diagnostics(timepoints, aruco_pos_raw, traj_interp, get_axis_value, res.x[0])
    log("INFO", f"算法2完成: best_offset={res.x[0]:.6f}, mse={final_error:.6f}, 误差评估次数={eval_counter['count']}")
    log(
        "INFO",
        (
            f"算法2最终点诊断: valid={final_diag['valid_count']}/{len(timepoints)}, "
            f"corr={final_diag['corr']}, reason={final_diag['reason']}"
        )
    )

    # 算法2诊断输出: Z-score形状对比（只在有效点>=3时有意义）
    arcap_samples_final = [get_axis_value(traj_interp, t + res.x[0]) for t in timepoints]
    valid_mask_final = np.array(
        [v is not None and np.isfinite(v) for v in arcap_samples_final],
        dtype=bool
    )
    if int(np.sum(valid_mask_final)) >= 3:
        zscore_plot_path = latency_calib_dir.joinpath('latency_algo2_zscore_shape_compare.pdf')
        save_zscore_shape_plot(
            zscore_plot_path,
            np.array(timepoints, dtype=float)[valid_mask_final],
            aruco_pos_raw[valid_mask_final],
            np.array([v for v, m in zip(arcap_samples_final, valid_mask_final) if m], dtype=float),
            title=f"Algo2 Z-score Shape Compare @ offset={res.x[0]:.6f}",
        )
        log("INFO", f"已保存算法2 Z-Score 形状对比图: {zscore_plot_path}")
    else:
        log("WARN", "算法2 Z-Score图未生成: 有效点不足3个")
        

    latency_of_arcap_result.update({
        "mean_2": res.x[0],
        "error": final_error,
    })
    if "mean" not in latency_of_arcap_result:
        latency_of_arcap_result["mean"] = latency_of_arcap_result["mean_2"]
        
    latency_json_path = latency_calib_dir.joinpath('latency_of_arcap.json')
    with open(str(latency_json_path), "w") as fp:
        json.dump(latency_of_arcap_result, fp)
    log("INFO", f"已保存延迟结果: {latency_json_path} -> {latency_of_arcap_result}")

    lat = latency_of_arcap_result["mean_2"]
    calibrated_arcap_trajectory = []
    arcap_trajectory_final = [
        (t + lat, get_axis_value(traj_interp, t + lat))
        for t in timepoints
    ]
    arcap_trajectory_final = [
        (t, v) for t, v in arcap_trajectory_final
        if v is not None and np.isfinite(v)
    ]
    for t, v in arcap_trajectory_final:
        calibrated_arcap_trajectory.append((t - lat, v))
    plot_trajectories(aruco_trajectory, calibrated_arcap_trajectory, 
                    [], [], 
                    latency_calib_dir.joinpath(f'latency_trajectory_final_algo_2.pdf'))
    log("INFO", f"已保存算法2最终图: {latency_calib_dir.joinpath('latency_trajectory_final_algo_2.pdf')}")
    log_section("处理完成")
    log("INFO", "ARCap时间延迟对齐执行成功")


    
    




## %%
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR", f"执行失败: {type(exc).__name__}: {exc}")
        raise
