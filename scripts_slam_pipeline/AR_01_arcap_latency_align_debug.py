"""
AR_01_arcap_latency_align_debug.py - ARCap延迟对齐可视化调试版

目标:
- 复用 AR_01_arcap_latency_align.py 的数据处理思路
- 在对齐前/对齐中输出更细粒度可视化和诊断报告
- 快速定位“为什么匹配不到”

用法:
python scripts_slam_pipeline/AR_01_arcap_latency_align_debug.py <session_dir> -c <calibration_dir> \
    --calibration_axis x --init_offset 0.0 --scan_min -6 --scan_max 6 --scan_step 0.05
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import pickle
import subprocess
import time

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from umi.common.timecode_util import mp4_get_start_datetime

from scripts_slam_pipeline.utils.constants import ARUCO_ID
from scripts_slam_pipeline.utils.data_loading import load_proprio_interp
from scripts_slam_pipeline.utils.misc import (
    custom_minimize,
    get_single_path,
    plot_long_horizon_trajectory,
    plot_trajectories,
)


def log(level, message):
    print(f"[ARCAP_ALIGN_DEBUG][{level}] {message}")


def log_section(title):
    print(f"\n[ARCAP_ALIGN_DEBUG][STEP] {'=' * 12} {title} {'=' * 12}")


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
            f"时长={ts.max() - ts.min():.3f}s, 数值范围={vals.min():.6f}->{vals.max():.6f}, "
            f"均值={vals.mean():.6f}, 标准差={vals.std():.6f}"
        ),
    )


def detect_turning_points(data, name="trajectory", verbose=False):
    valid_data = [(t, p) for t, p in data if p is not None and np.isfinite(p)]
    if len(valid_data) < 3:
        log("WARN", f"{name}: 有效点不足3个，无法检测转向点 (valid={len(valid_data)})")
        return []

    timestamp = [d[0] for d in valid_data]
    position = np.array([d[1] for d in valid_data], dtype=float)

    max_diff = np.max(position) - np.min(position)
    stable_diff = 0.1 * max_diff

    if max_diff <= 1e-12:
        log("WARN", f"{name}: 轨迹变化幅度过小(max_diff={max_diff:.6e})，无法检测转向点")
        return []

    left = 0
    window_max, window_min = position[0], position[0]
    while left < len(position) - 1 and window_max - window_min < stable_diff:
        left += 1
        window_max = max(window_max, position[left])
        window_min = min(window_min, position[left])

    right = len(position) - 1
    window_max, window_min = position[right], position[right]
    while right > 0 and window_max - window_min < stable_diff:
        right -= 1
        window_max = max(window_max, position[right])
        window_min = min(window_min, position[right])

    position = position[left : right + 1]
    timestamp = timestamp[left : right + 1]

    if len(position) < 3:
        log("WARN", f"{name}: 去除稳定区后点数不足3个，无法检测转向点")
        return []

    position = gaussian_filter(position, sigma=1)

    turning_points = []
    for i in range(1, len(position) - 1):
        dt1 = position[i] - position[i - 1]
        dt2 = position[i + 1] - position[i]
        if np.sign(dt1) != np.sign(dt2):
            direction_change = np.sign(dt2) - np.sign(dt1)
            turning_points.append((timestamp[i], direction_change, position[i]))

    if verbose:
        log("INFO", f"{name}: 检测到转向点 {len(turning_points)} 个")
    return turning_points


def evaluate_offset(aruco_timepoints, aruco_values, traj_interp, get_axis_value, offset):
    arcap_samples = [get_axis_value(traj_interp, t + offset) for t in aruco_timepoints]
    valid_mask = np.array([v is not None and np.isfinite(v) for v in arcap_samples], dtype=bool)
    valid_count = int(np.sum(valid_mask))

    mse = 1e6
    corr = np.nan
    arcap_std = np.nan
    aruco_std = np.nan

    if valid_count >= 3:
        arcap_pos = np.array([v for v, m in zip(arcap_samples, valid_mask) if m], dtype=float)
        aruco_pos = np.array([v for v, m in zip(aruco_values, valid_mask) if m], dtype=float)

        arcap_std = float(np.std(arcap_pos))
        aruco_std = float(np.std(aruco_pos))
        if arcap_std > 1e-12 and aruco_std > 1e-12:
            arcap_norm = (arcap_pos - np.mean(arcap_pos)) / arcap_std
            aruco_norm = (aruco_pos - np.mean(aruco_pos)) / aruco_std
            mse = float(np.mean((aruco_norm - arcap_norm) ** 2))
            corr = float(np.corrcoef(aruco_norm, arcap_norm)[0, 1])

    return {
        "offset": float(offset),
        "valid_count": valid_count,
        "valid_ratio": float(valid_count / len(aruco_timepoints)),
        "mse": float(mse),
        "corr": corr,
        "arcap_std": arcap_std,
        "aruco_std": aruco_std,
    }


def save_timeline_debug_plot(
    out_path,
    aruco_start,
    aruco_end,
    proprio_start,
    proprio_end,
    init_offset,
    scan_offsets,
    valid_counts,
):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=False)

    axes[0].set_title("Time Range Overlap")
    axes[0].hlines(1.0, aruco_start, aruco_end, colors="#4c72b0", linewidth=8, label="ArUco")
    axes[0].hlines(0.5, proprio_start, proprio_end, colors="#dd8452", linewidth=8, label="ARCap(Proprio)")
    axes[0].hlines(
        1.5,
        aruco_start + init_offset,
        aruco_end + init_offset,
        colors="#55a868",
        linewidth=5,
        label=f"ArUco shifted by init_offset={init_offset:.3f}",
    )
    axes[0].set_yticks([0.5, 1.0, 1.5])
    axes[0].set_yticklabels(["ARCap", "ArUco", "ArUco+init_offset"])
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Offset vs Valid Sample Count")
    axes[1].plot(scan_offsets, valid_counts, color="#c44e52")
    axes[1].axhline(3, color="black", linestyle="--", linewidth=1, label="min valid=3")
    axes[1].axvline(init_offset, color="#55a868", linestyle="--", linewidth=1, label="init_offset")
    axes[1].set_xlabel("Offset (seconds)")
    axes[1].set_ylabel("Valid samples")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_error_curve_plot(out_path, stats, init_offset, best_offset):
    offsets = np.array([x["offset"] for x in stats], dtype=float)
    mses = np.array([x["mse"] for x in stats], dtype=float)
    corrs = np.array([x["corr"] for x in stats], dtype=float)
    valid = np.array([x["valid_count"] for x in stats], dtype=float)

    finite_mask = np.isfinite(mses) & (mses < 1e6)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].set_title("Offset Scan Diagnostics")
    axes[0].plot(offsets, valid, color="#4c72b0", label="valid_count")
    axes[0].axhline(3, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(init_offset, color="#55a868", linestyle=":", linewidth=1, label="init_offset")
    axes[0].axvline(best_offset, color="#c44e52", linestyle="--", linewidth=1.5, label="best_offset")
    axes[0].set_ylabel("Valid")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(offsets, mses, color="#dd8452", label="mse")
    if np.any(finite_mask):
        ymin = np.nanmin(mses[finite_mask])
        ymax = np.nanmax(mses[finite_mask])
        pad = max(0.05 * (ymax - ymin), 0.05)
        axes[1].set_ylim(ymin - pad, ymax + pad)
    axes[1].axvline(best_offset, color="#c44e52", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(offsets, corrs, color="#8172b3", label="corr")
    axes[2].axvline(best_offset, color="#c44e52", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("Offset (seconds)")
    axes[2].set_ylabel("Pearson r")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_turning_points_count_plot(out_path, stats_algo1, init_offset):
    offsets = np.array([x["offset"] for x in stats_algo1], dtype=float)
    valid_counts = np.array([x["valid_count"] for x in stats_algo1], dtype=float)
    turn_counts = np.array([x["turn_count"] for x in stats_algo1], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax2 = ax1.twinx()

    ax1.plot(offsets, valid_counts, color="#4c72b0", label="valid_count")
    ax1.axhline(3, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel("Valid count", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")

    ax2.plot(offsets, turn_counts, color="#c44e52", label="turning_points")
    ax2.set_ylabel("Turning points", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")

    ax1.axvline(init_offset, color="#55a868", linestyle="--", linewidth=1)
    ax1.set_xlabel("Offset (seconds)")
    ax1.set_title("Algorithm-1 Diagnoser: Valid points and turning points")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


@click.command()
@click.argument("session_dir")
@click.option("-c", "--calibration_dir", required=True, help="标定文件目录路径")
@click.option("--calibration_axis", type=str, default="x", help="用于对齐的坐标轴(x/y/z)")
@click.option("--init_offset", type=float, default=0.0, help="初始时间偏移估计值(秒)")
@click.option("--scan_min", type=float, default=-6.0, help="可视化扫描最小偏移(秒)")
@click.option("--scan_max", type=float, default=6.0, help="可视化扫描最大偏移(秒)")
@click.option("--scan_step", type=float, default=0.05, help="可视化扫描步长(秒)")
@click.option("--force_detect", is_flag=True, help="强制重新运行ArUco检测")
def main(session_dir, calibration_dir, calibration_axis, init_offset, scan_min, scan_max, scan_step, force_detect):
    log_section("参数与路径初始化")
    axis_map = {"x": 0, "y": 1, "z": 2}
    if calibration_axis not in axis_map:
        raise ValueError(f"无效 calibration_axis={calibration_axis}，仅支持 x/y/z")
    calibration_axis_index = axis_map[calibration_axis]

    def get_axis_value(interp, value):
        try:
            pose = interp(value)
            return -pose[calibration_axis_index]
        except ValueError:
            return None

    calibration_dir = pathlib.Path(calibration_dir)
    session_dir = pathlib.Path(__file__).parent.joinpath(session_dir).absolute()
    latency_calib_dir = session_dir.joinpath("latency_calibration")
    debug_out_dir = latency_calib_dir.joinpath("latency_debug")
    debug_out_dir.mkdir(parents=True, exist_ok=True)

    log("INFO", f"session_dir={session_dir}")
    log("INFO", f"calibration_dir={calibration_dir}")
    log("INFO", f"latency_calibration_dir={latency_calib_dir}")
    log("INFO", f"debug_out_dir={debug_out_dir}")
    log(
        "INFO",
        f"axis={calibration_axis}(index={calibration_axis_index}) init_offset={init_offset} "
        f"scan=[{scan_min}, {scan_max}] step={scan_step}",
    )

    if not latency_calib_dir.is_dir():
        raise FileNotFoundError(f"标定目录不存在: {latency_calib_dir}")

    mp4_candidates = list(latency_calib_dir.glob("**/*.MP4")) + list(latency_calib_dir.glob("**/*.mp4"))
    calibration_mp4 = get_single_path(mp4_candidates)
    log("INFO", f"使用标定视频: {calibration_mp4}")

    log_section("加载VR姿态轨迹插值器")
    traj_interp, _ = load_proprio_interp(session_dir, latency=0.0, extend_boundary=10)
    proprio_start_ts = traj_interp.left_max
    proprio_end_ts = traj_interp.right_min
    log("INFO", f"VR轨迹有效时间范围: {proprio_start_ts:.6f} -> {proprio_end_ts:.6f}")

    log_section("ArUco检测与轨迹加载")
    script_path = pathlib.Path(__file__).parent.parent.joinpath("scripts", "detect_aruco_newversion.py")
    camera_intrinsics = calibration_dir.joinpath("gopro_intrinsics_2_7k.json")
    aruco_config = calibration_dir.joinpath("aruco_config.yaml")
    aruco_pickle_path = latency_calib_dir.joinpath("tag_detection.pkl")

    if not script_path.is_file():
        raise FileNotFoundError(f"ArUco检测脚本不存在: {script_path}")
    if not camera_intrinsics.is_file():
        raise FileNotFoundError(f"相机内参文件不存在: {camera_intrinsics}")
    if not aruco_config.is_file():
        raise FileNotFoundError(f"ArUco配置文件不存在: {aruco_config}")

    if force_detect or (not aruco_pickle_path.is_file()):
        cmd = [
            "python",
            str(script_path),
            "--input",
            str(calibration_mp4),
            "--output",
            str(aruco_pickle_path),
            "--intrinsics_json",
            str(camera_intrinsics),
            "--aruco_yaml",
            str(aruco_config),
            "--num_workers",
            "1",
        ]
        log("INFO", "执行ArUco检测")
        t0 = time.time()
        result = subprocess.run(cmd)
        dt = time.time() - t0
        log("INFO", f"ArUco检测结束: returncode={result.returncode}, 耗时={dt:.2f}s")
        if result.returncode != 0:
            raise RuntimeError("ArUco检测失败")
    else:
        log("INFO", f"复用已有检测结果: {aruco_pickle_path}")

    video_start_time = mp4_get_start_datetime(str(calibration_mp4)).timestamp()
    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)

    aruco_trajectory = []
    frame_total = 0
    frame_has_tag = 0
    for frame in aruco_pkl:
        frame_total += 1
        if ARUCO_ID in frame["tag_dict"]:
            frame_has_tag += 1
            x = frame["tag_dict"][ARUCO_ID]["tvec"][0]
            aruco_trajectory.append((frame["time"] + video_start_time, x))

    log("INFO", f"Aruco结果帧数: total={frame_total}, 含目标tag={frame_has_tag}, 轨迹点数={len(aruco_trajectory)}")
    if len(aruco_trajectory) < 3:
        raise ValueError(f"Aruco轨迹点不足(<3): {len(aruco_trajectory)}")

    summarize_trajectory("aruco_trajectory", aruco_trajectory)

    aruco_trajectory = sorted(aruco_trajectory, key=lambda x: x[0])
    aruco_timepoints = np.array([t for t, _ in aruco_trajectory], dtype=float)
    aruco_values = np.array([v for _, v in aruco_trajectory], dtype=float)
    aruco_start_ts = float(aruco_timepoints[0])
    aruco_end_ts = float(aruco_timepoints[-1])

    overlap_start = max(proprio_start_ts, aruco_start_ts + init_offset)
    overlap_end = min(proprio_end_ts, aruco_end_ts + init_offset)
    overlap_sec = max(0.0, overlap_end - overlap_start)
    log("INFO", f"overlap_sec@init_offset={init_offset:.3f} -> {overlap_sec:.3f}s")

    # 长时域图
    extend_range = 15.0
    long_times = np.concatenate(
        [
            np.linspace(aruco_timepoints[0] - extend_range, aruco_timepoints[0], 100),
            aruco_timepoints,
            np.linspace(aruco_timepoints[-1], aruco_timepoints[-1] + extend_range, 100),
        ]
    )
    arcap_trajectory_for_plotting = [(t, get_axis_value(traj_interp, t + init_offset)) for t in long_times]
    plot_long_horizon_trajectory(
        aruco_trajectory,
        arcap_trajectory_for_plotting,
        title=f"Offset {init_offset:.3f}",
        save_dir=debug_out_dir.joinpath("01_long_horizon_offset_init.pdf"),
    )

    # 全局偏移扫描
    log_section("全局偏移扫描诊断")
    scan_offsets = np.arange(scan_min, scan_max + 0.5 * scan_step, scan_step)
    if len(scan_offsets) < 3:
        raise ValueError("扫描点太少，请增大扫描区间或减小步长")

    scan_stats = [
        evaluate_offset(aruco_timepoints, aruco_values, traj_interp, get_axis_value, offset)
        for offset in scan_offsets
    ]

    valid_counts = [x["valid_count"] for x in scan_stats]
    save_timeline_debug_plot(
        debug_out_dir.joinpath("02_timeline_and_valid_count.pdf"),
        aruco_start_ts,
        aruco_end_ts,
        proprio_start_ts,
        proprio_end_ts,
        init_offset,
        scan_offsets,
        valid_counts,
    )

    finite_candidates = [x for x in scan_stats if x["mse"] < 1e6 and np.isfinite(x["mse"])]
    if not finite_candidates:
        log("WARN", "扫描区间内没有任何offset达到有效匹配(>=3有效点 + 非零方差)")
        best_scan_offset = float(init_offset)
    else:
        best_scan_offset = float(min(finite_candidates, key=lambda x: x["mse"])["offset"])
        best_scan_mse = float(min(finite_candidates, key=lambda x: x["mse"])["mse"])
        log("INFO", f"扫描最佳offset={best_scan_offset:.6f}, mse={best_scan_mse:.6f}")

    # 算法1诊断图
    log_section("算法1诊断(转向点)")
    turn_point_aruco = detect_turning_points(aruco_trajectory, name="aruco", verbose=True)
    algo1_offsets = np.arange(init_offset - 3.0, init_offset + 3.0 + 0.25, 0.25)
    algo1_stats = []
    for offset in algo1_offsets:
        arcap_trajectory = [(t + offset, get_axis_value(traj_interp, t + offset)) for t in aruco_timepoints]
        arcap_valid = [(t, v) for t, v in arcap_trajectory if v is not None and np.isfinite(v)]
        turn_point_arcap = detect_turning_points(arcap_valid, name=f"arcap@{offset:.2f}", verbose=False)
        algo1_stats.append(
            {
                "offset": float(offset),
                "valid_count": len(arcap_valid),
                "turn_count": len(turn_point_arcap),
            }
        )

    save_turning_points_count_plot(
        debug_out_dir.joinpath("03_algo1_turning_points_vs_offset.pdf"),
        algo1_stats,
        init_offset,
    )

    # 选一个候选offset画算法1对齐图
    if len(scan_stats) > 0:
        plot_offset_for_algo1 = best_scan_offset
    else:
        plot_offset_for_algo1 = init_offset

    arcap_plot = [(t + plot_offset_for_algo1, get_axis_value(traj_interp, t + plot_offset_for_algo1)) for t in aruco_timepoints]
    arcap_plot_valid = [(t, v) for t, v in arcap_plot if v is not None and np.isfinite(v)]
    turn_point_arcap_plot = detect_turning_points(arcap_plot_valid, name="arcap_plot", verbose=False)
    plot_trajectories(
        aruco_trajectory,
        arcap_plot_valid,
        turn_point_aruco,
        turn_point_arcap_plot,
        debug_out_dir.joinpath("04_algo1_overlay_best_scan_offset.pdf"),
    )

    # 算法2: 与原01一致，继续使用 custom_minimize 做局部优化
    log_section("算法2诊断(最小二乘)")

    def mse_error(x):
        stat = evaluate_offset(aruco_timepoints, aruco_values, traj_interp, get_axis_value, x[0])
        return stat["mse"]

    left_offset_bound = -1.0 + init_offset
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

        if shift_iter > 20:
            log("WARN", "算法2搜索超过20轮，提前停止以避免无限漂移")
            break

    best_opt_offset = float(res.x[0])
    best_opt_stat = evaluate_offset(aruco_timepoints, aruco_values, traj_interp, get_axis_value, best_opt_offset)
    log(
        "INFO",
        f"算法2结果: offset={best_opt_offset:.6f}, mse={best_opt_stat['mse']:.6f}, "
        f"valid_count={best_opt_stat['valid_count']}",
    )

    save_error_curve_plot(
        debug_out_dir.joinpath("05_algo2_error_curve.pdf"),
        scan_stats,
        init_offset,
        best_opt_offset,
    )

    # 算法2最终叠加图
    arcap_final = [(t + best_opt_offset, get_axis_value(traj_interp, t + best_opt_offset)) for t in aruco_timepoints]
    arcap_final_valid = [(t, v) for t, v in arcap_final if v is not None and np.isfinite(v)]
    calibrated_arcap_trajectory = [(t - best_opt_offset, v) for t, v in arcap_final_valid]
    plot_trajectories(
        aruco_trajectory,
        calibrated_arcap_trajectory,
        [],
        [],
        debug_out_dir.joinpath("06_algo2_overlay_final.pdf"),
    )

    diagnosis = {
        "inputs": {
            "session_dir": str(session_dir),
            "calibration_dir": str(calibration_dir),
            "calibration_axis": calibration_axis,
            "init_offset": init_offset,
            "scan_min": scan_min,
            "scan_max": scan_max,
            "scan_step": scan_step,
        },
        "time_ranges": {
            "aruco_start_ts": aruco_start_ts,
            "aruco_end_ts": aruco_end_ts,
            "proprio_start_ts": proprio_start_ts,
            "proprio_end_ts": proprio_end_ts,
            "overlap_at_init_offset_sec": overlap_sec,
        },
        "counts": {
            "aruco_frame_total": frame_total,
            "aruco_frame_has_tag": frame_has_tag,
            "aruco_trajectory_points": len(aruco_trajectory),
            "aruco_turning_points": len(turn_point_aruco),
        },
        "scan_summary": {
            "max_valid_count": int(np.max([x["valid_count"] for x in scan_stats])),
            "min_valid_count": int(np.min([x["valid_count"] for x in scan_stats])),
            "num_finite_candidates": len(finite_candidates),
            "best_scan_offset": best_scan_offset,
        },
        "algo2_result": {
            "best_opt_offset": best_opt_offset,
            "best_opt_mse": best_opt_stat["mse"],
            "best_opt_valid_count": best_opt_stat["valid_count"],
        },
        "scan_stats": scan_stats,
        "algo1_stats": algo1_stats,
    }

    diagnosis_path = debug_out_dir.joinpath("latency_diagnosis_report.json")
    with open(str(diagnosis_path), "w") as fp:
        json.dump(diagnosis, fp, indent=2)

    log("INFO", f"诊断报告已保存: {diagnosis_path}")
    log("INFO", f"可视化目录: {debug_out_dir}")
    log_section("处理完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR", f"执行失败: {type(exc).__name__}: {exc}")
        raise
