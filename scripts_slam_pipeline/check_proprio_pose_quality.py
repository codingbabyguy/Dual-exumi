"""
check_proprio_pose_quality.py - 原始 proprio(pose) 质量检查

用途:
- 检查 tactile_*/angle_*.json 中的 timestamps 是否异常(重复/乱序/大跳变)
- 检查 pose 数据是否含 NaN/Inf
- 检查 x/y/z 目标轴是否全部无效
- 输出可视化图和 JSON 报告, 方便定位 AR_01 中 traj_interp 取值失败原因

用法:
python scripts_slam_pipeline/check_proprio_pose_quality.py <session_dir>
python scripts_slam_pipeline/check_proprio_pose_quality.py <session_dir> --jump_sigma 8 --min_jump_sec 0.2
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np


def log(level, msg):
    print(f"[PROPRIO_CHECK][{level}] {msg}")


def robust_jump_threshold(dt_pos, jump_sigma, min_jump_sec):
    if dt_pos.size == 0:
        return min_jump_sec
    med = float(np.median(dt_pos))
    mad = float(np.median(np.abs(dt_pos - med)))
    robust_std = 1.4826 * mad
    if robust_std < 1e-12:
        robust_std = float(np.std(dt_pos))
    return max(min_jump_sec, med + jump_sigma * robust_std)


def save_timestamp_plots(out_dir, all_t, all_dt, jump_threshold, jump_mask):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图1: dt 直方图
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    dt_pos = all_dt[all_dt > 0]
    if dt_pos.size > 0:
        ax.hist(dt_pos, bins=80, color="#4c72b0", alpha=0.85)
    ax.axvline(jump_threshold, color="red", linestyle="--", linewidth=1.5, label=f"jump_threshold={jump_threshold:.6f}s")
    ax.set_xlabel("delta_t (s)")
    ax.set_ylabel("count")
    ax.set_title("Timestamp Delta Histogram")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_dir.joinpath("01_timestamp_dt_hist.pdf")))
    plt.close(fig)

    # 图2: dt 随索引变化
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))
    x = np.arange(len(all_dt))
    ax.plot(x, all_dt, color="#4c72b0", linewidth=1, label="delta_t")
    if np.any(jump_mask):
        ax.scatter(x[jump_mask], all_dt[jump_mask], color="red", s=12, label="large_jump")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(jump_threshold, color="red", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.set_xlabel("sample index")
    ax.set_ylabel("delta_t (s)")
    ax.set_title("Timestamp Delta Timeline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_dir.joinpath("02_timestamp_dt_timeline.pdf")))
    plt.close(fig)

    # 图3: 时间戳本身随索引变化
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))
    ax.plot(np.arange(len(all_t)), all_t, color="#55a868", linewidth=1)
    ax.set_xlabel("sample index")
    ax.set_ylabel("timestamp (s)")
    ax.set_title("Timestamp Sequence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_dir.joinpath("03_timestamp_sequence.pdf")))
    plt.close(fig)


def save_pose_plots(out_dir, pose_arr):
    out_dir.mkdir(parents=True, exist_ok=True)

    n = pose_arr.shape[0]
    idx = np.arange(n)

    axis_names = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for k in range(3):
        if pose_arr.shape[1] <= k:
            axes[k].text(0.5, 0.5, f"axis {axis_names[k]} not found", ha="center", va="center", transform=axes[k].transAxes)
            axes[k].set_axis_off()
            continue
        vals = pose_arr[:, k]
        valid = np.isfinite(vals)
        axes[k].plot(idx[valid], vals[valid], color="#4c72b0", linewidth=0.8, label=f"{axis_names[k]} valid")
        if np.any(~valid):
            axes[k].scatter(idx[~valid], np.zeros(np.sum(~valid)), color="red", s=8, label="invalid")
        axes[k].set_ylabel(axis_names[k])
        axes[k].grid(True, alpha=0.25)
        axes[k].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("sample index")
    fig.suptitle("Pose Axis Timeline (x/y/z)")
    fig.tight_layout()
    fig.savefig(str(out_dir.joinpath("04_pose_axis_timeline.pdf")))
    plt.close(fig)

    # 各轴有效率
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    labels = []
    ratios = []
    for k, name in enumerate(axis_names):
        labels.append(name)
        if pose_arr.shape[1] <= k:
            ratios.append(0.0)
        else:
            ratios.append(float(np.mean(np.isfinite(pose_arr[:, k]))))
    ax.bar(labels, ratios, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("finite ratio")
    ax.set_title("Pose Axis Finite Ratio")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(str(out_dir.joinpath("05_pose_axis_finite_ratio.pdf")))
    plt.close(fig)


@click.command()
@click.argument("session_dir")
@click.option("--jump_sigma", type=float, default=8.0, help="大跳变阈值: median + jump_sigma * robust_std")
@click.option("--min_jump_sec", type=float, default=0.2, help="大跳变最小秒数阈值")
def main(session_dir, jump_sigma, min_jump_sec):
    session_dir = pathlib.Path(session_dir).expanduser()
    if not session_dir.is_absolute():
        session_dir = pathlib.Path(__file__).parent.joinpath(session_dir)
    session_dir = session_dir.absolute()

    log("INFO", f"session_dir={session_dir}")

    proprio_files = sorted(session_dir.glob("tactile_*/angle_*.json"))
    if len(proprio_files) == 0:
        raise FileNotFoundError(f"No tactile_*/angle_*.json found in {session_dir}")

    out_dir = session_dir.joinpath("latency_calibration", "proprio_quality_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_t = []
    all_pose = []
    file_stats = []

    for fp in proprio_files:
        with open(str(fp), "r") as f:
            data = json.load(f)

        ts = np.array(data.get("timestamps", []), dtype=float)
        pose = np.array(data.get("data", []), dtype=float)

        if len(ts) != len(pose):
            log("WARN", f"len mismatch: {fp.name} timestamps={len(ts)} data={len(pose)}")

        n = min(len(ts), len(pose))
        ts = ts[:n]
        pose = pose[:n]

        dt = np.diff(ts) if n >= 2 else np.array([], dtype=float)
        duplicated = int(np.sum(dt == 0))
        non_monotonic = int(np.sum(dt < 0))

        finite_mask = np.isfinite(pose)
        pose_nan_inf = int(np.size(pose) - np.sum(finite_mask))

        file_stats.append(
            {
                "file": str(fp),
                "samples": int(n),
                "duplicated_timestamps": duplicated,
                "non_monotonic_timestamps": non_monotonic,
                "pose_nan_inf_count": pose_nan_inf,
            }
        )

        all_t.append(ts)
        all_pose.append(pose)

    all_t = np.concatenate(all_t) if all_t else np.array([], dtype=float)
    all_pose = np.concatenate(all_pose, axis=0) if all_pose else np.zeros((0, 0), dtype=float)

    if all_t.size < 2:
        raise ValueError("Not enough timestamps to analyze.")

    all_dt = np.diff(all_t)
    dt_pos = all_dt[all_dt > 0]
    jump_threshold = robust_jump_threshold(dt_pos, jump_sigma=jump_sigma, min_jump_sec=min_jump_sec)
    jump_mask = all_dt > jump_threshold

    dup_total = int(np.sum(all_dt == 0))
    non_mono_total = int(np.sum(all_dt < 0))
    jump_total = int(np.sum(jump_mask))

    # pose 维度统计
    pose_dim = int(all_pose.shape[1]) if all_pose.ndim == 2 else 0
    axis_stats = {}
    axis_names = ["x", "y", "z"]
    for i, name in enumerate(axis_names):
        if pose_dim <= i:
            axis_stats[name] = {
                "exists": False,
                "valid_count": 0,
                "invalid_count": int(all_pose.shape[0]) if all_pose.ndim == 2 else 0,
                "finite_ratio": 0.0,
                "all_invalid": True,
            }
            continue
        vals = all_pose[:, i]
        valid = np.isfinite(vals)
        vc = int(np.sum(valid))
        ic = int(np.sum(~valid))
        axis_stats[name] = {
            "exists": True,
            "valid_count": vc,
            "invalid_count": ic,
            "finite_ratio": float(vc / len(vals)) if len(vals) > 0 else 0.0,
            "all_invalid": vc == 0,
            "min": float(np.nanmin(vals)) if vc > 0 else None,
            "max": float(np.nanmax(vals)) if vc > 0 else None,
            "std": float(np.nanstd(vals)) if vc > 0 else None,
        }

    # 汇总输出
    report = {
        "session_dir": str(session_dir),
        "num_files": len(proprio_files),
        "num_samples": int(len(all_t)),
        "timestamp_stats": {
            "duplicated_total": dup_total,
            "non_monotonic_total": non_mono_total,
            "large_jump_total": jump_total,
            "jump_threshold_sec": float(jump_threshold),
            "dt_median_sec": float(np.median(dt_pos)) if dt_pos.size > 0 else None,
            "dt_p95_sec": float(np.percentile(dt_pos, 95)) if dt_pos.size > 0 else None,
            "dt_max_sec": float(np.max(all_dt)) if all_dt.size > 0 else None,
            "top10_jump_indices": np.where(jump_mask)[0][:10].tolist(),
            "top10_jump_values_sec": all_dt[jump_mask][:10].tolist(),
        },
        "pose_stats": {
            "pose_dim": pose_dim,
            "axis": axis_stats,
            "nan_inf_total": int(np.size(all_pose) - np.sum(np.isfinite(all_pose))),
        },
        "file_stats": file_stats,
    }

    report_path = out_dir.joinpath("proprio_quality_report.json")
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)

    save_timestamp_plots(out_dir, all_t, all_dt, jump_threshold, jump_mask)
    if all_pose.ndim == 2 and all_pose.shape[0] > 0:
        save_pose_plots(out_dir, all_pose)

    log("INFO", f"报告已保存: {report_path}")
    log("INFO", f"可视化目录: {out_dir}")
    log("INFO", f"timestamps异常: duplicate={dup_total}, non_monotonic={non_mono_total}, large_jump={jump_total}")
    for name in ["x", "y", "z"]:
        st = axis_stats[name]
        log(
            "INFO",
            f"axis={name}: exists={st['exists']} valid={st['valid_count']} invalid={st['invalid_count']} all_invalid={st['all_invalid']}",
        )


if __name__ == "__main__":
    main()
