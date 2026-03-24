#!/usr/bin/env python3
"""
将 exUMI session 数据转换为 LeRobot Diffusion Policy 可直接训练的数据格式。

转换目标
--------
1) 输入整个 session 目录（包含 demos 和 tactile_*/angle_*.json）
2) 使用每个 demo 的 raw_video.mp4 + aligned_arcap_poses.json
3) 生成按帧索引的数据集，核心字段：
   - observation.image
   - action (10维: 3平移 + 6D旋转 + 1夹爪)

动作定义
--------
- 平移 (x, y, z): 使用训练集全局 Z-score 标准化
- 旋转 (qx, qy, qz, qw): 转换为 6D rotation 表示
- 夹爪 (gripper): 来自 angle_*.json，归一化到 [0,1]

严格对齐原则
------------
视频帧和控制序列使用同一时间轴，且逐帧构建：
  t_i = t0 + i / fps

其中：
- t0 来自 mp4 timecode 解析（与现有 pipeline 保持一致）
- i 为视频解码帧索引
- 同一个 t_i 同时用于：取 observation.image、取 pose[i]、插值 gripper

这样可以保证视频抽帧逻辑与位姿/夹爪重采样逻辑完全一致。

示例
----
python scripts/convert_session_to_lerobot_dp.py \
    --session_dir data/my_session/batch_1 \
    --output_dir data/my_session/batch_1/lerobot_dp_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import av
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from datasets import Dataset, DatasetDict, Features, Image, Sequence as HFSequence, Value
except ImportError as e:
    raise ImportError(
        "未检测到 huggingface datasets。请先安装: pip install datasets"
    ) from e

from umi.common.timecode_util import mp4_get_start_datetime


@dataclass
class EpisodeMeta:
    """描述一个 demo episode 的元信息。"""

    episode_name: str
    episode_index: int
    video_path: Path
    aligned_pose_path: Path
    start_timestamp: float
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 session 数据转换为 LeRobot DP 可训练的 Hugging Face datasets 格式"
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="session 目录路径（例如 data/xxx/batch_1）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（将写入 hf_dataset 与统计信息）",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="按 episode 级别划分训练集比例，默认 0.9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="episode 划分随机种子",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="开启严格检查：视频解码帧数必须与 aligned pose 条目数一致（默认开启）",
    )
    return parser.parse_args()


def _safe_angle_to_scalar(v) -> float:
    """将 angle JSON 中可能是标量或长度为1的数组安全转换为 float。"""
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            raise ValueError("检测到空 angle 元素，无法转换为标量")
        return float(v[0])
    return float(v)


def load_angle_stream(session_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取整个 session 的 angle 数据流。

    数据来源：session_dir/tactile_*/angle_*.json
    每个文件结构：
    {
      "angles": [[...], [...], ...],
      "data": [...],
      "timestamps": [...]
    }

    返回：
    - times: shape (N,), 绝对时间戳（秒）
    - vals : shape (N,), 对应 angle 标量
    """
    angle_files = sorted(session_dir.glob("tactile_*/angle_*.json"))
    if len(angle_files) == 0:
        raise FileNotFoundError(
            f"未在 {session_dir} 下找到 tactile_*/angle_*.json，无法构建 gripper 序列"
        )

    all_times: List[float] = []
    all_vals: List[float] = []

    for p in angle_files:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        timestamps = obj.get("timestamps", [])
        angles = obj.get("angles", [])
        if len(timestamps) != len(angles):
            raise ValueError(
                f"angle 文件长度不一致: {p} | timestamps={len(timestamps)} angles={len(angles)}"
            )
        all_times.extend(float(x) for x in timestamps)
        all_vals.extend(_safe_angle_to_scalar(a) for a in angles)

    if len(all_times) == 0:
        raise ValueError("angle 数据为空，无法进行 gripper 构建")

    times = np.asarray(all_times, dtype=np.float64)
    vals = np.asarray(all_vals, dtype=np.float64)

    # 按时间排序，保证后续插值稳定。
    order = np.argsort(times)
    times = times[order]
    vals = vals[order]

    # 去除重复时间戳（保留最后一次观测）。
    # np.unique 返回首个索引，因此我们先反转再取 unique 再还原。
    rev_times = times[::-1]
    rev_vals = vals[::-1]
    uniq_rev_times, uniq_rev_idx = np.unique(rev_times, return_index=True)
    uniq_times = uniq_rev_times[::-1]
    uniq_vals = rev_vals[uniq_rev_idx][::-1]

    if len(uniq_times) < 2:
        raise ValueError("有效 angle 时间点少于2个，无法插值")
    return uniq_times, uniq_vals


def interpolate_angle(times_ref: np.ndarray, vals_ref: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """
    在统一时间轴 query_ts 上插值 gripper angle。

    这里使用 np.interp 做一维线性插值。
    对于边界外的 query_ts，按边界值外推（left/right）。
    """
    return np.interp(query_ts, times_ref, vals_ref, left=vals_ref[0], right=vals_ref[-1])


def discover_episodes(session_dir: Path) -> List[EpisodeMeta]:
    """
    发现 session 下所有可用 demo，并提取时间基准信息。

    预期结构：
    session_dir/
      demos/
        demo_xxx/
          raw_video.mp4
          aligned_arcap_poses.json
    """
    demos_dir = session_dir / "demos"
    if not demos_dir.is_dir():
        raise FileNotFoundError(f"未找到 demos 目录: {demos_dir}")

    metas: List[EpisodeMeta] = []
    demo_video_paths = sorted(demos_dir.glob("*/raw_video.mp4"))
    if len(demo_video_paths) == 0:
        raise FileNotFoundError(f"在 {demos_dir} 下未找到任何 raw_video.mp4")

    for epi, video_path in enumerate(demo_video_paths):
        aligned_path = video_path.parent / "aligned_arcap_poses.json"
        if not aligned_path.is_file():
            # 直接跳过，避免混入未对齐 demo。
            continue

        with av.open(str(video_path), "r") as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)

        start_ts = mp4_get_start_datetime(str(video_path)).timestamp()
        metas.append(
            EpisodeMeta(
                episode_name=video_path.parent.name,
                episode_index=epi,
                video_path=video_path,
                aligned_pose_path=aligned_path,
                start_timestamp=float(start_ts),
                fps=float(fps),
            )
        )

    if len(metas) == 0:
        raise FileNotFoundError(
            "没有可用 demo（可能缺少 aligned_arcap_poses.json，请先运行 AR_03_align_trajectory）"
        )
    return metas


def split_episodes(metas: List[EpisodeMeta], train_ratio: float, seed: int) -> Tuple[List[EpisodeMeta], List[EpisodeMeta]]:
    """按 episode 级别做 train/val 划分，避免同一轨迹泄漏到不同集合。"""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio 必须在 (0,1) 之间，当前为 {train_ratio}")

    order = list(range(len(metas)))
    rng = random.Random(seed)
    rng.shuffle(order)

    # 至少保证 train 和 val 都非空（当 episode 数 >= 2 时）。
    n_total = len(metas)
    n_train = int(math.floor(n_total * train_ratio))
    if n_total >= 2:
        n_train = min(max(n_train, 1), n_total - 1)
    else:
        n_train = n_total

    train_idx = set(order[:n_train])
    train_eps = [m for i, m in enumerate(metas) if i in train_idx]
    val_eps = [m for i, m in enumerate(metas) if i not in train_idx]
    return train_eps, val_eps


def load_pose_array(aligned_pose_path: Path) -> np.ndarray:
    """
    加载 aligned_arcap_poses.json 中的 pose 列表。

    预期：obj["pose"] 是 N x 7 的数组，顺序为 [x,y,z,qx,qy,qz,qw]。
    """
    with aligned_pose_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    pose = np.asarray(obj["pose"], dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(
            f"pose 维度错误: {aligned_pose_path} | shape={pose.shape}，期望 (N,7)"
        )
    return pose


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    将四元数 (x,y,z,w) 转成 6D rotation 表示。

    实现方式：
    1) 归一化四元数
    2) 转旋转矩阵 R (3x3)
    3) 取前两列并展平得到 6 维向量
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("检测到接近零范数四元数，无法转换为旋转")
    q = q / n
    rot_m = R.from_quat(q).as_matrix()  # 3x3
    rot6d = np.concatenate([rot_m[:, 0], rot_m[:, 1]], axis=0)
    return rot6d.astype(np.float32)


def build_frame_timestamps(start_ts: float, fps: float, n: int) -> np.ndarray:
    """构建长度为 n 的逐帧绝对时间戳数组。"""
    idx = np.arange(n, dtype=np.float64)
    return start_ts + idx / fps


def compute_train_stats(
    train_eps: Sequence[EpisodeMeta],
    angle_times: np.ndarray,
    angle_vals: np.ndarray,
) -> Dict[str, List[float]]:
    """
    用训练集全局统计量计算标准化参数：
    - 平移 xyz 的 mean/std
    - gripper angle 的 min/max
    """
    xyz_sum = np.zeros(3, dtype=np.float64)
    xyz_sq_sum = np.zeros(3, dtype=np.float64)
    total_count = 0

    g_min = np.inf
    g_max = -np.inf

    for ep in train_eps:
        pose = load_pose_array(ep.aligned_pose_path)
        n = pose.shape[0]
        xyz = pose[:, :3]

        xyz_sum += xyz.sum(axis=0)
        xyz_sq_sum += np.square(xyz).sum(axis=0)
        total_count += n

        ts = build_frame_timestamps(ep.start_timestamp, ep.fps, n)
        g = interpolate_angle(angle_times, angle_vals, ts)
        g_min = min(g_min, float(np.min(g)))
        g_max = max(g_max, float(np.max(g)))

    if total_count == 0:
        raise ValueError("训练集为空，无法计算标准化参数")

    mean = xyz_sum / total_count
    var = xyz_sq_sum / total_count - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    # 若 gripper 全程恒定，避免除0。
    if not np.isfinite(g_min) or not np.isfinite(g_max):
        raise ValueError("未能从训练集计算 gripper min/max")
    if abs(g_max - g_min) < 1e-12:
        g_max = g_min + 1.0

    return {
        "xyz_mean": mean.tolist(),
        "xyz_std": std.tolist(),
        "gripper_min": [float(g_min)],
        "gripper_max": [float(g_max)],
    }


def normalize_xyz(xyz: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """对 xyz 进行 Z-score 标准化。"""
    return (xyz - mean) / std


def normalize_gripper(v: np.ndarray, g_min: float, g_max: float) -> np.ndarray:
    """将 gripper angle 线性归一化到 [0,1]，并做 clip。"""
    out = (v - g_min) / (g_max - g_min)
    return np.clip(out, 0.0, 1.0)


def iter_episode_samples(
    ep: EpisodeMeta,
    angle_times: np.ndarray,
    angle_vals: np.ndarray,
    xyz_mean: np.ndarray,
    xyz_std: np.ndarray,
    g_min: float,
    g_max: float,
    strict: bool,
) -> Iterator[Dict]:
    """
    逐帧迭代一个 episode 的样本。

    说明：
    - 该函数是“严格一致性”的核心：
      每解码一帧，立刻在同一帧索引 i 上构建 action。
    - 不先整段缓存，避免中间处理改变帧计数。
    """
    pose = load_pose_array(ep.aligned_pose_path)
    n_pose = pose.shape[0]

    # 先构建这一条 episode 的 gripper 时间序列，长度与 pose 对齐。
    ts_arr = build_frame_timestamps(ep.start_timestamp, ep.fps, n_pose)
    g_raw = interpolate_angle(angle_times, angle_vals, ts_arr)
    g_norm = normalize_gripper(g_raw, g_min, g_max)

    decoded = 0
    with av.open(str(ep.video_path), "r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for i, frame in enumerate(container.decode(stream)):
            decoded += 1
            if i >= n_pose:
                if strict:
                    raise ValueError(
                        f"视频帧数多于 pose 条目: {ep.episode_name} | decoded>{n_pose}"
                    )
                break

            img = frame.to_ndarray(format="rgb24")

            xyz = pose[i, :3]
            quat = pose[i, 3:7]
            xyz_n = normalize_xyz(xyz, xyz_mean, xyz_std).astype(np.float32)
            rot6d = quat_xyzw_to_rot6d(quat)
            grip = np.array([g_norm[i]], dtype=np.float32)

            action = np.concatenate([xyz_n, rot6d, grip], axis=0)
            if action.shape[0] != 10:
                raise RuntimeError(
                    f"动作维度错误: {action.shape[0]}，期望 10 (3+6+1)"
                )

            yield {
                "observation.image": img,
                "action": action.tolist(),
                "episode_index": int(ep.episode_index),
                "frame_index": int(i),
                "timestamp": float(ts_arr[i]),
            }

    if strict and decoded != n_pose:
        raise ValueError(
            f"视频帧数与 pose 条目不一致: {ep.episode_name} | decoded={decoded}, pose={n_pose}"
        )


def build_split_dataset(
    episodes: Sequence[EpisodeMeta],
    angle_times: np.ndarray,
    angle_vals: np.ndarray,
    stats: Dict[str, List[float]],
    strict: bool,
) -> Dataset:
    """
    构建单个 split 的 HF Dataset。

    这里使用 from_generator，避免一次性将所有帧加载到内存。
    """
    xyz_mean = np.asarray(stats["xyz_mean"], dtype=np.float64)
    xyz_std = np.asarray(stats["xyz_std"], dtype=np.float64)
    g_min = float(stats["gripper_min"][0])
    g_max = float(stats["gripper_max"][0])

    def gen() -> Iterator[Dict]:
        for ep in episodes:
            for sample in iter_episode_samples(
                ep=ep,
                angle_times=angle_times,
                angle_vals=angle_vals,
                xyz_mean=xyz_mean,
                xyz_std=xyz_std,
                g_min=g_min,
                g_max=g_max,
                strict=strict,
            ):
                yield sample

    features = Features(
        {
            "observation.image": Image(),
            "action": HFSequence(feature=Value("float32"), length=10),
            "episode_index": Value("int32"),
            "frame_index": Value("int32"),
            "timestamp": Value("float64"),
        }
    )
    return Dataset.from_generator(gen, features=features)


def main() -> None:
    args = parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Session: {session_dir}")
    print(f"[INFO] Output : {output_dir}")

    # 步骤1: 发现可用 episode
    episodes = discover_episodes(session_dir)
    print(f"[INFO] Found episodes: {len(episodes)}")

    # 步骤2: 读取 angle 全局流，用于构建 gripper 维度
    angle_times, angle_vals = load_angle_stream(session_dir)
    print(
        "[INFO] Angle stream loaded: "
        f"N={len(angle_times)}, range=[{angle_times[0]:.6f}, {angle_times[-1]:.6f}]"
    )

    # 步骤3: episode 级别划分 train/validation
    train_eps, val_eps = split_episodes(episodes, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] Train episodes: {len(train_eps)} | Val episodes: {len(val_eps)}")

    # 步骤4: 只用训练集计算标准化参数
    stats = compute_train_stats(train_eps, angle_times, angle_vals)
    print("[INFO] Normalization stats:")
    print(f"       xyz_mean={stats['xyz_mean']}")
    print(f"       xyz_std ={stats['xyz_std']}")
    print(f"       g_min   ={stats['gripper_min'][0]:.6f}")
    print(f"       g_max   ={stats['gripper_max'][0]:.6f}")

    # 步骤5: 构建 HF datasets（按帧索引，严格对齐）
    train_ds = build_split_dataset(
        episodes=train_eps,
        angle_times=angle_times,
        angle_vals=angle_vals,
        stats=stats,
        strict=args.strict,
    )
    val_ds = build_split_dataset(
        episodes=val_eps,
        angle_times=angle_times,
        angle_vals=angle_vals,
        stats=stats,
        strict=args.strict,
    )

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # 步骤6: 保存结果
    hf_dir = output_dir / "hf_dataset"
    ds_dict.save_to_disk(str(hf_dir))

    stats_path = output_dir / "normalization_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    split_meta = {
        "train_episodes": [ep.episode_name for ep in train_eps],
        "validation_episodes": [ep.episode_name for ep in val_eps],
        "session_dir": str(session_dir),
    }
    split_path = output_dir / "split_meta.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] HF dataset saved to: {hf_dir}")
    print(f"[DONE] Stats saved to     : {stats_path}")
    print(f"[DONE] Split meta saved to: {split_path}")


if __name__ == "__main__":
    main()
