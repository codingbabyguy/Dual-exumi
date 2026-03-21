"""
Unified controller for VR pose and tactile collection with batch-based saving.

Usage:
    python do_umi.py

Controls:
    w: start a new batch (creates batch_N subfolder)
    e: end current batch (flush and stop saving)
    q: quit

Behavior:
    - VR and tactile pipelines are started immediately and keep receiving data.
    - Data is only saved while a batch is active.
    - Each batch has its own subdirectory under the user-provided root folder.
"""

import os
import sys
import time
import errno
import threading
import select
import tty
import termios
import socket
from datetime import datetime

import numpy as np

from ARCap.data_collection_umi import DataChunker
from ARCap.quest_robot_module_clean import QuestRightArmLeapModule
from ARCap.ip_config import (
    VR_HOST,
    LOCAL_HOST,
    POSE_CMD_PORT,
    IK_RESULT_PORT,
    TACTILE_CAMERA,
)
from ARCap.tactile_collection_process import TactileCollectionEnv, UsbCamera
from multiprocessing.managers import SharedMemoryManager


def save_worldframe_if_ready(quest: QuestRightArmLeapModule, save_dir: str, marker: dict):
    """Save the latest worldframe only once into pose/worldframe_<ts>/WorldFrame.npy."""
    if not save_dir:
        return
    world_frame = getattr(quest, "latest_world_frame", None)
    world_ts = getattr(quest, "latest_world_frame_ts", None)
    if world_frame is None or world_ts is None:
        return
    # Unity sends worldframe only once before the first batch.
    # Keep a single persisted copy for the whole session.
    if marker.get("saved"):
        return

    world_dir = os.path.join(save_dir, f"worldframe_{world_ts}")
    os.makedirs(world_dir, exist_ok=True)
    save_path = os.path.join(world_dir, "WorldFrame.npy")
    np.save(save_path, world_frame)
    marker["saved"] = True
    print(f"[VR] WorldFrame saved successfully: {save_path}")


# ---------- terminal utilities ----------
class RawMode:
    """Context manager to put stdin into raw mode for single-key capture."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.prev_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.prev_attrs)


def read_key(timeout=0.1):
    """Non-blocking single-key read; returns None if no key within timeout."""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return None


# ---------- shared state ----------
class SharedState:
    def __init__(self, root_dir):
        self.lock = threading.Lock()
        self.batch_id = 1
        self.batch_active = False
        self.root_dir = root_dir
        self.current_batch_dir = None
        self.vr_dir = None
        self.tactile_dir = None
        self.tactile_folder_name = None

    def start_batch(self):
        with self.lock:
            if self.batch_active:
                return None
            batch_dir = os.path.join(self.root_dir, f"batch_{self.batch_id}")
            
            # 创建符合数据处理流程的文件夹结构
            vr_dir = os.path.join(batch_dir, "pose")
            self.tactile_folder_name = f"tactile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tactile_dir = os.path.join(batch_dir, self.tactile_folder_name)
            
            # 创建所有必需的文件夹
            os.makedirs(vr_dir, exist_ok=True)
            os.makedirs(tactile_dir, exist_ok=True)
            os.makedirs(os.path.join(batch_dir, "latency_calibration"), exist_ok=True)
            os.makedirs(os.path.join(batch_dir, "raw_videos"), exist_ok=True)
            
            self.batch_active = True
            self.current_batch_dir = batch_dir
            self.vr_dir = vr_dir
            self.tactile_dir = tactile_dir
            print(f"[INFO] Batch {self.batch_id} started. Directories created:")
            print(f"  - VR数据: {vr_dir}")
            print(f"  - 触觉数据: {tactile_dir}")
            print(f"  - 标定视频: {batch_dir}/latency_calibration")
            print(f"  - 原始视频: {batch_dir}/raw_videos")
            return batch_dir

    def end_batch(self):
        with self.lock:
            if not self.batch_active:
                return None
            self.batch_active = False
            finished_dir = self.current_batch_dir
            
            # 输出批次的完整结构信息
            print(f"[INFO] Batch {self.batch_id} ended. 数据保存结构:")
            print(f"  - 批次目录: {finished_dir}")
            print(f"  - VR姿态数据: {finished_dir}/pose/")
            print(f"  - 触觉数据: {finished_dir}/{self.tactile_folder_name}/")
            print(f"  - 标定视频: {finished_dir}/latency_calibration/")
            print(f"  - 原始视频: {finished_dir}/raw_videos/")
            
            self.current_batch_dir = None
            self.vr_dir = None
            self.tactile_dir = None
            self.tactile_folder_name = None
            self.batch_id += 1
            return finished_dir

    def snapshot(self):
        with self.lock:
            return {
                "batch_active": self.batch_active,
                "vr_dir": self.vr_dir,
                "tactile_dir": self.tactile_dir,
            }


# ---------- worker threads ----------
def vr_worker(
    state: SharedState,
    stop_event: threading.Event,
    chunker: DataChunker,
    quest: QuestRightArmLeapModule,
    freq_hz: int = 30,
):
    print("[VR] VR worker thread starting...")
    print("[VR] Reusing calibrated Quest socket for data collection.")
    last_tick = time.time()
    worldframe_marker = {"saved": False, "last_pending_log": 0.0}
    try:
        while not stop_event.is_set():
            try:
                wrist, head_pose = quest.receive()
            except socket.error:
                print("[VR] Socket error, skipping...")
                continue

            snap = state.snapshot()
            save_dir = snap["vr_dir"] if snap["batch_active"] else None
            if save_dir:
                if getattr(quest, "latest_world_frame", None) is None:
                    now = time.time()
                    if now - worldframe_marker["last_pending_log"] > 2.0:
                        print("[VR] Batch active but worldframe not received yet, waiting...")
                        worldframe_marker["last_pending_log"] = now
                save_worldframe_if_ready(quest, save_dir, worldframe_marker)

            if wrist:
                pos, quat = wrist
                data = np.concatenate((pos, quat))
                chunker.put(data, save_dir)
                if save_dir:
                    print(f"[VR] Data saved to batch: {save_dir} | Timestamp: {time.time():.2f}")
                else:
                    print("[VR] Receiving data, batch not active. Data not saved.")

            now = time.time()
            target_dt = 1.0 / float(freq_hz)
            if now - last_tick < target_dt:
                time.sleep(target_dt - (now - last_tick))
            last_tick = time.time()
    finally:
        # Flush any remaining chunk when stopping
        if chunker.last_save_dir is not None:
            print("[VR] Flushing remaining VR chunk...")
            chunker.save_and_reset()
        quest.close()
        print("[VR] VR worker thread exited.")


def tactile_worker(state: SharedState, stop_event: threading.Event, save_interval: float = 5.0, fps: int = 30):
    print("[Tactile] Tactile worker thread starting...")
    camera_dev_path_dict = {side: (UsbCamera, device) for side, device in TACTILE_CAMERA.items()}
    
    # 初始保存目录设为None，将在批次激活时设置
    current_save_dir = None
    
    with SharedMemoryManager() as shm_manager:
        env = TactileCollectionEnv(
            camera_dev_path_dict=camera_dev_path_dict,
            save_dir=state.root_dir,  # 初始目录，批次激活后会更新
            shm_manager=shm_manager,
            S_N="placeholder",
            resolution=(640, 480),
            fps=fps,
            buffer_size=350,
        )
        env.start(wait=True)
        print("[Tactile] Tactile environment started.")
        try:
            while not stop_event.is_set():
                snap = state.snapshot()
                if snap["batch_active"] and snap["tactile_dir"]:
                    # 只有在批次激活且目录变化时才更新保存目录
                    if current_save_dir != snap["tactile_dir"]:
                        env.save_dir = snap["tactile_dir"]
                        current_save_dir = snap["tactile_dir"]
                        print(f"[Tactile] 切换到新的触觉数据目录: {env.save_dir}")
                    
                    print(f"[Tactile] 保存触觉数据到: {env.save_dir}")
                    start_ts = time.monotonic()
                    env.get_and_save_data()
                    duration = time.monotonic() - start_ts
                    print(f"[Tactile] 批次持续时间: {duration:.2f}s (间隔: {save_interval}s)")
                    if duration < save_interval:
                        time.sleep(save_interval - duration)
                else:
                    # 批次未激活时，重置保存目录
                    if current_save_dir is not None:
                        print("[Tactile] 批次未激活，停止保存触觉数据")
                        current_save_dir = None
                    time.sleep(0.2)
        finally:
            env.stop(wait=True)
            print("[Tactile] Tactile worker thread exited.")


# ---------- main ----------
def main():
    dataset_name = input("请输入采集名称(文件夹名): ").strip()
    if not dataset_name:
        dataset_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    root_dir = os.path.join("data", dataset_name)
    os.makedirs(root_dir, exist_ok=True)
    print(f"[MAIN] 主目录: {root_dir}")
    print("[MAIN] 进入启动标定流程（Origin, +X, +Y, +Z验证）...")

    try:
        quest = QuestRightArmLeapModule(VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(
                f"[MAIN][ERROR] 端口 {POSE_CMD_PORT} 已被占用，请先关闭其它采集进程后重试。"
            )
            return
        raise
    try:
        quest.run_axis_calibration(min_move=0.03)
        if quest.manual_origin is not None and quest.manual_rotation is not None:
            print("[MAIN] 启动标定完成，将在后续采集中使用该 worldframe。")
        else:
            print("[MAIN] 标定未成功，将回退到原有 worldframe/打桩逻辑。")
    except Exception:
        quest.close()
        raise
    print("[MAIN] 数据将按照以下结构组织:")
    print("  batch_1/")
    print("  ├── pose/               # VR姿态数据 (.npz文件)")
    print("  ├── tactile_YYYYMMDD_HHMMSS/  # 触觉数据文件夹")
    print("  ├── latency_calibration/ # 标定视频(预留)")
    print("  └── raw_videos/         # 原始视频(预留)")
    print("[MAIN] 按 'w' 开始batch, 'e' 结束当前batch, 'q' 退出")

    state = SharedState(root_dir)
    stop_event = threading.Event()
    chunker = DataChunker(chunksize=600)

    vr_thread = threading.Thread(
        target=vr_worker,
        args=(state, stop_event, chunker, quest),
        kwargs={"freq_hz": 30},
        daemon=True,
    )
    tactile_thread = threading.Thread(target=tactile_worker, args=(state, stop_event), daemon=True)
    vr_thread.start()
    tactile_thread.start()
    print("[MAIN] VR和Tactile采集线程已启动。")

    with RawMode():
        try:
            while True:
                key = read_key(timeout=0.1)
                if key is None:
                    continue
                if key == "w":
                    batch_dir = state.start_batch()
                    if batch_dir:
                        print(f"[MAIN] 开始 batch: {batch_dir}")
                elif key == "e":
                    finished_dir = state.end_batch()
                    if finished_dir:
                        if chunker.last_save_dir is not None:
                            print("[MAIN] Flushing VR chunk for batch end...")
                            chunker.save_and_reset()
                        print(f"[MAIN] 结束 batch: {finished_dir}")
                elif key == "q":
                    print("[MAIN] 退出中...")
                    break
        finally:
            stop_event.set()
            print("[MAIN] 等待采集线程退出...")
            vr_thread.join()
            tactile_thread.join()
    print("[MAIN] 采集完成。")


if __name__ == "__main__":
    main()
