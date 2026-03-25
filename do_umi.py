"""
Unified controller for VR pose and tactile collection with batch-based saving.

Usage:
    python do_umi.py

Controls:
    w: start a new batch (creates batch_N subfolder)
    e: end current batch (flush and stop saving)
    d: delete current batch data and restart current batch id
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
import shutil
import threading
import select
import tty
import termios
import socket
from datetime import datetime

import numpy as np
import pybullet as pb


def timestamp_to_readable(ts_unix: float) -> str:
    """将Unix时间戳转换为可读的标准时间格式"""
    dt = datetime.fromtimestamp(ts_unix)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 精确到毫秒

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
                "batch_id": self.batch_id,
                "batch_dir": self.current_batch_dir,
            }

    def status_text(self):
        with self.lock:
            mode = "采集中" if self.batch_active else "待机"
            return f"当前batch: {self.batch_id} | 状态: {mode}"

    def delete_and_restart_current_batch(self):
        with self.lock:
            if not self.batch_active or self.current_batch_dir is None:
                return None

            batch_dir = self.current_batch_dir
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)

            vr_dir = os.path.join(batch_dir, "pose")
            self.tactile_folder_name = f"tactile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tactile_dir = os.path.join(batch_dir, self.tactile_folder_name)

            os.makedirs(vr_dir, exist_ok=True)
            os.makedirs(tactile_dir, exist_ok=True)
            os.makedirs(os.path.join(batch_dir, "latency_calibration"), exist_ok=True)
            os.makedirs(os.path.join(batch_dir, "raw_videos"), exist_ok=True)

            self.vr_dir = vr_dir
            self.tactile_dir = tactile_dir
            return {
                "batch_dir": batch_dir,
                "vr_dir": vr_dir,
                "tactile_dir": tactile_dir,
                "batch_id": self.batch_id,
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
    
    # 时间戳统计
    vr_timestamps = []
    ts_last_print = time.time()
    
    try:
        while not stop_event.is_set():
            try:
                wrist, head_pose = quest.receive()
            except socket.error:
                print("[VR] Socket error, skipping...")
                continue

            snap = state.snapshot()
            save_dir = snap["vr_dir"] if snap["batch_active"] else None
            
            current_ts = time.time()
            if wrist:
                vr_timestamps.append(current_ts)
            
            # 每秒输出一次时间戳统计
            if current_ts - ts_last_print >= 1.0:
                if vr_timestamps:
                    avg_ts = sum(vr_timestamps) / len(vr_timestamps)
                    ts_range = [vr_timestamps[0], vr_timestamps[-1]]
                    readable_first = timestamp_to_readable(ts_range[0])
                    readable_last = timestamp_to_readable(ts_range[1])
                    print(f"\n{'='*80}")
                    print(f"[VR位姿数据] 当前时间段统计:")
                    print(f"  采样数量: {len(vr_timestamps)} 帧")
                    print(f"  起始时间: {readable_first}")
                    print(f"  结束时间: {readable_last}")
                    print(f"  Unix时间: {ts_range[0]:.7f} ~ {ts_range[1]:.7f}")
                    print(f"  时间跨度: {ts_range[1]-ts_range[0]:.4f} 秒")
                    print(f"{'='*80}\n")
                else:
                    print(f"\n{'='*80}")
                    print(f"[VR位姿数据] 本秒无数据接收")
                    print(f"{'='*80}\n")
                vr_timestamps = []
                ts_last_print = current_ts
            
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
                    print(f"[VR] Data saved to batch: {save_dir} | Timestamp: {current_ts:.7f}")
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
    next_collect_at = 0.0
    
    # 时间戳统计
    tactile_timestamps = {side: [] for side in camera_dev_path_dict.keys()}
    angle_timestamps = []
    ts_last_print = time.time()
    
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
                current_time = time.time()
                
                # 每秒输出一次时间戳统计
                if current_time - ts_last_print >= 1.0:
                    print(f"\n{'#'*80}")
                    print(f"### 触觉和角度传感器 - 当前时间统计")
                    print(f"{'#'*80}")
                    
                    # 触觉摄像头时间戳
                    for side in tactile_timestamps.keys():
                        if tactile_timestamps[side]:
                            ts_list = tactile_timestamps[side]
                            readable_first = timestamp_to_readable(ts_list[0])
                            readable_last = timestamp_to_readable(ts_list[-1])
                            print(f"\n[触觉相机-{side.upper()}] 当前时间:")
                            print(f"  帧数: {len(ts_list)} 帧")
                            print(f"  起始时间: {readable_first}")
                            print(f"  结束时间: {readable_last}")
                            print(f"  Unix时间: {ts_list[0]:.7f} ~ {ts_list[-1]:.7f}")
                            print(f"  时间跨度: {ts_list[-1]-ts_list[0]:.4f} 秒")
                        else:
                            print(f"\n[触觉相机-{side.upper()}] 本秒无数据")
                    
                    # Angle时间戳
                    if angle_timestamps:
                        readable_first = timestamp_to_readable(angle_timestamps[0])
                        readable_last = timestamp_to_readable(angle_timestamps[-1])
                        print(f"\n[旋转传感器-ANGLE] 当前时间:")
                        print(f"  采样数量: {len(angle_timestamps)} 个")
                        print(f"  起始时间: {readable_first}")
                        print(f"  结束时间: {readable_last}")
                        print(f"  Unix时间: {angle_timestamps[0]:.7f} ~ {angle_timestamps[-1]:.7f}")
                        print(f"  时间跨度: {angle_timestamps[-1]-angle_timestamps[0]:.4f} 秒")
                    else:
                        print(f"\n[旋转传感器-ANGLE] 本秒无数据")
                    
                    print(f"{'#'*80}\n")
                    
                    # 重置统计
                    tactile_timestamps = {side: [] for side in camera_dev_path_dict.keys()}
                    angle_timestamps = []
                    ts_last_print = current_time
                
                if snap["batch_active"] and snap["tactile_dir"]:
                    # 批次激活且目录变化时更新保存目录
                    if current_save_dir != snap["tactile_dir"]:
                        env.save_dir = snap["tactile_dir"]
                        current_save_dir = snap["tactile_dir"]
                        next_collect_at = 0.0
                        print(f"[Tactile] 切换到新的触觉数据目录: {env.save_dir}")

                    now = time.monotonic()
                    if now >= next_collect_at:
                        print(f"[Tactile] 保存触觉数据到: {env.save_dir}")
                        start_ts = time.monotonic()
                        
                        # 采集前记录时间戳
                        pre_collect_time = time.time()
                        env.get_and_save_data()
                        post_collect_time = time.time()
                        
                        # 从env对象中获取最新的时间戳数据
                        if hasattr(env, 'last_camera_data'):
                            for side in camera_dev_path_dict.keys():
                                if side in env.last_camera_data:
                                    camera_data = env.last_camera_data[side]
                                    if camera_data and 'timestamp' in camera_data:
                                        tactile_timestamps[side].extend(camera_data['timestamp'].tolist())
                        
                        if hasattr(env, 'last_angle_data') and env.last_angle_data:
                            if 'timestamp' in env.last_angle_data:
                                angle_timestamps.extend(env.last_angle_data['timestamp'].tolist())
                        
                        duration = time.monotonic() - start_ts
                        print(f"[Tactile] 批次持续时间: {duration:.2f}s (间隔: {save_interval}s)")
                        next_collect_at = time.monotonic() + save_interval
                    else:
                        # 使用短sleep，避免错过短batch切换
                        time.sleep(min(0.1, max(0.0, next_collect_at - now)))
                else:
                    # 批次结束时做一次强制落盘，避免短 batch 没有 left/right 数据
                    if current_save_dir is not None:
                        print(f"[Tactile] 批次结束，执行最后一次触觉落盘: {current_save_dir}")
                        env.save_dir = current_save_dir
                        try:
                            env.get_and_save_data()
                        except Exception as e:
                            print(f"[Tactile][WARN] batch结束落盘失败: {e}")
                        print("[Tactile] 批次未激活，停止保存触觉数据")
                        current_save_dir = None
                    next_collect_at = 0.0
                    time.sleep(0.1)
        finally:
            # 线程退出前再尝试一次落盘，降低中断造成的数据丢失概率
            if current_save_dir is not None:
                print(f"[Tactile] 线程退出前最后落盘: {current_save_dir}")
                env.save_dir = current_save_dir
                try:
                    env.get_and_save_data()
                except Exception as e:
                    print(f"[Tactile][WARN] 退出前落盘失败: {e}")

            # 限时停止，避免单个采集子进程卡住导致 q 无法完全退出
            env.stop(wait=False)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                cam_alive = [name for name, cam in env.camera_dict.items() if cam.is_alive()]
                angle_alive = env.angle_sensor.is_alive()
                if not cam_alive and not angle_alive:
                    break
                time.sleep(0.1)

            for name, cam in env.camera_dict.items():
                if cam.is_alive():
                    print(f"[Tactile][WARN] 相机进程未退出，强制结束: {name}")
                    cam.terminate()
                    cam.join(timeout=0.5)
            if env.angle_sensor.is_alive():
                print("[Tactile][WARN] Angle进程未退出，强制结束")
                env.angle_sensor.terminate()
                env.angle_sensor.join(timeout=0.5)

            print("[Tactile] Tactile worker thread exited.")


# ---------- main ----------
def main():
    physics_client = None
    if not pb.isConnected():
        physics_client = pb.connect(pb.DIRECT)
        print(f"[MAIN] PyBullet connected (client_id={physics_client}).")

    dataset_name = input("请输入采集名称(文件夹名): ").strip()
    if not dataset_name:
        dataset_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    root_dir = os.path.join("data", dataset_name)
    os.makedirs(root_dir, exist_ok=True)
    print(f"[MAIN] 主目录: {root_dir}")
    print("[MAIN] 进入启动标定流程（Origin, +X, +Y, +Z验证）...")

    try:
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
                # 🔧 保存标定参数到根目录，用于可视化验证
                calib_file = os.path.join(root_dir, "calibration_params.npz")
                np.savez(
                    calib_file,
                    manual_origin=quest.manual_origin,
                    manual_rotation=quest.manual_rotation,
                    calibration_timestamp=datetime.now().isoformat()
                )
                print(f"[MAIN] ✅ 标定参数已保存: {calib_file}")
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
        print("[MAIN] 按 'w' 开始batch, 'e' 结束当前batch, 'd' 删除并重采当前batch, 'q' 退出")

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
        print(f"[MAIN][STATUS] {state.status_text()}")

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
                        print(f"[MAIN][STATUS] {state.status_text()}")
                    elif key == "e":
                        finished_dir = state.end_batch()
                        if finished_dir:
                            if chunker.last_save_dir is not None:
                                print("[MAIN] Flushing VR chunk for batch end...")
                                chunker.save_and_reset()
                            print(f"[MAIN] 结束 batch: {finished_dir}")
                        print(f"[MAIN][STATUS] {state.status_text()}")
                    elif key == "d":
                        print("[MAIN][CONFIRM] 确认删除当前 batch 数据并重采？按 y 确认，按 n 取消。")
                        confirm = None
                        while confirm is None:
                            c = read_key(timeout=0.1)
                            if c is None:
                                continue
                            lc = c.lower()
                            if lc in ("y", "n"):
                                confirm = lc

                        if confirm == "y":
                            reset_info = state.delete_and_restart_current_batch()
                            if reset_info:
                                chunker.discard_and_reset(reason="Current batch deleted by user.")
                                print(
                                    f"[MAIN] 已删除并重建 batch_{reset_info['batch_id']}:\n"
                                    f"  - VR数据: {reset_info['vr_dir']}\n"
                                    f"  - 触觉数据: {reset_info['tactile_dir']}"
                                )
                            else:
                                print("[MAIN] 当前没有进行中的 batch，无法删除重采。")
                        else:
                            print("[MAIN] 已取消删除当前 batch。")
                        print(f"[MAIN][STATUS] {state.status_text()}")
                    elif key == "q":
                        print("[MAIN] 退出中...")
                        break
            finally:
                stop_event.set()
                print("[MAIN] 等待采集线程退出...")
                vr_thread.join(timeout=5.0)
                tactile_thread.join(timeout=5.0)
                if vr_thread.is_alive():
                    print("[MAIN][WARN] VR线程未在超时内退出，将随主进程结束。")
                if tactile_thread.is_alive():
                    print("[MAIN][WARN] Tactile线程未在超时内退出，将随主进程结束。")
        print("[MAIN] 采集完成。")
    finally:
        if physics_client is not None and pb.isConnected():
            pb.disconnect(physics_client)
            print("[MAIN] PyBullet disconnected.")


if __name__ == "__main__":
    main()
