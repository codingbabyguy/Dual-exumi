import time
import cv2
import os
import json
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from .camera_process import UsbCamera, CsiCamera, TactileCamera
from .angle_process import AngleSensor

import threading

from .ip_config import TACTILE_CAMERA


def _fmt_ts(ts):
    return f"{ts:.6f}" if ts is not None else "None"


def timestamp_to_readable(ts_unix: float) -> str:
    """将Unix时间戳转换为可读的标准时间格式"""
    dt = datetime.fromtimestamp(ts_unix)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 精确到毫秒


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

class TactileCollectionEnv:
    def __init__(self, camera_dev_path_dict, save_dir, shm_manager,S_N, *, 
                        resolution=(640, 480), fps=30, buffer_size=350):
        self.resolution = resolution
        self.fps = fps
        

        camera_dict = {}
        print("initializing angle")
        self.angle_sensor = AngleSensor(
            shm_manager=shm_manager,
            capture_fps=fps,
            put_fps=fps,
            get_max_k=buffer_size,
            verbose=False
        )
        print("initializing camera")
        for camera_name, (camera_cls, v4l_path) in camera_dev_path_dict.items():
            camera_dict[camera_name] = TactileCamera(
                camera_cls=camera_cls,
                dev_video_path=v4l_path,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=fps,
                put_fps=fps,
                get_max_k=buffer_size,
                cap_buffer_size=1,
                # put_downsample=False,
                verbose=True
            )
        




        self.camera_dict = camera_dict
        self.last_camera_data = {k: None for k in camera_dict.keys()}
        self.last_angle_data = None
        self.save_dir = save_dir
        self.global_step = 0
        
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = True
        for camera in self.camera_dict.values():
            ready_flag = ready_flag and camera.is_ready
        ready_flag = ready_flag and self.angle_sensor.is_ready
        return ready_flag
    
    def start(self, wait=True):
        
        self.angle_sensor.start(wait=False)
        for camera in self.camera_dict.values():
            camera.start(wait=False)
        
        
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.camera_dict.values():
            camera.stop(wait=False)
        self.angle_sensor.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.angle_sensor.start_wait()
        for camera in self.camera_dict.values():
            camera.start_wait()
    def stop_wait(self):
        for camera in self.camera_dict.values():
            camera.end_wait()
        self.angle_sensor.end_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_and_save_data(self):
        assert self.is_ready

        start_time = time.monotonic()
        print(f"[COLLECT] ===== Step {self.global_step} begin | save_dir={self.save_dir} =====")
 
        
        for camera_name, camera in self.camera_dict.items():
            print(f"[CAMERA] Fetching data: {camera_name}")
            self.last_camera_data[camera_name] = camera.get_all()
            camera_data = self.last_camera_data[camera_name]
            if camera_data is None:
                print(f"[CAMERA][WARN] {camera_name}: get_all() returned None")
            else:
                frame_count = _safe_len(camera_data.get('color', []))
                ts = camera_data.get('timestamp', [])
                ts_count = _safe_len(ts)
                if ts_count > 0:
                    first_ts = float(ts[0])
                    last_ts = float(ts[-1])
                    print(
                        f"[CAMERA] {camera_name}: frames={frame_count}, timestamps={ts_count}, "
                        f"range=[{_fmt_ts(first_ts)}, {_fmt_ts(last_ts)}], "
                        f"span={last_ts - first_ts:.3f}s"
                    )
                else:
                    print(f"[CAMERA][WARN] {camera_name}: empty timestamp list")
            print(f"[COLLECT] elapsed={time.monotonic()-start_time:.3f}s")
        

        
        print("[ANGLE] Fetching angle data")
        self.last_angle_data = self.angle_sensor.get_all()

        if self.last_angle_data is None:
            print("[ANGLE][WARN] No angle data received in this step (get_all returned None)")
        else:
            angles = self.last_angle_data.get('angle', [])
            poses = self.last_angle_data.get('data', [])
            timestamps = self.last_angle_data.get('timestamp', [])
            n_angles = _safe_len(angles)
            n_poses = _safe_len(poses)
            n_ts = _safe_len(timestamps)

            print(f"[ANGLE] Received samples: angles={n_angles}, poses={n_poses}, timestamps={n_ts}")

            if n_ts > 0:
                first_ts = float(timestamps[0])
                last_ts = float(timestamps[-1])
                print(
                    f"[ANGLE] Time range=[{_fmt_ts(first_ts)}, {_fmt_ts(last_ts)}], "
                    f"span={last_ts - first_ts:.3f}s"
                )

            if n_angles > 0:
                angle_vals = [int(a[0]) if hasattr(a, '__len__') else int(a) for a in angles]
                print(
                    f"[ANGLE] Angle stats: min={min(angle_vals)}, max={max(angle_vals)}, "
                    f"first={angle_vals[0]}, last={angle_vals[-1]}"
                )

            if n_angles == 0 or n_poses == 0 or n_ts == 0:
                print("[ANGLE][WARN] Incomplete angle payload; angle_*.json may be skipped or invalid")
        
            
        print(f"[COLLECT] elapsed={time.monotonic() - start_time:.3f}s")
        threads = []
        for camera_name, camera_data in self.last_camera_data.items():
            
            thread = threading.Thread(target=self._save_video_and_json, args=(camera_name, camera_data))
            threads.append(thread)
            thread.start()

        if self.last_angle_data:
            angle_thread = threading.Thread(target=self._save_angle_data, args=(self.last_angle_data,))
            threads.append(angle_thread)
            angle_thread.start()
        else:
            print(f"[ANGLE][WARN] Skip angle save for step {self.global_step}: no data")
        
        for thread in threads:
            thread.join()
        print(f"[COLLECT] ===== Step {self.global_step} done | elapsed={time.monotonic() - start_time:.3f}s =====")
        
        self.global_step += 1
        self.last_camera_data = {k: None for k in self.camera_dict.keys()}

    def _save_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        if camera_data is None:
            print(f"[SAVE][CAMERA][WARN] {camera_name}: camera_data is None, skip save")
            return

        video_path = os.path.join(self.save_dir, f"{camera_name}_{self.global_step}.mp4")
        json_path = os.path.join(self.save_dir, f"{camera_name}_{self.global_step}.json")

        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)

        video_shape = camera_data['color'].shape
        
        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        
        for frame in camera_data['color']:
            video_writer.write(frame)
        video_writer.release()

        with open(json_path, 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)

        frame_count = _safe_len(camera_data["color"])
        if len(timestamps) > 0:
            readable_first = timestamp_to_readable(timestamps[0])
            readable_last = timestamp_to_readable(timestamps[-1])
            print(f"\n[触觉相机保存] {camera_name.upper()}:")
            print(f"  视频文件: {video_path}")
            print(f"  帧数: {frame_count} 帧")
            print(f"  起始时间: {readable_first}")
            print(f"  结束时间: {readable_last}")
            print(f"  Unix时间: {_fmt_ts(timestamps[0])} ~ {_fmt_ts(timestamps[-1])}")
            print(f"  时间戳文件: {json_path}")
        else:
            print(f"\n[触觉相机保存][警告] {camera_name}: 空时间戳列表 -> {json_path}")
        print(f"  保存耗时: {time.monotonic() - start_time:.3f}s\n")

    def _save_realsense_video_and_json(self, camera_name, camera_data):
        start_time = time.monotonic()
        video_writer_colored = cv2.VideoWriter(
            os.path.join(self.save_dir, f"{camera_name}_colored_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)
        video_writer_depth = cv2.VideoWriter(
            os.path.join(self.save_dir, f"{camera_name}_depth_{self.global_step}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, self.resolution)

        video_shape = camera_data['color'].shape
        
        assert len(video_shape) == 4, str(video_shape)  # T,w,h,c
        
        for frame in camera_data['color']:
            video_writer_colored.write(frame)
        video_writer_colored.release()
        for frame in camera_data['depth']:
            video_writer_depth.write(frame)
        video_writer_depth.release()

        with open(os.path.join(self.save_dir, f"{camera_name}_{self.global_step}.json"), 'w') as json_file:
            timestamps = camera_data["timestamp"].tolist()
            json.dump(timestamps, json_file)
        print("Saved realsese data")
    
    def _save_angle_data(self, angle_data):
        start_time = time.monotonic()
        angle_path = os.path.join(self.save_dir, f"angle_{self.global_step}.json")

        with open(angle_path, 'w') as json_file:
            angles = angle_data['angle'].tolist()
            data = angle_data['data'].tolist()
            timestamps = angle_data["timestamp"].tolist()
            json.dump({'angles': angles, 'data':data, 'timestamps': timestamps}, json_file)
        n = len(timestamps)
        if n > 0:
            angle_vals = [int(a[0]) if hasattr(a, '__len__') else int(a) for a in angles]
            readable_first = timestamp_to_readable(timestamps[0])
            readable_last = timestamp_to_readable(timestamps[-1])
            print(f"\n[旋转传感器保存] ANGLE:")
            print(f"  数据文件: {angle_path}")
            print(f"  采样数量: {n} 个")
            print(f"  起始时间: {readable_first}")
            print(f"  结束时间: {readable_last}")
            print(f"  Unix时间: {_fmt_ts(timestamps[0])} ~ {_fmt_ts(timestamps[-1])}")
            print(f"  角度范围: min={min(angle_vals)}, max={max(angle_vals)}")
        else:
            print(f"\n[旋转传感器保存][警告] 空角度数据文件 -> {angle_path}")
        print(f"  保存耗时: {time.monotonic() - start_time:.3f}s\n")
        

        
        


def main():
    
    camera_dev_path_dict = {}
    for side, device in TACTILE_CAMERA.items():
        camera_dev_path_dict[side] = [UsbCamera, device]
    save_dir = "data/tactile_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    assert not os.path.exists(save_dir), "path exists"
    os.makedirs(save_dir)
    save_per_second = 5.0
    S_N = "327122079322"


    with SharedMemoryManager() as shm_manager:
        with TactileCollectionEnv(camera_dev_path_dict, save_dir, shm_manager,S_N) as env:
            print('[DEBUG] Start collecting')
            try:
                
                while True:
                    print('[DEBUG] Saving')
                    start_time = time.monotonic()
                    env.get_and_save_data()
                    duration = time.monotonic()-start_time
                    if duration > save_per_second:
                        print(f"WARNING: save duration exeeds the save gap: {duration} > {save_per_second}")
                    else:
                        print(f"[DEBUG] {duration} {save_per_second}")
                        time.sleep(save_per_second - duration)

            except KeyboardInterrupt:
                print("Interrupted")
                env.get_and_save_data()

            print("End.")




if __name__ == '__main__':
    main()
