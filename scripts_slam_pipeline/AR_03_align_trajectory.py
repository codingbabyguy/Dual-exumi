"""
AR_03_align_trajectory.py - 多模态数据对齐与处理

功能说明:
1. 对每个demo视频，按时间戳对齐VR位姿、夹持宽度、触觉图像
2. 根据时延校准参数和触觉标定参数，统一输出对齐后的数据
3. 支持批量处理所有demo，输出json和mp4格式

用法:
python scripts_slam_pipeline/AR_03_align_trajectory.py \
    -i <demos目录> \
    -calib <arcap_latency_calibration_path> \
    -tactile_calib <tactile_calibration_path>

主要流程:
1. 加载时延校准参数和触觉标定参数
2. 遍历所有demo视频，逐帧对齐并处理
3. 输出对齐后的姿态json和触觉mp4
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
import json
import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import copy
import av
from scipy.spatial.transform import Rotation as R
import yaml

# 工具函数：提取视频起始时间戳
from umi.common.timecode_util import mp4_get_start_datetime
# 触觉传感器处理类
from scripts_slam_pipeline.utils.dtact_sensor import Sensor
# 数据插值与保存工具
from scripts_slam_pipeline.utils.data_loading import (
    load_proprio_interp, load_tactile_interp, 
    save_frames_to_mp4_with_av,
)
# 字典转换工具
from scripts_slam_pipeline.utils.misc import listOfDict_to_dictOfList
# 坐标变换矩阵
from scripts_slam_pipeline.utils.constants import (
    tx_arhand_inv,
    tx_arbase_at_flexivbase,
    tx_flexivobj_at_arobj,
    tx_flexivcamera_at_flexivobj,
)


def is_calibration_demo(video_dir: pathlib.Path):
    meta_path = video_dir.joinpath('segment_meta.json')
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, 'r') as fp:
            meta = json.load(fp)
    except Exception:
        return False
    return bool(meta.get('is_calibration', False))




# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='demos文件夹路径')
@click.option('-calib', '--arcap_latency_calibration_path', required=True, help='ARCap时延校准参数路径')
@click.option('-tactile_calib', '--tactile_calibration_path', required=True, help='触觉标定参数路径')
@click.option('-n', '--num_workers', type=int, default=8, help='并行处理线程数')
def main(input_dir, arcap_latency_calibration_path, tactile_calibration_path, num_workers):
    """
    主处理流程：
    1. 加载校准参数
    2. 遍历所有demo视频，逐帧对齐VR位姿、夹持宽度、触觉图像
    3. 输出对齐后的json和mp4
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    # 获取所有demo视频目录
    all_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
    input_video_dirs = [x for x in all_video_dirs if not is_calibration_demo(x)]
    print(f'Found {len(all_video_dirs)} video dirs, {len(input_video_dirs)} used (calibration segments skipped)')

    session_dir = input_dir.parent

    # 步骤1: 加载时延校准参数和插值器
    latency = json.load(open(arcap_latency_calibration_path))['mean']
    traj_interp, width_interp = load_proprio_interp(session_dir, latency, extend_boundary=0.5)
    tactile_interp = load_tactile_interp(session_dir, latency, extend_boundary=0.5)

    # 步骤2: 加载触觉传感器标定参数
    sensor_postprocess = {}
    with open(tactile_calibration_path, "r") as f:
        cfg_raw = yaml.load(f, Loader=yaml.FullLoader)
        for side in ["left", "right"]:
            cfg = copy.deepcopy(cfg_raw)
            cfg['sensor_id'] = side
            sensor_postprocess[side] = Sensor(cfg)

    # 步骤3: 检查每个视频帧数和fps
    fps = None
    n_frames_list = []
    for video_dir in tqdm(input_video_dirs, desc="check fps"):
        mp4_path = video_dir.joinpath('raw_video.mp4')
        with av.open(str(mp4_path), 'r') as container:
            stream = container.streams.video[0]
            n_frames = stream.frames
            if fps is None:
                fps = stream.average_rate
            else:
                if fps != stream.average_rate:
                    print(f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}")
                    exit(1)
            n_frames_list.append(n_frames)
            print(n_frames)

    # 步骤4: 逐个处理视频
    def process_video(video_dir, n_frames):
        mp4_path = video_dir.joinpath('raw_video.mp4')

        # 获取视频起始时间戳
        segment_meta_path = video_dir.joinpath('segment_meta.json')
        if segment_meta_path.is_file():
            with open(segment_meta_path, 'r') as fp:
                segment_meta = json.load(fp)
            start_timestamp = float(segment_meta['start_timestamp'])
        else:
            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()

        # 对齐后的数据存储
        aligned_proprio_data = []
        aligned_tactile_data = []

        drop_demo = False  # 标记本demo是否丢弃
        for i_frame in range(n_frames):
            timestamp = start_timestamp + i_frame / fps
            try:
                _pose = traj_interp(timestamp)
                _width = width_interp(timestamp)
            except ValueError as e:
                print(e)
                drop_demo = True
                break

            # 姿态数据后处理：坐标变换到flexivbase
            _pose = np.array(_pose)
            assert _pose.shape == (7,), f"Invalid pose shape: {_pose.shape}, expected (6,)"
            mat = np.identity(4)
            mat[:3, 3] = _pose[:3]
            mat[:3, :3] = R.from_quat(_pose[3:]).as_matrix()
            # 坐标变换链：arhand->arbase->flexivbase->obj->camera
            tx_obj = mat @ tx_arhand_inv
            mat = tx_arbase_at_flexivbase @ tx_obj @ tx_flexivobj_at_arobj @ tx_flexivcamera_at_flexivobj
            _pose[:3] = mat[:3, 3]
            _pose[3:] = R.from_matrix(mat[:3, :3]).as_quat()

            aligned_proprio_data.append( {
                'pose': _pose.tolist(),
                'width': float(_width),  # (1,) ->  float
            } )

            # 触觉数据后处理
            _tact_data = {}
            for side in ["left", "right"]:
                _sensor = sensor_postprocess[side]
                try:
                    raw_img = tactile_interp[side](timestamp)
                except ValueError as e:
                    print(e)
                    drop_demo = True
                    break

                # 第一帧更新参考图像
                if i_frame == 0:
                    _sensor.update_ref(raw_img)

                img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                img = _sensor.raw_image_2_xy_repr(img_GRAY)
                img = np.round(img).astype(np.uint8)
                img = _sensor.get_rectify_crop_image(img)

                _tact_data.update({
                    f"tactile_{side}": img,
                })
            if drop_demo:
                break
            aligned_tactile_data.append(_tact_data)

        if drop_demo:
            return False

        # 保存对齐后的姿态数据为json
        with open(video_dir.joinpath('aligned_arcap_poses.json'), 'w') as fp:
            proprio_data = listOfDict_to_dictOfList(aligned_proprio_data)
            json.dump(proprio_data, fp)

        # 保存对齐后的触觉数据为mp4
        tactile_data = listOfDict_to_dictOfList(aligned_tactile_data)
        for key, frames in tactile_data.items():
            save_frames_to_mp4_with_av(frames, video_dir.joinpath(f'{key}.mp4'), fps=int(fps))

        return True

    # 顺序处理所有视频
    completed = []
    for vid_dir, n_frame in tqdm(zip(input_video_dirs, n_frames_list),
                                total=len(input_video_dirs), ncols=60):
        completed.append(process_video(vid_dir, n_frame))

    # 并行处理（注释掉，需同步）
    # with tqdm(total=len(input_video_dirs), ncols=60) as pbar:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = set()
    #         for vid_dir, n_frame in zip(input_video_dirs, n_frames_list):
    #             if len(futures) >= num_workers:
    #                 completed, futures = concurrent.futures.wait(futures, 
    #                     return_when=concurrent.futures.FIRST_COMPLETED)
    #                 pbar.update(len(completed))
    #             futures.add(executor.submit(process_video, vid_dir, n_frame))
    #         completed, futures = concurrent.futures.wait(futures)
    #         pbar.update(len(completed))

    print("Done!")
    num_of_skip = len([x for x in completed if not x])
    if num_of_skip > 0:
        print(f"Skipped {num_of_skip} demos from {len(input_video_dirs)}")

# %%
if __name__ == "__main__":
    main()
