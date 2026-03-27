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
from scipy.ndimage import gaussian_filter

from umi.common.timecode_util import mp4_get_start_datetime


from scripts_slam_pipeline.utils.constants import ARUCO_ID
from scripts_slam_pipeline.utils.misc import (
    get_single_path, custom_minimize, 
    plot_trajectories, plot_long_horizon_trajectory
)
from scripts_slam_pipeline.utils.data_loading import load_proprio_interp


def find_calibration_video(session_dir: pathlib.Path):
    """Find calibration video path from latency_calibration first, then demos with segment_meta."""
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
        f"No calibration video found in {latency_calib_dir} and no is_calibration segment in {demos_dir}"
    )


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
        print(f"执行ArUco检测命令: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        assert result.returncode == 0, "ArUco检测失败"
    else:
        print(f"tag_detection.pkl已存在，跳过ArUco检测: {calibration_mp4}")


    print("align visual and trajectory")

    # get aruco trajectory
    video_start_time = mp4_get_start_datetime(str(calibration_mp4)).timestamp()

    aruco_pickle_path = latency_calib_dir.joinpath('tag_detection.pkl')
    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)
        aruco_trajectory = []
        for frame in aruco_pkl:
            if ARUCO_ID in frame['tag_dict']:
                x = frame['tag_dict'][ARUCO_ID]['tvec'][0]    # x-axis
                aruco_trajectory.append((frame['time']+video_start_time, x))

    
    # get arcap trajectory for plotting
    extend_range = 15
    timepoints = sorted([t for t, _ in aruco_trajectory])
    time_extend_before = np.linspace(timepoints[0]-extend_range, timepoints[0], 100).tolist()
    time_extend_after = np.linspace(timepoints[-1], timepoints[-1]+extend_range, 100).tolist()
    timepoints = time_extend_before + timepoints + time_extend_after

    arcap_trajectory_for_plotting = [
        (t, get_axis_value(traj_interp, t+init_offset)) 
        for t in timepoints
    ]
    plot_long_horizon_trajectory(
        aruco_trajectory, arcap_trajectory_for_plotting,
        title=f"Offset {init_offset:.2f} (move arcap {'left' if init_offset > 0 else 'right'})",
        save_dir=latency_calib_dir.joinpath(f'latency_trajectory_offset{init_offset}_long.pdf')
    )
    


    ###### algorithm 1: align by turning points
    turn_point_aruco = detect_turning_points(aruco_trajectory)
    turn_point_arcap = None

    for arcap_time_offset in [0.0, 1.0, -1.0, 2.0, -2.0]:
        arcap_time_offset += init_offset

        # get arcap trajectory
        arcap_trajectory = []
        timepoints = [t+arcap_time_offset for t, _ in aruco_trajectory]
        timepoints = sorted(timepoints)
        for t in timepoints:
            arcap_trajectory.append( ( t, get_axis_value(traj_interp, t) ) )
        turn_point_arcap = detect_turning_points(arcap_trajectory)
        
        # pickle.dump({
        #     "aruco": aruco_trajectory,
        #     "arcap": arcap_trajectory,
        # }, open(str(latency_calib_dir.joinpath(f'latency_data_{arcap_time_offset}.pkl')), 'wb'))
        
        plot_trajectories(aruco_trajectory, arcap_trajectory, 
                        [], [],
                        #turn_point_aruco, turn_point_arcap, 
                        latency_calib_dir.joinpath(f'latency_trajectory_offset{arcap_time_offset}.pdf'))

        if len(turn_point_aruco) == len(turn_point_arcap):
            break
        else:
            turn_point_arcap = None


    latency_of_arcap_result = {}
        
    if turn_point_arcap is not None:
        # algorithm 1 is good
        latency_of_arcap = []
        for (t1, d1, _), (t2, d2, _) in zip(turn_point_aruco, turn_point_arcap):
            if(d1 != d2):
                print("Direction mismatch")
                break
            latency_of_arcap.append(t2 - t1)
        else:
            if np.std(latency_of_arcap) < 0.1:
                # everything is good
                print(np.mean(latency_of_arcap), np.std(latency_of_arcap))

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
                
    

    #### algorithm 2: align by cross correlation

    print("switch to algorithm 2")

    aruco_trajectory = sorted(aruco_trajectory, key=lambda x: x[0])
    timepoints = [t for t, v in aruco_trajectory]
    aruco_pos = np.array([v for t, v in aruco_trajectory])
    aruco_pos = (aruco_pos - np.mean(aruco_pos)) / np.std(aruco_pos)  # normalized


    def mse_error(x):
        arcap_pos = np.array([get_axis_value(traj_interp, t+x[0]) for t in timepoints])
        arcap_pos = (arcap_pos - np.mean(arcap_pos)) / np.std(arcap_pos)
        return np.mean((aruco_pos - arcap_pos)**2)

    left_offset_bound  = -1.0 + init_offset
    right_offset_bound = 1.0 + init_offset
    epsilon = 0.01
    while True:
        res = custom_minimize(mse_error, 0.0, bounds=[(left_offset_bound, right_offset_bound)])
        if res.x - left_offset_bound < epsilon:
            left_offset_bound -= 1.0
            right_offset_bound -= 1.0
        elif right_offset_bound - res.x < epsilon:
            left_offset_bound += 1.0
            right_offset_bound += 1.0
        else:
            break

    print(res.x, mse_error(res.x))
        

    latency_of_arcap_result.update({
        "mean_2": res.x[0],
        "error": mse_error(res.x),
    })
    if "mean" not in latency_of_arcap_result:
        latency_of_arcap_result["mean"] = latency_of_arcap_result["mean_2"]
        
    with open(str(latency_calib_dir.joinpath('latency_of_arcap.json')), "w") as fp:
        json.dump(latency_of_arcap_result, fp)

    lat = latency_of_arcap_result["mean_2"]
    calibrated_arcap_trajectory = []
    for t, v in arcap_trajectory:
        calibrated_arcap_trajectory.append((t-lat, v))
    plot_trajectories(aruco_trajectory, calibrated_arcap_trajectory, 
                    [], [], 
                    latency_calib_dir.joinpath(f'latency_trajectory_final_algo_2.pdf'))


    
    




## %%
if __name__ == "__main__":
    main()
