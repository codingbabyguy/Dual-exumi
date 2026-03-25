"""
exUMI 数据采集服务器端脚本

功能说明:
- 该脚本运行在 Orange Pi 上，作为数据采集的服务器端
- 接收来自 Meta Quest VR 头显的手部/夹持器位姿数据
- 将位姿数据分块保存到本地文件系统中
- 支持实时显示数据接收状态

使用方式:
    python data_collection_umi.py [--frequency 30]

数据流向:
    VR头显 -> QuestRightArmLeapModule -> DataChunker -> .npz文件
"""

import socket
import time
import threading
from argparse import ArgumentParser
import pybullet as pb
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    # Package import path (used by do_umi.py)
    from ARCap.ip_config import *
    from ARCap.quest_robot_module_clean import QuestRightArmLeapModule
except ModuleNotFoundError:
    # Direct script execution path (used by `python ARCap/data_collection_umi.py`)
    from ip_config import *
    from quest_robot_module_clean import QuestRightArmLeapModule



class DataChunker:
    """
    数据分块器
    
    功能说明:
    - 用于将接收到的位姿数据分块保存到磁盘
    - 当累积的数据量达到 chunksize 时自动保存到文件
    - 支持动态切换保存目录（对应不同的采集批次）
    - 每个 .npz 文件包含 pose 和 time 两个数组
    
    数据格式:
    - pose: 7维向量 [x, y, z, qx, qy, qz, qw]，xyz为位置，xyzw为四元数
    - time: Unix时间戳
    
    保存策略:
    - 正常情况下，当记录数超过 chunksize 时保存
    - 当切换保存目录时，先保存当前块再切换
    - 当传入 save_dir=None 时，保存当前块并停止保存
    """
    
    def __init__(self, chunksize=1200):
        """
        初始化数据分块器
        
        参数:
            chunksize: 每个数据块的最大记录数，默认为1200
                      约对应 30Hz 采样率下 40 秒的数据量
        """
        self.chunksize = chunksize
        self.records = []  # 存储位姿数据 [x,y,z,qx,qy,qz,qw]
        self.timestamps = []  # 存储对应的时间戳

        self.last_save_dir = None  # 上一次使用的保存目录
        self._lock = threading.Lock()

    def put(self, x, save_dir):
        """
        添加一条位姿数据记录
        
        参数:
            x: 位姿数据，7维numpy数组 [x, y, z, qx, qy, qz, qw]
            save_dir: 保存目录路径，当为 None 时停止保存
        
        工作流程:
        1. 如果 save_dir 不为 None:
           - 如果是第一次保存或目录未变化，正常添加数据
           - 如果目录变化，先保存当前块，再开始新块
        2. 如果 save_dir 为 None:
           - 如果之前有保存数据，保存当前块
           - 清空缓冲区
        """
        with self._lock:
            if save_dir is not None:

                if self.last_save_dir is None or self.last_save_dir == save_dir:
                    self.last_save_dir = save_dir
                    self.records.append(x)
                    self.timestamps.append(time.time())

                    if len(self.records) > self.chunksize:
                        self._save_and_reset_locked()

                else:
                    self._save_and_reset_locked()

                    self.records.append(x)
                    self.timestamps.append(time.time())
                    self.last_save_dir = save_dir

            else:

                if self.last_save_dir is not None:
                    self._save_and_reset_locked()

                else:
                    self.records = []
                    self.timestamps = []

                self.last_save_dir = None
            

    def save_and_reset(self):
        """
        将当前缓冲区中的数据保存到文件并重置缓冲区
        
        保存格式:
        - 文件名: chunk_{起始时间}_{结束时间}.npz
        - pose: 存储所有位姿数据的二维数组，每行一条记录
        - time: 存储对应时间戳的一维数组
        
        注意:
        - 使用 np.savez 保存为 NumPy 的压缩格式
        - 保存后清空缓冲区，准备接收新数据
        """
        with self._lock:
            self._save_and_reset_locked()

    def _save_and_reset_locked(self):
        if self.last_save_dir is None or len(self.records) == 0 or len(self.timestamps) == 0:
            self.records = []
            self.timestamps = []
            self.last_save_dir = None
            return

        path = f"{self.last_save_dir}/chunk_{self.timestamps[0]}_{self.timestamps[-1]}.npz"
        np.savez(
            path,
            pose=self.records,
            time=self.timestamps,
        )

        self.records = []
        self.timestamps = []
        self.last_save_dir = None

        print(f"Saved chunk to {path}")

    def discard_and_reset(self, reason=""):
        """Drop buffered records without saving, used when deleting current batch data."""
        with self._lock:
            dropped = len(self.records)
            self.records = []
            self.timestamps = []
            self.last_save_dir = None
        if dropped > 0:
            print(f"[VR] Dropped {dropped} buffered pose samples. {reason}".strip())

            


if __name__ == "__main__":
    """
    主程序入口
    
    程序工作流程:
    1. 解析命令行参数
    2. 初始化 PyBullet 仿真环境（DIRECT 模式，不显示窗口）
    3. 创建 Quest VR 模块连接
    4. 创建数据分块器
    5. 进入主循环，持续接收和处理位姿数据
    
    通信端口说明:
    - VR_HOST: VR 头显的 IP 地址
    - LOCAL_HOST: 本地（Orange Pi）的 IP 地址
    - POSE_CMD_PORT: 位姿命令传输端口
    - IK_RESULT_PORT: IK 结果接收端口
    """
    
    parser = ArgumentParser(description="exUMI 数据采集服务器端")
    
    parser.add_argument(
        "--frequency", 
        type=int, 
        default=30,
        help="数据接收频率（Hz），默认30Hz"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="../gloveDemo/leap_assets/leap_hand/robot.urdf",
        help="URDF 模型路径，用于 PyBullet 仿真"
    )
    parser.add_argument(
        "--serial_port", 
        type=str, 
        default="COM3",
        help="串口名称，用于连接手套设备（Windows格式）"
    )
    parser.add_argument(
        "--serial_baud", 
        type=int, 
        default=115200,
        help="串口波特率"
    )
    parser.add_argument(
        "--axis_calibration",
        action="store_true",
        help="启动后先进行交互式 XYZ 轴标定（origin,+X,+Y,+Z验证）"
    )
    args = parser.parse_args()

    # 初始化 PyBullet 仿真环境
    # 使用 DIRECT 模式，不显示图形窗口，仅用于计算
    c = pb.connect(pb.DIRECT)
    
    # 可视化相关配置（当前未使用）
    vis_sp = []
    c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
        
    # 创建 Quest 右手 Leap 模块连接
    # 该模块负责与 VR 头显通信，接收手部追踪数据
    # 参数:
    #   - VR_HOST: VR 头显IP
    #   - LOCAL_HOST: 本地IP
    #   - POSE_CMD_PORT: 位姿命令端口
    #   - IK_RESULT_PORT: IK结果端口
    quest = QuestRightArmLeapModule(
        VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None
    )

    if args.axis_calibration:
        print("[MAIN] 启动交互式轴标定流程...")
        quest.run_axis_calibration(min_move=0.03)
        print("[MAIN] 轴标定完成，后续位姿将转换到该标定 worldframe。")

    # 性能统计变量
    start_time = time.time()  # 用于计算每秒接收数据包数
    fps_counter = 0  # 当前秒已接收的数据包数
    packet_counter = 0  # 累计接收的数据包总数
    print("Initialization completed")
    
    # 频率控制：记录上一次处理数据的时间
    current_ts = time.time()

    # 创建数据分块器，chunksize=600 约等于 20 秒的数据（30Hz）
    data_chunker = DataChunker(chunksize=600)
    
    # 用于控制状态打印频率（每秒打印一次）
    last_print = time.time()


    while True:
        """
        主循环
        
        循环执行以下操作:
        1. 频率控制：按指定频率处理数据
        2. 接收 Quest 数据：获取手腕位姿和头部姿态
        3. 数据存储：将位姿数据存入分块器
        4. 状态显示：每秒打印一次接收状态
        5. 异常处理：处理网络错误和用户中断
        """
        now = time.time()
        
        # 频率控制：如果当前时间间隔小于目标间隔，则跳过
        # TODO: May cause communication issues, need to tune on AR side.
        # 注意：这种方式可能导致通信问题，需要在 AR 端进行调优
        if now - current_ts < 1 / args.frequency:
            continue
        else:
            current_ts = now

        try:
            # 从 Quest 模块接收数据
            # 返回值:
            #   - wrist: 手腕位姿 (position, quaternion) 元组
            #   - head_pose: 头部姿态
            wrist, head_pose = quest.receive()

            # 如果成功接收到手腕位姿数据
            if wrist:
                # 将位姿数据（位置+四元数）合并为一维数组
                # 格式: [x, y, z, qx, qy, qz, qw]
                # 传入分块器进行保存
                # quest.data_dir 包含当前的数据保存目录路径
                data_chunker.put(np.concatenate(wrist), quest.data_dir)

            # 每秒打印一次接收状态
            if time.time() - last_print > 1.0:
                last_print = time.time()
                if wrist is not None:
                    # 解析位置和四元数
                    pos, quat = wrist
                    # 将四元数转换为欧拉角（XYZ顺序，角度制）
                    rot = R.from_quat(quat)
                    print("Data:", pos, rot.as_euler("xyz", degrees=True))
                else:
                    print("Data: None")

        # 网络 socket 错误处理
        except socket.error as e:
            print(e)
            pass

        # 用户按 Ctrl+C 中断处理
        except KeyboardInterrupt:
            # 关闭 Quest 连接
            quest.close()
            break

        else:
            # 统计数据包接收情况
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            # 每秒计算并显示接收频率
            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0
