"""
AR_00_process_videos.py - 视频预处理模块

功能说明:
- 该脚本是exUMI数据处理流水线的第一步，负责预处理采集的原始视频文件
- 主要任务：将raw_videos中的MP4文件重新组织到demos文件夹中，并按相机序列号和开始时间命名
- 为后续的SLAM流程准备标准化的视频数据格式

使用方式:
    python scripts_slam_pipeline/AR_00_process_videos.py <session_dir>

数据流向:
    raw_videos/*.mp4 → 提取元数据 → 按规则重命名 → demos/demo_<camera>_<timestamp>/raw_video.mp4

注意事项:
- 假设视频文件位于 <session_dir>/raw_videos 目录下
- 需要触觉数据文件夹 tactile_* 存在
- 使用ExifTool读取视频元数据，需要安装exiftool
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
import shutil
from exiftool import ExifToolHelper

from umi.common.timecode_util import mp4_get_start_datetime

from scripts_slam_pipeline.utils.misc import get_single_path



# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    """
    视频预处理主函数
    
    参数:
        session_dir: 一个或多个会话目录路径，每个目录对应一个数据批次
        
    处理流程:
        1. 检查输入目录结构
        2. 查找所有MP4视频文件
        3. 提取视频元数据（相机序列号、开始时间）
        4. 按规则重命名和组织视频文件
        5. 创建符号链接保持原始位置访问
    """
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        
        # 硬编码的目录结构 - 符合exUMI数据采集标准
        input_dir = session.joinpath('raw_videos')      # 原始视频输入目录
        output_dir = session.joinpath('demos')          # 处理后的输出目录
        tactile_dir = get_single_path(session.glob('tactile_*'))  # 触觉数据目录

        
        # 检查raw_videos目录是否存在
        if not input_dir.is_dir():
            raise FileNotFoundError(f"{input_dir.name} subdir don't exits")
            
        # 在input_dir及其所有子目录中查找MP4视频文件
        # 支持大小写不同的扩展名(.MP4和.mp4)
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        # 使用ExifTool读取视频元数据
        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                # 跳过已处理的符号链接文件
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                # 提取视频的开始时间和相机序列号
                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                
                # 生成标准化的输出目录名格式: demo_<相机序列号>_<时间戳>
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # 特殊文件夹处理 - 映射和夹爪校准视频
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # 创建输出目录
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # 移动视频文件到新位置并重命名为标准名称
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # 创建符号链接回到原始位置，保持向后兼容性
                # 由于Python 3.12之前没有relative_to的walk_up参数，手动计算相对路径
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
