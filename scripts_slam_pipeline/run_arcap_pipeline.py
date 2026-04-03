"""
Main script for exUMI data process pipeline.
python run_arcap_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-gth', '--gripper_threshold', type=float, default=None, help="")
@click.option('--calibration_axis', type=str, default="x", help='')
@click.option('--init_offset', type=float, default=0)
@click.option('--only_calib', is_flag=True, default=False, help="only run calibration")
@click.option('--skip_calib', is_flag=True, default=False, help="skip the calibration")
@click.option(
    '--legacy_flexiv_transform/--no-legacy_flexiv_transform',
    default=False,
    help="AR_03 是否启用历史 Flexiv 固定外参。默认关闭，保持 manual frame 相对位姿。",
)
@click.option(
    '--save_pose_debug/--no-save_pose_debug',
    default=True,
    help="AR_03 输出 aligned_pose_summary.json + aligned_pose_debug.png。",
)
def main(
    session_dir,
    calibration_dir,
    gripper_threshold,
    calibration_axis,
    init_offset,
    only_calib,
    skip_calib,
    legacy_flexiv_transform,
    save_pose_debug,
):
    root_dir = pathlib.Path(ROOT_DIR)
    script_dir = root_dir.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = root_dir.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = root_dir.joinpath(session).absolute()

        print("############## AR_00_process_videos #############")
        script_path = script_dir.joinpath("AR_00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        
        demo_dir = session.joinpath('demos')


        if not skip_calib:
            print("############## AR_01_arcap_latency_align #############")
            script_path = script_dir.joinpath("AR_01_arcap_latency_align.py")
            assert script_path.is_file()
            cmd = [
                'python', str(script_path),
                '--calibration_dir', str(calibration_dir),
                '--calibration_axis', calibration_axis,
                '--init_offset', str(init_offset),
                str(session)
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0

        if only_calib:
            continue
        

        print("############# AR_03_align_trajectory ###########")
        script_path = script_dir.joinpath("AR_03_align_trajectory.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '-calib', str(session.joinpath('latency_calibration/latency_of_arcap.json')),
            '-tactile_calib', 'ARCap/tactile_calib/shape_config.yaml',
        ]
        if legacy_flexiv_transform:
            cmd.append('--legacy_flexiv_transform')
        if save_pose_debug:
            cmd.append('--save_pose_debug')
        else:
            cmd.append('--no-save_pose_debug')
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# AR_06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("AR_06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session),
            # "--tx_slam_tag", "example/tx_slam_tag_identity.json",
            # "--use_arcap_trajectory",
        ]
        if legacy_flexiv_transform:
            cmd.append('--allow_legacy_frame')
        if gripper_threshold:
            cmd.extend(["--gripper_threshold", str(gripper_threshold)])
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    main()
