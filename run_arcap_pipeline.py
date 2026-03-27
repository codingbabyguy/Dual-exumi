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
import shlex
import time


def run_step(step_name, cmd):
    print(f"[START] {step_name}")
    print(f"[CMD] {' '.join(shlex.quote(item) for item in cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    print(f"[END] {step_name} | returncode={result.returncode} | elapsed={elapsed:.2f}s")
    return result

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-gth', '--gripper_threshold', type=float, default=None, help="")
@click.option('--calibration_axis', type=str, default="x", help='')
@click.option('--init_offset', type=float, default=0)
@click.option('--only_calib', is_flag=True, default=False, help="only run calibration")
@click.option('--skip_calib', is_flag=True, default=False, help="skip the calibration")
def main(session_dir, calibration_dir, gripper_threshold, calibration_axis, init_offset, only_calib, skip_calib):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()
    print(f"[INFO] script_dir={script_dir}")
    print(f"[INFO] calibration_dir={calibration_dir}")
    print(f"[INFO] skip_calib={skip_calib}, only_calib={only_calib}, calibration_axis={calibration_axis}, init_offset={init_offset}, gripper_threshold={gripper_threshold}")

    for session in session_dir:
        session = pathlib.Path(__file__).parent.joinpath(session).absolute()
        print(f"\n[SESSION] {session}")

        print("############## AR_00_process_videos #############")
        script_path = script_dir.joinpath("AR_00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = run_step("AR_00_process_videos", cmd)
        assert result.returncode == 0
        
        demo_dir = session.joinpath('demos')
        print(f"[INFO] demo_dir={demo_dir}")


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
            result = run_step("AR_01_arcap_latency_align", cmd)
            assert result.returncode == 0
        else:
            print("[SKIP] AR_01_arcap_latency_align (skip_calib=True)")

        if only_calib:
            print("[SKIP] Remaining steps (only_calib=True)")
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
        result = run_step("AR_03_align_trajectory", cmd)
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
        if gripper_threshold:
            cmd.extend(["--gripper_threshold", str(gripper_threshold)])
        result = run_step("AR_06_generate_dataset_plan", cmd)
        assert result.returncode == 0

        print("############# AR_08_make_pose_demo_preview ###########")
        script_path = script_dir.joinpath("AR_08_make_pose_demo_preview.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--full_length',
        ]
        result = run_step("AR_08_make_pose_demo_preview", cmd)
        if result.returncode != 0:
            print("[WARN] AR_08 preview generation failed, continuing...")
        
        print(f"[SESSION DONE] {session}")

## %%
if __name__ == "__main__":
    main()
