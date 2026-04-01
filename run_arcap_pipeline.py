"""
Main script for exUMI data process pipeline.

流程概述：
  AR_00: 处理原始视频
  AR_01: 时间延迟校准（不生成PDF，仅输出latency_of_arcap.json）
  AR_03: 多模态数据对齐（默认保持采集端 manual frame 相对pose，仅时间对齐）
  AR_06: 生成数据集计划
  AR_08: 生成预览视频（使用改进的相对坐标系位姿可视化）

用法：
  python run_arcap_pipeline.py <session_dir> [options]
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
import json


def run_step(step_name, cmd):
    print(f"[START] {step_name}")
    print(f"[CMD] {' '.join(shlex.quote(item) for item in cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    print(f"[END] {step_name} | returncode={result.returncode} | elapsed={elapsed:.2f}s")
    return result


def load_alignment_window(session: pathlib.Path):
    path = session.joinpath('latency_calibration', 'alignment_window.json')
    if not path.is_file():
        return None, path
    with open(path, 'r') as fp:
        return json.load(fp), path


def confirm_after_alignment(session: pathlib.Path):
    report, report_path = load_alignment_window(session)
    if report is not None:
        print("[ALIGN][REPORT] 对齐窗口摘要:")
        print(f"  - 标定视频: {report.get('calibration_video')}")
        print(
            "  - 视频内对齐区间: "
            f"{report.get('aruco_rel_start_sec', 0.0):.3f}s ~ {report.get('aruco_rel_end_sec', 0.0):.3f}s "
            f"(总长 {report.get('video_duration_sec', 0.0):.3f}s)"
        )
        print(
            "  - 位于整段视频前: "
            f"{report.get('aruco_window_start_pct', 0.0):.2f}% ~ {report.get('aruco_window_end_pct', 0.0):.2f}%"
        )
        if report.get('matched_pose_start_unix') is not None and report.get('matched_pose_end_unix') is not None:
            print(
                "  - 匹配到的VR时间段: "
                f"{report.get('matched_pose_start_unix'):.6f} ~ {report.get('matched_pose_end_unix'):.6f}"
            )
        if report.get('mse_error') is not None:
            print(f"  - 匹配误差MSE: {report.get('mse_error'):.6f}")
        print(f"  - 报告文件: {report_path}")
    else:
        print(f"[ALIGN][WARN] 未找到对齐窗口报告: {report_path}")

    confirm = input("[PROMPT] 输入 yes 继续后续切分与处理，输入其它任意内容跳过该 session: ").strip()
    return confirm.lower() == 'yes'

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-gth', '--gripper_threshold', type=float, default=None, help="")
@click.option('--calibration_axis', type=str, default="y", help='')
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

        latency_json_path = session.joinpath('latency_calibration', 'latency_of_arcap.json')
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

        if not latency_json_path.is_file():
            raise FileNotFoundError(
                f"Missing latency file: {latency_json_path}. Please run AR_01 first."
            )

        if only_calib:
            print("[SKIP] Remaining steps (only_calib=True)")
            continue

        if not confirm_after_alignment(session):
            print("[ABORT] 用户未输入 yes，跳过该 session 后续流程。")
            continue

        print("############## AR_00_process_videos #############")
        script_path = script_dir.joinpath("AR_00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--latency_json', str(latency_json_path),
            str(session)
        ]
        result = run_step("AR_00_process_videos", cmd)
        assert result.returncode == 0

        demo_dir = session.joinpath('demos')
        print(f"[INFO] demo_dir={demo_dir}")

        print("############# AR_03_align_trajectory ###########")
        script_path = script_dir.joinpath("AR_03_align_trajectory.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '-calib', str(latency_json_path),
            '-tactile_calib', 'ARCap/tactile_calib/shape_config.yaml',
        ]
        if legacy_flexiv_transform:
            cmd.append('--legacy_flexiv_transform')
        if save_pose_debug:
            cmd.append('--save_pose_debug')
        else:
            cmd.append('--no-save_pose_debug')
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
