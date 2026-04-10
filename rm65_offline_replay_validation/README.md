# RM65 Offline Replay Validation

目标：在非真机环境中，用你已有的 `aligned_arcap_poses.json` 做离线回放验证，输出：
- 限位裕度（joint limit margin）
- 奇异值（sigma_min）
- 分支切换次数（branch switch count）

并可用 MuJoCo 将求得的关节轨迹可视化成视频。

## 目录结构

```text
rm65_offline_replay_validation/
  configs/
    rm65_default.yaml
  scripts/
    run_offline_validation.py
    visualize_first_frame_alignment.py
    optimize_mapping_params.py
    replay_mujoco.py
  src/rm65_offline_replay/
    config.py
    io_pose.py
    math3d.py
    pin_solver.py
    report.py
    rm_baseline.py
  requirements-server.txt
```

## 输入数据要求

默认读取 `run_arcap_pipeline` 后处理结果：

`<session_dir>/demos/demo_*/aligned_arcap_poses.json`

文件应包含：
- `pose`: `N x 7`，格式 `[x,y,z,qx,qy,qz,qw]`
- `timestamp`: `N`
- `width`: `N`

## 关键坐标系（B/F/T）在本工程的落地

- `B`: 机械臂 base（URDF base frame）
- `F`: 法兰（默认 `Link6`）
- `T`: 策略TCP（工具尖端）

配置中三个变换：
- `T_B_from_pose_frame`: 输入位姿坐标系 -> 机器人 base
- `T_pose_to_tcp`: 输入位姿 -> 策略TCP
- `T_flange_to_tcp`: 法兰 -> TCP（默认填了 RM-65 flange xacro 值）

`robot.input_pose_represents`：
- `tcp`：输入pose就是TCP
- `flange`：输入pose就是法兰（你的数据通常是这个）

主流程内部使用：
- `T_B_tcp = T_B_from_pose_frame * T_pose * T_pose_to_tcp`
- 若 `solve_frame=flange`，则求解目标为 `T_B_flange = T_B_tcp * inv(T_flange_to_tcp)`

## 服务器运行步骤

1. 安装核心依赖（不含 MuJoCo）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi/rm65_offline_replay_validation
pip install -r requirements-server.txt
```

2. 修改配置

编辑：
`configs/rm65_default.yaml`

至少确认：
- `robot.urdf_path`
- `robot.ee_frame_name`
- `robot.input_pose_represents`（若采集的是法兰pose，设为 `flange`）
- `frames.T_B_from_pose_frame`
- `frames.T_pose_to_tcp`
- `frames.T_flange_to_tcp`
- `selection.home_q_deg`（标准初始位形，单位度）
- `selection.hard_max_step_rad`（单步关节突变硬阈值）
- `selection.w_shape_joint_range / w_elbow_sign / w_elbow_halfspace / w_wrist_flip`（姿态先验）

3. 执行离线验证

```bash
python scripts/run_offline_validation.py \
  --config configs/rm65_default.yaml \
  --session_dir <你的session目录> \
  --output_dir <输出目录>
```

4. 输出结果

每个 demo 会生成：
- `summary.json`
- `per_frame_metrics.csv`
- `selected_trajectory.npz`

全局汇总：
- `global_summary.json`

4.5 两阶段流程（推荐）

先做第一阶段：确认 `T_B_from_pose_frame` 是否把首帧法兰姿态正确映射到 base 坐标系。

```bash
python scripts/visualize_first_frame_alignment.py \
  --config configs/rm65_default.yaml \
  --session_dir <你的session目录> \
  --demo_index 0 \
  --frame_index 0 \
  --output_dir <首帧可视化输出目录>
```

输出：
- `*_flange_axes.mp4`：首帧静态视频，法兰坐标轴颜色为 `X=红, Y=绿, Z=蓝`
- `*_alignment_debug.json`：首帧目标/IK结果/误差明细

第二阶段：固定你在 config 中给定的 `T_B_from_pose_frame`，再做搜索。

注意：当 `robot.input_pose_represents=flange` 时，`T_pose_to_tcp` 不参与搜索；这时建议开启 `--optimize_home_q`。

```bash
python scripts/optimize_mapping_params.py \
  --config configs/rm65_default.yaml \
  --session_dir <你的session目录> \
  --output_dir <优化输出目录> \
  --max_trials 120 \
  --frame_stride 2 \
  --max_demos 2 \
  --optimize_home_q
```

优化输出：
- `best_config.yaml`（可直接用于 run_offline_validation）
- `best_metrics.json`
- `search_history.json`

然后用最优配置重跑：

```bash
python scripts/run_offline_validation.py \
  --config <优化输出目录>/best_config.yaml \
  --session_dir <你的session目录> \
  --output_dir <验证输出目录>
```

5. MuJoCo 回放（可选）

先安装可视化依赖：

```bash
pip install -r requirements-mujoco.txt
```

```bash
python scripts/replay_mujoco.py \
  --urdf_path "<urdf路径>" \
  --trajectory_npz "<某个demo>/selected_trajectory.npz" \
  --output_mp4 "<输出mp4>"
```

如果服务器是无头环境，可先设置：

```bash
export MUJOCO_GL=osmesa
```

若 `pip install mujoco` 失败，通常是 Python 版本或平台 wheel 不匹配。建议：
- 换 Python 3.10/3.11 虚拟环境后再安装
- 或使用 conda-forge 安装 MuJoCo

## 可选：RM_API2 基线对照

`configs/rm65_default.yaml` 中：
- `rm_baseline.enable: true`
- `rm_baseline.rm_api_python_dir` 指向 `RM_API2/Python`

若 SDK 依赖完整，主脚本会附带输出 RM baseline IK 的成功率对照。


cd /home/icrlab/tactile_work_Wy/Dual-exumi/rm65_offline_replay_validation

python scripts/optimize_mapping_params.py \
  --config configs/rm65_default.yaml \
  --session_dir /home/icrlab/tactile_work_Wy/data/grip_red/batch_3 \
  --output_dir /home/icrlab/tactile_work_Wy/data/grip_red/batch_3/demos/optimize \
  --max_trials 120 \
  --frame_stride 2 \
  --max_demos 2 \
  --optimize_home_q

python scripts/run_offline_validation.py \
  --config /home/icrlab/tactile_work_Wy/data/grip_red/batch_3/demos/optimize/best_config.yaml \
  --session_dir /home/icrlab/tactile_work_Wy/data/grip_red/batch_3 \
  --output_dir /home/icrlab/tactile_work_Wy/data/grip_red/batch_3/optimized_work

python /home/icrlab/tactile_work_Wy/Dual-exumi/rm65_offline_replay_validation/scripts/replay_mujoco.py \
  --urdf_path "/home/icrlab/tactile_work_Wy/Dual-exumi/RM-65/ROS/rm_65_robot/rm_65_description/urdf/rm_65_description.urdf" \
  --trajectory_npz "/home/icrlab/tactile_work_Wy/data/grip_red/batch_3/optimized_work/demo_C3441130003646_2026.04.03_14.39.48.318421_seg026/selected_trajectory.npz" \
  --output_mp4 "/home/icrlab/tactile_work_Wy/Dual-exumi/rm65_offline_replay_validation/trajectory_best.mp4"
