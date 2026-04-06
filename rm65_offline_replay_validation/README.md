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

主流程内部使用：
- `T_B_tcp = T_B_from_pose_frame * T_pose * T_pose_to_tcp`
- 若 `solve_frame=flange`，则求解目标为 `T_B_flange = T_B_tcp * inv(T_flange_to_tcp)`

## 服务器运行步骤

1. 安装依赖

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
- `frames.T_B_from_pose_frame`
- `frames.T_pose_to_tcp`
- `frames.T_flange_to_tcp`

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

5. MuJoCo 回放（可选）

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

## 可选：RM_API2 基线对照

`configs/rm65_default.yaml` 中：
- `rm_baseline.enable: true`
- `rm_baseline.rm_api_python_dir` 指向 `RM_API2/Python`

若 SDK 依赖完整，主脚本会附带输出 RM baseline IK 的成功率对照。

