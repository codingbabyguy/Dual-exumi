"""
exUMI 采集数据可视化脚本

功能：
- 加载并显示采集到的位姿轨迹（3D 路径）
- 绘制末端执行器的旋转姿态（RGB 三轴坐标）
- 导出静态图片、MP4 视频或 GIF 动画
- 支持相对坐标和绝对坐标两种显示模式

用法：
  1. 修改代码顶部的 DATA_FOLDER 变量，指定你的 pose 文件夹路径
  2. 根据需要调整 USE_ABSOLUTE, OUTPUT_FORMAT, SKIP_VIDEO 等参数
  3. 运行: python visualize_pose.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# ================= 可视化参数 =================
# 🔧 修改这里以指定你的数据文件夹路径
DATA_FOLDER = "data/testXYZ/batch_1/pose"  # 写死的目标文件夹路径

VIDEO_FPS = 30          # 视频帧率
DOWNSAMPLE = 1          # 数据抽稀倍数（1=不抽稀，2=每两帧取一帧）
GIF_MAX_FRAMES = 300    # GIF 最大帧数限制
GIF_FPS = 12            # GIF 输出帧率
GIF_DPI = 80            # GIF 渲染分辨率缩放

USE_ABSOLUTE = False    # 是否使用绝对坐标（False=相对坐标）
OUTPUT_FORMAT = "mp4"   # 输出格式: "mp4", "gif", "both"
SKIP_VIDEO = False      # 是否跳过视频生成
# ===============================================


def quats_to_rotmats(quats):
    """将 Nx4 四元数 [qx, qy, qz, qw] 转为 Nx3x3 旋转矩阵"""
    q = np.asarray(quats, dtype=np.float64)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q = q / q_norm

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[:, 0, 1] = 2.0 * (xy - wz)
    rot[:, 0, 2] = 2.0 * (xz + wy)
    rot[:, 1, 0] = 2.0 * (xy + wz)
    rot[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[:, 1, 2] = 2.0 * (yz - wx)
    rot[:, 2, 0] = 2.0 * (xz - wy)
    rot[:, 2, 1] = 2.0 * (yz + wx)
    rot[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rot


def load_worldframe(folder_path, parent_folder_path=None):
    """加载 WorldFrame 或标定参数（优先级顺序）
    
    1. 首先查找 calibration_params.npz（最精确的标定参数）
    2. 再查找 worldframe_*/WorldFrame.npy（备选方案）
    3. 都找不到就用单位矩阵
    """
    print("[STEP 2] 🔍 正在搜索坐标系标定参数...")
    
    # 1️⃣ 优先加载明确保存的标定参数
    if parent_folder_path:
        calib_param_path = os.path.join(parent_folder_path, "calibration_params.npz")
        if os.path.exists(calib_param_path):
            try:
                data = np.load(calib_param_path)
                manual_origin = data['manual_origin']
                manual_rotation = data['manual_rotation']
                print(f"   ✅ 成功加载标定参数: {os.path.basename(calib_param_path)}")
                print(f"      标定原点: [{manual_origin[0]:.6f}, {manual_origin[1]:.6f}, {manual_origin[2]:.6f}]")
                print(f"      标定时间: {data['calibration_timestamp']}")
                # 构造 4x4 变换矩阵
                T = np.eye(4)
                T[:3, :3] = manual_rotation
                T[:3, 3] = manual_origin
                return T, True  # True 表示这是手动标定的 manual 参数
            except Exception as e:
                print(f"   ⚠️ 加载标定参数失败: {e}，尝试备选方案...")
    
    # 2️⃣ 备选：从 worldframe_*/WorldFrame.npy 加载
    world_frame_paths = [
        os.path.join(folder_path, "WorldFrame.npy"),
        glob.glob(os.path.join(folder_path, "worldframe_*/WorldFrame.npy"))[0]
        if glob.glob(os.path.join(folder_path, "worldframe_*/WorldFrame.npy"))
        else None,
    ]

    for path in world_frame_paths:
        if path and os.path.exists(path):
            try:
                world_frame = np.load(path)
                if world_frame.shape == (7,):
                    print(f"   - 检测到 7D 格式 (pos + quat)，正在转换为 4x4 变换矩阵...")
                    pos = world_frame[:3]
                    quat = world_frame[3:7]
                    rot = rotmat_from_quat(quat)
                    T = np.eye(4)
                    T[:3, :3] = rot
                    T[:3, 3] = pos
                    world_frame = T
                print(f"   ✅ 成功加载 WorldFrame: {os.path.basename(path)}")
                print(f"      原点: [{world_frame[0,3]:.6f}, {world_frame[1,3]:.6f}, {world_frame[2,3]:.6f}]")
                return world_frame, False  # False 表示这是从 worldframe_* 加载的
            except Exception as e:
                print(f"   ⚠️ 加载 {path} 失败: {e}")

    print("   ⚠️ 未找到标定参数，将使用单位矩阵作为基准。")
    return np.eye(4), False


def rotmat_from_quat(quat):
    """四元数 [qx, qy, qz, qw] 转旋转矩阵"""
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3)
    q = q / norm

    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return rot


def load_and_process_data(folder_path, use_absolute=False):
    """
    加载采集数据并处理坐标系
    
    【关键说明】
    采集阶段：
    - 原始数据来自 VR (Unity坐标系)
    - 经过 _compute_rel_transform_manual() 变换后，得到【相对于manual_origin/manual_rotation的相对pose】
    - 保存的chunk数据就是这个相对pose
    
    可视化阶段：
    - use_absolute=False（默认）：直接显示保存的相对pose ✅ 与采集完全一致
    - use_absolute=True：先变换回绝对坐标，再显示（需要worldframe参数）
    
    参数：
        folder_path: pose 数据文件夹路径
        use_absolute: 是否转换到绝对坐标
    
    返回：
        (轨迹点, 旋转矩阵, 绘图边界) 元组
    """
    print(f"\n[STEP 1] 📂 正在读取数据文件夹...")
    print(f"         {folder_path}")

    # 获取父目录（session根目录），用于查找calibration_params.npz
    parent_folder = os.path.dirname(folder_path)

    # 1. 加载 WorldFrame / 标定参数
    world_frame, is_manual_calib = load_worldframe(folder_path, parent_folder)
    world_rot = world_frame[:3, :3]
    world_pos = world_frame[:3, 3]

    # 2. 获取并排序 chunk 文件
    print(f"\n[STEP 3] 📦 正在扫描 chunk 文件...")
    chunk_files = sorted(glob.glob(os.path.join(folder_path, "chunk_*.npz")))
    if not chunk_files:
        print(f"   ❌ 未找到 chunk 文件！")
        return None, None, None

    print(f"   ✅ 找到 {len(chunk_files)} 个 chunk 文件")
    for i, f in enumerate(chunk_files[:3], 1):  # 显示前 3 个文件
        print(f"      - {os.path.basename(f)}")
    if len(chunk_files) > 3:
        print(f"      - ... 还有 {len(chunk_files) - 3} 个文件")

    all_pts = []
    all_rotmats = []

    # 3. 遍历读取并解析数据 (Nx7: [x,y,z,qx,qy,qz,qw])
    print(f"\n[STEP 4] 🔄 正在加载并处理 chunk 数据...")
    print(f"         数据坐标模式: {'相对坐标（来自采集）' if not use_absolute else '绝对坐标（变换后）'}")
    
    for idx, file in enumerate(chunk_files):
        try:
            data = np.load(file)
            pose_data = data['pose']
            num_poses = len(pose_data)

            # 提取 [x, y, z] 位置和 [qx, qy, qz, qw] 旋转
            pts = pose_data[:, :3]
            quats = pose_data[:, 3:7]
            
            # 【核心变换逻辑】
            if use_absolute:
                # 用户要求绝对坐标：P_abs = world_pos + world_rot @ P_rel
                pts = (world_rot @ pts.T).T + world_pos
                local_rotmats = quats_to_rotmats(quats)
                # R_abs = world_rot @ R_rel
                rotmats = np.einsum('ij,njk->nik', world_rot, local_rotmats)
            else:
                # 保持相对坐标（直接使用采集数据）✅ 这是最精确的模式
                rotmats = quats_to_rotmats(quats)
            
            all_pts.append(pts)
            all_rotmats.append(rotmats)
            
            print(f"   ✓ 处理第 {idx+1}/{len(chunk_files)} 个chunk ({num_poses} 个位姿点)")
        except Exception as e:
            print(f"   ⚠️ 读取文件 {os.path.basename(file)} 失败: {e}")

    if not all_pts:
        print(f"   ❌ 未加载到任何数据！")
        return None, None, None

    # 拼接所有点和旋转矩阵
    print(f"\n[STEP 5] 📊 正在合并数据...")
    full_trajectory = np.vstack(all_pts)
    full_rotmats = np.vstack(all_rotmats)
    print(f"   ✅ 总共加载: {len(full_trajectory)} 个位姿点")

    # 计算绘图边界，保持 XYZ 比例一致
    print(f"\n[STEP 6] 📐 正在计算坐标范围...")
    max_range = np.ptp(full_trajectory, axis=0).max() / 2.0
    mid_pts = np.median(full_trajectory, axis=0)
    bounds = (mid_pts - max_range, mid_pts + max_range)

    if use_absolute:
        coord_info = "绝对坐标（已变换）"
    else:
        calib_type = "XYZ标定" if is_manual_calib else "自动打桩/WorldFrame"
        coord_info = f"相对坐标（{calib_type}）"
    
    print(f"   ✅ 坐标类型: {coord_info}")
    print(f"   ✅ 轨迹范围:")
    print(f"      X: [{full_trajectory[:, 0].min():.4f}, {full_trajectory[:, 0].max():.4f}]")
    print(f"      Y: [{full_trajectory[:, 1].min():.4f}, {full_trajectory[:, 1].max():.4f}]")
    print(f"      Z: [{full_trajectory[:, 2].min():.4f}, {full_trajectory[:, 2].max():.4f}]")

    return full_trajectory, full_rotmats, bounds


def save_static_image(pts, bounds, folder_path, use_absolute=False):
    """保存静态轨迹图片"""
    output_path = os.path.join(folder_path, 
                              "trajectory_plot_absolute.png" if use_absolute 
                              else "trajectory_plot_relative.png")
    print(f"\n[STEP 7] 📸 正在生成静态轨迹图...")
    print(f"         这可能需要几秒钟，请稍候...")

    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整轨迹线
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#1f77b4', linewidth=1.5, 
            alpha=0.6, label='Trajectory')

    # 标出起点和终点
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='green', s=100, 
               edgecolors='black', label='Start', zorder=5)
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='red', s=100, 
               edgecolors='black', label='End', zorder=5)

    # 设置坐标轴属性
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_zlim(bounds[0][2], bounds[1][2])
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)

    coord_type = "Absolute" if use_absolute else "Relative"
    ax.set_title(f"exUMI Pose Trajectory ({coord_type})\n{len(pts)} points",
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"   ✅ 静态图片已保存: {os.path.basename(output_path)}")


def save_animated_video(pts, rotmats, bounds, folder_path, use_absolute=False, output_format="mp4"):
    """
    生成轨迹运动回放视频或 GIF
    
    参数：
        output_format: "mp4" 或 "gif"
    """
    coord_type = "absolute" if use_absolute else "relative"
    base_name = f"trajectory_animation_{coord_type}"
    mp4_path = os.path.join(folder_path, f"{base_name}.mp4")
    gif_path = os.path.join(folder_path, f"{base_name}.gif")

    # 数据抽稀
    plot_pts = pts[::DOWNSAMPLE]
    plot_rotmats = rotmats[::DOWNSAMPLE]
    num_frames = len(plot_pts)

    print(f"\n[STEP 8] 📽️  正在生成动态视频...")
    print(f"         总帧数: {num_frames}，数据抽稀倍数: {DOWNSAMPLE}")
    print(f"         这可能需要几分钟，请耐心等待...")

    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴属性
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_zlim(bounds[0][2], bounds[1][2])
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)

    coord_type_str = "Absolute" if use_absolute else "Relative"
    ax.set_title(f"exUMI Pose Animation ({coord_type_str})", fontsize=12, fontweight='bold')

    # 初始化绘图元素
    full_line, = ax.plot([], [], [], color='gray', linewidth=0.5, alpha=0.2)
    head_dot = ax.scatter([], [], [], color='red', s=80, edgecolors='black', zorder=10)
    trace_line, = ax.plot([], [], [], color='#e377c2', linewidth=2, alpha=0.8)
    axis_artists = []

    # 姿态坐标轴长度
    axis_len = max(np.ptp(plot_pts, axis=0).max() * 0.08, 0.02)

    # 时间戳与统计文本
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=10)
    info_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes, fontsize=9)

    # 绘制灰色背景完整轨迹
    full_line.set_data(plot_pts[:, 0], plot_pts[:, 1])
    full_line.set_3d_properties(plot_pts[:, 2])

    def update(i):
        nonlocal axis_artists

        # 更新头部点
        current_pt = plot_pts[i, :]
        head_dot._offsets3d = ([current_pt[0]], [current_pt[1]], [current_pt[2]])

        # 更新尾迹（最近 100 帧）
        trace_start = max(0, i - 100)
        trace_data = plot_pts[trace_start:i+1, :]
        trace_line.set_data(trace_data[:, 0], trace_data[:, 1])
        trace_line.set_3d_properties(trace_data[:, 2])

        # 清理上一帧姿态坐标轴
        for artist in axis_artists:
            artist.remove()
        axis_artists = []

        # 绘制当前帧姿态坐标轴: X-红, Y-绿, Z-蓝
        basis = plot_rotmats[i]
        axis_colors = ['r', 'g', 'b']
        axis_labels = ['X', 'Y', 'Z']
        for axis_idx, (color, label) in enumerate(zip(axis_colors, axis_labels)):
            vec = basis[:, axis_idx] * axis_len
            axis_quiver = ax.quiver(
                current_pt[0], current_pt[1], current_pt[2],
                vec[0], vec[1], vec[2],
                color=color, linewidth=2.5, arrow_length_ratio=0.2, alpha=0.8
            )
            axis_artists.append(axis_quiver)

        # 更新文本
        progress = (i + 1) / num_frames * 100.0
        time_text.set_text(f'Frame: {i+1}/{num_frames} ({progress:.1f}%)')

        # 距离统计
        disp = np.linalg.norm(plot_pts[i] - plot_pts[0])
        info_text.set_text(f'Distance from start: {disp:.4f} m')

        return head_dot, trace_line, time_text, info_text, *axis_artists

    # 创建动画对象
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/VIDEO_FPS, blit=False)

    def make_progress_callback(tag):
        def _cb(i, n):
            if n <= 0:
                return
            if i == 0 or i == n - 1 or i % max(1, n // 20) == 0:
                pct = (i + 1) / n * 100.0
                bar_len = 30
                filled = int(bar_len * (i + 1) / n)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"         {tag} [{bar}] {pct:6.1f}% ({i+1}/{n})")
        return _cb

    # 保存为指定格式
    if output_format == "mp4":
        try:
            from matplotlib.animation import FFMpegWriter
            print("         🎬 使用 FFmpeg 编码 MP4...")
            writer = FFMpegWriter(fps=VIDEO_FPS, metadata=dict(artist='exUMI'), bitrate=2000)
            ani.save(mp4_path, writer=writer, progress_callback=make_progress_callback("MP4"))
            print(f"   ✅ MP4 视频已保存: {os.path.basename(mp4_path)}")
            return mp4_path
        except FileNotFoundError:
            print("   ⚠️ 未找到 FFmpeg，正在降级为 GIF...")
            output_format = "gif"

    if output_format == "gif":
        try:
            gif_step = max(1, int(np.ceil(num_frames / GIF_MAX_FRAMES)))
            gif_indices = np.arange(0, num_frames, gif_step)
            print(f"         🎬 GIF 将使用 {len(gif_indices)} 帧（原始 {num_frames} 帧，步长 {gif_step}）")
            ani_gif = FuncAnimation(fig, update, frames=gif_indices, interval=1000/GIF_FPS, blit=False)
            gif_writer = PillowWriter(fps=GIF_FPS)
            ani_gif.save(gif_path, writer=gif_writer, dpi=GIF_DPI, progress_callback=make_progress_callback("GIF"))
            print(f"   ✅ GIF 动画已保存: {os.path.basename(gif_path)}")
            return gif_path
        except Exception as e:
            print(f"   ❌ 生成 GIF 失败: {e}")

    plt.close(fig)
    return None


def save_trajectory_stats(pts, folder_path, use_absolute=False):
    """保存轨迹统计信息（详细版 + 坐标系验证）"""
    output_path = os.path.join(folder_path,
                              "trajectory_stats_absolute.txt" if use_absolute
                              else "trajectory_stats_relative.txt")

    print(f"\n[STEP 9] 📊 正在保存轨迹统计信息...")

    start_pos = pts[0]
    end_pos = pts[-1]
    rel_disp = end_pos - start_pos
    
    # 计算总长和速度
    frame_deltas = np.diff(pts, axis=0)
    frame_distances = np.linalg.norm(frame_deltas, axis=1)
    total_dist = np.sum(frame_distances)
    
    # 计算轨迹方向和各轴运动量
    disp_magnitude = np.linalg.norm(rel_disp)
    x_range = np.ptp(pts[:, 0])
    y_range = np.ptp(pts[:, 1])
    z_range = np.ptp(pts[:, 2])
    
    # 确定主要运动方向
    abs_disp = np.abs(rel_disp)
    primary_axis = np.argmax(abs_disp)
    axis_names = ['X', 'Y', 'Z']
    primary_direction = axis_names[primary_axis]
    primary_sign = '正向' if rel_disp[primary_axis] > 0 else '负向'
    
    # 分段分析（分成5-10段）
    num_segments = min(10, max(5, len(pts) // 100))
    segment_size = len(pts) // num_segments
    
    with open(output_path, 'w', encoding='utf-8') as f:
        coord_type = "绝对坐标" if use_absolute else "相对坐标"
        
        f.write("=" * 80 + "\n")
        f.write(f"exUMI 轨迹详细统计报告 ({coord_type})\n")
        f.write("=" * 80 + "\n\n")
        
        # 【坐标系说明】
        f.write("[0️⃣] 坐标系说明\n")
        f.write("-" * 80 + "\n")
        if use_absolute:
            f.write("📍 绝对坐标：已使用 calibration_params.npz 或 WorldFrame 变换\n")
            f.write("   - 原点对应采集时的 manual_origin（或自动打桩的起点）\n")
            f.write("   - 坐标轴沿标定时的 X/Y/Z 方向\n")
        else:
            f.write("📍 相对坐标：直接使用采集保存的数据（最精确）\n")
            f.write("   - 完全对应采集时的坐标系\n")
            f.write("   - 原点 = 采集时的 manual_origin\n")
            f.write("   - X/Y/Z 轴 = 采集时的标定轴\n")
            f.write("   ⚠️  如果 X 方向平移看起来不是沿 X 轴：\n")
            f.write("       → 检查采集时的 XYZ 标定是否正确\n")
            f.write("       → 回看可视化的 RGB 坐标轴（动画视频中的箭头）是否指向正确\n")
        f.write("\n\n")
        
        # 第一部分：基本信息
        f.write("[1] 基本信息\n")
        f.write("-" * 80 + "\n")
        f.write(f"总采样点数:        {len(pts)} 个\n")
        f.write(f"数据采集时长:      约 {len(pts) / 30:.1f} 秒 (假设采样率 30Hz)\n")
        f.write(f"起始位置:          [{start_pos[0]:8.6f}, {start_pos[1]:8.6f}, {start_pos[2]:8.6f}] m\n")
        f.write(f"终止位置:          [{end_pos[0]:8.6f}, {end_pos[1]:8.6f}, {end_pos[2]:8.6f}] m\n\n")
        
        # 第二部分：轨迹总体特征
        f.write("[2] 轨迹总体特征\n")
        f.write("-" * 80 + "\n")
        f.write(f"直线位移向量:      [{rel_disp[0]:8.6f}, {rel_disp[1]:8.6f}, {rel_disp[2]:8.6f}] m\n")
        f.write(f"直线位移大小:      {disp_magnitude:.6f} m\n")
        f.write(f"轨迹总长度:        {total_dist:.6f} m\n")
        f.write(f"轨迹冗余度:        {total_dist / max(disp_magnitude, 1e-6):.2f}x (越接近1越直线)\n")
        f.write(f"\n主运动方向:        {primary_direction} 轴 {primary_sign}\n")
        f.write(f"运动量分布:        X轴 {abs_disp[0]:.6f} m | Y轴 {abs_disp[1]:.6f} m | Z轴 {abs_disp[2]:.6f} m\n")
        f.write(f"范围分布:          X轴 {x_range:.6f} m | Y轴 {y_range:.6f} m | Z轴 {z_range:.6f} m\n\n")
        
        # 第三部分：速度分析
        f.write("[3] 速度统计\n")
        f.write("-" * 80 + "\n")
        velocities = frame_distances * 30  # 假设 30Hz 采样率，转换为 m/s
        f.write(f"平均速度:          {np.mean(velocities):.6f} m/s\n")
        f.write(f"最大速度:          {np.max(velocities):.6f} m/s (帧 {np.argmax(velocities)})\n")
        f.write(f"最小速度:          {np.min(velocities):.6f} m/s\n")
        f.write(f"速度标准差:        {np.std(velocities):.6f} m/s\n\n")
        
        # 第四部分：分段运动分析
        f.write("[4] 分段运动分析 (共 {} 段)\n".format(num_segments))
        f.write("-" * 80 + "\n")
        
        for seg in range(num_segments):
            start_idx = seg * segment_size
            end_idx = min((seg + 1) * segment_size, len(pts))
            
            seg_start = pts[start_idx]
            seg_end = pts[end_idx - 1]
            seg_disp = seg_end - seg_start
            seg_dist = np.sum(np.linalg.norm(np.diff(pts[start_idx:end_idx], axis=0), axis=1))
            seg_mag = np.linalg.norm(seg_disp)
            
            # 确定主运动方向
            seg_abs_disp = np.abs(seg_disp)
            if np.max(seg_abs_disp) > 1e-6:
                seg_primary = np.argmax(seg_abs_disp)
                seg_primary_name = axis_names[seg_primary]
                seg_primary_sign = '正' if seg_disp[seg_primary] > 0 else '负'
                direction_str = f"{seg_primary_name}轴{seg_primary_sign}"
            else:
                direction_str = "基本静止"
            
            time_start = start_idx / 30
            time_end = (end_idx - 1) / 30
            
            f.write(f"\n  第 {seg + 1} 段 (时刻 {time_start:.2f}s ~ {time_end:.2f}s, 帧 {start_idx:5d} ~ {end_idx-1:5d}):\n")
            f.write(f"    位移方向:      {direction_str}\n")
            f.write(f"    位移向量:      [{seg_disp[0]:8.6f}, {seg_disp[1]:8.6f}, {seg_disp[2]:8.6f}] m\n")
            f.write(f"    直线位移:      {seg_mag:.6f} m\n")
            f.write(f"    路径长度:      {seg_dist:.6f} m\n")
            f.write(f"    冗余度:        {seg_dist / max(seg_mag, 1e-6):.2f}x\n")
        
        f.write("\n\n")
        
        # 第五部分：各轴独立分析
        f.write("[5] 各轴详细分析\n")
        f.write("-" * 80 + "\n")
        
        for axis_idx, axis_name in enumerate(axis_names):
            axis_data = pts[:, axis_idx]
            axis_min = np.min(axis_data)
            axis_max = np.max(axis_data)
            axis_mean = np.mean(axis_data)
            axis_std = np.std(axis_data)
            axis_change = axis_data[-1] - axis_data[0]
            
            # 识别主要运动阶段
            axis_deltas = np.diff(axis_data)
            positive_move = np.sum(axis_deltas[axis_deltas > 0])
            negative_move = np.sum(np.abs(axis_deltas[axis_deltas < 0]))
            
            f.write(f"\n{axis_name} 轴:\n")
            f.write(f"    范围:          [{axis_min:.6f}, {axis_max:.6f}] m\n")
            f.write(f"    总变化:        {axis_change:+.6f} m\n")
            f.write(f"    正向移动:      {positive_move:.6f} m\n")
            f.write(f"    负向移动:      {negative_move:.6f} m\n")
            f.write(f"    平均值:        {axis_mean:.6f} m\n")
            f.write(f"    标准差:        {axis_std:.6f} m\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

    print(f"   ✅ 详细统计信息已保存: {os.path.basename(output_path)}")


def main():
    print(f"\n{'='*70}")
    print(f"   exUMI 轨迹可视化工具 v1.0")
    print(f"{'='*70}")

    # 使用全局定义的参数（写死在代码顶部）
    print(f"\n[初始化] 🔧 读取配置...")
    print(f"         数据目录: {DATA_FOLDER}")
    print(f"         坐标模式: {'绝对坐标' if USE_ABSOLUTE else '相对坐标'}")
    print(f"         视频格式: {OUTPUT_FORMAT}")
    print(f"         跳过视频: {SKIP_VIDEO}")

    if not os.path.exists(DATA_FOLDER):
        print(f"\n❌ [错误] 文件夹不存在: {DATA_FOLDER}")
        return

    print(f"\n{'='*70}\n")

    # 加载数据
    trajectory, rotmats, bounds = load_and_process_data(DATA_FOLDER, 
                                                         use_absolute=USE_ABSOLUTE)

    if trajectory is not None:
        # 保存统计信息
        save_trajectory_stats(trajectory, DATA_FOLDER, use_absolute=USE_ABSOLUTE)

        # 生成静态图
        save_static_image(trajectory, bounds, DATA_FOLDER, use_absolute=USE_ABSOLUTE)

        # 生成动态视频
        if not SKIP_VIDEO:
            if OUTPUT_FORMAT in ["mp4", "both"]:
                save_animated_video(trajectory, rotmats, bounds, DATA_FOLDER,
                                  use_absolute=USE_ABSOLUTE, output_format="mp4")
            if OUTPUT_FORMAT in ["gif", "both"]:
                save_animated_video(trajectory, rotmats, bounds, DATA_FOLDER,
                                  use_absolute=USE_ABSOLUTE, output_format="gif")

        print(f"\n{'='*70}")
        print(f"✅ [完成] 可视化处理已全部完成！")
        print(f"         所有文件已保存到:")
        print(f"         {DATA_FOLDER}")
        print(f"{'='*70}\n")
    else:
        print(f"\n❌ [错误] 无法加载轨迹数据，请检查数据文件夹。\n")


if __name__ == "__main__":
    main()
