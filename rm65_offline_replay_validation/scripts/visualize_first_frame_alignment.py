#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rm65_offline_replay.config import load_config, transform_from_cfg
from rm65_offline_replay.io_pose import find_aligned_pose_jsons, load_aligned_pose_json
from rm65_offline_replay.math3d import (
    compose,
    matrix_to_pose7,
    pose7_to_matrix,
    pose_error_components,
    transform_inverse,
)
from rm65_offline_replay.pin_solver import PinocchioIKBatchSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 alignment check: render one frame with flange XYZ axes in MuJoCo."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--session_dir", type=str, default=None, help="Session dir with demos/demo_*/aligned_arcap_poses.json")
    parser.add_argument("--demo_json", type=str, nargs="*", default=None, help="Optional explicit demo json list.")
    parser.add_argument("--demo_index", type=int, default=0, help="Demo index in selected list.")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index inside selected demo.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--output_mp4", type=str, default=None, help="Optional explicit output mp4 path.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional explicit output debug json path.")
    parser.add_argument("--fps", type=int, default=30, help="Render fps.")
    parser.add_argument("--hold_sec", type=float, default=3.0, help="Static video duration in seconds.")
    parser.add_argument("--width", type=int, default=1280, help="Frame width.")
    parser.add_argument("--height", type=int, default=720, help="Frame height.")
    parser.add_argument("--axis_length_m", type=float, default=0.10, help="Flange XYZ axis length in meters.")
    parser.add_argument("--axis_radius_m", type=float, default=0.004, help="Flange XYZ axis radius in meters.")
    parser.add_argument("--camera_distance", type=float, default=0.90, help="Free-camera distance.")
    parser.add_argument("--camera_azimuth", type=float, default=125.0, help="Free-camera azimuth in degrees.")
    parser.add_argument("--camera_elevation", type=float, default=-20.0, help="Free-camera elevation in degrees.")
    return parser.parse_args()


def _rewrite_urdf_package_paths(urdf_path: Path) -> Path:
    urdf_path = urdf_path.expanduser().resolve()
    text = urdf_path.read_text(encoding="utf-8")

    package_root = urdf_path.parent.parent
    package_name = package_root.name
    workspace_root = package_root.parent

    tmp_dir = Path(tempfile.mkdtemp(prefix="mujoco_rm65_align_"))
    tmp_urdf = tmp_dir / "model.urdf"

    mesh_ref_pat = re.compile(r'filename="([^"]+)"')
    matches = list(mesh_ref_pat.finditer(text))
    new_text = text
    offset = 0

    def _resolve_source(raw_ref: str) -> Path | None:
        if raw_ref.startswith("package://"):
            rest = raw_ref[len("package://") :]
            if "/" not in rest:
                return None
            pkg, rel = rest.split("/", 1)
            if pkg == package_name:
                cand = package_root / rel
            else:
                cand = workspace_root / pkg / rel
            return cand if cand.is_file() else None
        p = Path(raw_ref)
        if p.is_absolute():
            return p if p.is_file() else None
        cand = urdf_path.parent / p
        return cand if cand.is_file() else None

    copied_names: set[str] = set()
    for m in matches:
        raw_ref = m.group(1)
        src = _resolve_source(raw_ref)
        if src is None:
            continue
        basename = src.name
        dst = tmp_dir / basename
        if basename not in copied_names:
            shutil.copy2(src, dst)
            copied_names.add(basename)
        start, end = m.span(1)
        start += offset
        end += offset
        new_text = new_text[:start] + basename + new_text[end:]
        offset += len(basename) - (end - start)

    replacement = package_root.as_posix().rstrip("/") + "/"
    new_text = new_text.replace(f"package://{package_name}/", replacement)

    def _generic_replace(match: re.Match[str]) -> str:
        pkg = match.group(1)
        cand = workspace_root / pkg
        if cand.is_dir():
            return cand.as_posix().rstrip("/") + "/"
        return match.group(0)

    new_text = re.sub(r"package://([^/]+)/", _generic_replace, new_text)

    tmp_urdf.write_text(new_text, encoding="utf-8")
    return tmp_urdf


def _build_target_one(pose7: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames_cfg = cfg["frames"]
    input_pose_represents = str(cfg["robot"].get("input_pose_represents", "tcp")).strip().lower()
    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    if input_pose_represents not in {"tcp", "flange"}:
        raise ValueError(f"robot.input_pose_represents must be tcp|flange, got {input_pose_represents}")
    if solve_frame not in {"flange", "tcp"}:
        raise ValueError(f"robot.solve_frame must be flange|tcp, got {solve_frame}")

    T_B_from_pose = transform_from_cfg(frames_cfg["T_B_from_pose_frame"])
    T_pose_to_tcp = transform_from_cfg(frames_cfg["T_pose_to_tcp"])
    T_flange_to_tcp = transform_from_cfg(frames_cfg["T_flange_to_tcp"])
    T_pose_to_flange = transform_inverse(T_flange_to_tcp)

    T_pose = pose7_to_matrix(pose7)
    if input_pose_represents == "flange":
        T_B_flange = compose(T_B_from_pose, T_pose)
        T_B_tcp = compose(T_B_flange, T_flange_to_tcp)
    else:
        T_B_tcp = compose(compose(T_B_from_pose, T_pose), T_pose_to_tcp)
        T_B_flange = compose(T_B_tcp, T_pose_to_flange)

    if solve_frame == "flange":
        T_B_solve = T_B_flange
    else:
        T_B_solve = T_B_tcp
    return T_B_solve, T_B_flange, T_B_tcp


def _find_named_pose(
    mujoco: Any,
    model: Any,
    data: Any,
    frame_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_name = str(frame_name).strip()
    if len(frame_name) == 0:
        return None
    obj_candidates = [
        (mujoco.mjtObj.mjOBJ_BODY, "xpos", "xmat"),
        (mujoco.mjtObj.mjOBJ_SITE, "site_xpos", "site_xmat"),
        (mujoco.mjtObj.mjOBJ_GEOM, "geom_xpos", "geom_xmat"),
    ]
    for obj_type, pos_attr, mat_attr in obj_candidates:
        idx = int(mujoco.mj_name2id(model, obj_type, frame_name))
        if idx < 0:
            continue
        pos = np.asarray(getattr(data, pos_attr)[idx], dtype=np.float64).copy()
        mat = np.asarray(getattr(data, mat_attr)[idx], dtype=np.float64).reshape(3, 3).copy()
        return pos, mat
    return None


def _add_axis_markers(
    mujoco: Any,
    scene: Any,
    origin_xyz: np.ndarray,
    rot_mat: np.ndarray,
    axis_length_m: float,
    axis_radius_m: float,
) -> None:
    def _call_connector(
        geom: Any,
        radius_m: float,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
    ) -> None:
        fn = getattr(mujoco, "mjv_makeConnector", None)
        if fn is None:
            fn = getattr(mujoco, "mjv_connector", None)
        if fn is None:
            raise RuntimeError(
                "MuJoCo python API missing connector function: "
                "expected mjv_makeConnector or mjv_connector."
            )
        sx, sy, sz = [float(x) for x in np.asarray(start_xyz, dtype=np.float64).reshape(3)]
        ex, ey, ez = [float(x) for x in np.asarray(end_xyz, dtype=np.float64).reshape(3)]
        try:
            # Most bindings: (geom, type, width, from_x, from_y, from_z, to_x, to_y, to_z)
            fn(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(radius_m),
                sx,
                sy,
                sz,
                ex,
                ey,
                ez,
            )
            return
        except TypeError:
            # Some bindings may expose vector endpoints.
            fn(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(radius_m),
                np.asarray([sx, sy, sz], dtype=np.float64),
                np.asarray([ex, ey, ez], dtype=np.float64),
            )

    colors = np.asarray(
        [
            [1.0, 0.15, 0.15, 1.0],  # X: red
            [0.15, 1.0, 0.15, 1.0],  # Y: green
            [0.20, 0.45, 1.0, 1.0],  # Z: blue
        ],
        dtype=np.float32,
    )
    size = np.asarray([axis_radius_m, axis_radius_m, axis_radius_m], dtype=np.float64)
    zero_pos = np.zeros((3,), dtype=np.float64)
    eye9 = np.eye(3, dtype=np.float64).reshape(-1)
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    rot = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)

    for axis in range(3):
        if int(scene.ngeom) >= int(scene.maxgeom):
            return
        geom = scene.geoms[scene.ngeom]
        end = origin + rot[:, axis] * float(axis_length_m)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            size,
            zero_pos,
            eye9,
            colors[axis],
        )
        _call_connector(
            geom=geom,
            radius_m=float(axis_radius_m),
            start_xyz=origin,
            end_xyz=end,
        )
        scene.ngeom += 1

    if int(scene.ngeom) >= int(scene.maxgeom):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([axis_radius_m * 1.5, 0.0, 0.0], dtype=np.float64),
        origin,
        eye9,
        np.asarray([1.0, 0.9, 0.2, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def _render_static_pose_video(
    mujoco: Any,
    imageio: Any,
    urdf_path: Path,
    q_single: np.ndarray,
    ee_frame_name: str,
    flange_pose_fallback: np.ndarray,
    output_mp4: Path,
    fps: int,
    hold_sec: float,
    width: int,
    height: int,
    axis_length_m: float,
    axis_radius_m: float,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
) -> None:
    rewritten_urdf = _rewrite_urdf_package_paths(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(rewritten_urdf))
    data = mujoco.MjData(model)

    off_w = int(getattr(model.vis.global_, "offwidth", 640))
    off_h = int(getattr(model.vis.global_, "offheight", 480))
    render_w = min(int(width), off_w)
    render_h = min(int(height), off_h)
    if render_w != int(width) or render_h != int(height):
        print(
            f"[WARN] requested {width}x{height} exceeds offscreen {off_w}x{off_h}, fallback to {render_w}x{render_h}",
            flush=True,
        )

    renderer = mujoco.Renderer(model, width=render_w, height=render_h)
    scene_option = mujoco.MjvOption()
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = float(camera_distance)
    camera.azimuth = float(camera_azimuth)
    camera.elevation = float(camera_elevation)

    nq_use = min(int(model.nq), int(np.asarray(q_single).shape[0]))
    if nq_use <= 0:
        renderer.close()
        raise RuntimeError(f"Invalid nq_use={nq_use}, model.nq={model.nq}, q_single.shape={q_single.shape}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_mp4), fps=max(1, int(fps)), codec="libx264", quality=8)
    n_frames = max(1, int(round(max(0.2, float(hold_sec)) * max(1, int(fps)))))
    try:
        for _ in range(n_frames):
            data.qpos[:nq_use] = q_single[:nq_use]
            mujoco.mj_forward(model, data)

            named_pose = _find_named_pose(mujoco=mujoco, model=model, data=data, frame_name=ee_frame_name)
            if named_pose is None:
                origin = np.asarray(flange_pose_fallback[:3, 3], dtype=np.float64)
                rot = np.asarray(flange_pose_fallback[:3, :3], dtype=np.float64)
            else:
                origin, rot = named_pose
            camera.lookat[:] = origin

            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            _add_axis_markers(
                mujoco=mujoco,
                scene=renderer.scene,
                origin_xyz=origin,
                rot_mat=rot,
                axis_length_m=float(axis_length_m),
                axis_radius_m=float(axis_radius_m),
            )
            frame = renderer.render()
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    urdf_path = Path(cfg["robot"]["urdf_path"]).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"robot.urdf_path must be a file, got: {urdf_path}")

    demo_paths: list[Path]
    if args.demo_json:
        demo_paths = [Path(x).expanduser().resolve() for x in args.demo_json]
    elif args.session_dir:
        demo_paths = find_aligned_pose_jsons(args.session_dir)
    else:
        raise ValueError("Either --session_dir or --demo_json must be provided.")
    if len(demo_paths) == 0:
        raise RuntimeError("No demos found.")

    demo_index = int(args.demo_index)
    if demo_index < 0 or demo_index >= len(demo_paths):
        raise IndexError(f"demo_index={demo_index} out of range [0, {len(demo_paths)-1}]")
    demo_path = demo_paths[demo_index]
    demo = load_aligned_pose_json(demo_path)
    pose_arr = np.asarray(demo["pose"], dtype=np.float64)

    frame_index = int(args.frame_index)
    if frame_index < 0 or frame_index >= pose_arr.shape[0]:
        raise IndexError(f"frame_index={frame_index} out of range [0, {pose_arr.shape[0]-1}]")
    pose7 = pose_arr[frame_index]

    ee_frame_name = str(cfg["robot"]["ee_frame_name"])
    solver = PinocchioIKBatchSolver(urdf_path=str(urdf_path), ee_frame_name=ee_frame_name)

    T_B_solve, T_B_flange_target, _ = _build_target_one(pose7=pose7, cfg=cfg)
    ik_out = solver.solve_sequence([T_B_solve], cfg)
    q_selected = np.asarray(ik_out["q_selected"], dtype=np.float64)
    success = bool(int(np.asarray(ik_out["success"], dtype=np.int32)[0])) if q_selected.shape[0] > 0 else False
    if q_selected.shape[0] <= 0:
        raise RuntimeError("IK returned empty sequence for single-frame input.")
    q0 = q_selected[0]

    solve_frame = str(cfg["robot"]["solve_frame"]).strip().lower()
    achieved_solve_T = ik_out["achieved_T"][0]
    T_flange_to_tcp = transform_from_cfg(cfg["frames"]["T_flange_to_tcp"])
    if solve_frame == "flange":
        T_B_flange_achieved = achieved_solve_T
    else:
        T_tcp_to_flange = transform_inverse(T_flange_to_tcp)
        T_B_flange_achieved = compose(achieved_solve_T, T_tcp_to_flange)

    pos_err_m, rot_err_rad = pose_error_components(T_B_flange_target, T_B_flange_achieved)
    output_mp4 = (
        Path(args.output_mp4).expanduser().resolve()
        if args.output_mp4
        else out_dir / f"{demo['demo_name']}_frame{frame_index:04d}_flange_axes.mp4"
    )
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else out_dir / f"{demo['demo_name']}_frame{frame_index:04d}_alignment_debug.json"
    )

    try:
        import mujoco
    except ModuleNotFoundError as e:
        raise RuntimeError("MuJoCo python package missing. Install with: pip install mujoco") from e
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as e:
        raise RuntimeError("imageio missing. Install with: pip install imageio imageio-ffmpeg") from e

    print(f"[INFO] demo={demo['demo_name']} frame_index={frame_index} success={int(success)}", flush=True)
    print("[INFO] axis colors: X=red, Y=green, Z=blue", flush=True)
    _render_static_pose_video(
        mujoco=mujoco,
        imageio=imageio,
        urdf_path=urdf_path,
        q_single=q0,
        ee_frame_name=ee_frame_name,
        flange_pose_fallback=T_B_flange_achieved,
        output_mp4=output_mp4,
        fps=int(args.fps),
        hold_sec=float(args.hold_sec),
        width=int(args.width),
        height=int(args.height),
        axis_length_m=float(args.axis_length_m),
        axis_radius_m=float(args.axis_radius_m),
        camera_distance=float(args.camera_distance),
        camera_azimuth=float(args.camera_azimuth),
        camera_elevation=float(args.camera_elevation),
    )

    payload = {
        "config_path": str(cfg.get("config_path", args.config)),
        "demo_json": str(demo_path),
        "demo_name": str(demo["demo_name"]),
        "frame_index": int(frame_index),
        "input_pose_represents": str(cfg["robot"].get("input_pose_represents", "tcp")),
        "solve_frame": str(cfg["robot"]["solve_frame"]),
        "ee_frame_name": ee_frame_name,
        "ik_success": bool(success),
        "flange_pos_err_m": float(pos_err_m),
        "flange_rot_err_deg": float(np.rad2deg(rot_err_rad)),
        "q_selected_rad": [float(x) for x in q0.tolist()],
        "q_selected_deg": [float(x) for x in np.rad2deg(q0).tolist()],
        "target_flange_pose7": [float(x) for x in matrix_to_pose7(T_B_flange_target).tolist()],
        "achieved_flange_pose7": [float(x) for x in matrix_to_pose7(T_B_flange_achieved).tolist()],
        "axis_color_legend": {"x": "red", "y": "green", "z": "blue"},
        "artifacts": {"mp4": str(output_mp4), "json": str(output_json)},
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[DONE] video: {output_mp4}", flush=True)
    print(f"[DONE] debug json: {output_json}", flush=True)


if __name__ == "__main__":
    main()
