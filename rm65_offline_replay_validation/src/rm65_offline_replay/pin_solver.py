from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from .math3d import matrix_to_pose7, pose_error_components


@dataclass
class IKCandidate:
    q: np.ndarray
    success: bool
    pos_err_m: float
    rot_err_rad: float
    limit_margin_rad: float
    sigma_min: float
    local_cost: float


def _dedup_candidates(candidates: list[IKCandidate], tol_rad: float) -> list[IKCandidate]:
    out: list[IKCandidate] = []
    for cand in candidates:
        keep = True
        for ref in out:
            if np.max(np.abs(cand.q - ref.q)) <= tol_rad:
                keep = False
                break
        if keep:
            out.append(cand)
    return out


class PinocchioIKBatchSolver:
    def __init__(self, urdf_path: str, ee_frame_name: str):
        try:
            import pinocchio as pin
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Pinocchio is required. Install on server, e.g. conda/pip package `pin`."
            ) from e
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame `{ee_frame_name}` not found in URDF model.")

        self.nq = int(self.model.nq)
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64).copy()
        self.joint_names = self._extract_joint_names()
        self._frame_id_cache: dict[str, int] = {}

    def _extract_joint_names(self) -> list[str]:
        # Universe joint is index 0 in pinocchio.
        names: list[str] = []
        for jid, joint_model in enumerate(self.model.joints):
            if jid == 0:
                continue
            if joint_model.nq == 1:
                names.append(self.model.names[jid])
            elif joint_model.nq > 1:
                names.extend([f"{self.model.names[jid]}[{k}]" for k in range(joint_model.nq)])
        if len(names) != self.nq:
            names = [f"q{i}" for i in range(self.nq)]
        return names

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.lower), self.upper)

    def fk_matrix(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        oMf = self.data.oMf[self.ee_frame_id]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def _get_frame_id(self, frame_name: str) -> int | None:
        key = str(frame_name).strip()
        if len(key) == 0:
            return None
        cached = self._frame_id_cache.get(key, None)
        if cached is not None:
            return cached
        fid = self.model.getFrameId(key)
        if fid >= len(self.model.frames):
            return None
        self._frame_id_cache[key] = int(fid)
        return int(fid)

    def frame_translation(self, q: np.ndarray, frame_name: str) -> np.ndarray | None:
        fid = self._get_frame_id(frame_name)
        if fid is None:
            return None
        q = np.asarray(q, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacement(self.model, self.data, fid)
        return np.asarray(self.data.oMf[fid].translation, dtype=np.float64).copy()

    def jacobian_sigma_min(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64)
        J = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[-1])

    def limit_margin(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64)
        margin = np.minimum(q - self.lower, self.upper - q)
        return float(np.min(margin))

    def _resolve_home_q(self, cfg: dict) -> np.ndarray:
        sel = cfg.get("selection", {})
        home_q_rad = sel.get("home_q_rad", None)
        home_q_deg = sel.get("home_q_deg", None)
        if isinstance(home_q_rad, list) and len(home_q_rad) == self.nq:
            home_q = np.asarray(home_q_rad, dtype=np.float64)
            return self.clip_q(home_q)
        if isinstance(home_q_deg, list) and len(home_q_deg) == self.nq:
            home_q = np.deg2rad(np.asarray(home_q_deg, dtype=np.float64))
            return self.clip_q(home_q)
        return self.neutral.copy()

    def _home_bias_cost(self, q: np.ndarray, frame_idx: int, cfg: dict, home_q: np.ndarray) -> float:
        sel = cfg["selection"]
        q = np.asarray(q, dtype=np.float64)
        home_q = np.asarray(home_q, dtype=np.float64)
        dist = float(np.linalg.norm(q - home_q))
        w_home = float(sel.get("w_home", 0.0))
        cost = w_home * dist
        start_window = int(sel.get("start_home_window", 0))
        w_start_home = float(sel.get("w_start_home", 0.0))
        if start_window > 0 and frame_idx < start_window:
            alpha = float(start_window - frame_idx) / float(start_window)
            cost += w_start_home * alpha * dist
        return float(cost)

    def _center_cost(self, q: np.ndarray) -> float:
        # Encourage mid-range joint posture to avoid living near limits.
        center = 0.5 * (self.lower + self.upper)
        half_span = 0.5 * np.maximum(self.upper - self.lower, 1e-6)
        normed = (np.asarray(q, dtype=np.float64) - center) / half_span
        return float(np.linalg.norm(normed) / np.sqrt(self.nq))

    def _normalize_joint_index(self, idx_raw: Any) -> int | None:
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            idx += self.nq
        if idx < 0 or idx >= self.nq:
            return None
        return idx

    def _joint_range_prior_cost(self, q: np.ndarray, cfg: dict) -> tuple[float, bool]:
        sel = cfg.get("selection", {})
        ranges = sel.get("joint_preferred_ranges_deg", None)
        if not isinstance(ranges, list):
            return 0.0, False
        q = np.asarray(q, dtype=np.float64)
        acc = 0.0
        used = 0
        violated = False
        for i in range(min(self.nq, len(ranges))):
            item = ranges[i]
            if item is None:
                continue
            if isinstance(item, dict):
                lo_deg = item.get("min_deg", item.get("min", None))
                hi_deg = item.get("max_deg", item.get("max", None))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lo_deg, hi_deg = item[0], item[1]
            else:
                continue
            try:
                lo = float(np.deg2rad(float(lo_deg)))
                hi = float(np.deg2rad(float(hi_deg)))
            except (TypeError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            span = max(hi - lo, 1e-6)
            qi = float(q[i])
            if qi < lo:
                v = (lo - qi) / span
                violated = True
            elif qi > hi:
                v = (qi - hi) / span
                violated = True
            else:
                v = 0.0
            acc += v * v
            used += 1
        if used <= 0:
            return 0.0, False
        return float(acc / used), violated

    def _elbow_sign_cost(self, q: np.ndarray, cfg: dict) -> tuple[float, bool]:
        sel = cfg.get("selection", {})
        idx = self._normalize_joint_index(sel.get("elbow_joint_index", 2))
        if idx is None:
            return 0.0, False
        pref = 1.0 if float(sel.get("elbow_preferred_sign", 1.0)) >= 0.0 else -1.0
        deadband = max(float(sel.get("elbow_sign_deadband_rad", 0.0)), 0.0)
        signed = pref * float(np.asarray(q, dtype=np.float64)[idx])
        if signed >= deadband:
            return 0.0, False
        viol = deadband - signed
        scale = np.pi
        cost = (viol / scale) ** 2
        return float(cost), bool(signed < 0.0)

    def _elbow_halfspace_cost(self, q: np.ndarray, cfg: dict) -> tuple[float, bool]:
        sel = cfg.get("selection", {})
        frame_name = str(sel.get("elbow_frame_name", "")).strip()
        if len(frame_name) == 0:
            return 0.0, False
        p_elbow = self.frame_translation(q=q, frame_name=frame_name)
        if p_elbow is None:
            return 0.0, False
        normal = np.asarray(sel.get("elbow_halfspace_normal_xyz", [0.0, 0.0, 1.0]), dtype=np.float64)
        if normal.shape != (3,):
            return 0.0, False
        n_norm = float(np.linalg.norm(normal))
        if n_norm < 1e-12:
            return 0.0, False
        normal = normal / n_norm
        offset = float(sel.get("elbow_halfspace_offset_m", 0.0))
        pref = 1.0 if float(sel.get("elbow_halfspace_preferred_sign", 1.0)) >= 0.0 else -1.0
        signed = pref * (float(np.dot(normal, p_elbow)) + offset)
        if signed >= 0.0:
            return 0.0, False
        viol_m = -signed
        scale_m = max(float(sel.get("elbow_halfspace_scale_m", 0.10)), 1e-4)
        cost = (viol_m / scale_m) ** 2
        return float(cost), True

    def _wrist_flip_transition_cost(
        self,
        q_prev: np.ndarray,
        q_now: np.ndarray,
        cfg: dict,
    ) -> tuple[float, bool]:
        sel = cfg.get("selection", {})
        idx_list = sel.get("wrist_joint_indices", [-2, -1])
        if not isinstance(idx_list, list):
            idx_list = [idx_list]
        step_th = max(float(sel.get("wrist_flip_step_threshold_rad", 0.8)), 1e-6)
        sign_eps = max(float(sel.get("wrist_flip_sign_epsilon_rad", 0.15)), 0.0)
        q_prev = np.asarray(q_prev, dtype=np.float64)
        q_now = np.asarray(q_now, dtype=np.float64)
        raw = 0.0
        flagged = False
        for idx_raw in idx_list:
            idx = self._normalize_joint_index(idx_raw)
            if idx is None:
                continue
            a = float(q_prev[idx])
            b = float(q_now[idx])
            dq = abs(b - a)
            if dq > step_th:
                raw += (dq - step_th) / step_th
                flagged = True
            if abs(a) > sign_eps and abs(b) > sign_eps and (a * b < 0.0):
                raw += 1.0
                flagged = True
        return float(raw), flagged

    def _ik_residual(
        self, q: np.ndarray, T_target: np.ndarray, pos_w: float = 1.0, rot_w: float = 1.0
    ) -> np.ndarray:
        T_actual = self.fk_matrix(q)
        dp = T_actual[:3, 3] - T_target[:3, 3]
        r_err = R.from_matrix(T_actual[:3, :3].T @ T_target[:3, :3]).as_rotvec()
        return np.concatenate([pos_w * dp, rot_w * r_err], axis=0)

    def solve_ik(
        self,
        T_target: np.ndarray,
        q_seed: np.ndarray,
        max_nfev: int,
        pos_tol_m: float,
        rot_tol_rad: float,
    ) -> tuple[np.ndarray, bool, float, float]:
        q_seed = self.clip_q(np.asarray(q_seed, dtype=np.float64))
        result = least_squares(
            fun=lambda q: self._ik_residual(q, T_target),
            x0=q_seed,
            bounds=(self.lower, self.upper),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
        )
        q_sol = self.clip_q(result.x)
        T_actual = self.fk_matrix(q_sol)
        pos_err_m, rot_err_rad = pose_error_components(T_target, T_actual)
        ok = bool(pos_err_m <= pos_tol_m and rot_err_rad <= rot_tol_rad)
        return q_sol, ok, pos_err_m, rot_err_rad

    def _make_local_cost(self, cand: IKCandidate, cfg: dict, frame_idx: int, home_q: np.ndarray) -> float:
        sel = cfg["selection"]
        cost = 0.0
        cost += float(sel["w_local_pos"]) * cand.pos_err_m
        cost += float(sel["w_local_rot"]) * cand.rot_err_rad
        cost += float(sel["w_local_limit"]) / (cand.limit_margin_rad + 1e-6)
        cost += float(sel["w_local_sing"]) / (cand.sigma_min + 1e-6)
        cost += float(sel.get("w_local_center", 0.0)) * self._center_cost(cand.q)
        cost += self._home_bias_cost(cand.q, frame_idx=frame_idx, cfg=cfg, home_q=home_q)
        w_shape_joint_range = float(sel.get("w_shape_joint_range", 0.0))
        if w_shape_joint_range > 0.0:
            shape_cost, _ = self._joint_range_prior_cost(cand.q, cfg)
            cost += w_shape_joint_range * shape_cost
        w_elbow_sign = float(sel.get("w_elbow_sign", 0.0))
        if w_elbow_sign > 0.0:
            elbow_sign_cost, _ = self._elbow_sign_cost(cand.q, cfg)
            cost += w_elbow_sign * elbow_sign_cost
        w_elbow_halfspace = float(sel.get("w_elbow_halfspace", 0.0))
        if w_elbow_halfspace > 0.0:
            elbow_halfspace_cost, _ = self._elbow_halfspace_cost(cand.q, cfg)
            cost += w_elbow_halfspace * elbow_halfspace_cost
        if not cand.success:
            cost += float(sel["failed_candidate_penalty"])
        return float(cost)

    def generate_candidates(
        self,
        T_target: np.ndarray,
        q_prev: np.ndarray | None,
        rng: np.random.Generator,
        cfg: dict,
        frame_idx: int,
        home_q: np.ndarray,
    ) -> list[IKCandidate]:
        ik_cfg = cfg["ik"]
        max_nfev = int(ik_cfg["max_nfev"])
        pos_tol_m = float(ik_cfg["pos_tol_m"])
        rot_tol_rad = float(np.deg2rad(ik_cfg["rot_tol_deg"]))

        seeds: list[np.ndarray] = []
        if q_prev is not None:
            seeds.append(np.asarray(q_prev, dtype=np.float64))
            seeds.append(self.clip_q(q_prev + rng.normal(scale=0.04, size=self.nq)))
            seeds.append(self.clip_q(q_prev + rng.normal(scale=0.08, size=self.nq)))
        seeds.append(np.asarray(home_q, dtype=np.float64))
        if frame_idx <= 3:
            seeds.append(self.clip_q(home_q + rng.normal(scale=0.03, size=self.nq)))
            seeds.append(self.clip_q(home_q + rng.normal(scale=0.06, size=self.nq)))
        seeds.append(self.neutral)

        n_random = int(ik_cfg["n_random_seeds"])
        for _ in range(n_random):
            seeds.append(rng.uniform(self.lower, self.upper))

        cands: list[IKCandidate] = []
        for seed in seeds:
            q, ok, pos_err_m, rot_err_rad = self.solve_ik(
                T_target=T_target,
                q_seed=seed,
                max_nfev=max_nfev,
                pos_tol_m=pos_tol_m,
                rot_tol_rad=rot_tol_rad,
            )
            cand = IKCandidate(
                q=q,
                success=ok,
                pos_err_m=float(pos_err_m),
                rot_err_rad=float(rot_err_rad),
                limit_margin_rad=self.limit_margin(q),
                sigma_min=self.jacobian_sigma_min(q),
                local_cost=0.0,
            )
            cand.local_cost = self._make_local_cost(cand, cfg, frame_idx=frame_idx, home_q=home_q)
            cands.append(cand)

        cands = _dedup_candidates(cands, tol_rad=float(ik_cfg["dedup_joint_tol_rad"]))
        cands = sorted(cands, key=lambda x: x.local_cost)
        max_keep = int(ik_cfg["max_candidates_per_frame"])
        cands = cands[:max_keep]
        return cands

    def _transition_cost(self, q_prev: np.ndarray, q_now: np.ndarray, cfg: dict) -> tuple[float, bool, bool]:
        sel = cfg["selection"]
        dq = np.asarray(q_now - q_prev, dtype=np.float64)
        max_jump = float(np.max(np.abs(dq)))
        jump_flag = max_jump > float(sel["branch_jump_rad"])
        wrist_flip_flag = False
        w_l2 = float(sel.get("w_transition_l2", sel.get("w_transition_smooth", 0.0)))
        w_linf = float(sel.get("w_transition_linf", 0.0))
        cost = w_l2 * float(np.linalg.norm(dq))
        cost += w_linf * max_jump

        w_wrist_flip = float(sel.get("w_wrist_flip", 0.0))
        if w_wrist_flip > 0.0:
            wrist_raw_cost, wrist_flip_flag = self._wrist_flip_transition_cost(q_prev=q_prev, q_now=q_now, cfg=cfg)
            cost += w_wrist_flip * wrist_raw_cost
            if wrist_flip_flag:
                jump_flag = True

        hard_max_step = float(sel.get("hard_max_step_rad", 0.0))
        if hard_max_step > 0.0 and max_jump > hard_max_step:
            jump_flag = True
            excess = max_jump - hard_max_step
            cost += float(sel.get("hard_step_penalty", 200.0)) * (1.0 + excess / hard_max_step)
        if jump_flag:
            cost += float(sel["branch_penalty"])
        return cost, jump_flag, wrist_flip_flag

    def _evaluate_sequence(
        self, q_selected: np.ndarray, target_T_list: list[np.ndarray], cfg: dict, home_q: np.ndarray
    ) -> dict[str, Any]:
        ik_cfg = cfg["ik"]
        pos_tol_m = float(ik_cfg["pos_tol_m"])
        rot_tol_rad = float(np.deg2rad(ik_cfg["rot_tol_deg"]))

        n = int(q_selected.shape[0])
        success = np.zeros((n,), dtype=np.int32)
        pos_err_m = np.zeros((n,), dtype=np.float64)
        rot_err_rad = np.zeros((n,), dtype=np.float64)
        limit_margin_rad = np.zeros((n,), dtype=np.float64)
        sigma_min = np.zeros((n,), dtype=np.float64)
        local_cost = np.zeros((n,), dtype=np.float64)
        joint_pref_cost = np.zeros((n,), dtype=np.float64)
        elbow_sign_cost = np.zeros((n,), dtype=np.float64)
        elbow_halfspace_cost = np.zeros((n,), dtype=np.float64)
        joint_pref_violation = np.zeros((n,), dtype=np.int32)
        elbow_sign_violation = np.zeros((n,), dtype=np.int32)
        elbow_halfspace_violation = np.zeros((n,), dtype=np.int32)

        achieved_T: list[np.ndarray] = []
        for i in range(n):
            q = q_selected[i]
            T_actual = self.fk_matrix(q)
            achieved_T.append(T_actual)
            p_err, r_err = pose_error_components(target_T_list[i], T_actual)
            pos_err_m[i] = p_err
            rot_err_rad[i] = r_err
            limit_margin_rad[i] = self.limit_margin(q)
            sigma_min[i] = self.jacobian_sigma_min(q)
            success[i] = 1 if (p_err <= pos_tol_m and r_err <= rot_tol_rad) else 0
            cand = IKCandidate(
                q=q,
                success=bool(success[i]),
                pos_err_m=float(p_err),
                rot_err_rad=float(r_err),
                limit_margin_rad=float(limit_margin_rad[i]),
                sigma_min=float(sigma_min[i]),
                local_cost=0.0,
            )
            local_cost[i] = self._make_local_cost(cand, cfg, frame_idx=i, home_q=home_q)
            shape_cost_i, shape_bad_i = self._joint_range_prior_cost(q, cfg)
            elbow_sign_cost_i, elbow_sign_bad_i = self._elbow_sign_cost(q, cfg)
            elbow_hs_cost_i, elbow_hs_bad_i = self._elbow_halfspace_cost(q, cfg)
            joint_pref_cost[i] = shape_cost_i
            elbow_sign_cost[i] = elbow_sign_cost_i
            elbow_halfspace_cost[i] = elbow_hs_cost_i
            joint_pref_violation[i] = 1 if shape_bad_i else 0
            elbow_sign_violation[i] = 1 if elbow_sign_bad_i else 0
            elbow_halfspace_violation[i] = 1 if elbow_hs_bad_i else 0

        branch_flags = np.zeros((n,), dtype=np.int32)
        wrist_flip_flags = np.zeros((n,), dtype=np.int32)
        for i in range(1, n):
            _, jump_flag, wrist_flip_flag = self._transition_cost(q_selected[i - 1], q_selected[i], cfg)
            branch_flags[i] = 1 if jump_flag else 0
            wrist_flip_flags[i] = 1 if wrist_flip_flag else 0
        branch_count = int(np.sum(branch_flags))
        wrist_flip_count = int(np.sum(wrist_flip_flags))
        achieved_pose7 = np.stack([matrix_to_pose7(T) for T in achieved_T], axis=0)
        return {
            "success": success,
            "pos_err_m": pos_err_m,
            "rot_err_rad": rot_err_rad,
            "limit_margin_rad": limit_margin_rad,
            "sigma_min": sigma_min,
            "branch_flags": branch_flags,
            "branch_count": branch_count,
            "local_cost": local_cost,
            "wrist_flip_flags": wrist_flip_flags,
            "wrist_flip_count": wrist_flip_count,
            "joint_pref_cost": joint_pref_cost,
            "joint_pref_violation": joint_pref_violation,
            "joint_pref_violation_ratio": float(np.mean(joint_pref_violation)) if n > 0 else 0.0,
            "elbow_sign_cost": elbow_sign_cost,
            "elbow_sign_violation": elbow_sign_violation,
            "elbow_sign_violation_ratio": float(np.mean(elbow_sign_violation)) if n > 0 else 0.0,
            "elbow_halfspace_cost": elbow_halfspace_cost,
            "elbow_halfspace_violation": elbow_halfspace_violation,
            "elbow_halfspace_violation_ratio": float(np.mean(elbow_halfspace_violation)) if n > 0 else 0.0,
            "achieved_T": achieved_T,
            "achieved_pose7": achieved_pose7,
        }

    def _post_stabilize(
        self,
        q_selected: np.ndarray,
        target_T_list: list[np.ndarray],
        cfg: dict,
    ) -> tuple[np.ndarray, int]:
        sel = cfg["selection"]
        max_step = float(sel.get("hard_max_step_rad", 0.0))
        passes = int(sel.get("post_stabilize_passes", 0))
        if max_step <= 0.0 or passes <= 0:
            return q_selected, 0

        ik_cfg = cfg["ik"]
        max_nfev = int(ik_cfg["max_nfev"])
        pos_tol_m = float(ik_cfg["pos_tol_m"])
        rot_tol_rad = float(np.deg2rad(ik_cfg["rot_tol_deg"]))
        scale = float(sel.get("post_stabilize_dq_scale", 1.0))
        scale = max(0.1, min(1.0, scale))

        q_out = np.asarray(q_selected, dtype=np.float64).copy()
        fix_count = 0
        for _ in range(passes):
            changed = False
            for i in range(1, q_out.shape[0]):
                dq = q_out[i] - q_out[i - 1]
                old_jump = float(np.max(np.abs(dq)))
                if old_jump <= max_step:
                    continue
                limited = np.clip(dq, -max_step * scale, max_step * scale)
                q_seed = self.clip_q(q_out[i - 1] + limited)
                q_new, ok, _, _ = self.solve_ik(
                    T_target=target_T_list[i],
                    q_seed=q_seed,
                    max_nfev=max_nfev,
                    pos_tol_m=pos_tol_m,
                    rot_tol_rad=rot_tol_rad,
                )
                if not ok:
                    continue
                new_jump = float(np.max(np.abs(q_new - q_out[i - 1])))
                if new_jump < old_jump:
                    q_out[i] = q_new
                    fix_count += 1
                    changed = True
            if not changed:
                break
        return q_out, fix_count

    def select_sequence(
        self, candidates_per_frame: list[list[IKCandidate]], cfg: dict
    ) -> tuple[list[IKCandidate], int, np.ndarray]:
        n = len(candidates_per_frame)
        if n == 0:
            return [], 0, np.zeros((0,), dtype=np.float64)

        back_ptr: list[np.ndarray] = []
        dp_prev = np.array([c.local_cost for c in candidates_per_frame[0]], dtype=np.float64)

        for i in range(1, n):
            cur = candidates_per_frame[i]
            prev = candidates_per_frame[i - 1]
            trans = np.zeros((len(prev), len(cur)), dtype=np.float64)
            for p_idx, p in enumerate(prev):
                for c_idx, c in enumerate(cur):
                    t_cost, _, _ = self._transition_cost(p.q, c.q, cfg)
                    trans[p_idx, c_idx] = t_cost

            total = dp_prev[:, None] + trans + np.array([c.local_cost for c in cur])[None, :]
            best_prev = np.argmin(total, axis=0)
            dp_cur = total[best_prev, np.arange(len(cur))]
            back_ptr.append(best_prev)
            dp_prev = dp_cur

        last_idx = int(np.argmin(dp_prev))
        chosen_idx = [last_idx]
        for i in range(n - 2, -1, -1):
            last_idx = int(back_ptr[i][last_idx])
            chosen_idx.append(last_idx)
        chosen_idx.reverse()

        chosen = [candidates_per_frame[i][chosen_idx[i]] for i in range(n)]

        branch_flags = np.zeros((n,), dtype=np.int32)
        for i in range(1, n):
            _, jump_flag, _ = self._transition_cost(chosen[i - 1].q, chosen[i].q, cfg)
            branch_flags[i] = 1 if jump_flag else 0
        branch_count = int(np.sum(branch_flags))
        return chosen, branch_count, branch_flags

    def solve_sequence(self, target_T_list: list[np.ndarray], cfg: dict) -> dict[str, Any]:
        ik_seed = int(cfg["ik"]["random_seed"])
        rng = np.random.default_rng(ik_seed)
        home_q = self._resolve_home_q(cfg)

        candidates_per_frame: list[list[IKCandidate]] = []
        q_prev: np.ndarray | None = None
        for i, T_target in enumerate(target_T_list):
            local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i)
            cands = self.generate_candidates(
                T_target,
                q_prev=q_prev,
                rng=local_rng,
                cfg=cfg,
                frame_idx=i,
                home_q=home_q,
            )
            if not cands:
                # Defensive fallback, should rarely happen.
                q_fb = self.neutral if q_prev is None else q_prev
                T_fb = self.fk_matrix(q_fb)
                pos_err_m, rot_err_rad = pose_error_components(T_target, T_fb)
                fallback = IKCandidate(
                    q=q_fb.copy(),
                    success=False,
                    pos_err_m=pos_err_m,
                    rot_err_rad=rot_err_rad,
                    limit_margin_rad=self.limit_margin(q_fb),
                    sigma_min=self.jacobian_sigma_min(q_fb),
                    local_cost=float(cfg["selection"]["failed_candidate_penalty"]) + 100.0,
                )
                cands = [fallback]
            q_prev = cands[0].q
            candidates_per_frame.append(cands)

        chosen, branch_count, branch_flags = self.select_sequence(candidates_per_frame, cfg)
        q_selected_raw = np.stack([c.q for c in chosen], axis=0)
        q_selected, post_stabilize_fix_count = self._post_stabilize(
            q_selected=q_selected_raw,
            target_T_list=target_T_list,
            cfg=cfg,
        )
        eval_out = self._evaluate_sequence(
            q_selected=q_selected,
            target_T_list=target_T_list,
            cfg=cfg,
            home_q=home_q,
        )

        return {
            "q_selected": q_selected,
            "q_selected_raw": q_selected_raw,
            "post_stabilize_fix_count": int(post_stabilize_fix_count),
            "success": eval_out["success"],
            "pos_err_m": eval_out["pos_err_m"],
            "rot_err_rad": eval_out["rot_err_rad"],
            "limit_margin_rad": eval_out["limit_margin_rad"],
            "sigma_min": eval_out["sigma_min"],
            "branch_flags": eval_out["branch_flags"],
            "branch_count": eval_out["branch_count"],
            "branch_count_raw": int(branch_count),
            "branch_flags_raw": branch_flags,
            "wrist_flip_flags": eval_out["wrist_flip_flags"],
            "wrist_flip_count": int(eval_out["wrist_flip_count"]),
            "joint_pref_cost": eval_out["joint_pref_cost"],
            "joint_pref_violation": eval_out["joint_pref_violation"],
            "joint_pref_violation_ratio": float(eval_out["joint_pref_violation_ratio"]),
            "elbow_sign_cost": eval_out["elbow_sign_cost"],
            "elbow_sign_violation": eval_out["elbow_sign_violation"],
            "elbow_sign_violation_ratio": float(eval_out["elbow_sign_violation_ratio"]),
            "elbow_halfspace_cost": eval_out["elbow_halfspace_cost"],
            "elbow_halfspace_violation": eval_out["elbow_halfspace_violation"],
            "elbow_halfspace_violation_ratio": float(eval_out["elbow_halfspace_violation_ratio"]),
            "local_cost": eval_out["local_cost"],
            "achieved_T": eval_out["achieved_T"],
            "achieved_pose7": eval_out["achieved_pose7"],
            "joint_names": self.joint_names,
            "home_q_rad": home_q,
        }
