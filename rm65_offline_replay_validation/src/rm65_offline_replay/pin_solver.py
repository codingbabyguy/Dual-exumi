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

    def _make_local_cost(self, cand: IKCandidate, cfg: dict) -> float:
        sel = cfg["selection"]
        cost = 0.0
        cost += float(sel["w_local_pos"]) * cand.pos_err_m
        cost += float(sel["w_local_rot"]) * cand.rot_err_rad
        cost += float(sel["w_local_limit"]) / (cand.limit_margin_rad + 1e-6)
        cost += float(sel["w_local_sing"]) / (cand.sigma_min + 1e-6)
        if not cand.success:
            cost += float(sel["failed_candidate_penalty"])
        return float(cost)

    def generate_candidates(
        self,
        T_target: np.ndarray,
        q_prev: np.ndarray | None,
        rng: np.random.Generator,
        cfg: dict,
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
            cand.local_cost = self._make_local_cost(cand, cfg)
            cands.append(cand)

        cands = _dedup_candidates(cands, tol_rad=float(ik_cfg["dedup_joint_tol_rad"]))
        cands = sorted(cands, key=lambda x: x.local_cost)
        max_keep = int(ik_cfg["max_candidates_per_frame"])
        cands = cands[:max_keep]
        return cands

    def _transition_cost(self, q_prev: np.ndarray, q_now: np.ndarray, cfg: dict) -> tuple[float, bool]:
        sel = cfg["selection"]
        dq = np.asarray(q_now - q_prev, dtype=np.float64)
        max_jump = float(np.max(np.abs(dq)))
        jump_flag = max_jump > float(sel["branch_jump_rad"])
        cost = float(sel["w_transition_smooth"]) * float(np.linalg.norm(dq))
        if jump_flag:
            cost += float(sel["branch_penalty"])
        return cost, jump_flag

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
                    t_cost, _ = self._transition_cost(p.q, c.q, cfg)
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
            _, jump_flag = self._transition_cost(chosen[i - 1].q, chosen[i].q, cfg)
            branch_flags[i] = 1 if jump_flag else 0
        branch_count = int(np.sum(branch_flags))
        return chosen, branch_count, branch_flags

    def solve_sequence(self, target_T_list: list[np.ndarray], cfg: dict) -> dict[str, Any]:
        ik_seed = int(cfg["ik"]["random_seed"])
        rng = np.random.default_rng(ik_seed)

        candidates_per_frame: list[list[IKCandidate]] = []
        q_prev: np.ndarray | None = None
        for i, T_target in enumerate(target_T_list):
            local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i)
            cands = self.generate_candidates(T_target, q_prev=q_prev, rng=local_rng, cfg=cfg)
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
        q_selected = np.stack([c.q for c in chosen], axis=0)
        success = np.array([1 if c.success else 0 for c in chosen], dtype=np.int32)
        pos_err_m = np.array([c.pos_err_m for c in chosen], dtype=np.float64)
        rot_err_rad = np.array([c.rot_err_rad for c in chosen], dtype=np.float64)
        limit_margin_rad = np.array([c.limit_margin_rad for c in chosen], dtype=np.float64)
        sigma_min = np.array([c.sigma_min for c in chosen], dtype=np.float64)
        local_cost = np.array([c.local_cost for c in chosen], dtype=np.float64)

        achieved_T = [self.fk_matrix(q) for q in q_selected]
        achieved_pose7 = np.stack([matrix_to_pose7(T) for T in achieved_T], axis=0)

        return {
            "q_selected": q_selected,
            "success": success,
            "pos_err_m": pos_err_m,
            "rot_err_rad": rot_err_rad,
            "limit_margin_rad": limit_margin_rad,
            "sigma_min": sigma_min,
            "branch_flags": branch_flags,
            "branch_count": branch_count,
            "local_cost": local_cost,
            "achieved_T": achieved_T,
            "achieved_pose7": achieved_pose7,
            "joint_names": self.joint_names,
        }

