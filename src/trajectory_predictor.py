"""Trajectory predictor for Task 5.

Two motion models behind a unified interface:
  cv   — constant velocity, ego-motion-compensated, multi-step EKF predict
  ctrv — constant turn rate + velocity; same ego-motion compensation

Both models loop predict_state() (CV) or the CTRV kinematic update in
dt_substep_s steps so process noise Q scales correctly with each sub-interval
and per-substep ego-frame rotations are applied incrementally.

CTRV kinematic update (in world-aligned ego frame at each substep):
    v   = sqrt(vx² + vy²)
    ω   = estimated turn rate (rad/s) from yaw history
    if |ω| > 1e-3:
        x'   = x + (v/ω)·(sin(yaw + ω·Δt) − sin(yaw))
        y'   = y + (v/ω)·(−cos(yaw + ω·Δt) + cos(yaw))
    else:
        x'   = x + v·cos(yaw)·Δt
        y'   = y + v·sin(yaw)·Δt
    yaw' = yaw + ω·Δt
    vx'  = v·cos(yaw');  vy' = v·sin(yaw')
Then rotate into the new ego frame exactly as predict_state() does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ego_trajectory import EgoTrajectory
from src.motion_model import predict_state, _rot2d
from src.task2_fusion import _X, _Y, _VX, _VY, _YAW, _L, _W, STATE_DIM
from src.tf_utils import EgoStateLog


@dataclass
class Prediction:
    track_id: int
    source_ts_ns: int
    horizon_s: float
    target_ts_ns: int
    state: np.ndarray       # (7,) in predicted-future ego frame
    covariance: np.ndarray  # (7, 7)
    world_xy: tuple[float, float]
    model: str              # "cv" or "ctrv"


class TrajectoryPredictor:
    """Predict future states for a single confirmed track.

    Parameters
    ----------
    ego_traj     : EgoTrajectory — used to convert predicted ego-frame state → world XY.
    ego_log      : EgoStateLog  — supplies ego velocity/heading at each substep.
    model        : "cv" or "ctrv".
    dt_substep_s : substep size in seconds (default 0.05 s = 50 ms).
    """

    def __init__(
        self,
        ego_traj: EgoTrajectory,
        ego_log: EgoStateLog,
        model: str = "cv",
        dt_substep_s: float = 0.05,
    ) -> None:
        if model not in ("cv", "ctrv"):
            raise ValueError(f"model must be 'cv' or 'ctrv', got {model!r}")
        self._ego_traj = ego_traj
        self._ego_log = ego_log
        self._model = model
        self._dt_sub = dt_substep_s
        # track_id → DataFrame sorted by timestamp_ns (set via set_track_history)
        self._history: dict[int, pd.DataFrame] = {}

    def set_track_history(self, history: dict[int, pd.DataFrame]) -> None:
        """Supply per-track history DataFrames for CTRV ω estimation."""
        self._history = history

    # ── public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        track_id: int,
        source_state: np.ndarray,
        source_P: np.ndarray,
        source_ts_ns: int,
        horizons_s: list[float],
    ) -> list[Prediction]:
        """Return one Prediction per horizon."""
        horizons_s = sorted(horizons_s)
        max_h = horizons_s[-1]
        n_steps = math.ceil(max_h / self._dt_sub)
        dt_sub_ns = int(self._dt_sub * 1e9)

        omega = self._estimate_omega(track_id, source_ts_ns)

        state = source_state.copy()
        P = source_P.copy()
        results: list[Prediction] = []

        for step in range(n_steps):
            t_prev = source_ts_ns + step * dt_sub_ns
            t_now  = t_prev + dt_sub_ns
            elapsed_s = (step + 1) * self._dt_sub

            ego1 = self._ego_log.interpolate(t_prev)
            ego2 = self._ego_log.interpolate(t_now)

            if self._model == "cv":
                state, P = predict_state(state, P, ego1, ego2, self._dt_sub)
            else:
                state, P = self._ctrv_step(state, P, ego1, ego2, omega, self._dt_sub)

            # Check if elapsed_s is close enough to a horizon
            for h in horizons_s:
                if abs(elapsed_s - h) < self._dt_sub * 0.5 + 1e-9:
                    target_ts = source_ts_ns + int(h * 1e9)
                    wx, wy = self._to_world(state, target_ts)
                    results.append(Prediction(
                        track_id=track_id,
                        source_ts_ns=source_ts_ns,
                        horizon_s=h,
                        target_ts_ns=target_ts,
                        state=state.copy(),
                        covariance=P.copy(),
                        world_xy=(wx, wy),
                        model=self._model,
                    ))

        return results

    # ── private helpers ───────────────────────────────────────────────────────

    def _to_world(self, state: np.ndarray, target_ts_ns: int) -> tuple[float, float]:
        pt = np.array([[state[_X], state[_Y]]])
        w = self._ego_traj.ego_to_world(pt, target_ts_ns)
        return float(w[0, 0]), float(w[0, 1])

    def _estimate_omega(self, track_id: int, source_ts_ns: int) -> float:
        """Estimate turn rate ω (rad/s) from recent yaw history.

        Uses linear regression over the last ≤5 yaw samples at or before
        source_ts_ns. Falls back to 0 (CV behaviour) when < 3 samples exist.
        """
        if track_id not in self._history:
            return 0.0
        df = self._history[track_id]
        past = df[df["timestamp_ns"] <= source_ts_ns].tail(5)
        if len(past) < 3:
            return 0.0
        times_s = (past["timestamp_ns"].values - past["timestamp_ns"].values[0]) / 1e9
        yaws = np.unwrap(past["yaw"].values)
        if times_s[-1] - times_s[0] < 1e-6:
            return 0.0
        omega, _ = np.polyfit(times_s, yaws, 1)
        return float(omega)

    def _ctrv_step(
        self,
        state: np.ndarray,
        P: np.ndarray,
        ego1: dict[str, float],
        ego2: dict[str, float],
        omega: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One CTRV substep with ego-motion compensation."""
        x   = state[_X]
        y   = state[_Y]
        vx  = state[_VX]
        vy  = state[_VY]
        yaw = state[_YAW]
        v   = math.hypot(vx, vy)

        if abs(omega) > 1e-3:
            x_new = x + (v / omega) * (math.sin(yaw + omega * dt) - math.sin(yaw))
            y_new = y + (v / omega) * (-math.cos(yaw + omega * dt) + math.cos(yaw))
        else:
            x_new = x + v * math.cos(yaw) * dt
            y_new = y + v * math.sin(yaw) * dt

        yaw_new = _wrap_angle(yaw + omega * dt)
        vx_new  = v * math.cos(yaw_new)
        vy_new  = v * math.sin(yaw_new)

        h1 = float(ego1["heading"])
        h2 = float(ego2["heading"])
        R1 = _rot2d(h1)
        R2 = _rot2d(h2)

        vx_e1, vy_e1 = float(ego1["vx"]), float(ego1["vy"])
        vx_e2, vy_e2 = float(ego2["vx"]), float(ego2["vy"])
        delta_e = np.array([
            0.5 * (vx_e1 + vx_e2) * dt,
            0.5 * (vy_e1 + vy_e2) * dt,
        ])

        p_local = np.array([x_new, y_new])
        v_local = np.array([vx_new, vy_new])

        p_world = R1 @ p_local - delta_e
        p_ego2  = R2.T @ p_world

        v_ego2  = R2.T @ R1 @ v_local
        delta_h = h2 - h1
        yaw_ego2 = _wrap_angle(yaw_new - delta_h)

        state_new = np.array([
            p_ego2[0], p_ego2[1],
            v_ego2[0], v_ego2[1],
            yaw_ego2,
            state[_L], state[_W],
        ], dtype=np.float64)

        F = _ctrv_jacobian(x, y, vx, vy, yaw, omega, dt, R1, R2)
        from src.motion_model import _Q_DIAG_PER_SEC
        Q = np.diag(_Q_DIAG_PER_SEC * dt)
        P_new = F @ P @ F.T + Q

        return state_new, P_new


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _ctrv_jacobian(
    x: float, y: float, vx: float, vy: float, yaw: float,
    omega: float, dt: float,
    R1: np.ndarray, R2: np.ndarray,
) -> np.ndarray:
    """Analytical Jacobian of the CTRV + ego-motion-compensation step.

    State vector: [x, y, vx, vy, yaw, L, W]
    """
    v = math.hypot(vx, vy)
    v = max(v, 1e-6)

    dp_dp = np.eye(2)

    dvx_over_v = vx / v
    dvy_over_v = vy / v

    if abs(omega) > 1e-3:
        sin0 = math.sin(yaw)
        sin1 = math.sin(yaw + omega * dt)
        cos0 = math.cos(yaw)
        cos1 = math.cos(yaw + omega * dt)
        dx_dv = (sin1 - sin0) / omega
        dy_dv = (-cos1 + cos0) / omega
        dp_dv = np.array([[dx_dv], [dy_dv]])
        dp_dvx = dp_dv * dvx_over_v
        dp_dvy = dp_dv * dvy_over_v
        dx_dyaw = (v / omega) * (cos1 - cos0)
        dy_dyaw = (v / omega) * (sin1 - sin0)
        dp_dyaw = np.array([[dx_dyaw], [dy_dyaw]])
    else:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dp_dvx = np.array([[cos_yaw * dt * dvx_over_v],
                           [sin_yaw * dt * dvx_over_v]])
        dp_dvy = np.array([[cos_yaw * dt * dvy_over_v],
                           [sin_yaw * dt * dvy_over_v]])
        dp_dyaw = np.array([[-v * math.sin(yaw) * dt],
                             [v * math.cos(yaw) * dt]])

    yaw_new = yaw + omega * dt
    cos_yn, sin_yn = math.cos(yaw_new), math.sin(yaw_new)
    dv_dvx = np.array([[cos_yn * dvx_over_v], [sin_yn * dvx_over_v]])
    dv_dvy = np.array([[cos_yn * dvy_over_v], [sin_yn * dvy_over_v]])
    dv_dyaw = np.array([[-v * sin_yn], [v * cos_yn]])

    J_local = np.zeros((5, 5))
    J_local[0:2, 0:2] = dp_dp
    J_local[0:2, 2:3] = dp_dvx
    J_local[0:2, 3:4] = dp_dvy
    J_local[0:2, 4:5] = dp_dyaw
    J_local[2:4, 2:3] = dv_dvx
    J_local[2:4, 3:4] = dv_dvy
    J_local[2:4, 4:5] = dv_dyaw
    J_local[4, 4] = 1.0

    R2T_R1 = R2.T @ R1
    F = np.eye(STATE_DIM)
    F[np.ix_([_X, _Y], [_X, _Y])]   = R2T_R1 @ J_local[0:2, 0:2]
    F[np.ix_([_X, _Y], [_VX, _VY])] = R2T_R1 @ J_local[0:2, 2:4]
    F[_X, _YAW] = (R2T_R1 @ J_local[0:2, 4:5])[0, 0]
    F[_Y, _YAW] = (R2T_R1 @ J_local[0:2, 4:5])[1, 0]
    F[np.ix_([_VX, _VY], [_VX, _VY])] = R2T_R1 @ J_local[2:4, 2:4]
    F[_VX, _YAW] = (R2T_R1 @ J_local[2:4, 4:5])[0, 0]
    F[_VY, _YAW] = (R2T_R1 @ J_local[2:4, 4:5])[1, 0]
    F[_YAW, _YAW] = 1.0
    return F
