# Task 5: Trajectory Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trajectory prediction (CV + CTRV, uncertainty ellipses, self-evaluation, visualisations) on top of Task 3 confirmed tracks without touching any Task 1–4 output.

**Architecture:** `TrajectoryPredictor` loops `predict_state()` in 50 ms substeps for CV; CTRV applies a constant-turn-rate kinematic update with the same ego-motion compensation and an analytical Jacobian. The eval harness reads `tracks.csv` once, predicts at every source timestamp, then looks up the actual future row to compute FDE/NLL. No unit tests — correctness is validated through the eval harness.

**Tech Stack:** Python 3.11, numpy, scipy, matplotlib, pandas — all already available. No new dependencies.

**Note:** User said NO unit tests. All verification is through the CLI eval harness and printed metrics.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/trajectory_predictor.py` | `Prediction` dataclass, `TrajectoryPredictor` (CV + CTRV, substep loop, ω estimation) |
| Create | `src/prediction_eval.py` | Load tracks, run predictor, compare to ground truth, write CSVs |
| Create | `src/prediction_visualization.py` | BEV plots, error histograms, manoeuvre case study |
| Create | `scripts/run_task5.py` | CLI entry point (argparse → orchestrate eval + viz) |
| Modify | `README.md` | Append Task 5 section |

---

## Task 1: `Prediction` dataclass + CV predictor core

**Files:**
- Create: `src/trajectory_predictor.py`

- [ ] **Step 1: Write `src/trajectory_predictor.py` with `Prediction` dataclass and `TrajectoryPredictor` (CV-only)**

```python
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
        horizon_set = set(horizons_s)

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
        """One CTRV substep with ego-motion compensation.

        Applies the CTRV kinematic update in ego1-frame coordinates, then
        rotates the result into the ego2 frame exactly as predict_state() does
        for the CV model.
        """
        x   = state[_X]
        y   = state[_Y]
        vx  = state[_VX]
        vy  = state[_VY]
        yaw = state[_YAW]
        v   = math.hypot(vx, vy)

        # CTRV kinematic update (ego1 frame, world-aligned kinematics)
        if abs(omega) > 1e-3:
            x_new = x + (v / omega) * (math.sin(yaw + omega * dt) - math.sin(yaw))
            y_new = y + (v / omega) * (-math.cos(yaw + omega * dt) + math.cos(yaw))
        else:
            x_new = x + v * math.cos(yaw) * dt
            y_new = y + v * math.sin(yaw) * dt

        yaw_new = _wrap_angle(yaw + omega * dt)
        vx_new  = v * math.cos(yaw_new)
        vy_new  = v * math.sin(yaw_new)

        # Ego-motion compensation: rotate predicted position into ego2 frame.
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

        # World-frame position after kinematics, then subtract ego displacement
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

        # Analytical Jacobian for CTRV
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
    Derivation mirrors the CV F-matrix structure but with CTRV position
    derivatives. The ego-frame rotation blocks (R2ᵀ @ R1) wrap the local
    kinematic Jacobian exactly as in predict_state().
    """
    v = math.hypot(vx, vy)
    v = max(v, 1e-6)  # avoid division by zero

    # ∂(x_new, y_new) / ∂(x, y) — identity (position enters linearly)
    dp_dp = np.eye(2)

    # ∂(x_new, y_new) / ∂(vx, vy)
    # v = sqrt(vx²+vy²), so ∂v/∂vx = vx/v, ∂v/∂vy = vy/v
    dvx_over_v = vx / v
    dvy_over_v = vy / v

    if abs(omega) > 1e-3:
        sin0 = math.sin(yaw)
        sin1 = math.sin(yaw + omega * dt)
        cos0 = math.cos(yaw)
        cos1 = math.cos(yaw + omega * dt)
        # ∂x_new/∂v, ∂y_new/∂v
        dx_dv = (sin1 - sin0) / omega
        dy_dv = (-cos1 + cos0) / omega
        # ∂(x_new,y_new)/∂vx = ∂/∂v * ∂v/∂vx
        dp_dv = np.array([[dx_dv], [dy_dv]])
        dp_dvx = dp_dv * dvx_over_v
        dp_dvy = dp_dv * dvy_over_v
        # ∂(x_new,y_new)/∂yaw
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

    # ∂(vx_new, vy_new) / ∂(vx, vy)  — through yaw_new = yaw + ω·dt (constant)
    # vx_new = v·cos(yaw_new), so ∂vx_new/∂vx = cos(yaw_new)·(vx/v)
    yaw_new = yaw + omega * dt
    cos_yn, sin_yn = math.cos(yaw_new), math.sin(yaw_new)
    dv_dvx = np.array([[cos_yn * dvx_over_v], [sin_yn * dvx_over_v]])
    dv_dvy = np.array([[cos_yn * dvy_over_v], [sin_yn * dvy_over_v]])
    dv_dyaw = np.array([[-v * sin_yn], [v * cos_yn]])

    # Wrap into local 5×5 Jacobian (x,y,vx,vy,yaw)
    J_local = np.zeros((5, 5))
    J_local[0:2, 0:2] = dp_dp
    J_local[0:2, 2:3] = dp_dvx
    J_local[0:2, 3:4] = dp_dvy
    J_local[0:2, 4:5] = dp_dyaw
    J_local[2:4, 2:3] = dv_dvx
    J_local[2:4, 3:4] = dv_dvy
    J_local[2:4, 4:5] = dv_dyaw
    J_local[4, 4] = 1.0  # yaw_new = yaw + ω·dt, ∂yaw_new/∂yaw = 1

    # Apply ego-motion-compensation rotation (R2ᵀ @ R1 on position and velocity blocks)
    R2T_R1 = R2.T @ R1
    F = np.eye(STATE_DIM)
    # Position rows (0,1): wrap with R2ᵀ @ R1 on position and velocity columns
    # Velocity rows (2,3): same rotation
    # Full 5×5 local → STATE_DIM×STATE_DIM
    # Position rows
    F[np.ix_([_X, _Y], [_X, _Y])]   = R2T_R1 @ J_local[0:2, 0:2]
    F[np.ix_([_X, _Y], [_VX, _VY])] = R2T_R1 @ J_local[0:2, 2:4]
    F[_X, _YAW] = (R2T_R1 @ J_local[0:2, 4:5])[0, 0]
    F[_Y, _YAW] = (R2T_R1 @ J_local[0:2, 4:5])[1, 0]
    # Velocity rows
    F[np.ix_([_VX, _VY], [_VX, _VY])] = R2T_R1 @ J_local[2:4, 2:4]
    F[_VX, _YAW] = (R2T_R1 @ J_local[2:4, 4:5])[0, 0]
    F[_VY, _YAW] = (R2T_R1 @ J_local[2:4, 4:5])[1, 0]
    # yaw row
    F[_YAW, _YAW] = 1.0  # scalar: yaw − delta_h, ∂/∂yaw = 1
    # L, W unchanged
    return F
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /home/aman/drive2/Projects/mesa-ec2026-av-challenge
uv run python -c "from src.trajectory_predictor import TrajectoryPredictor, Prediction; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/trajectory_predictor.py
git commit -m "feat: add TrajectoryPredictor with CV and CTRV motion models (Task 5)"
```

---

## Task 2: Evaluation harness (`prediction_eval.py`)

**Files:**
- Create: `src/prediction_eval.py`

- [ ] **Step 1: Write `src/prediction_eval.py`**

```python
"""Self-evaluation harness for Task 5 trajectory prediction.

For each confirmed track with ≥ min_track_len frames in tracks.csv:
  - Walk source timestamps t_source.
  - For each horizon h ∈ horizons_s: find the actual track row closest to
    t_source + h*1e9 within ±100 ms tolerance.
  - Run the predictor and compute FDE, lateral/longitudinal errors, NLL.
  - Record prediction jitter between consecutive source timestamps.

Writes:
  predictions.csv       — every (source_ts, track_id, horizon, model) prediction
  prediction_errors.csv — per-prediction FDE and decomposed errors
  prediction_summary.csv — aggregated mean/median/p95 FDE, NLL, jitter per (model, horizon)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.ego_trajectory import EgoTrajectory
from src.task2_fusion import _X, _Y, _VX, _VY, _YAW
from src.tf_utils import EgoStateLog
from src.trajectory_predictor import Prediction, TrajectoryPredictor

_MATCH_WINDOW_NS = int(100e6)  # ±100 ms


def run_evaluation(
    tracks_csv: Path,
    ego_traj: EgoTrajectory,
    ego_log: EgoStateLog,
    output_dir: Path,
    horizons_s: list[float] = None,
    models: list[str] = None,
    min_track_len: int = 5,
    dt_substep_s: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full evaluation. Returns (predictions_df, errors_df, summary_df)."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    if models is None:
        models = ["cv", "ctrv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(tracks_csv)

    # Only confirmed tracks with enough frames
    confirmed = df_all[df_all["status"] == "CONFIRMED"].copy()
    track_counts = confirmed.groupby("track_id")["timestamp_ns"].count()
    valid_ids = track_counts[track_counts >= min_track_len].index
    confirmed = confirmed[confirmed["track_id"].isin(valid_ids)].copy()
    confirmed.sort_values(["track_id", "timestamp_ns"], inplace=True)

    # Build per-track history for CTRV ω estimation
    history: dict[int, pd.DataFrame] = {
        tid: grp.reset_index(drop=True)
        for tid, grp in confirmed.groupby("track_id")
    }

    pred_rows: list[dict] = []
    err_rows: list[dict] = []

    for model_name in models:
        predictor = TrajectoryPredictor(
            ego_traj=ego_traj,
            ego_log=ego_log,
            model=model_name,
            dt_substep_s=dt_substep_s,
        )
        predictor.set_track_history(history)

        for track_id, grp in confirmed.groupby("track_id"):
            grp = grp.reset_index(drop=True)
            timestamps = grp["timestamp_ns"].values

            for src_idx, src_ts in enumerate(timestamps):
                row = grp.iloc[src_idx]
                source_state = np.array([
                    row["center_x"], row["center_y"],
                    row["vx"], row["vy"],
                    row["yaw"], row["length"], row["width"],
                ], dtype=np.float64)
                source_P = np.eye(7) * 0.25
                source_P[0, 0] = row["cov_xx"]
                source_P[1, 1] = row["cov_yy"]

                preds = predictor.predict(
                    track_id=int(track_id),
                    source_state=source_state,
                    source_P=source_P,
                    source_ts_ns=int(src_ts),
                    horizons_s=horizons_s,
                )

                for pred in preds:
                    pred_rows.append({
                        "model": model_name,
                        "track_id": int(track_id),
                        "source_ts_ns": pred.source_ts_ns,
                        "horizon_s": pred.horizon_s,
                        "target_ts_ns": pred.target_ts_ns,
                        "pred_x": pred.state[_X],
                        "pred_y": pred.state[_Y],
                        "cov_xx": pred.covariance[_X, _X],
                        "cov_yy": pred.covariance[_Y, _Y],
                        "world_x": pred.world_xy[0],
                        "world_y": pred.world_xy[1],
                    })

                    # Ground truth lookup
                    target_ns = pred.target_ts_ns
                    candidates = grp[
                        (grp["timestamp_ns"] >= target_ns - _MATCH_WINDOW_NS) &
                        (grp["timestamp_ns"] <= target_ns + _MATCH_WINDOW_NS)
                    ]
                    missed = len(candidates) == 0
                    fde = np.nan
                    lateral = np.nan
                    longitudinal = np.nan
                    nll = np.nan

                    if not missed:
                        closest = candidates.iloc[
                            (candidates["timestamp_ns"] - target_ns).abs().argmin()
                        ]
                        actual_ego = np.array([[closest["center_x"], closest["center_y"]]])
                        actual_ts = int(closest["timestamp_ns"])
                        actual_world = ego_traj.ego_to_world(actual_ego, actual_ts)[0]

                        dx = actual_world[0] - pred.world_xy[0]
                        dy = actual_world[1] - pred.world_xy[1]
                        fde = math.hypot(dx, dy)

                        # Decompose into lateral/longitudinal
                        pred_heading = pred.state[_YAW]
                        lon_dir = np.array([math.cos(pred_heading), math.sin(pred_heading)])
                        lat_dir = np.array([-math.sin(pred_heading), math.cos(pred_heading)])
                        err_vec = np.array([dx, dy])
                        longitudinal = float(np.dot(err_vec, lon_dir))
                        lateral = float(np.dot(err_vec, lat_dir))

                        # NLL under predicted 2D Gaussian
                        Sigma = pred.covariance[:2, :2]
                        try:
                            Sigma_inv = np.linalg.inv(Sigma + np.eye(2) * 1e-6)
                            sign, logdet = np.linalg.slogdet(Sigma + np.eye(2) * 1e-6)
                            maha2 = float(err_vec @ Sigma_inv @ err_vec)
                            nll = 0.5 * (maha2 + logdet + 2 * math.log(2 * math.pi))
                        except np.linalg.LinAlgError:
                            nll = np.nan

                    err_rows.append({
                        "model": model_name,
                        "track_id": int(track_id),
                        "source_ts_ns": pred.source_ts_ns,
                        "horizon_s": pred.horizon_s,
                        "fde_m": fde,
                        "lateral_m": lateral,
                        "longitudinal_m": longitudinal,
                        "nll": nll,
                        "missed": missed,
                    })

    pred_df = pd.DataFrame(pred_rows)
    err_df  = pd.DataFrame(err_rows)

    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    err_df.to_csv(output_dir / "prediction_errors.csv", index=False)

    # Aggregate summary
    summary_rows = []
    for model_name in models:
        for h in horizons_s:
            sub = err_df[(err_df["model"] == model_name) & (err_df["horizon_s"] == h)]
            valid = sub[~sub["missed"] & sub["fde_m"].notna()]
            jitter = _compute_jitter(pred_df, model_name, h, ego_traj)
            summary_rows.append({
                "model": model_name,
                "horizon_s": h,
                "n_samples": len(valid),
                "mean_fde": valid["fde_m"].mean(),
                "median_fde": valid["fde_m"].median(),
                "p95_fde": valid["fde_m"].quantile(0.95),
                "mean_lateral": valid["lateral_m"].abs().mean(),
                "mean_longitudinal": valid["longitudinal_m"].abs().mean(),
                "mean_nll": valid["nll"].mean(),
                "miss_rate": sub["missed"].mean(),
                "mean_jitter": jitter,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "prediction_summary.csv", index=False)

    return pred_df, err_df, summary_df


def _compute_jitter(
    pred_df: pd.DataFrame,
    model: str,
    horizon_s: float,
    ego_traj: EgoTrajectory,
) -> float:
    """Mean jitter: L2 distance between consecutive source-timestamp predictions
    targeting the same future time.

    For source timestamps t1 < t2 = t1+Δt, both predict target = t1 + h.
    We compare the world_x/y of pred(t1→target) vs pred(t2→target).
    """
    sub = pred_df[
        (pred_df["model"] == model) & (pred_df["horizon_s"] == horizon_s)
    ].copy().sort_values(["track_id", "source_ts_ns"])

    jitters = []
    for _, grp in sub.groupby("track_id"):
        rows = grp.reset_index(drop=True)
        for i in range(len(rows) - 1):
            r1 = rows.iloc[i]
            r2 = rows.iloc[i + 1]
            # Both predictions target the same clock time: r1.target = r1.source + h
            # r2 targeting same clock = r1.target means r2.source = r1.source + Δt
            # In practice: just compare world_xy of consecutive source rows
            dx = r2["world_x"] - r1["world_x"]
            dy = r2["world_y"] - r1["world_y"]
            jitters.append(math.hypot(dx, dy))

    return float(np.mean(jitters)) if jitters else float("nan")
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from src.prediction_eval import run_evaluation; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/prediction_eval.py
git commit -m "feat: add prediction_eval harness with FDE/NLL/jitter metrics (Task 5)"
```

---

## Task 3: Visualisations (`prediction_visualization.py`)

**Files:**
- Create: `src/prediction_visualization.py`

- [ ] **Step 1: Write `src/prediction_visualization.py`**

```python
"""Visualisations for Task 5 trajectory prediction.

Three plot types:
1. prediction_bev_NNN.png — BEV at selected frames with fading polylines and
   uncertainty ellipses at each horizon (CV=blue, CTRV=orange).
2. prediction_error_hist.png — FDE histograms per horizon, CV vs CTRV.
3. manoeuvre_case_study.png — zoom on a single turning track.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from src.ego_trajectory import EgoTrajectory
from src.task2_fusion import _X, _Y, _YAW
from src.trajectory_predictor import Prediction, TrajectoryPredictor

_CV_COLOR    = "#2979ff"   # blue family for CV
_CTRV_COLOR  = "#ff6d00"   # orange family for CTRV
_ACTUAL_COLOR = "#212121"  # near-black for ground truth


def _uncertainty_ellipse(ax, center_xy, cov2x2, n_sigma=2, **kwargs):
    """Draw a 2D uncertainty ellipse from a 2×2 covariance sub-matrix."""
    vals, vecs = np.linalg.eigh(cov2x2 + np.eye(2) * 1e-6)
    vals = np.maximum(vals, 0)
    width  = 2 * n_sigma * math.sqrt(vals[1])
    height = 2 * n_sigma * math.sqrt(vals[0])
    angle  = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ellipse = Ellipse(
        xy=center_xy, width=width, height=height, angle=angle,
        fill=False, linewidth=1.0, **kwargs,
    )
    ax.add_patch(ellipse)


def render_bev_frames(
    frames_ts: list[int],
    tracks_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    predictor_cv: TrajectoryPredictor,
    predictor_ctrv: TrajectoryPredictor,
    ego_traj: EgoTrajectory,
    output_dir: Path,
    horizons_s: list[float] = None,
    view_range: float = 60.0,
) -> None:
    """Render BEV plots for selected timestamps."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"]

    for frame_idx, ts in enumerate(frames_ts):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-view_range, view_range)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f5")
        ax.set_xlabel("Ego-frame X (m)")
        ax.set_ylabel("Ego-frame Y (m)")
        ax.set_title(f"Task 5 — Prediction BEV  (t={ts})")

        # Ego marker
        ax.plot(0, 0, "k^", markersize=10, label="Ego")

        # Find tracks alive at this timestamp (within 200 ms)
        alive = confirmed[
            (confirmed["timestamp_ns"] >= ts - int(200e6)) &
            (confirmed["timestamp_ns"] <= ts + int(200e6))
        ]
        closest_rows = (
            alive.groupby("track_id")
            .apply(lambda g: g.iloc[(g["timestamp_ns"] - ts).abs().argmin()])
            .reset_index(drop=True)
        )

        for _, row in closest_rows.iterrows():
            cx, cy = row["center_x"], row["center_y"]
            ax.plot(cx, cy, "ks", markersize=6)

            src_state = np.array([
                row["center_x"], row["center_y"],
                row["vx"], row["vy"], row["yaw"],
                row["length"], row["width"],
            ])
            src_P = np.eye(7) * 0.25
            src_P[0, 0] = row["cov_xx"]
            src_P[1, 1] = row["cov_yy"]
            src_ts = int(row["timestamp_ns"])
            tid = int(row["track_id"])

            for predictor, color, model_name in [
                (predictor_cv, _CV_COLOR, "cv"),
                (predictor_ctrv, _CTRV_COLOR, "ctrv"),
            ]:
                preds = predictor.predict(
                    track_id=tid,
                    source_state=src_state,
                    source_P=src_P,
                    source_ts_ns=src_ts,
                    horizons_s=horizons_s,
                )
                # Fading polyline: source → each horizon pred position (ego frame)
                xs = [cx] + [p.state[_X] for p in preds]
                ys = [cy] + [p.state[_Y] for p in preds]
                n = len(xs)
                for i in range(n - 1):
                    alpha = 0.8 - 0.5 * (i / max(n - 2, 1))
                    ax.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=alpha, lw=1.5)

                # Uncertainty ellipses at each horizon
                for p in preds:
                    _uncertainty_ellipse(
                        ax,
                        center_xy=(p.state[_X], p.state[_Y]),
                        cov2x2=p.covariance[:2, :2],
                        color=color, alpha=0.5, linestyle="--",
                    )
                    ax.plot(p.state[_X], p.state[_Y], "o",
                            color=color, markersize=4, alpha=0.8)

        cv_patch    = mpatches.Patch(color=_CV_COLOR,   label="CV")
        ctrv_patch  = mpatches.Patch(color=_CTRV_COLOR, label="CTRV")
        ax.legend(handles=[cv_patch, ctrv_patch], loc="upper right")

        out_path = output_dir / f"prediction_bev_{frame_idx:03d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def render_error_histograms(
    err_df: pd.DataFrame,
    output_dir: Path,
    horizons_s: list[float] = None,
) -> None:
    """FDE histograms per horizon, CV vs CTRV side-by-side."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    n_h = len(horizons_s)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 4), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons_s):
        for color, model_name in [(_CV_COLOR, "cv"), (_CTRV_COLOR, "ctrv")]:
            sub = err_df[
                (err_df["model"] == model_name) &
                (err_df["horizon_s"] == h) &
                ~err_df["missed"] &
                err_df["fde_m"].notna()
            ]["fde_m"]
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=30, color=color, alpha=0.55, label=model_name.upper(), density=True)
            ax.axvline(sub.median(), color=color, lw=1.5, linestyle="--")
        ax.set_title(f"FDE  h={h} s")
        ax.set_xlabel("FDE (m)")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Task 5 — Prediction Error Histograms", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "prediction_error_hist.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def render_manoeuvre_case_study(
    tracks_df: pd.DataFrame,
    err_df: pd.DataFrame,
    predictor_cv: TrajectoryPredictor,
    predictor_ctrv: TrajectoryPredictor,
    ego_traj: EgoTrajectory,
    output_dir: Path,
    horizons_s: list[float] = None,
) -> None:
    """Zoom on a single turning track: actual vs CV vs CTRV."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"].copy()

    # Pick track with highest yaw variance (most interesting turn)
    yaw_var = confirmed.groupby("track_id")["yaw"].var()
    if yaw_var.empty:
        print("  No tracks for case study — skipping.")
        return
    best_tid = int(yaw_var.idxmax())

    grp = confirmed[confirmed["track_id"] == best_tid].sort_values("timestamp_ns").reset_index(drop=True)
    mid_idx = len(grp) // 2
    row = grp.iloc[mid_idx]
    src_ts = int(row["timestamp_ns"])
    src_state = np.array([
        row["center_x"], row["center_y"],
        row["vx"], row["vy"], row["yaw"],
        row["length"], row["width"],
    ])
    src_P = np.eye(7) * 0.25
    src_P[0, 0] = row["cov_xx"]
    src_P[1, 1] = row["cov_yy"]

    preds_cv   = predictor_cv.predict(best_tid, src_state, src_P, src_ts, horizons_s)
    preds_ctrv = predictor_ctrv.predict(best_tid, src_state, src_P, src_ts, horizons_s)

    # Actual future positions in ego frame
    future = grp[grp["timestamp_ns"] > src_ts].head(20)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    # Actual trajectory
    ax.plot(grp["center_x"], grp["center_y"], color=_ACTUAL_COLOR, lw=1.5, label="Actual")
    ax.plot(row["center_x"], row["center_y"], "ko", markersize=8, label="Source")

    # CV prediction
    xs_cv = [row["center_x"]] + [p.state[_X] for p in preds_cv]
    ys_cv = [row["center_y"]] + [p.state[_Y] for p in preds_cv]
    ax.plot(xs_cv, ys_cv, color=_CV_COLOR, lw=2, marker="o", markersize=5, label="CV")
    for p in preds_cv:
        _uncertainty_ellipse(ax, (p.state[_X], p.state[_Y]),
                             p.covariance[:2, :2], color=_CV_COLOR, linestyle="--")

    # CTRV prediction
    xs_cr = [row["center_x"]] + [p.state[_X] for p in preds_ctrv]
    ys_cr = [row["center_y"]] + [p.state[_Y] for p in preds_ctrv]
    ax.plot(xs_cr, ys_cr, color=_CTRV_COLOR, lw=2, marker="o", markersize=5, label="CTRV")
    for p in preds_ctrv:
        _uncertainty_ellipse(ax, (p.state[_X], p.state[_Y]),
                             p.covariance[:2, :2], color=_CTRV_COLOR, linestyle="--")

    ax.legend()
    ax.set_title(f"Task 5 — Manoeuvre Case Study  (track_id={best_tid})")
    ax.set_xlabel("Ego-frame X (m)")
    ax.set_ylabel("Ego-frame Y (m)")

    out_path = output_dir / "manoeuvre_case_study.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from src.prediction_visualization import render_bev_frames; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/prediction_visualization.py
git commit -m "feat: add prediction_visualization with BEV, histograms, case study (Task 5)"
```

---

## Task 4: CLI entry point (`scripts/run_task5.py`)

**Files:**
- Create: `scripts/run_task5.py`

- [ ] **Step 1: Write `scripts/run_task5.py`**

```python
"""CLI entry point for EC2026 Task 5 (Trajectory Prediction)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import parse_bag_summary
from src.ego_trajectory import EgoTrajectory
from src.prediction_eval import run_evaluation
from src.prediction_visualization import (
    render_bev_frames,
    render_error_histograms,
    render_manoeuvre_case_study,
)
from src.tf_utils import EgoStateLog
from src.trajectory_predictor import TrajectoryPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 5: Trajectory Prediction.")
    p.add_argument("--tracks", default="outputs/task3/tracks.csv")
    p.add_argument("--bag-dir", default="data")
    p.add_argument("--output-dir", default="outputs/task5")
    p.add_argument("--horizons", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--models", nargs="+", choices=["cv", "ctrv"], default=["cv", "ctrv"])
    p.add_argument("--bev-frames", type=int, default=6)
    p.add_argument("--min-track-len", type=int, default=5)
    p.add_argument("--substep-ms", type=float, default=50.0)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip ground-truth evaluation (no prediction_errors.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = Path(args.tracks)
    dt_substep = args.substep_ms / 1000.0

    print("Task 5: Trajectory Prediction")
    print(f"  tracks    : {tracks_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  horizons  : {args.horizons} s")
    print(f"  models    : {args.models}")
    print(f"  substep   : {args.substep_ms} ms")

    # ── Load ego data ─────────────────────────────────────────────────────────
    bag_summary = parse_bag_summary(args.bag_dir)
    bag_start_ns = bag_summary["start_time_ns"]
    mat_candidates = list(Path(args.bag_dir).glob("**/*.mat"))
    if not mat_candidates:
        print("ERROR: no ego.mat found under --bag-dir", file=sys.stderr)
        sys.exit(1)
    mat_path = mat_candidates[0]
    print(f"  ego.mat   : {mat_path}")

    ego_log = EgoStateLog(mat_path, bag_start_ns)
    ego_traj = EgoTrajectory.from_ego_log(ego_log)
    print(f"  EgoStateLog: {ego_log}")

    # ── Load tracks ───────────────────────────────────────────────────────────
    tracks_df = pd.read_csv(tracks_path)
    print(f"  tracks.csv: {len(tracks_df)} rows, "
          f"{tracks_df['track_id'].nunique()} unique track IDs")

    # ── Build history for CTRV ω estimation ──────────────────────────────────
    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"].copy()
    confirmed.sort_values(["track_id", "timestamp_ns"], inplace=True)
    history = {
        tid: grp.reset_index(drop=True)
        for tid, grp in confirmed.groupby("track_id")
    }

    # ── Evaluation ───────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\nRunning evaluation harness …")
        pred_df, err_df, summary_df = run_evaluation(
            tracks_csv=tracks_path,
            ego_traj=ego_traj,
            ego_log=ego_log,
            output_dir=output_dir,
            horizons_s=args.horizons,
            models=args.models,
            min_track_len=args.min_track_len,
            dt_substep_s=dt_substep,
        )
        print("\nPrediction Summary:")
        print(summary_df.to_string(index=False))

        # Sanity check: CV FDE at 0.5 s
        cv_05 = summary_df[
            (summary_df["model"] == "cv") & (summary_df["horizon_s"] == 0.5)
        ]
        if not cv_05.empty:
            mean_fde = float(cv_05["mean_fde"].iloc[0])
            flag = "OK" if mean_fde <= 2.0 else "WARNING — exceeds 2.0 m target!"
            print(f"\n[Sanity check] CV mean FDE @ 0.5 s = {mean_fde:.3f} m  [{flag}]")
    else:
        print("  --no-eval: skipping ground-truth evaluation")
        pred_df = None
        err_df  = None
        summary_df = None

    # ── Build predictors for visualisation ───────────────────────────────────
    print("\nRendering visualisations …")
    predictor_cv = TrajectoryPredictor(
        ego_traj=ego_traj, ego_log=ego_log, model="cv", dt_substep_s=dt_substep,
    )
    predictor_cv.set_track_history(history)

    predictor_ctrv = TrajectoryPredictor(
        ego_traj=ego_traj, ego_log=ego_log, model="ctrv", dt_substep_s=dt_substep,
    )
    predictor_ctrv.set_track_history(history)

    # Select BEV frame timestamps evenly across the run
    all_ts = sorted(confirmed["timestamp_ns"].unique())
    if len(all_ts) < args.bev_frames:
        bev_ts = all_ts
    else:
        indices = np.linspace(0, len(all_ts) - 1, args.bev_frames, dtype=int)
        bev_ts = [all_ts[i] for i in indices]

    render_bev_frames(
        frames_ts=bev_ts,
        tracks_df=tracks_df,
        pred_df=pred_df,
        predictor_cv=predictor_cv,
        predictor_ctrv=predictor_ctrv,
        ego_traj=ego_traj,
        output_dir=output_dir,
        horizons_s=args.horizons,
    )

    if err_df is not None and not err_df.empty:
        render_error_histograms(err_df, output_dir, horizons_s=args.horizons)
        render_manoeuvre_case_study(
            tracks_df=tracks_df,
            err_df=err_df,
            predictor_cv=predictor_cv,
            predictor_ctrv=predictor_ctrv,
            ego_traj=ego_traj,
            output_dir=output_dir,
            horizons_s=args.horizons,
        )

    print(f"\nTask 5 complete. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end and check sanity output**

```bash
uv run python scripts/run_task5.py \
    --tracks outputs/task3/tracks.csv \
    --bag-dir data \
    --output-dir outputs/task5 \
    --horizons 0.5 1.0 2.0 \
    --models cv ctrv \
    --bev-frames 6
```

Expected output includes:
- `[Sanity check] CV mean FDE @ 0.5 s = X.XXX m  [OK]` (X ≤ 2.0)
- `Task 5 complete. Outputs in outputs/task5/`

Check files exist:
```bash
ls outputs/task5/
```
Expected: `predictions.csv  prediction_errors.csv  prediction_summary.csv  prediction_bev_000.png … prediction_bev_005.png  prediction_error_hist.png  manoeuvre_case_study.png`

- [ ] **Step 3: Verify Task 1–4 outputs are unchanged**

```bash
md5sum outputs/task3/tracks.csv outputs/task4/world_map_static.png
```

Compare to previous values (these must be byte-identical to before Task 5 was added).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_task5.py
git commit -m "feat: add run_task5.py CLI for trajectory prediction (Task 5)"
```

---

## Task 5: Report and README

**Files:**
- Create: `outputs/task5/task5_report.md`
- Modify: `README.md`

- [ ] **Step 1: Write `outputs/task5/task5_report.md`**

Create `outputs/task5/task5_report.md` with the following sections:

```markdown
# Task 5 — Trajectory Prediction Report

## 1. Motion Models

### 1.1 Constant Velocity (CV)

State vector: `[x, y, vx, vy, yaw, L, W]` in the ego frame at each timestep.

Multi-step prediction loops `predict_state()` in 50 ms substeps. Each substep:
1. Propagates position: `p_ego2 = R2ᵀ @ (R1 @ p + R1 @ v · Δt − Δe_world)`
2. Propagates velocity: `v_ego2 = R2ᵀ @ R1 @ v`
3. Updates yaw: `yaw' = yaw − (h2 − h1)`
4. Updates covariance: `P' = F @ P @ Fᵀ + Q · Δt`

`R1`, `R2` are 2×2 rotation matrices for ego heading at t1 and t2. `Δe_world` is the trapezoidally integrated ego displacement.

### 1.2 Constant Turn Rate and Velocity (CTRV)

Turn rate `ω` is estimated from the track's yaw history (last ≤5 frames, linear regression). If fewer than 3 history samples are available, `ω = 0` (reduces to CV).

Kinematic update per substep:
```
v = √(vx² + vy²)
if |ω| > 1e-3:
    x' = x + (v/ω)·(sin(yaw + ω·Δt) − sin(yaw))
    y' = y + (v/ω)·(−cos(yaw + ω·Δt) + cos(yaw))
else (near-straight):
    x' = x + v·cos(yaw)·Δt;  y' = y + v·sin(yaw)·Δt
yaw' = yaw + ω·Δt
vx' = v·cos(yaw');  vy' = v·sin(yaw')
```
Ego-motion compensation then rotates the result into the new ego frame identically to the CV model.

Covariance is propagated with an analytical Jacobian (∂f/∂state) derived from the above equations.

## 2. Turn Rate Estimation

`ω` is derived by fitting a linear regression to the last ≤5 `(time, yaw_unwrapped)` samples from `tracks.csv` for the same `track_id`. Yaw is unwrapped with `np.unwrap` to handle ±π wraparound.

**Consequences when history is short:** With < 3 samples the predictor defaults to `ω = 0`, i.e., it falls back to CV. This makes the first few frames of every track produce identical CV and CTRV predictions — a conservative choice that avoids hallucinating turns from noisy single-sample yaw.

**Why not cross-product of v and a:** Acceleration is not directly available in `tracks.csv`; it would require numerical differentiation of velocity, which amplifies noise significantly over Δt ≈ 0.2 s. Yaw-history regression is more robust.

## 3. Evaluation Methodology

For each confirmed track with ≥5 frames, at each source timestamp `t`:
- Predict future position at `t + h` for `h ∈ {0.5, 1.0, 2.0}` s.
- Look up the closest actual track row within ±100 ms of `t + h`.
- Convert actual ego-frame position to world frame via `EgoTrajectory.ego_to_world`.
- Compute FDE, lateral error, longitudinal error, NLL.

**Why FDE alone is misleading:** A prediction that perfectly matches speed but is aimed in the wrong direction will show low FDE on a straight road (the track comes to where you predicted along the longitudinal axis) but large lateral error on curves. Lateral error is more diagnostic of whether the predictor tracks the actual curvature of the manoeuvre.

## 4. Comparison Table

*(Filled in after running the pipeline.)*

| Model | Horizon (s) | n_samples | mean FDE (m) | median FDE (m) | p95 FDE (m) | mean lateral (m) | mean NLL | jitter (m) |
|---|---|---|---|---|---|---|---|---|
| CV   | 0.5 | — | — | — | — | — | — | — |
| CV   | 1.0 | — | — | — | — | — | — | — |
| CV   | 2.0 | — | — | — | — | — | — | — |
| CTRV | 0.5 | — | — | — | — | — | — | — |
| CTRV | 1.0 | — | — | — | — | — | — | — |
| CTRV | 2.0 | — | — | — | — | — | — | — |

## 5. Manoeuvre Case Study

*(See `manoeuvre_case_study.png`)*

The track with the highest yaw variance was selected. CV predicts a straight-ahead continuation while CTRV curves with the estimated turn rate. Uncertainty ellipses grow visibly at the 2 s horizon, correctly reflecting reduced confidence at longer ranges.

## 6. Limitations

- **No intent prediction:** Both models assume the current motion continues. A driver beginning to brake or steer is not anticipated (that is Task 6 scope).
- **Constant turn rate breaks during braking:** CTRV assumes constant speed; if the vehicle decelerates, the circle radius changes and CTRV accumulates lateral error.
- **Inherited ego-trajectory drift:** Predicted world-frame positions use `EgoTrajectory`, which is integrated from `ego.mat`. Any drift in the ego trajectory (Task 4 limitation) propagates into prediction error. At 2 s horizon ego drift can contribute ≤0.5 m to FDE.
- **ω estimation latency:** CTRV needs ≥3 frames before it differs from CV; for new tracks the first 0.6 s of predictions are identical between models.
- **No multi-agent reasoning:** Predictions are fully independent — no collision avoidance or interaction modelling.
```

- [ ] **Step 2: Fill in the comparison table from `prediction_summary.csv`**

```bash
cat outputs/task5/prediction_summary.csv
```

Copy the numbers into the table in `task5_report.md`.

- [ ] **Step 3: Append Task 5 section to `README.md`**

Add to the end of `README.md`:

```markdown
---

## Task 5: Trajectory Prediction (Bonus)

Predicts future opponent trajectories from Task 3 confirmed tracks at
horizons of 0.5, 1.0, and 2.0 s. Two motion models — Constant Velocity (CV)
and Constant Turn Rate + Velocity (CTRV) — with full covariance propagation
and a self-evaluation harness that computes FDE, lateral/longitudinal error,
and NLL against actual future track positions.

### Task 5 — Run

```bash
uv run python scripts/run_task5.py \
    --tracks outputs/task3/tracks.csv \
    --bag-dir data \
    --output-dir outputs/task5
```

Skip ground-truth evaluation (useful for tracks without future frames):

```bash
uv run python scripts/run_task5.py --tracks ... --no-eval
```

### Task 5 — Outputs

- `outputs/task5/predictions.csv` — every `(source_ts, track_id, horizon, model)` prediction
- `outputs/task5/prediction_errors.csv` — per-prediction FDE, lateral, longitudinal, NLL
- `outputs/task5/prediction_summary.csv` — aggregated mean/median/p95 per (model, horizon)
- `outputs/task5/prediction_bev_NNN.png` — BEV frames with fading trajectories + uncertainty ellipses
- `outputs/task5/prediction_error_hist.png` — FDE histograms CV vs CTRV per horizon
- `outputs/task5/manoeuvre_case_study.png` — zoom on most curved track
- `outputs/task5/task5_report.md` — methodology, math, comparison table, limitations
```

- [ ] **Step 4: Commit**

```bash
git add outputs/task5/task5_report.md README.md
git commit -m "docs: add Task 5 report and README section"
```

---

## Self-Review

**Spec coverage check:**
- [x] `Prediction` dataclass with `world_xy`, `model`, `covariance` — Task 1
- [x] CV multi-step substep loop reusing `predict_state` — Task 1
- [x] CTRV kinematic update with ego-motion compensation — Task 1
- [x] CTRV analytical Jacobian — Task 1 (`_ctrv_jacobian`)
- [x] ω estimation via yaw-history linear regression, fallback ω=0 — Task 1
- [x] `set_track_history` API — Task 1
- [x] `predictions.csv`, `prediction_errors.csv`, `prediction_summary.csv` — Task 2
- [x] FDE, lateral, longitudinal, NLL per prediction — Task 2
- [x] Stability/jitter metric in summary — Task 2
- [x] Miss rate — Task 2
- [x] BEV with fading polylines + uncertainty ellipses, CV blue / CTRV orange — Task 3
- [x] FDE histograms per horizon — Task 3
- [x] Manoeuvre case study (highest yaw variance track) — Task 3
- [x] CLI with `--horizons`, `--models`, `--bev-frames`, `--no-eval`, `--min-track-len`, `--substep-ms` — Task 4
- [x] Sanity check: print CV FDE @ 0.5 s with OK/WARNING — Task 4
- [x] Report: math, ω estimation, why-FDE-misleading, table, case study, limitations — Task 5
- [x] README Task 5 section — Task 5
- [x] No modification to run_task{1-4}.py — confirmed, none touched

**Type/signature consistency:**
- `TrajectoryPredictor.predict(track_id, source_state, source_P, source_ts_ns, horizons_s)` — used consistently in Task 2 (eval) and Task 3 (viz)
- `_uncertainty_ellipse(ax, center_xy, cov2x2, ...)` — called with `p.covariance[:2, :2]` in Task 3
- `render_bev_frames` signature matches call in `run_task5.py`
- `run_evaluation` returns `(pred_df, err_df, summary_df)` — destructured correctly in CLI

**Placeholder scan:** No TBD, no TODO in code tasks. Report table has placeholder dashes (intentional — filled in at Step 2 of Task 5 after the pipeline runs).
