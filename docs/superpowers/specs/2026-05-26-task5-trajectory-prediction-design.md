# Task 5 — Trajectory Prediction Design

**Date:** 2026-05-26  
**Status:** Approved

---

## 1. Scope

Implement trajectory prediction for confirmed opponent tracks produced by Task 3. Given a track state at time *t*, predict its future position at horizons h ∈ {0.5, 1.0, 2.0} s using two motion models (CV and CTRV), propagate uncertainty as a covariance ellipse, self-evaluate against ground-truth future frames, and produce BEV visualisations.

No unit tests (per user instruction). Correctness is validated through the eval harness on real tracks data and a final FDE sanity check.

---

## 2. Architecture

### 2.1 Modules

| File | Role |
|---|---|
| `src/trajectory_predictor.py` | `Prediction` dataclass + `TrajectoryPredictor` class with CV and CTRV models |
| `src/prediction_eval.py` | Self-evaluation harness: predict at *t*, compare to actual track at *t+h* |
| `src/prediction_visualization.py` | BEV plots with fading polylines + uncertainty ellipses; histograms; case study |
| `scripts/run_task5.py` | CLI entry point analogous to run_task3/4 |

### 2.2 Data flow

```
tracks.csv  ──→  TrajectoryPredictor.predict()  ──→  predictions.csv
                        │                                    │
              EgoStateLog (ego.mat)                  prediction_eval.py
              EgoTrajectory (pose_at)                        │
                                                     prediction_errors.csv
                                                     prediction_summary.csv
                                                             │
                                                  prediction_visualization.py
                                                             │
                                                     BEV PNGs, histograms
```

---

## 3. Predictor Design (`trajectory_predictor.py`)

### 3.1 `Prediction` dataclass

```python
@dataclass
class Prediction:
    track_id: int
    source_ts_ns: int
    horizon_s: float
    target_ts_ns: int
    state: np.ndarray       # (7,) [x, y, vx, vy, yaw, L, W] in predicted-future ego frame
    covariance: np.ndarray  # (7, 7)
    world_xy: tuple[float, float]  # world-frame position for unambiguous eval
    model: str              # "cv" or "ctrv"
```

### 3.2 `TrajectoryPredictor`

```python
class TrajectoryPredictor:
    def __init__(self, ego_traj, ego_log, model="cv", dt_substep_s=0.05)
    def predict(self, source_state, source_P, source_ts_ns, horizons_s) -> list[Prediction]
    def set_track_history(self, track_history: dict[int, pd.DataFrame])  # for CTRV ω estimation
```

### 3.3 CV model (multi-step loop)

For each horizon `h`, loop from `t = source_ts_ns` to `t + h * 1e9` in substeps of `dt_substep_s`:

```python
for step in range(n_steps):
    t_prev = source_ts_ns + step * dt_sub_ns
    t_now  = t_prev + dt_sub_ns
    ego1 = ego_log.interpolate(t_prev)
    ego2 = ego_log.interpolate(t_now)
    state, P = predict_state(state, P, ego1, ego2, dt_substep_s)
```

This is correct because `predict_state` scales Q by Δt per substep and applies incremental frame rotations.

`world_xy` at `target_ts_ns = source_ts_ns + int(h * 1e9)` is obtained via `ego_traj.ego_to_world(state[[X,Y]].reshape(1,2), target_ts_ns)[0]`.

### 3.4 CTRV model

Same substep loop as CV. At each substep, instead of the CV position update:

```
v = sqrt(vx² + vy²)
if |ω| > 1e-3:
    x' = x + (v/ω)·(sin(yaw + ω·Δt) − sin(yaw))
    y' = y + (v/ω)·(−cos(yaw + ω·Δt) + cos(yaw))
else:
    x' = x + v·cos(yaw)·Δt
    y' = y + v·sin(yaw)·Δt
yaw' = yaw + ω·Δt
vx' = v·cos(yaw'); vy' = v·sin(yaw')
```

Then apply ego-motion compensation (rotate predicted world-frame position into the new ego frame) using the same `R2ᵀ @ R1` logic from `predict_state`. The CTRV Jacobian is computed analytically (∂f/∂state) and used to propagate covariance: `P' = F @ P @ F.T + Q·Δt`.

### 3.5 Turn rate ω estimation

Loaded once into `dict[track_id → pd.DataFrame]` at startup.

For each `(track_id, source_ts_ns)`:
1. Take rows where `timestamp_ns ≤ source_ts_ns`, last N=5 rows.
2. If < 3 rows: `ω = 0.0` (falls back to CV).
3. Else: unwrap yaw, `np.polyfit(Δt_seconds, yaw_unwrapped, 1)` → slope = ω.

---

## 4. Evaluation Harness (`prediction_eval.py`)

For each confirmed track with ≥ 5 observed frames:
- Walk source timestamps `t_source`.
- For each horizon `h ∈ {0.5, 1.0, 2.0}` s, find actual row closest to `t_source + h·1e9` within ±100 ms.
- Run predictor (both models) from `t_source`.
- Convert actual ego-frame center to world via `ego_traj.ego_to_world`.
- Compute:
  - **FDE**: `‖pred_world_xy − actual_world_xy‖`
  - **Lateral error**: component perpendicular to predicted heading
  - **Longitudinal error**: component along predicted heading
  - **NLL**: `−log N(actual | pred_world_xy, Σ_2d)` where `Σ_2d = covariance[:2,:2]`
- If no actual row within ±100 ms: record as missed.

**Stability / jitter**: for consecutive source timestamps `t1, t2 = t1 + Δt`, both predicting the same target time `t1 + h`, compute `‖pred1_xy − pred2_xy‖`. Mean per horizon reported in summary.

**Output CSVs**:
- `prediction_errors.csv`: `model, track_id, source_ts_ns, horizon_s, fde_m, lateral_m, longitudinal_m, nll, missed`
- `prediction_summary.csv`: 6 rows (2 models × 3 horizons): `model, horizon_s, n_samples, mean_fde, median_fde, p95_fde, mean_lateral, mean_longitudinal, mean_nll, miss_rate, mean_jitter`
- `predictions.csv`: every `(source_ts, track_id, horizon, model)` combination with `pred_x, pred_y, cov_xx, cov_yy, world_x, world_y`

---

## 5. Visualisations (`prediction_visualization.py`)

### 5.1 `prediction_bev_NNN.png` (5–8 frames)
- BEV centred on ego, ±60 m.
- For each confirmed track: draw actual track box + velocity arrow.
- For each model: draw fading polyline (alpha ∝ distance from source), with **uncertainty ellipses** at h ∈ {0.5, 1.0, 2.0} s. Ellipse axes = 2·√eigenvalues of `cov[:2,:2]` (~95% confidence). CV = blue family, CTRV = orange family.

### 5.2 `prediction_error_hist.png`
- 2×3 subplot grid (model × horizon).
- FDE histograms with vertical median lines. CV blue, CTRV orange.

### 5.3 `manoeuvre_case_study.png`
- Pick track with highest yaw variance or best CTRV-vs-CV FDE improvement.
- Zoom BEV showing: actual trajectory (black), CV prediction (blue + ellipses), CTRV prediction (orange + ellipses), horizon markers.

---

## 6. CLI (`scripts/run_task5.py`)

```bash
uv run python scripts/run_task5.py \
    --tracks outputs/task3/tracks.csv \
    --bag-dir data \
    --output-dir outputs/task5 \
    --horizons 0.5 1.0 2.0 \
    --models cv ctrv \
    --bev-frames 6
```

Flags: `--no-eval`, `--min-track-len 5`, `--substep-ms 50`.

---

## 7. Constraints

- Do NOT modify `run_task{1-4}.py` or any Task 1–4 output.
- Reuse `predict_state`, `_rot2d`, `_X/_Y/_VX/_VY/_YAW/_L/_W`, `STATE_DIM`, `EgoTrajectory`, `EgoStateLog` — no redefinition.
- No new top-level dependencies. Only `numpy`, `scipy`, `matplotlib`, `pandas`, `h5py` used.
- Append Task 5 section to `README.md`.

---

## 8. Implementation Phases

1. **`trajectory_predictor.py`** — CV model only, substep loop, world_xy conversion.
2. **Add CTRV** — analytical Jacobian, ω estimation from yaw history.
3. **`prediction_eval.py`** — eval harness, all 3 CSVs. Print FDE@0.5s sanity check before proceeding.
4. **`prediction_visualization.py`** — 3 plot types.
5. **`scripts/run_task5.py`** + **`task5_report.md`** + README section.

---

## 9. Definition of Done

- `run_task5.py` runs end-to-end without errors.
- `predictions.csv` has predictions for every confirmed track at every source timestamp, all horizons, both models.
- `prediction_summary.csv` has 6 non-NaN rows.
- Mean FDE (CV) ≤ 2.0 m at 0.5 s horizon — verified and documented in report.
- `prediction_bev_004.png` shows tracks with fading trajectories and visible ellipses.
- Tasks 1–4 outputs byte-identical.
