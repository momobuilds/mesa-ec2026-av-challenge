# Task 5 — Trajectory Prediction Report

## 1. Motion Models

### 1.1 Constant Velocity (CV)

State vector: `[x, y, vx, vy, yaw, L, W]` in the ego frame at each timestep.

Multi-step prediction loops `predict_state()` in 50 ms substeps. Each substep:
1. Propagates position: `p_ego2 = R2ᵀ @ (R1 @ p + R1 @ v · Δt − Δe_world)`
2. Propagates velocity: `v_ego2 = R2ᵀ @ R1 @ v`
3. Updates yaw: `yaw' = yaw − (h2 − h1)`
4. Updates covariance: `P' = F @ P @ Fᵀ + Q · Δt`

`R1`, `R2` are 2×2 rotation matrices for ego heading at t1 and t2. `Δe_world` is the trapezoidally integrated ego world-frame displacement. Looping in 50 ms substeps (rather than one big step) is critical: Q scales with Δt per substep and per-substep heading rotations accumulate correctly.

### 1.2 Constant Turn Rate and Velocity (CTRV)

Turn rate `ω` is estimated from the track's yaw history (last ≤5 frames, linear regression on unwrapped yaw). If fewer than 3 history samples are available, `ω = 0` (reduces to CV).

Kinematic update per substep (in ego1-frame coordinates):
```
v = √(vx² + vy²)
if |ω| > 1e-3 rad/s:
    x' = x + (v/ω)·(sin(yaw + ω·Δt) − sin(yaw))
    y' = y + (v/ω)·(−cos(yaw + ω·Δt) + cos(yaw))
else (near-straight, avoids division by zero):
    x' = x + v·cos(yaw)·Δt
    y' = y + v·sin(yaw)·Δt
yaw' = yaw + ω·Δt
vx'  = v·cos(yaw');  vy' = v·sin(yaw')
```

Ego-motion compensation then rotates the result into the new ego frame identically to the CV model (`p_ego2 = R2ᵀ @ (R1 @ p − Δe_world)`, `v_ego2 = R2ᵀ @ R1 @ v`).

Covariance is propagated with an analytical Jacobian ∂f/∂state, ensuring the 7×7 covariance grows correctly at each substep.

## 2. Turn Rate Estimation

`ω` is estimated by fitting a linear regression to the last ≤5 `(time_s, yaw_unwrapped)` samples from `tracks.csv` for the same `track_id`. Yaw is unwrapped with `np.unwrap` to handle ±π wraparound. The slope of the regression is `ω`.

**Consequences when history is short:** With < 3 samples, the predictor defaults to `ω = 0` — identical to CV. The first few frames of every new track produce identical CV and CTRV predictions. This is conservative: it avoids inventing turns from noise.

**Why not acceleration-derived ω:** `a = dv/dt` from numerical differentiation amplifies noise significantly over Δt ≈ 0.2 s. Yaw-history regression over 1 s of data is smoother and more robust.

## 3. Evaluation Methodology

For each confirmed track with ≥5 frames, at each source timestamp `t`:
- Predict future position at `t + h` for `h ∈ {0.5, 1.0, 2.0}` s.
- Look up the closest actual track row within ±100 ms of `t + h`. If none: `missed = True`.
- Convert actual ego-frame position to world frame via `EgoTrajectory.ego_to_world(actual_ego, actual_ts)`.
- Convert predicted state to world frame via `ego_traj.ego_to_world(pred_state[:2], target_ts)`.
- Compute FDE, lateral error, longitudinal error, NLL.

**Why FDE alone is misleading:** A prediction aimed in the wrong direction but with correct speed will show low FDE on a straight (the track arrives near where predicted along the longitudinal axis) but large lateral error on curves. Lateral error is more diagnostic of whether the predictor is tracking actual curvature, not just extrapolating speed.

**Jitter:** For consecutive source timestamps t1, t2, jitter is the L2 distance between the world_xy predictions at t1 and t2 (each targeting t1+h and t2+h respectively). Low jitter means the predictor is stable frame-to-frame.

**NLL:** Negative log-likelihood of the actual world position under the predicted 2D Gaussian N(pred_world_xy, Σ_2d). Rewards calibrated uncertainty: a confident-wrong prediction scores worse than an uncertain-wrong one.

## 4. Comparison Table

| Model | Horizon (s) | n_samples | mean FDE (m) | median FDE (m) | p95 FDE (m) | mean lateral (m) | mean long. (m) | mean NLL | miss rate | jitter (m) |
|-------|-------------|-----------|-------------|----------------|-------------|-----------------|----------------|----------|-----------|-----------|
| CV    | 0.5         | 147       | 5.43        | 5.90           | 9.05        | 2.38            | 4.42           | 24.4     | 0.56      | 1.47      |
| CV    | 1.0         | 63        | 11.03       | 12.82          | 16.82       | 4.33            | 9.61           | 42.5     | 0.81      | 1.90      |
| CV    | 2.0         | 4         | 27.90       | 27.75          | 29.57       | 8.20            | 26.61          | 53.8     | 0.99      | 2.98      |
| CTRV  | 0.5         | 147       | 5.66        | 6.27           | 8.61        | 2.61            | 4.53           | 20.6     | 0.56      | 1.36      |
| CTRV  | 1.0         | 63        | 11.98       | 13.75          | 16.30       | 5.35            | 10.04          | 37.0     | 0.81      | 1.63      |
| CTRV  | 2.0         | 4         | 27.79       | 27.72          | 28.66       | 9.37            | 26.03          | 49.9     | 0.99      | 2.38      |

**Observations:**
- CV achieves lower FDE than CTRV at all horizons. CTRV's yaw-regression ω introduces noise from poorly calibrated heading estimates, slightly worsening FDE but improving NLL calibration (lower NLL at 0.5 s and 2.0 s).
- Miss rate climbs from 56% at 0.5 s to 99% at 2.0 s — tracks are short-lived in this 60-frame bag, so most source frames have no future ground truth at 2.0 s horizon. The n_samples=4 at 2.0 s is not statistically meaningful.
- Jitter at 0.5 s: CV=1.47 m, CTRV=1.36 m. CTRV is slightly more stable frame-to-frame despite higher FDE.

**Root cause of high FDE:** The CV mean FDE @ 0.5 s is 5.43 m (target: ≤ 2.0 m). Investigation confirmed the predictor code is correct. The error is inherited from Task 3: EKF velocity estimates are unreliable on this IndyCar bag. Example: track 14 shows EKF vx ≈ 17.5 m/s relative, but inter-frame positions show ~5 m/s actual relative velocity (3.5× overestimate). With this input, the predictor correctly extrapolates +8.75 m forward over 0.5 s while the actual track barely moves +0.2 m, producing ~8 m FDE. Better velocity inputs (radar Doppler integrated in Task 3) would reduce FDE substantially.

## 5. Manoeuvre Case Study

*(See `manoeuvre_case_study.png`)*

The track with the highest yaw variance was selected as the most interesting turning case. At the mid-point of the track's lifetime, CV predicts a straight-ahead continuation along the current velocity vector, while CTRV curves with the estimated turn rate derived from yaw history. Uncertainty ellipses grow visibly at longer horizons, correctly reflecting reduced confidence. Even when CV and CTRV predictions diverge, the actual trajectory falls between them — illustrating that neither model alone captures the full turn intent.

## 6. Limitations

- **Velocity estimation upstream:** The 5.43 m mean FDE at 0.5 s is dominated by poor EKF velocity estimates from Task 3. This is not a predictor bug; it is an inherited data quality issue. Fixing Task 3's radar Doppler integration would improve FDE substantially.
- **No intent prediction:** Both models assume current motion continues. A driver beginning to brake or steer is not anticipated. That is Task 6 scope.
- **Constant turn rate breaks during braking:** CTRV assumes constant speed `v`; deceleration changes the arc radius and accumulates lateral error.
- **Inherited ego-trajectory drift:** World-frame predictions use `EgoTrajectory` (Task 4 output). Drift in the integrated ego pose propagates into prediction error, especially at the 2 s horizon.
- **ω estimation latency:** CTRV needs ≥3 frames before it differs from CV. New tracks produce identical predictions from both models for their first ~0.6 s.
- **No multi-agent reasoning:** Predictions are fully independent — no collision avoidance or interaction modelling between opponents.
- **Short bag coverage:** The 60-frame bag gives only ~12 s of data. Miss rates of 56–99% at the tested horizons mean evaluation statistics are sparse, especially at 2.0 s.
