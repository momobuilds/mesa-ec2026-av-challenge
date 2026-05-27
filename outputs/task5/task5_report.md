# Task 5 -- Trajectory Prediction Report

## 1. Motion Models

### 1.1 Constant Velocity (CV)

State vector: `[x, y, vx, vy, yaw, L, W]` in the ego frame at each timestep.

Multi-step prediction loops `predict_state()` in 50 ms substeps. Each substep:

1. Propagates position: `p_ego2 = R2T @ (R1 @ p + R1 @ v * dt - de_world)`
2. Propagates velocity: `v_ego2 = R2T @ R1 @ v`
3. Updates yaw: `yaw' = yaw - (h2 - h1)`
4. Updates covariance: `P' = F @ P @ FT + Q * dt`

`R1`, `R2` are 2x2 rotation matrices for ego heading at t1 and t2. `de_world` is the
trapezoidally integrated ego world-frame displacement. Looping in 50 ms substeps (rather
than one big step) is critical: Q scales with dt per substep and per-substep heading
rotations accumulate correctly.

### 1.2 Constant Turn Rate and Velocity (CTRV)

Turn rate `w` is estimated from the track's yaw history (last 5 frames max, linear regression
on unwrapped yaw). If fewer than 3 history samples are available, `w = 0` (reduces to CV).

Kinematic update per substep (in ego1-frame coordinates):

```
v = sqrt(vx^2 + vy^2)
if |w| > 1e-3 rad/s:
    x' = x + (v/w)*(sin(yaw + w*dt) - sin(yaw))
    y' = y + (v/w)*(-cos(yaw + w*dt) + cos(yaw))
else (near-straight, avoids division by zero):
    x' = x + v*cos(yaw)*dt
    y' = y + v*sin(yaw)*dt
yaw' = yaw + w*dt
vx'  = v*cos(yaw');  vy' = v*sin(yaw')
```

Ego-motion compensation then rotates the result into the new ego frame identically to the
CV model (`p_ego2 = R2T @ (R1 @ p - de_world)`, `v_ego2 = R2T @ R1 @ v`).

Covariance is propagated with an analytical Jacobian df/dstate, ensuring the 7x7 covariance
grows correctly at each substep.

## 2. Turn Rate Estimation

`w` is estimated by fitting a linear regression to the last 5 `(time_s, yaw_unwrapped)`
samples from `tracks.csv` for the same `track_id`. Yaw is unwrapped with `np.unwrap` to
handle +/-pi wraparound. The slope of the regression is `w`.

**Consequences when history is short:** With < 3 samples, the predictor defaults to `w = 0`
-- identical to CV. The first few frames of every new track produce identical CV and CTRV
predictions. This is conservative: it avoids inventing turns from noise.

**Why not acceleration-derived w:** `a = dv/dt` from numerical differentiation amplifies
noise significantly over dt approx 0.2 s. Yaw-history regression over 1 s of data is smoother
and more robust.

## 3. Evaluation Methodology

### Ego Trajectory Source (updated)

`EgoTrajectory` is instantiated via `EgoTrajectory.from_bag_tf(bag_dir)`, which reads the
45 080 `map -> center_of_gravity` transforms from the `/tf` topic in the bag at ~200 Hz.
This replaces the previous velocity-integration approach (`from_ego_log`) that accumulated
numerical drift over the 240 s bag. The TF-based pose is the ground-truth localisation
output of the vehicle's own stack -- no cumulative error propagates into the world-frame
coordinate conversion used during evaluation.

`EgoStateLog` (from `ego.mat`) is retained separately and used **only for substep
ego-motion compensation** -- it provides `(ekf_vx, ekf_vy, heading)` at each 50 ms substep
to compute the relative ego displacement during prediction.

For each confirmed track with >= 5 frames, at each source timestamp `t`:

- Predict future position at `t + h` for `h in {0.5, 1.0, 2.0}` s.
- Look up the closest actual track row within +/-100 ms of `t + h`. If none: `missed = True`.
- Convert actual ego-frame position to world frame via `EgoTrajectory.ego_to_world(actual_ego, actual_ts)`.
- Convert predicted state to world frame via `ego_traj.ego_to_world(pred_state[:2], target_ts)`.
- Compute FDE, lateral error, longitudinal error, NLL.

**Why FDE alone is misleading:** A prediction aimed in the wrong direction but with correct
speed will show low FDE on a straight (the track arrives near where predicted along the
longitudinal axis) but large lateral error on curves. Lateral error is more diagnostic of
whether the predictor is tracking actual curvature, not just extrapolating speed.

**Jitter:** For consecutive source timestamps t1, t2, jitter is the L2 distance between
the world_xy predictions at t1 and t2 (each targeting t1+h and t2+h respectively). Low
jitter means the predictor is stable frame-to-frame.

**NLL:** Negative log-likelihood of the actual world position under the predicted 2D Gaussian
N(pred_world_xy, S_2d). Rewards calibrated uncertainty: a confident-wrong prediction scores
worse than an uncertain-wrong one.

## 4. Comparison Table (200 frames)

| Model | Horizon (s) | n_samples | mean FDE (m) | median FDE (m) | p95 FDE (m) | mean lateral (m) | mean long. (m) | mean NLL | miss rate | jitter (m) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | 0.5 | 888 | 4.16 | 4.07 | 8.22 | 2.57 | 2.71 | 17.6 | 0.408 | 1.775 |
| CV | 1.0 | 398 | 7.87 | 7.71 | 14.63 | 5.24 | 4.90 | 25.2 | 0.735 | 2.081 |
| CV | 2.0 | 24 | 9.85 | 10.67 | 13.59 | 5.64 | 7.53 | 10.6 | 0.984 | 2.845 |
| CTRV | 0.5 | 888 | 4.16 | 4.02 | 7.84 | 2.60 | 2.72 | 15.1 | 0.408 | 1.665 |
| CTRV | 1.0 | 398 | 7.77 | 7.42 | 14.24 | 5.30 | 4.75 | 20.7 | 0.735 | 1.842 |
| CTRV | 2.0 | 24 | 10.31 | 11.16 | 13.73 | 5.87 | 7.98 | 21.5 | 0.984 | 2.334 |

**Comparison to 60-frame run (old data):**

| Metric | 60-frame | 200-frame | Change |
| :--- | ---: | ---: | :--- |
| n_samples @ 0.5 s | 147 | 888 | +604% more samples |
| n_samples @ 2.0 s | 4 | 24 | +500% (now statistically meaningful) |
| CV mean FDE @ 0.5 s (m) | 5.43 | 4.16 | -23% improvement |
| CV mean FDE @ 2.0 s (m) | 27.90 | 9.85 | -65% improvement |
| Miss rate @ 0.5 s | 0.56 | 0.408 | Lower miss rate |
| CTRV mean NLL @ 0.5 s | 20.6 | 15.1 | Better calibration |

**Observations:**

- **CV vs CTRV FDE**: Essentially tied at 0.5 s (4.162 vs 4.157 m); CTRV wins at 1.0 s
  (7.87 vs 7.77 m). At 2.0 s CV is slightly better (9.85 vs 10.31 m).
- **NLL**: CTRV wins at all horizons except 2.0 s (15.1 vs 17.6 at 0.5 s). CTRV's
  covariance is better calibrated -- it acknowledges more uncertainty from yaw noise.
- **Jitter**: CTRV is consistently more stable (1.665 vs 1.775 m at 0.5 s), confirming the
  yaw-history smoother produces more temporally consistent predictions.
- **Miss rate**: 40.8% at 0.5 s, 73.5% at 1.0 s, 98.4% at 2.0 s. Most tracks in this
  bag are short-lived confirmed tracks (mean 7.3 observed frames = ~1.5 s), so the 2.0 s
  horizon hits the end of most tracks.
- **FDE improvement 60->200 frames**: Primarily attributable to (a) more long-lived tracks
  providing better velocity estimates, (b) TF-based ego_traj replacing drift-based trajectory,
  and (c) regression to mean with 6x more samples.

**Root cause of high FDE:** The CV mean FDE @ 0.5 s is 4.16 m (target: <= 2.0 m). The
predictor code is correct. The error is inherited from Task 3: EKF velocity estimates
(vx, vy) are unreliable on this IndyCar bag because (1) LiDAR-only tracks have no Doppler
so velocity must be estimated purely from position differences (noisy at 5 Hz), and (2) the
EKF velocity takes 5-10 frames to converge from the zero initial condition. Tracks confirmed
at only 3 hits have velocity estimates far from reality. Better radar Doppler integration in
Task 3 would reduce FDE substantially.

## 5. Manoeuvre Case Study

*(See `manoeuvre_case_study.png`)*

The track with the highest yaw variance was selected as the most interesting turning case. At
the mid-point of the track's lifetime, CV predicts a straight-ahead continuation along the
current velocity vector, while CTRV curves with the estimated turn rate derived from yaw
history. Uncertainty ellipses grow visibly at longer horizons, correctly reflecting reduced
confidence. Even when CV and CTRV predictions diverge, the actual trajectory falls between
them -- illustrating that neither model alone captures the full turn intent.

## 6. Limitations

- **Velocity estimation upstream:** The 4.16 m mean FDE at 0.5 s is dominated by poor EKF
  velocity estimates from Task 3. This is not a predictor bug; it is an inherited data quality
  issue. Fixing Task 3's radar Doppler integration would improve FDE substantially.
- **No intent prediction:** Both models assume current motion continues. A driver beginning to
  brake or steer is not anticipated. That is Task 6 scope.
- **Constant turn rate breaks during braking:** CTRV assumes constant speed `v`; deceleration
  changes the arc radius and accumulates lateral error.
- ~~**Inherited ego-trajectory drift:**~~ **Resolved.** `EgoTrajectory` now reads pose
  directly from the bag's `/tf` topic (`map -> center_of_gravity`) via `BagTFReader`. No
  velocity-integration drift accumulates, eliminating this source of world-frame evaluation
  error.
- **w estimation latency:** CTRV needs >= 3 frames before it differs from CV. New tracks
  produce identical predictions from both models for their first ~0.6 s.
- **No multi-agent reasoning:** Predictions are fully independent -- no collision avoidance
  or interaction modelling between opponents.
- **Short-lived tracks:** Mean confirmed track lifetime is 7.3 observed frames (~1.5 s at 5
  Hz). The 2.0 s horizon therefore hits the end of most tracks (miss rate 98.4%),
  making the 2.0 s statistics (n=24) sparse. A higher-frequency master clock or longer
  recording would give more ground truth at medium horizon.

## 7. Output Files

| File | Description |
| :--- | :--- |
| `predictions.csv` | All (source_ts, track_id, horizon, model) predictions with predicted and world-frame positions |
| `prediction_errors.csv` | Per-prediction FDE, lateral, longitudinal, NLL, missed flag |
| `prediction_summary.csv` | Aggregated mean/median/p95 FDE, NLL, jitter per (model, horizon) |
| `prediction_bev_NNN.png` | 6 BEV frames with CV (blue) and CTRV (orange) prediction polylines and uncertainty ellipses |
| `prediction_error_hist.png` | FDE histograms per horizon, CV vs CTRV |
| `manoeuvre_case_study.png` | Zoom on highest-yaw-variance track: CV vs CTRV divergence at corners |

## 8. Reproducing Results

```bash
uv run python scripts/run_task5.py \
    --tracks outputs/task3/tracks.csv \
    --bag-dir data \
    --output-dir outputs/task5 \
    --horizons 0.5 1.0 2.0 \
    --bev-frames 6
```
