# Task 3 Continuous Target Tracking Report

## Pipeline Overview

Task 3 adds a persistent multi-object tracker on top of the Task 2 per-frame sensor detection
pipeline. Every opponent detection from all 7 sensors (3 LiDARs + 4 radars) is fed into a
Mahalanobis-gated, Hungarian-matched Extended Kalman Filter (EKF) tracker that emits trajectories
with stable integer track IDs across the entire bag.

```text
per-sensor DBSCAN detection (Task 2)
       │
       ▼
SensorDetection index (flat, sorted by timestamp)
       │
       ▼  ±50 ms window gather per master-clock tick (lidar_front)
MultiObjectTracker.step(timestamp, detections)
  ├─ predict_state()      ← CV model with ego-motion compensation
  ├─ Two-stage Hungarian  ← Stage 1: confirmed↔LiDAR  Stage 2: confirmed↔radar
  ├─ EKF update           ← task2_fusion._ekf_update (reused)
  └─ lifecycle manage     ← tentative → confirmed → coasting → deleted
       │
       ▼
tracks.csv  /  tracks_summary.csv  /  BEV plots  /  timeline
```

## Sensor Extrinsics (updated)

All sensor → ego (`center_of_gravity`) transforms are now derived from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, auto-loaded at import time. This resolves the sensor
orientation ambiguity raised with the professor.

| Sensor | x (fwd, m) | y (lat, m) | Mounting yaw |
|---|---:|---:|---:|
| lidar_front | +0.662 | 0.000 | −179.5° |
| lidar_left | +0.002 | +0.135 | −59.4° |
| lidar_right | +0.002 | −0.135 | +60.0° |
| radar_front | +0.408 | 0.000 | −89.5° |
| radar_back | −2.550 | 0.000 | +91.7° |
| radar_left | +0.294 | +0.230 | 0.0° |
| radar_right | +0.294 | −0.230 | −180.0° |

## Motion Model

### Ego-motion source (updated)

The tracker uses two distinct ego-motion sources:

| Source | Used for |
|---|---|
| `BagTFReader` (`/tf` topic) | *(not used in tracking predict step — predict works in ego frame)* |
| `EgoStateLog` (`ego.mat`) | `predict_state()`: ego displacement Δe and heading change Δh between frames |

The tracker's `predict_state()` works entirely in the ego frame: it does not need the absolute
world pose. It needs only the *relative* ego displacement Δe and heading change Δh between two
consecutive ticks, both obtained from `EgoStateLog.interpolate()` via `ekf_vx`/`ekf_vy`.

`BagTFReader` (world-frame ego pose from `/tf`) is used by Tasks 4 and 5 for map building and
world-frame error evaluation — not needed for the ego-frame tracker predict step.

### Why ego-motion compensation is required

All detections are expressed in the **ego (`centre_of_gravity`) frame** at the moment of
measurement. Between consecutive master-clock ticks (~200 ms at 5 Hz), the ego car moves ~8–12 m
forward. A naive EKF predict step that adds `v · Δt` in ego coordinates treats the ego frame as
inertial, causing every stationary object to appear to drift −v_ego per step. Without
compensation, every track immediately loses association.

### Derivation

Let:

- R1 = rot2D(h1), R2 = rot2D(h2) — 2D rotation matrices for ego heading at t1, t2
- Δe = ½(v_ego1 + v_ego2) · Δt — ego world-frame displacement (trapezoidal)

For track state (p_ego1, v_ego1) at t1:

```python
# Target world-frame position at t2 (CV propagation):
p_w2 = R1 @ p_ego1 + R1 @ v_ego1 * dt - delta_e

# Convert back to ego2 frame:
p_ego2 = R2.T @ p_w2   # = R2.T @ R1 @ (p_ego1 + v_ego1 * dt) - R2.T @ delta_e

v_ego2 = R2.T @ R1 @ v_ego1
```

`EgoStateLog.interpolate(t)` provides (vx, vy, heading) for any ROS timestamp by linear
interpolation over `ego.mat`. These velocities are also used for Doppler correction.

### Jacobian and process noise

The linearised state-transition Jacobian for F = [I | Δt·I | ...] block structure becomes
F_pos_pos = F_vel_vel = R2ᵀ R1, F_pos_vel = R2ᵀ R1 Δt.
Process noise Q (scaled by Δt):

| State | σ²/s | Rationale |
| --- | --- | --- |
| x, y position | 0.5 m² | Lane-constrained opponent motion |
| vx, vy velocity | 2.0 m²/s² | Braking / acceleration bursts |
| yaw | (15°)²/s | Fast heading change in corners |
| length, width | 0.1 m² | Dimensions nearly constant |

## Track Lifecycle

| Status | Transition | Condition |
| --- | --- | --- |
| TENTATIVE | → CONFIRMED | ≥ 3 consecutive association hits |
| TENTATIVE | → (suppressed) | `|center_y| > 8 m` at promotion time |
| CONFIRMED | → COASTING | ≥ 1 missed frame |
| COASTING | → CONFIRMED | Association received |
| COASTING / CONFIRMED | → DELETED | `time_since_update > 5` frames |
| TENTATIVE | → DELETED | age > 10 frames without promotion |

Track IDs are monotonically increasing integers and are **never reused**.

## Association Strategy

**Stage 1**: Confirmed + coasting tracks ↔ LiDAR detections.
Gate: Mahalanobis distance d < 4.0 using the track's current covariance P
(not a fixed per-sensor σ — the covariance grows during coasting, widening the gate exactly
when it should).

**Stage 2**: Remaining confirmed + coasting tracks ↔ Radar detections.
Gate: d < 6.0 (wider; radar position noise σ² = 4.0 m² vs LiDAR σ² = 0.5 m²).

**Stage 3**: Unmatched detections ↔ tentative tracks (gate d < 4.0).

**Seeding**: Any remaining unmatched detection seeds a new tentative track. Tracks with
|center_y| > 8 m at seed time are immediately flagged `static_clutter = True` and never promoted.

Per-sensor measurement noise R (reused from Task 2):

| Sensor | σ²_x, σ²_y (m²) | σ²_vx, σ²_vy (m²/s²) |
| --- | --- | --- |
| LiDAR | 0.5 | 9.0 (no Doppler) |
| Radar | 4.0 | 0.4 (good Doppler) |

## Results (60 frames, all 7 sensors)

| Metric | Value |
| --- | --- |
| Master-clock frames | 60 |
| Raw sensor detections | 518 |
| Total unique track IDs created | 178 |
| Confirmed tracks (≥ 3 hits) | 98 |
| Track-rows emitted in tracks.csv | 784 |
| Tracks with > 30 observed frames | **2** (T13: 38 frames, T14: 34 frames) |
| Mean confirmed track lifetime | 4.9 frames |
| Probable ID switches (proxy) | 1 |
| Coasting fraction | 46% of emitted rows |

The two longest tracks (T13, T14) both have coasting segments, confirming that the tracker
successfully bridges short detection gaps. Mean speeds of 9–15 m/s in ego-frame relative speed
are consistent with opponent vehicles at similar absolute speeds to the ego car.

## Task 2 vs Task 3 comparison

| Property | Task 2 | Task 3 |
| --- | --- | --- |
| Output unit | per-frame detection | persistent track |
| Identity across frames | none (per-frame ID) | stable integer track_id |
| Occlusion handling | none | coasting up to max_age=5 frames |
| Ego-motion compensation | no (single-frame) | yes (world-frame predict) |
| Total rows (60 frames) | ~390 (≈6.5 dets/frame) | 784 (confirmed + coasting) |
| Velocity estimate | 0.0 (LiDAR-only, no Doppler match) | EKF-fused from track history |

## Occlusion Recovery

Running `--simulate-occlusion 5 --occlusion-start 20` blanks all detections for frames 20–24.
Track T101 survived the 5-frame gap with the same ID (n_frames_observed=25,
n_frames_coasted=10), demonstrating correct coasting behaviour.

The tracker uses max_age=5 missed frames before deletion, matching exactly the simulated gap
length. Extending to `--max-age 8` would allow recovery from longer occlusions.

## Known Limitations

1. **No cross-sensor matches.** Radar detections are barrier returns at >10 m lateral or ego
   self-reflections at <3 m range. The Mahalanobis gate correctly rejects them. Radar is not
   improving track quality in this bag.

2. **LiDAR velocity is zero-initialised.** No Doppler on LiDAR means EKF velocity converges
   slowly from track position changes, not from direct measurement. Velocity estimates are more
   accurate for longer tracks.

3. **High track count (178 created, 98 confirmed, 80 tentative).** Many short tentative tracks
   indicate DBSCAN over-segmentation or multi-bounce radar returns. Tightening `min_samples` or
   adding a minimum point-count filter would reduce noise.

4. **Static-clutter filter is lateral-only.** Only |y_ego| > 8 m is checked at seed time. A
   world-frame position-drift check (object didn't move in world frame over 10+ frames) would
   also suppress parked vehicles, but was omitted to avoid false-flagging slow opponents.

5. ~~**Hardcoded extrinsics.**~~ **Resolved.** Sensor → ego transforms are now loaded from
   `data/eva.urdf` via `tf_utils.init_extrinsics_from_urdf()`. The left/right LiDAR yaw
   ambiguity (120° vs 60° mounting) is resolved by the URDF's explicit rpy values.

## Output Files

| File | Description |
| --- | --- |
| `tracks.csv` | One row per (track_id, timestamp). Columns: timestamp_ns, track_id, status, age, time_since_update, center_x, center_y, vx, vy, yaw, length, width, cov_xx, cov_yy, n_lidar, n_radar, sensors, confidence |
| `tracks_summary.csv` | One row per track_id: start/end timestamps, n_frames_observed, n_frames_coasted, total_path_m, mean/max speed, last_status |
| `tracking_bev_NNN.png` | BEV at 5 evenly-spaced frames: raw sensor points, oriented boxes (solid=confirmed, dashed=coasting), velocity arrows, trail of last 20 positions, colour per track_id |
| `tracking_timeline.png` | Gantt chart: track_id on y-axis, frames on x-axis, solid=observed, hatched=coasting |

## Reproducing Results

```bash
# Full 60-frame run
uv run python scripts/run_task3.py --bag-dir data --max-frames 60 --bev-frames 5

# Occlusion recovery demo (5-frame blank at frame 20)
uv run python scripts/run_task3.py --bag-dir data --max-frames 60 \
    --simulate-occlusion 5 --occlusion-start 20 --bev-frames 0

# Motion model unit tests
uv run python -m src.motion_model
```
