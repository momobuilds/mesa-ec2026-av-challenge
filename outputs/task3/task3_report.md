# Task 3 Continuous Target Tracking Report

## Pipeline Overview

Task 3 adds a persistent multi-object tracker on top of the Task 2 per-frame sensor detection
pipeline. Every opponent detection from all 7 sensors (3 LiDARs + 4 radars) is fed into a
Mahalanobis-gated, Hungarian-matched Extended Kalman Filter (EKF) tracker that emits trajectories
with stable integer track IDs across the entire bag.

```text
per-sensor DBSCAN detection (Task 2)
       |
       v
SensorDetection index (flat, sorted by timestamp)
       |
       v  +/-50 ms window gather per master-clock tick (lidar_front)
MultiObjectTracker.step(timestamp, detections)
  +-- predict_state()      <- CV model with ego-motion compensation
  +-- Two-stage Hungarian  <- Stage 1: confirmed <-> LiDAR  Stage 2: confirmed <-> radar
  +-- EKF update           <- task2_fusion._ekf_update (reused)
  +-- lifecycle manage     <- tentative -> confirmed -> coasting -> deleted
       |
       v
tracks.csv  /  tracks_summary.csv  /  BEV plots  /  timeline
```

## Sensor Extrinsics (updated)

All sensor -> ego (`center_of_gravity`) transforms are now derived from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, auto-loaded at import time. This resolves the sensor
orientation ambiguity raised with the professor.

| Sensor      | x (fwd, m) | y (lat, m) | Mounting yaw  |
| :---------- | ---------: | ---------: | ------------: |
| lidar_front |     +0.662 |      0.000 | -179.5 deg    |
| lidar_left  |     +0.002 |     +0.135 |  -59.4 deg    |
| lidar_right |     +0.002 |     -0.135 |  +60.0 deg    |
| radar_front |     +0.408 |      0.000 |  -89.5 deg    |
| radar_back  |     -2.550 |      0.000 |  +91.7 deg    |
| radar_left  |     +0.294 |     +0.230 |    0.0 deg    |
| radar_right |     +0.294 |     -0.230 | -180.0 deg    |

## Motion Model

### Ego-motion source (updated)

The tracker uses two distinct ego-motion sources:

| Source                      | Used for                                                              |
| :-------------------------- | :-------------------------------------------------------------------- |
| `BagTFReader` (`/tf` topic) | (not used in tracking predict step -- predict works in ego frame)     |
| `EgoStateLog` (`ego.mat`)   | `predict_state()`: ego displacement and heading change between frames |

The tracker's `predict_state()` works entirely in the ego frame: it does not need the absolute
world pose. It needs only the *relative* ego displacement and heading change between two
consecutive ticks, both obtained from `EgoStateLog.interpolate()` via `ekf_vx`/`ekf_vy`.

`BagTFReader` (world-frame ego pose from `/tf`) is used by Tasks 4 and 5 for map building and
world-frame error evaluation -- not needed for the ego-frame tracker predict step.

### Why ego-motion compensation is required

All detections are expressed in the **ego (`centre_of_gravity`) frame** at the moment of
measurement. Between consecutive master-clock ticks (~200 ms at 5 Hz), the ego car moves ~8-12 m
forward. A naive EKF predict step that adds `v * dt` in ego coordinates treats the ego frame as
inertial, causing every stationary object to appear to drift -v_ego per step. Without
compensation, every track immediately loses association.

### Derivation

Let:

- R1 = rot2D(h1), R2 = rot2D(h2) -- 2D rotation matrices for ego heading at t1, t2
- de = trapezoidally integrated ego world-frame displacement

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

The linearised state-transition Jacobian for F = [I | dt*I | ...] block structure becomes
F_pos_pos = F_vel_vel = R2^T R1, F_pos_vel = R2^T R1 dt.
Process noise Q (scaled by dt):

| State         | sigma^2/s      | Rationale                             |
| :------------ | -------------: | :------------------------------------ |
| x, y          |       0.5 m^2  | Lane-constrained opponent motion      |
| vx, vy        |    2.0 m^2/s^2 | Braking / acceleration bursts         |
| yaw           |  (15 deg)^2/s  | Fast heading change in corners        |
| length, width |       0.1 m^2  | Dimensions nearly constant            |

## Track Lifecycle

| Status               | Transition   | Condition                                        |
| :------------------- | :----------- | :----------------------------------------------- |
| TENTATIVE            | -> CONFIRMED | >= 3 consecutive association hits                |
| TENTATIVE            | -> suppressed | `abs(center_y) > 8 m` at promotion time         |
| CONFIRMED            | -> COASTING  | >= 1 missed frame                                |
| COASTING             | -> CONFIRMED | Association received                             |
| COASTING / CONFIRMED | -> DELETED   | `time_since_update > 8` frames                  |
| TENTATIVE            | -> DELETED   | age > 16 frames without promotion               |

Track IDs are monotonically increasing integers and are **never reused**.

## Association Strategy

**Stage 1**: Confirmed + coasting tracks <-> LiDAR detections.
Gate: Mahalanobis distance d < 4.0 using the track's current covariance P
(not a fixed per-sensor sigma -- the covariance grows during coasting, widening the gate exactly
when it should).

**Stage 2**: Remaining confirmed + coasting tracks <-> Radar detections.
Gate: d < 6.0 (wider; radar position noise sigma^2 = 4.0 m^2 vs LiDAR sigma^2 = 0.5 m^2).

**Stage 3**: Unmatched detections <-> tentative tracks (gate d < 4.0).

**Seeding**: Any remaining unmatched detection seeds a new tentative track. Tracks with
`|center_y| > 8 m` at seed time are immediately flagged `static_clutter = True` and never
promoted.

Per-sensor measurement noise R (reused from Task 2):

| Sensor | sigma^2_x, sigma^2_y (m^2) | sigma^2_vx, sigma^2_vy (m^2/s^2) |
| :----- | -------------------------: | --------------------------------: |
| LiDAR  |                        0.5 | 9.0 (no Doppler)                  |
| Radar  |                        4.0 | 0.4 (good Doppler)                |

## Results (60 frames, all 7 sensors)

| Metric                                    |    Value |
| :---------------------------------------- | -------: |
| Master-clock frames                       |      60 |
| Raw sensor detections                     |     866 |
| Total unique track IDs created            |     218 |
| Confirmed tracks (>= 3 hits)              |     169 |
| Track-rows emitted in tracks.csv          |     864 |
| Tracks with >= 20 observed frames         |       7 |
| Tracks with >= 10 observed frames         |      54 |
| Mean confirmed track lifetime (obs)       | 6.9 frames |
| Mean coasting frames per track            | 1.6 frames |
| Coasting fraction of emitted rows         |    37.0% |
| Probable ID switches (proximity proxy)    |      12 |

The longest tracks are T27 (30 obs, 9 coasted), T88 (28 obs, 7 coasted), T82 (25 obs, 15
coasted), T143 (22 obs), and T68 (21 obs), all at relative speeds of 11-15 m/s consistent with
opponent vehicles in close convoy racing. Both LiDAR and radar contribute to the sensor mix for
multi-sensor tracks.

## Task 2 vs Task 3 comparison

| Property               | Task 2                            | Task 3                          |
| :--------------------- | :-------------------------------- | :------------------------------ |
| Output unit            | per-frame detection               | persistent track                |
| Identity across frames | none (per-frame ID)               | stable integer track_id         |
| Occlusion handling     | none                              | coasting up to max_age=8 frames |
| Ego-motion compensation | no (single-frame)                | yes (world-frame predict)       |
| Total rows (60 frames) | ~918 fused                        | 864 (confirmed + coasting)      |
| Velocity estimate      | 0.0 (LiDAR-only) or radar Doppler | EKF-fused from track history    |

## Occlusion Recovery

Running `--simulate-occlusion 5 --occlusion-start 20` blanks all detections for frames 20-24.
With `max_age=8` the tracker can now coast through up to 8 consecutive missed frames (1.6 s at
5 Hz), exceeding the 5-frame simulated gap. Tracks that were CONFIRMED before the gap survive
with the same ID and re-associate after the blank period ends, demonstrating correct coasting
behaviour.

Increasing `max_age` from 5 to 8 reduces the coasting fraction (46.6% -> 37.0%) and improves
mean observed lifetime (4.9 -> 6.9 frames) because the tracker can bridge longer detection
gaps without creating a new tentative track.

## Known Limitations

1. **No cross-sensor position fusion in Task 3.** Radar detections are primarily barrier returns
   at >10 m lateral or ego self-reflections at <3 m range. The Mahalanobis gate correctly
   rejects them for Stage 1 (LiDAR), but Stage 2 does admit some radar updates for tracks
   already established by LiDAR.

2. **LiDAR velocity is zero-initialised.** No Doppler on LiDAR means EKF velocity converges
   slowly from track position changes, not from direct measurement. Velocity estimates are more
   accurate for longer tracks.

3. **High track count (218 created, 169 confirmed).** Many short tentative tracks indicate
   DBSCAN over-segmentation or multi-bounce radar returns. Tightening `min_samples` or adding
   a minimum point-count filter would reduce noise.

4. **ID switch proxy is not ground truth.** The 12 probable ID switches are detected by a
   proximity heuristic (new track appears within 2 m of recently deleted confirmed track within
   1 s). The actual number of true identity swaps may be lower. The increase relative to the
   previous 60-frame run (1 switch) is partly explained by more total tracks created due to the
   now-active side LiDAR sensors producing additional detections.

5. **Static-clutter filter is lateral-only.** Only `|y_ego| > 8 m` is checked at seed time. A
   world-frame position-drift check (object did not move in world frame over 10+ frames) would
   also suppress parked vehicles, but was omitted to avoid false-flagging slow opponents.

6. ~~**Hardcoded extrinsics.**~~ **Resolved.** Sensor -> ego transforms are now loaded from
   `data/eva.urdf` via `tf_utils.init_extrinsics_from_urdf()`. The left/right LiDAR yaw
   ambiguity (120 deg vs 60 deg mounting) is resolved by the URDF's explicit rpy values.

7. ~~**n_frames_coasted off-by-one in summary metric.**~~ **Resolved.** `build_summary_df()`
   now correctly subtracts the creation history entry (`n_coasted = len(hist) - 1 - hits`)
   so that a track with no missed frames reports 0 coasted frames.

## Output Files

| File                   | Description                                                                                                                                                  |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tracks.csv`           | One row per (track_id, timestamp). Columns: timestamp_ns, track_id, status, age, time_since_update, center_x, center_y, vx, vy, yaw, length, width, cov_xx, cov_yy, n_lidar, n_radar, sensors, confidence |
| `tracks_summary.csv`   | One row per track_id: start/end timestamps, n_frames_observed, n_frames_coasted, total_path_m, mean/max speed, last_status                                   |
| `tracking_bev_NNN.png` | BEV at 5 evenly-spaced frames: raw sensor points, oriented boxes (solid=confirmed, dashed=coasting), velocity arrows, trail of last 20 positions             |
| `tracking_timeline.png` | Gantt chart: track_id on y-axis, frames on x-axis, solid=observed, hatched=coasting                                                                        |

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
