# Task 2 Report — Multi-Sensor Fusion

## Pipeline Overview

Task 2 extends the Task 1 DBSCAN detector to a full multi-sensor fusion pipeline across 3 LiDARs and 4 radars. The architecture:

1. **Per-sensor detection** — each of 7 PointCloud2 topics is transformed to the ego (`center_of_gravity`) frame and processed with the same DBSCAN + dimension filter from Task 1.
2. **Oriented bounding box** — cluster-masked BEV points are fed to Shapely's `minimum_rotated_rectangle` to extract heading (yaw) and tight dimensions instead of axis-aligned boxes.
3. **Radar Doppler velocity** — per-cluster (vx, vy) are estimated from the bag's radial `velocity` field by correcting for ego motion and projecting onto the ego x/y axes.
4. **EKF + Hungarian fusion** — master-clock LiDAR-front detections are matched with other-sensor detections within a ±50 ms window using Mahalanobis-gated Hungarian assignment, then refined with an EKF measurement update.
5. **Output** — `fused_detections.csv` and BEV plots per frame.

## Assumptions and Limitations

### Sensor Extrinsics

No calibration data exists in the bag (`/tf_static` has 0 messages; `/tf` only contains the `map → center_of_gravity` transform). All 7 sensor extrinsics are hardcoded based on typical racing-car geometry:

| Sensor       | Position (x fwd, y left, z up) | Mounting yaw |
|--------------|-------------------------------|--------------|
| lidar_front  | [+1.8, 0.0, +0.9]             | 0°           |
| lidar_left   | [+0.3, +0.8, +0.8]            | +90°         |
| lidar_right  | [+0.3, −0.8, +0.8]            | −90°         |
| radar_front  | [+2.0, 0.0, +0.4]             | 0°           |
| radar_back   | [−1.8, 0.0, +0.4]             | 180°         |
| radar_left   | [0.0, +0.8, +0.4]             | +90°         |
| radar_right  | [0.0, −0.8, +0.4]             | −90°         |

**Axis mappings** (confirmed from Task 1 empirical axis ranking):
- LiDAR: z = range/forward, x = lateral, y = vertical
- Radar: y = range/forward, x = lateral, z = vertical

These values are reasonable approximations for a DALLARA AV-21 but are not from ground-truth calibration. Position errors of 0.1–0.5 m in sensor mounting are plausible; they directly shift detected object positions in ego frame.

### Ego Velocity (Doppler Correction)

`ego.mat` is a MATLAB v7.3 HDF5 file read via `h5py`. The `stamp__tot` column is seconds elapsed since bag start (0–240 s); absolute ROS timestamps are reconstructed as `bag_start_ns + stamp_s × 10⁹`. Fields used: `vxEgo`, `vyEgo` (if present), otherwise `0.0`.

### Camera Intrinsics

No `camera_info` topics exist in the bag; all compressed image `frame_id` fields are empty. Camera `camera_fl` intrinsics are approximated assuming a 90° horizontal field of view:

```
fx = fy = width / (2 × tan(45°)) = 964 px
cx = 964,  cy = 250  (1928 × 500 image)
```

Camera mounting: t = [+1.5, +0.4, +0.8] m in ego frame, forward-facing. Camera-projection outputs (`verify_camera_projection_000.png`) are for sanity-checking only; the intrinsics are not used in the fusion pipeline.

### EKF Measurement Model

State vector: `[x, y, vx, vy, yaw, length, width]` (7-D).  
Observation matrix `H` selects `[x, y, vx, vy]`.  
Per-sensor diagonal measurement noise:

| Sensor | σ²_x, σ²_y (m²) | σ²_vx, σ²_vy (m²/s²) | Rationale |
|--------|----------------|----------------------|-----------|
| LiDAR  | 0.5            | 9.0                  | Good position, poor velocity |
| Radar  | 4.0            | 0.4                  | Poor position, good velocity (Doppler) |

### Known Limitations

1. **No sensor covariance calibration.** The R matrices are heuristic. A proper calibration against ground-truth tracks would improve position weighting.

2. **Radar sees track barriers.** Front radar detections at ~19 m lateral are barrier reflections at 60–90 m range — within the sensor's FoV but outside any realistic vehicle corridor. These fall outside the Mahalanobis gate (distance ≈ 9 σ) and are correctly rejected. A dedicated roadway filter (reject any ego-frame detection with |y_ego| > 10 m) would clean this up.

3. **No temporal tracking.** Each master-clock frame is fused independently; there is no inter-frame track management or Kalman prediction step. Adding a multi-frame tracker (e.g., SORT) is the natural next step.

4. **lidar_left / lidar_right produce zero detections.** Both sensors see the continuous armco barrier wall (a single large cluster that fails the DBSCAN vehicle-dimension filter). A separate "barrier segmentation" pass would be needed to make use of these sensors.

5. **Yaw from oriented bbox is ±π ambiguous.** The `minimum_rotated_rectangle` long-axis direction is resolved using the Doppler velocity direction when speed > 1 m/s. Stationary or slow detections retain the geometric yaw without disambiguation.

6. **Doppler correction approximation.** The radial-to-vector projection is computed point-by-point using hardcoded extrinsic rotations and aggregated via confidence-weighted mean. Without ground-truth ego velocity from GPS/IMU, the correction introduces errors proportional to ego speed uncertainty.

## Results Summary

Running on the first 5 LiDAR-front frames (out of 4800 total), the pipeline produces:

| Metric | Value |
|--------|-------|
| Raw detections (all sensors) | 44 across 7 sensors × 5 frames |
| LiDAR-front detections | 27 |
| Fused rows in CSV | 27 |
| Cross-sensor matches | 0 (radar-front detections are barrier returns) |
| BEV plots saved | 2 |

The 27 LiDAR-front detections are at 30–95 m forward, ±6 m lateral — consistent with vehicles on a racing straight ahead of the ego car.

## Output Files

| File | Description |
|------|-------------|
| `fused_detections.csv` | One row per fused detection. Columns: timestamp_ns, master_topic, detection_id, center_x, center_y, vx, vy, yaw, length, width, cov_xx, cov_yy, cov_vxvx, cov_vyvy, confidence, sensors, n_lidar, n_radar |
| `fusion_bev_NNN.png` | BEV plot: raw points (per-sensor colour), oriented fused boxes (pink), velocity arrows (pink), heading arrows (orange) |
| `verify_lidar_overlay_000.png` | 3-LiDAR ego-frame overlay (Phase 2 sanity check) |
| `verify_camera_projection_000.png` | camera_fl frame with projected LiDAR-front detections (Phase 5 sanity check) |

## Reproducing Results

```bash
# Full pipeline (default 20 frames, all 7 sensors)
uv run python scripts/run_task2.py --bag-dir data --max-frames 20 --output-dir outputs/task2

# Quick sanity check (5 frames, 2 BEV plots)
uv run python scripts/run_task2.py --bag-dir data --max-frames 5 --bev-frames 2

# Inspect bag only
uv run python scripts/run_task2.py --bag-dir data --inspect-only
```
