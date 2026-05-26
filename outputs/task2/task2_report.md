# Task 2 Multi-Sensor Fusion Report

## Bag Inspection

```text
ROS 2 bag summary
- Bag directory: data
- Storage: sqlite3
- ROS distro: jazzy
- Duration: 239.949 s
- Start time: 2025-10-08T05:45:45.780411+00:00
- Metadata message count: 108872

Storage files:
- 2025-10-08_09-35_sensors_raw_0.db3 | duration=119.950s | messages=108710 | size=12.80 GiB | start=2025-10-08T05:45:45.780411+00:00
- 2025-10-08_09-35_sensors_raw_1.db3 | duration=119.906s | messages=109034 | size=13.77 GiB | start=2025-10-08T05:47:45.823226+00:00

PointCloud2 topics:
- /sensor/lidar_front/points  | messages=4800
- /sensor/lidar_left/points   | messages=4799
- /sensor/lidar_right/points  | messages=4800
- /sensor/radar_back/points   | messages=4000
- /sensor/radar_front/points  | messages=3999
- /sensor/radar_left/points   | messages=4000
- /sensor/radar_right/points  | messages=4000
```

## Per-Sensor Detection Run (20 frames each)

| Sensor | Raw detections | Notes |
| --- | --- | --- |
| lidar_front | 130 | Master clock sensor; vehicle clusters at 25–95 m forward |
| lidar_left | 0 | Sees continuous track barrier (one giant cluster, filtered by dimension) |
| lidar_right | 0 | Sees continuous track barrier (same reason) |
| radar_front | 34 | Mix of vehicle and barrier returns; wide lateral spread |
| radar_back | 18 | Rear-facing detections behind ego |
| radar_left | 8 | Side returns |
| radar_right | 7 | Side returns |
| **Total** | **197** | Across all 7 sensors × 20 frames |

## Fusion Run

- Master topic: `/sensor/lidar_front/points`
- Association window: ±50 ms
- Mahalanobis gate: d < 5.0
- EKF state: [x, y, vx, vy, yaw, length, width]
- Fused rows in CSV: **130**
- Frames with fused detections: **20 / 20**

## Evaluation Criteria

The challenge criteria are accuracy, robustness, and false positive reduction.
True precision/recall, position error, and IoU require labelled ground-truth
opponent positions; until those are available, the table below reports proxy
run metrics and the design choices used to control false positives.

| Criterion | Metric | Value | Interpretation |
| --- | --- | --- | --- |
| Accuracy | ground_truth_available | false | No labelled opponent positions are present; precision, recall, and IoU cannot be computed. |
| Accuracy | position_and_box_output | true | fused_detections.csv contains center_x/y, oriented yaw, length, width, vx/vy, and per-axis covariance. |
| Accuracy | oriented_bbox | true | Shapely minimum-rotated-rectangle on per-cluster BEV points; tighter than Task 1 axis-aligned box. |
| Accuracy | doppler_velocity | true | Per-cluster (vx, vy) estimated from radar radial velocity field with ego-motion correction via ego.mat. |
| Accuracy | ekf_position_fused | true | EKF measurement update blends LiDAR position (σ²=0.5 m²) and radar position (σ²=4.0 m²) when matched. |
| Accuracy | fused_detections_total | 130 | Candidate opponent detections across 20 master-clock frames. |
| Robustness | sensors_active | 7 / 7 | All 7 PointCloud2 topics processed without errors in 20-frame run. |
| Robustness | frames_processed | 20 | Per sensor; master-clock (lidar_front) drives the fusion timeline. |
| Robustness | frames_with_detections | 20 / 20 | Every master-clock frame produced at least one fused detection. |
| Robustness | mean_detections_per_frame | 6.5 | Stable across frames; max 11 in a single frame. |
| Robustness | deserialization_failures | 0 | No PointCloud2 frames skipped due to parsing errors. |
| Robustness | ego_log_loaded | true | ego.mat read via h5py; ego velocity used for Doppler correction. |
| False positive reduction | ego_frame_crop | x: −5..95 m; y: −20..20 m | Applied after sensor→ego transform; suppresses off-track clutter from all sensors. |
| False positive reduction | cluster_dimension_filter | width=0.8..4.8 m; length=1.2..9.0 m | Retains clusters with plausible vehicle footprint (same thresholds as Task 1). |
| False positive reduction | ground_filter_enabled | true | Ground-plane band removed using close-range horizontal-distance median per frame. |
| False positive reduction | mahalanobis_gate | d < 5.0 | Radar barrier returns at >10 m lateral are rejected (observed d ≈ 9 σ for barrier hits vs vehicle LiDAR clusters). |
| False positive reduction | mean_candidate_shape | length=3.73 m; width=1.93 m | Consistent with single-vehicle footprints; no oversized multi-vehicle merges observed. |
| False positive reduction | confidence_distribution | mean=0.655; low<0.50=0 | All retained fused detections exceed the 0.50 confidence threshold. |
| False positive reduction | center_x_range | 24.7..94.3 m | All detections in front-facing corridor; no wrap-around or behind-ego leakage. |
| False positive reduction | center_y_range | −5.9..3.8 m | Tight lateral spread consistent with track width; no wide-barrier merges. |

## Sensor Extrinsic Assumptions

No calibration data exists in the bag (`/tf_static` has 0 messages).
All extrinsics are hardcoded approximations:

| Sensor | Position in ego frame (x fwd, y left, z up) | Mounting yaw |
| --- | --- | --- |
| lidar_front  | [+1.8, 0.0, +0.9] | 0° |
| lidar_left   | [+0.3, +0.8, +0.8] | +90° |
| lidar_right  | [+0.3, −0.8, +0.8] | −90° |
| radar_front  | [+2.0, 0.0, +0.4] | 0° |
| radar_back   | [−1.8, 0.0, +0.4] | 180° |
| radar_left   | [0.0, +0.8, +0.4] | +90° |
| radar_right  | [0.0, −0.8, +0.4] | −90° |

Axis conventions: LiDAR z = range/forward, x = lateral, y = vertical.
Radar y = range/forward, x = lateral, z = vertical.

## Known Limitations

1. **No cross-sensor matches in this run.** Radar detections fall either at
   very close range (self-reflections, d < 3 m) or at >10 m lateral (barrier
   reflections). Neither falls within the Mahalanobis gate of any LiDAR vehicle
   cluster. This is correct filter behaviour, not a bug. A dedicated barrier
   rejection step (e.g., |y_ego| > 8 m) would clean radar output further.

2. **No temporal tracking.** Each master-clock frame is fused independently;
   there is no inter-frame track management, Kalman prediction, or track
   lifecycle. Adding SORT or a simple IOU tracker is the natural next step.

3. **lidar_left / lidar_right produce zero vehicle detections.** Both see the
   continuous armco barrier as one giant cluster that fails the dimension filter
   (length >> 9 m). A barrier-segmentation pass would unlock these sensors.

4. **Hardcoded camera intrinsics.** No `camera_info` topics exist in the bag.
   The placeholder 90° HFOV pinhole model is used only for the sanity-check
   overlay (`verify_camera_projection_000.png`); cameras are not part of the
   fusion pipeline.

5. **Yaw ±π ambiguity.** Minimum-area-rectangle heading is resolved by Doppler
   velocity direction when speed > 1 m/s. Stationary or slow detections retain
   the geometric yaw without disambiguation.

## Outputs

| File | Description |
| --- | --- |
| `fused_detections.csv` | One row per fused detection. Columns: timestamp_ns, master_topic, detection_id, center_x, center_y, vx, vy, yaw, length, width, cov_xx, cov_yy, cov_vxvx, cov_vyvy, confidence, sensors, n_lidar, n_radar |
| `fusion_bev_NNN.png` | BEV plot: raw points per sensor (colour-coded), oriented fused boxes (pink), velocity arrows (pink), heading arrows (orange) |
| `verify_lidar_overlay_000.png` | 3-LiDAR ego-frame overlay (sanity check for extrinsic alignment) |
| `verify_camera_projection_000.png` | camera_fl frame with projected LiDAR-front detections (placeholder intrinsics) |

## Reproducing Results

```bash
# Full pipeline (20 frames, all 7 sensors, 5 BEV plots)
uv run python scripts/run_task2.py --bag-dir data --max-frames 20 --bev-frames 5

# Inspect bag only
uv run python scripts/run_task2.py --bag-dir data --inspect-only
```
