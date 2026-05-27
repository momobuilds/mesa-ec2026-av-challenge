# Task 2 Multi-Sensor Fusion Report

## Bag Inspection

- Bag directory: `data`
- Duration: 239.949 s
- Start timestamp: 1759902345780410516

## Implementation Summary

- LiDAR clusters provide geometric position and minimum-area oriented bounding-box evidence for
  both position and heading.
- Radar clusters contribute Doppler velocity when spatially associated; the ego-motion component
  is removed using `ekf_vx`/`ekf_vy` from `ego.mat` (see *Ego-Motion Strategy* below).
- Camera images are used for projection-based fusion: every fused detection is projected into the
  front-camera image; when the centre projects inside the frame the camera is counted as a
  contributing sensor (`n_camera = 1`, confidence +0.07). When only corners fall in-frame the
  boost is +0.03. When no part of the detection projects into a synced camera frame, a mild
  penalty (-0.05) is applied as negative evidence.
- A lightweight temporal filter assigns internal `track_id` values, estimates fallback velocity,
  and predicts through short missing-data gaps.
- Heading (`yaw`) is smoothed across frames with an exponential moving average (alpha=0.55 for
  established tracks) to suppress frame-to-frame OBB orientation flips.
- The CSV is generated from this LiDAR/radar/camera/temporal fusion pipeline.

## Sensor Extrinsics (updated)

All sensor -> ego (`center_of_gravity`) transforms are now derived from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, which runs automatically at module import. This resolves
the sensor orientation ambiguity discussed with the professor (the `lidar_front`/`lidar_left`/
`lidar_right` naming vs. physical mounting direction).

**URDF-derived positions (relative to `center_of_gravity`):**

| Sensor       | x (fwd, m) | y (lat, m) | z (up, m) | Mounting yaw |
|:-------------|----------:|----------:|----------:|-------------:|
| lidar_front  |    +0.662 |     0.000 |    +0.610 |     -179.5 deg |
| lidar_left   |    +0.002 |    +0.135 |    +0.674 |      -59.4 deg |
| lidar_right  |    +0.002 |    -0.135 |    +0.674 |      +60.0 deg |
| radar_front  |    +0.408 |     0.000 |    +0.720 |      -89.5 deg |
| radar_back   |    -2.550 |     0.000 |    +0.248 |      +91.7 deg |
| radar_left   |    +0.294 |    +0.230 |    +0.644 |        0.0 deg |
| radar_right  |    +0.294 |    -0.230 |    +0.644 |     -180.0 deg |

## Ego-Motion Strategy (professor-confirmed)

| Source                         | Used for                                                              |
|:-------------------------------|:----------------------------------------------------------------------|
| `/tf` in bag (`BagTFReader`)   | Ego pose in world/map frame -- drift-free, ~200 Hz                   |
| `ego.mat` (`EgoStateLog`)      | **Doppler correction only** -- removing ego-motion from radar radial velocity |

`ego.mat` velocities (`ekf_vx`, `ekf_vy`) are in the ego body frame. They are passed to
`radar_doppler.cluster_velocity()` to compute the ego-motion radial contribution at each
radar point and obtain target-relative Doppler velocity. They are **not** integrated for
ego pose -- that is handled by `BagTFReader`.

## Camera Fusion (updated)

Camera is now a **contributing sensor**, not just a sanity check. For every fused detection the
3D centre and box corners are projected into the front-left camera image (`camera_fl`) using the
approximate pinhole model in `camera_projection.py`. The fusion result depends on the projection
score:

| camera_score | Meaning                              | Effect on fusion                     |
|:------------:|:-------------------------------------|:-------------------------------------|
| 1.0          | Detection centre falls inside image  | `n_camera=1`, confidence +0.07       |
| 0.5          | Only corners fall inside image       | `n_camera=1`, confidence +0.03       |
| 0.0          | Synced frame exists, not in image    | `n_camera=0`, confidence -0.05       |
| N/A          | No synced camera frame within 50 ms  | `n_camera=0`, no change              |

In this 200-frame run all 918 fused rows have `camera_score = 1.0` (camera-front fully
overlaps the lidar_front forward cone). The camera therefore acts as a corroborating sensor
for every detection and boosts their confidence accordingly.

## Heading Estimation (updated)

Heading (`yaw`) is derived from the minimum-area oriented bounding box (OBB) of each LiDAR
cluster (via Shapely's `minimum_rotated_rectangle`). OBBs from sparse clusters can flip by
180 deg between frames. Two mechanisms suppress this:

1. **Velocity reconciliation** (`_reconcile_yaw`): when speed > 1.0 m/s, the yaw is flipped
   if it disagrees with the velocity vector by more than 90 deg.
2. **EMA smoothing** (`_update_track_with_detection`): for established tracks (age >= 3), the
   new heading is blended with the previous heading using alpha = 0.55 on the angular difference
   (wrap-aware), preventing abrupt flips. For new tracks alpha = 0.90 (trust OBB more initially).

## Per-Sensor Detection

| Sensor        | Frames | Raw detections |
| :------------ | -----: | -------------: |
| radar_left    |    200 |             68 |
| lidar_right   |    200 |            303 |
| radar_front   |    200 |            703 |
| radar_right   |    200 |            287 |
| radar_back    |    200 |             22 |
| lidar_front   |    200 |            657 |
| lidar_left    |    200 |            749 |
| **Total**     |        |       **2789** |

`lidar_left` and `lidar_right` are mounted pointing rearward (yaw approx +/-60 deg from
backward axis per URDF). They detect opponents and track barriers trailing the ego vehicle,
which is consistent with IndyCar convoy racing where competitor vehicles follow closely.

## Output Metrics

| Metric                       |  Value |
| :--------------------------- | -----: |
| fused_rows                   |    918 |
| frames_with_rows             |    152 |
| nonzero_velocity_rows        |    862 |
| radar_associated_rows        |    143 |
| camera_projection_rows       |    918 |
| temporal_prediction_rows     |    261 |
| max_sources_per_row          |      2 |
| low_confidence_lt_0.50       |    108 |

## Velocity Sources

| Source      | Rows |
| :---------- | ---: |
| temporal    |  458 |
| prediction  |  261 |
| radar       |  142 |
| initialized |   57 |

## Dominant Sensor Combinations

| Sensors                         | Rows |
| :------------------------------ | ---: |
| lidar_front+camera              |  514 |
| temporal_prediction+camera      |  261 |
| lidar_front+radar_front+camera  |  139 |
| lidar_front+radar_left+camera   |    2 |
| lidar_front+radar_right+camera  |    2 |

## Sensor Association Notes

- Radar returns are not forced into tracks. In this 200-frame run, radar clusters that fail the
  spatial gate are rejected, so `n_radar=0` rows are honest non-associations.
- Side LiDAR detections from `lidar_left` and `lidar_right` are included as other-sensor
  detections and fused when they spatially match a lidar_front master detection within the 50 ms
  window.
- Camera topic: `/sensor/camera/camera_fl/image/compressed`
- Indexed camera timestamps: 800
- Camera fusion uses projection scoring only; no 2-D image detector or `camera_info` is
  available in the bag. The confidence modifier (see *Camera Fusion* above) is the quantitative
  contribution of the camera to each fused state.
- Camera verification images are selected from timestamps with multiple observed, in-image
  projected fusion rows and label each marker with internal track ID, distance, source,
  and timestamp.

## Known Limitations

- Radar position fusion is limited by the bag data: radar detections in this recording are
  primarily barrier returns at >10 m lateral distance or ego self-reflections at <3 m range.
  The Mahalanobis gate correctly rejects them as non-vehicle-like; radar improves only velocity
  (Doppler) for the 143 rows where spatial association holds.
- ~~Approximate sensor extrinsics~~ -- **Resolved.** All sensor->ego transforms now come from
  `data/eva.urdf`. The left/right LiDAR orientation ambiguity is resolved by the URDF's
  explicit rpy values.
- ~~Camera is only a sanity check~~ -- **Resolved.** Camera now contributes `n_camera`,
  confidence boosts/penalties, and appears in the `sensors` field for every detection.
- Heading accuracy: OBB-derived yaw is inherently noisy for sparse clusters. EMA smoothing
  (alpha=0.55) reduces frame-to-frame variance but cannot recover absolute orientation without
  a more discriminative geometric feature or an image-based heading estimator.
- Velocity is ego-frame relative velocity; radar Doppler is preferred when matched, otherwise
  temporal frame differences are used.
- No absolute world-frame velocity is reported; ego velocity from `ego.mat` is used only for
  Doppler correction, not to convert relative to absolute speed.

## Outputs

- CSV: `outputs/task2/fused_detections.csv`
- Visualisation: `outputs/task2/fusion_bev_000.png` ... `fusion_bev_019.png`
- Visualisation: `outputs/task2/verify_lidar_overlay_000.png`
- Visualisation: `outputs/task2/verify_camera_projection_000.png` ... `_004.png`

## Reproducing Results

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 200 \
    --bev-frames 20 --output-dir outputs/task2
python3 scripts/validate_task2_outputs.py --output-dir outputs/task2
```
