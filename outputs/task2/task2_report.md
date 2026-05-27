# Task 2 Multi-Sensor Fusion Report

## Bag Inspection

- Bag directory: `data`
- Duration: 239.949 s
- Start timestamp: 1759902345780410516

## Implementation Summary

- LiDAR clusters provide geometric position and oriented bounding-box evidence.
- Radar clusters contribute Doppler velocity when spatially associated; the ego-motion
  component is removed using `ekf_vx`/`ekf_vy` from `ego.mat` (see *Sensor Extrinsics* below).
- Camera images are used only for projection/timestamp sanity checks because no camera detector
  or `camera_info` topics are available.
- A lightweight temporal filter assigns internal `track_id` values, estimates fallback velocity,
  and predicts through short missing-data gaps.
- The CSV is generated from this LiDAR/radar/temporal fusion pipeline; it does not read
  bag-provided object labels or image detections.

## Sensor Extrinsics (updated)

All sensor → ego (`center_of_gravity`) transforms are now derived from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, which runs automatically at module import. This resolves
the sensor orientation ambiguity discussed with the professor (the `lidar_front`/`lidar_left`/
`lidar_right` naming vs. physical mounting direction).

**URDF-derived positions (relative to `center_of_gravity`):**

| Sensor | x (fwd, m) | y (lat, m) | z (up, m) | Mounting yaw |
|---|---:|---:|---:|---:|
| lidar_front | +0.662 | 0.000 | +0.610 | −179.5° |
| lidar_left | +0.002 | +0.135 | +0.674 | −59.4° |
| lidar_right | +0.002 | −0.135 | +0.674 | +60.0° |
| radar_front | +0.408 | 0.000 | +0.720 | −89.5° |
| radar_back | −2.550 | 0.000 | +0.248 | +91.7° |
| radar_left | +0.294 | +0.230 | +0.644 | 0.0° |
| radar_right | +0.294 | −0.230 | +0.644 | −180.0° |

## Ego-Motion Strategy (professor-confirmed)

| Source | Used for |
|---|---|
| `/tf` in bag (`BagTFReader`) | Ego pose in world/map frame — drift-free, ~200 Hz |
| `ego.mat` (`EgoStateLog`) | **Doppler correction only** — removing ego-motion from radar radial velocity |

`ego.mat` velocities (`ekf_vx`, `ekf_vy`) are in the ego body frame. They are passed to
`radar_doppler.cluster_velocity()` to compute the ego-motion radial contribution at each
radar point and obtain target-relative Doppler velocity. They are **not** integrated for
ego pose — that is handled by `BagTFReader`.

## Per-Sensor Detection

| Sensor | Frames | Raw detections |
| --- | ---: | ---: |
| radar_left | 200 | 70 |
| lidar_right | 200 | 0 |
| radar_front | 200 | 767 |
| radar_right | 200 | 263 |
| radar_back | 200 | 140 |
| lidar_front | 200 | 1216 |
| lidar_left | 200 | 0 |
| **Total** |  | **2456** |

`lidar_left` and `lidar_right` produce zero detections. The URDF confirms these sensors are
mounted pointing rearward (yaw ≈ ±60° from backward direction), so opponent vehicles ahead of
the ego are outside their field of view in typical convoy racing scenarios.

## Output Metrics

| Metric | Value |
| --- | ---: |
| fused_rows | 1686 |
| frames_with_rows | 199 |
| nonzero_velocity_rows | 1599 |
| radar_associated_rows | 442 |
| camera_projection_rows | 1686 |
| temporal_prediction_rows | 470 |
| max_sources_per_row | 2 |
| low_confidence_lt_0.50 | 375 |

## Velocity Sources

| Source | Rows |
| --- | ---: |
| temporal | 700 |
| prediction | 470 |
| radar | 427 |
| initialized | 89 |

## Sensor Association Notes

- Radar returns are not forced into tracks. In this 200-frame run, radar clusters that fail the
  spatial gate are rejected, so `n_radar=0` rows are honest non-associations.
- Side LiDAR returns are accepted only when clusters pass the same vehicle-shape filters; long
  guardrail-like strips are rejected instead of being forced into false vehicle detections.
- Camera topic: `/sensor/camera/camera_fl/image/compressed`
- Indexed camera timestamps: 800
- Camera validation is projection gating only; it is not counted in `sensors` or `n_camera`
  because no image-object detector is available.
- Camera verification images are selected from timestamps with multiple observed, in-image
  projected fusion rows and label each marker with internal track ID, distance, source,
  and timestamp.

## Known Limitations

- Radar fusion depends on spatially plausible radar detections; current sample data does not
  provide radar matches to the LiDAR vehicle clusters.
- Camera projection uses approximate calibration and is reported only as a visual/timestamp
  sanity check; no 2-D camera boxes or ground-truth actor labels were found in the bag.
- Velocity is ego-frame relative velocity; radar Doppler is preferred when matched, otherwise
  temporal frame differences are used.
- ~~Approximate sensor extrinsics~~ — resolved: all sensor→ego transforms now come from
  `data/eva.urdf`. The left/right LiDAR orientation ambiguity is resolved by the URDF's
  explicit rpy values.

## Outputs

- CSV: `outputs/task2/fused_detections.csv`
- Visualisation: `outputs/task2/fusion_bev_000.png` … `fusion_bev_019.png`
- Visualisation: `outputs/task2/verify_lidar_overlay_000.png`
- Visualisation: `outputs/task2/verify_camera_projection_000.png` … `_004.png`

## Reproducing Results

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 200 \
    --bev-frames 20 --output-dir outputs/task2
python3 scripts/validate_task2_outputs.py --output-dir outputs/task2
```
