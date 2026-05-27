# Task 2 Multi-Sensor Fusion Report

## Bag Inspection

- Bag directory: `data`
- Duration: 239.949 s
- Start timestamp: 1759902345780410516

## Implementation Summary

- LiDAR clusters provide geometric position and oriented bounding-box evidence.
- Radar clusters are processed and can contribute Doppler velocity when spatial association is plausible.
- Camera images are used only for projection/timestamp sanity checks because no camera detector or camera_info topics are available.
- A lightweight temporal filter assigns internal `track_id` values, estimates fallback velocity, and predicts through short missing-data gaps.
- The CSV is generated from this LiDAR/radar/temporal fusion pipeline; it does not read bag-provided object labels or image detections.

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

- Radar returns are not forced into tracks. In this 200-frame run, radar clusters that fail the spatial gate are rejected, so `n_radar=0` rows are honest non-associations.
- Side LiDAR returns are accepted only when clusters pass the same vehicle-shape filters; long guardrail-like strips are rejected instead of being forced into false vehicle detections.
- Camera topic: `/sensor/camera/camera_fl/image/compressed`
- Indexed camera timestamps: 800
- Camera validation is projection gating only; it is not counted in `sensors` or `n_camera` because no image-object detector is available.
- Camera verification images are selected from timestamps with multiple observed, in-image projected fusion rows and label each marker with internal track ID, distance, source, and timestamp.

## Known Limitations

- Radar fusion depends on spatially plausible radar detections; current sample data does not provide radar matches to the LiDAR vehicle clusters.
- Camera projection uses approximate calibration and is reported only as a visual/timestamp sanity check; no 2-D camera boxes or ground-truth actor labels were found in the bag.
- Velocity is ego-frame relative velocity; radar Doppler is preferred when matched, otherwise temporal frame differences are used.

## Outputs

- CSV: `outputs/task2/fused_detections.csv`
- Visualisation: `outputs/task2/fusion_bev_000.png`
- Visualisation: `outputs/task2/fusion_bev_001.png`
- Visualisation: `outputs/task2/fusion_bev_002.png`
- Visualisation: `outputs/task2/fusion_bev_003.png`
- Visualisation: `outputs/task2/fusion_bev_004.png`
- Visualisation: `outputs/task2/fusion_bev_005.png`
- Visualisation: `outputs/task2/fusion_bev_006.png`
- Visualisation: `outputs/task2/fusion_bev_007.png`
- Visualisation: `outputs/task2/fusion_bev_008.png`
- Visualisation: `outputs/task2/fusion_bev_009.png`
- Visualisation: `outputs/task2/fusion_bev_010.png`
- Visualisation: `outputs/task2/fusion_bev_011.png`
- Visualisation: `outputs/task2/fusion_bev_012.png`
- Visualisation: `outputs/task2/fusion_bev_013.png`
- Visualisation: `outputs/task2/fusion_bev_014.png`
- Visualisation: `outputs/task2/fusion_bev_015.png`
- Visualisation: `outputs/task2/fusion_bev_016.png`
- Visualisation: `outputs/task2/fusion_bev_017.png`
- Visualisation: `outputs/task2/fusion_bev_018.png`
- Visualisation: `outputs/task2/fusion_bev_019.png`
- Visualisation: `outputs/task2/verify_lidar_overlay_000.png`
- Visualisation: `outputs/task2/verify_camera_projection_000.png`
- Visualisation: `outputs/task2/verify_camera_projection_001.png`
- Visualisation: `outputs/task2/verify_camera_projection_002.png`
- Visualisation: `outputs/task2/verify_camera_projection_003.png`
- Visualisation: `outputs/task2/verify_camera_projection_004.png`

## Reproducing Results

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 200 --bev-frames 20 --output-dir outputs/task2
python3 scripts/validate_task2_outputs.py --output-dir outputs/task2
```
