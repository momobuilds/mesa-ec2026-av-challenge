# Task 2 Multi-Sensor Fusion Report

## Bag Inspection

- Bag directory: `data`
- Duration: 239.949 s
- Start timestamp: 1759902345780410516

## Implementation Summary

- LiDAR clusters provide geometric position and oriented box evidence.
- Radar clusters are association candidates and provide Doppler velocity when they pass the Mahalanobis gate.
- Camera data is used through projection gating: a candidate must project into the selected camera image and have a nearby camera timestamp to be camera-validated.
- A lightweight temporal filter assigns `track_id`, smooths position, estimates fallback velocity from frame-to-frame motion, and emits short prediction rows through brief missed detections.

## Per-Sensor Detection

| Sensor | Frames | Raw detections |
| --- | ---: | ---: |
| radar_left | 20 | 8 |
| lidar_right | 20 | 0 |
| radar_front | 20 | 34 |
| radar_right | 20 | 7 |
| radar_back | 20 | 18 |
| lidar_front | 20 | 130 |
| lidar_left | 20 | 0 |
| **Total** |  | **197** |

## Output Metrics

| Metric | Value |
| --- | ---: |
| fused_rows | 168 |
| frames_with_rows | 20 |
| nonzero_velocity_rows | 152 |
| radar_associated_rows | 0 |
| camera_validated_rows | 168 |
| temporal_prediction_rows | 38 |
| max_sources_per_row | 2 |
| low_confidence_lt_0.50 | 24 |

## Velocity Sources

| Source | Rows |
| --- | ---: |
| temporal | 114 |
| prediction | 38 |
| initialized | 16 |

## Camera Validation

- Camera topic: `/sensor/camera/camera_fl/image/compressed`
- Indexed camera timestamps: 80
- Camera validation is geometric projection gating, not an image neural-network detector; no camera calibration topics were present.

## Known Limitations

- Radar associations are only used when spatially plausible; barrier/self-reflection radar detections are intentionally rejected.
- Camera contribution is FOV/timestamp validation because no camera object detector or `camera_info` topics are available in this repo.
- Velocity is ego-frame relative velocity. It uses radar Doppler when matched, otherwise temporal frame differences.
- This remains lighter than Task 3 tracking: tracks are short-lived and intended for Task 2 robustness, not full lifecycle evaluation.

## Outputs

- CSV: `outputs/task2/fused_detections.csv`
- Visualisation: `outputs/task2/fusion_bev_000.png`
- Visualisation: `outputs/task2/fusion_bev_001.png`
- Visualisation: `outputs/task2/fusion_bev_002.png`
- Visualisation: `outputs/task2/fusion_bev_003.png`
- Visualisation: `outputs/task2/fusion_bev_004.png`
- Visualisation: `outputs/task2/verify_lidar_overlay_000.png`
- Visualisation: `outputs/task2/verify_camera_projection_000.png`

## Reproducing Results

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 20 --bev-frames 5 --output-dir outputs/task2
uv run python scripts/validate_task2_outputs.py --output-dir outputs/task2
```
