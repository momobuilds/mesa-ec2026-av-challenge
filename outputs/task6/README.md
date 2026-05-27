# Task 6 - Behaviour Prediction

## Inputs Used

- Task 3 `tracks.csv`: position, velocity, heading/yaw, tracker status, confidence.
- Task 5 `predictions.csv`: CV/CTRV model disagreement and world-frame future points when available.
- Task 5 `prediction_errors.csv`: retrospective prediction error context when available.
- Task 4 `track_boundaries.geojson`: world-frame boundary overlay and future boundary distance when compatible with Task 5 world predictions.

## Features

- Sliding temporal window: 1.50 s.
- Minimum evidence points before non-unknown classification: 4.
- Speed, acceleration, heading, and yaw rate are inferred from positions when explicit columns are missing.
- Lateral displacement, longitudinal progress, curvature proxy, yaw peaks, speed change, and track-relative offset are computed per window.

## Rule Logic

- `lane_keeping`: stable heading, low yaw rate, low lateral displacement, and low acceleration.
- `braking`: sustained negative acceleration and speed drop with mostly stable heading.
- `overtaking_attempt`: lateral shift plus forward progress, maintained speed, yaw change, or Task 5 prediction disagreement.
- `trajectory_recovery`: previous yaw/offset instability followed by lower recent yaw and decreasing track-relative offset.
- `unknown`: too few points, low confidence, or weak rule evidence.

Temporal hysteresis is applied per track to reduce frame-by-frame flicker.

## Outputs

- `behaviour_predictions.csv`: per-frame scores, label, confidence, and explanation.
- `track_behaviour_segments.csv`: consecutive same-behaviour intervals.
- `behaviour_summary.csv`: counts, percentages, confidence, and missing-input notes.
- `behaviour_timeline.png`: behaviour class over time for each track.
- `behaviour_bev.png`: BEV track plot, with Task 4 boundary context when available.
- `track_*_behaviour_signals.png`: optional signal plots for interesting manoeuvre tracks.

## Limitations

- This is an explainable rule-based classifier because no labelled behaviour data is available.
- Task 3 tracks are ego-frame while Task 4 boundaries are world-frame; direct boundary distances use Task 5 world predictions rather than raw track positions.
- Short-lived or coasting tracks are intentionally assigned lower confidence.
- Thresholds are conservative to favor stable labels over noisy manoeuvre flicker.

## Run Command

```bash
uv run python src/task6_behavior_prediction.py --tracks outputs/task3/tracks.csv --predictions outputs/task5/predictions.csv --prediction-errors outputs/task5/prediction_errors.csv --map-dir outputs/task4 --output-dir outputs/task6
```

## Run Notes

- loaded tracks: 2752 rows from outputs\task3\tracks.csv
- loaded Task 5 predictions: 9000 rows
- loaded Task 5 prediction errors: 9000 rows
- loaded Task 4 boundaries: 12 polylines
- used_velocity_columns: True
- speed_note: inferred from vx/vy
- heading_note: used yaw
- acceleration_note: inferred from speed derivative
- yaw_rate_note: inferred from heading derivative
- confidence_note: used confidence
- status_note: used status
- prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used
- predictions_used: True
- prediction_errors_used: True
- used Task 4 boundaries through Task 5 world prediction distances

## Summary Snapshot

| number_of_tracks_analysed | number_of_frames_analysed | behaviour           | count | percentage | average_confidence | low_confidence_unknown_count | notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------- | ------------------------- | ------------------- | ----- | ---------- | ------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 95                        | 2752                      | lane_keeping        | 1199  | 43.568     | 0.511              | 928                          | loaded tracks: 2752 rows from outputs\task3\tracks.csv \| loaded Task 5 predictions: 9000 rows \| loaded Task 5 prediction errors: 9000 rows \| loaded Task 4 boundaries: 12 polylines \| used_velocity_columns: True \| speed_note: inferred from vx/vy \| heading_note: used yaw \| acceleration_note: inferred from speed derivative \| yaw_rate_note: inferred from heading derivative \| confidence_note: used confidence \| status_note: used status \| prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used \| predictions_used: True \| prediction_errors_used: True \| used Task 4 boundaries through Task 5 world prediction distances |
| 95                        | 2752                      | braking             | 465   | 16.897     | 0.629              | 928                          | loaded tracks: 2752 rows from outputs\task3\tracks.csv \| loaded Task 5 predictions: 9000 rows \| loaded Task 5 prediction errors: 9000 rows \| loaded Task 4 boundaries: 12 polylines \| used_velocity_columns: True \| speed_note: inferred from vx/vy \| heading_note: used yaw \| acceleration_note: inferred from speed derivative \| yaw_rate_note: inferred from heading derivative \| confidence_note: used confidence \| status_note: used status \| prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used \| predictions_used: True \| prediction_errors_used: True \| used Task 4 boundaries through Task 5 world prediction distances |
| 95                        | 2752                      | overtaking_attempt  | 444   | 16.134     | 0.418              | 928                          | loaded tracks: 2752 rows from outputs\task3\tracks.csv \| loaded Task 5 predictions: 9000 rows \| loaded Task 5 prediction errors: 9000 rows \| loaded Task 4 boundaries: 12 polylines \| used_velocity_columns: True \| speed_note: inferred from vx/vy \| heading_note: used yaw \| acceleration_note: inferred from speed derivative \| yaw_rate_note: inferred from heading derivative \| confidence_note: used confidence \| status_note: used status \| prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used \| predictions_used: True \| prediction_errors_used: True \| used Task 4 boundaries through Task 5 world prediction distances |
| 95                        | 2752                      | trajectory_recovery | 119   | 4.324      | 0.413              | 928                          | loaded tracks: 2752 rows from outputs\task3\tracks.csv \| loaded Task 5 predictions: 9000 rows \| loaded Task 5 prediction errors: 9000 rows \| loaded Task 4 boundaries: 12 polylines \| used_velocity_columns: True \| speed_note: inferred from vx/vy \| heading_note: used yaw \| acceleration_note: inferred from speed derivative \| yaw_rate_note: inferred from heading derivative \| confidence_note: used confidence \| status_note: used status \| prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used \| predictions_used: True \| prediction_errors_used: True \| used Task 4 boundaries through Task 5 world prediction distances |
| 95                        | 2752                      | unknown             | 525   | 19.077     | 0.164              | 928                          | loaded tracks: 2752 rows from outputs\task3\tracks.csv \| loaded Task 5 predictions: 9000 rows \| loaded Task 5 prediction errors: 9000 rows \| loaded Task 4 boundaries: 12 polylines \| used_velocity_columns: True \| speed_note: inferred from vx/vy \| heading_note: used yaw \| acceleration_note: inferred from speed derivative \| yaw_rate_note: inferred from heading derivative \| confidence_note: used confidence \| status_note: used status \| prediction_note: used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, so raw current-vs-future lateral subtraction is not used \| predictions_used: True \| prediction_errors_used: True \| used Task 4 boundaries through Task 5 world prediction distances |
