# MESA EC2026 AV Challenge

## Task 1: Opponent Vehicle Detection

This repository includes a Task 1 pipeline that inspects the ROS 2 bag metadata,
reads standard `sensor_msgs/msg/PointCloud2` messages directly from the sqlite3
`.db3` files, uses the provided vehicle URDF to transform LiDAR points into the
ego/base frame, clusters BEV points with DBSCAN, and writes candidate opponent
detections plus label-free evaluation evidence.

The code is designed to run without a full ROS 2 installation.

### Task 1 — Run

```powershell
uv run python src/task1_detection.py --bag-path data --urdf-path data/eva.urdf --output-dir outputs/task1 --max-frames all
```

For a quick debug run without processing the full bag:

```powershell
uv run python scripts/run_task1.py --bag-dir data --urdf-path data/eva.urdf --debug-max-frames 20 --output-dir outputs/task1_debug
```

To inspect the bag without running detection:

```powershell
uv run python scripts/run_task1.py --bag-dir data --inspect-only
```

To force a specific point cloud topic:

```powershell
uv run python scripts/run_task1.py --bag-dir data --urdf-path data/eva.urdf --topic /sensor/lidar_front/points --debug-max-frames 20 --output-dir outputs/task1_debug
```

### Task 1 — Outputs

The default run writes:

- `outputs/task1/detections.csv`
- `outputs/task1/evaluation_summary.csv`
- `outputs/task1/evaluation_summary.json`
- `outputs/task1/detection_count_over_time.png`
- `outputs/task1/frame_000.png`, `frame_001.png`, ...
- `outputs/task1/representative_frame_000_raw.png`, `_after_ego.png`, `_final.png`
- `outputs/task1/task1_report.md`

The CSV columns include timestamp, detection id, BEV center position, axis-aligned
bounding box limits, dimensions, point count, a simple confidence score, and
base-frame fields such as `sensor_frame`, `base_frame`, `center_x_base`,
`center_y_base`, and base-frame min/max box limits.

The evaluation summary maps the generated outputs to the challenge criteria:
accuracy, robustness, false positive reduction, and frame coverage. Because the
project still has no labelled opponent annotations, true precision/recall,
position error, and IoU are not reported. Instead, the JSON and markdown report
include total frames processed, detections-per-frame statistics, frames with
detections, DBSCAN cluster size statistics, bounding-box size statistics,
filter rejection counts/percentages for ground, ego/self, vehicle-size, and
point-count filters, before/after false-positive reduction evidence, sudden
jump/dropout checks, and the detection-count time plot.

For an additional honest pseudo-accuracy estimate, compare Task 1 detections
with temporal references:

```powershell
uv run python src/task1_pseudo_eval.py --detections outputs/task1/detections.csv --tracks outputs/task3/tracks.csv --output-dir outputs/task1
```

This writes `pseudo_accuracy_summary.json` using Task 3 confirmed tracks as a
pseudo-reference and `temporal_pseudo_accuracy_summary.json` using Task 1
temporal self-consistency. These are not labelled accuracy numbers.

### Task 1 — Notes

- The default topic is `/sensor/lidar_front/points` when available.
- `--max-frames all` processes all available LiDAR frames. Use
  `--debug-max-frames N` for fast checks.
- URDF transforms are required for geometry-backed Task 1 evidence. If a sensor
  frame cannot be resolved, the script fails clearly unless
  `--allow-identity-transform` is passed for debugging.
- The URDF references the ego mesh but does not provide exact mesh dimensions,
  so the self filter uses configurable defaults:
  `ego_length=5.0 m`, `ego_width=2.2 m`, `ego_height=1.6 m`,
  `safety_margin=0.25 m`.
- Ground-plane filtering is on by default; disable with `--no-ground-filter`.
- If a future bag uses custom message types, add a parser for the topic/message
  type listed in `outputs/task1/task1_report.md`.

---

## Task 2: Multi-Sensor Fusion

Extends Task 1 across all 7 PointCloud2 sensors (3 LiDARs + 4 radars) with:

- Ego-frame coordinate unification via hardcoded sensor extrinsics
- Oriented minimum-area bounding boxes (Shapely)
- Radar Doppler velocity extraction when spatial association is plausible
- Projection-based camera validation using compressed camera images
- Lightweight temporal filtering with `track_id`, velocity, heading, and short-gap prediction
- BEV plots, camera/lidar verification images, and fused detections CSV

### Task 2 — Run

Quick validation run:

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 20 --bev-frames 5 --output-dir outputs/task2
python3 scripts/validate_task2_outputs.py --output-dir outputs/task2
```

Full available-frame run for submission artifacts:

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 4800 --bev-frames 20 --output-dir outputs/task2
python3 scripts/validate_task2_outputs.py --output-dir outputs/task2
```

To inspect the bag without running detection:

```bash
uv run python scripts/run_task2.py --bag-dir data --inspect-only
```

To process a subset of sensors:

```bash
uv run python scripts/run_task2.py --sensors /sensor/lidar_front/points /sensor/radar_front/points
```

### Task 2 — Outputs

- `outputs/task2/fused_detections.csv` — one row per fused detection per master timestamp
- `outputs/task2/fusion_bev_NNN.png` — BEV plots (raw points + oriented boxes + velocity arrows)
- `outputs/task2/verify_lidar_overlay_000.png` — regenerated LiDAR/frame-level verification view
- `outputs/task2/verify_camera_projection_000.png` — regenerated camera projection verification view
- `outputs/task2/task2_report.md` — full assumptions, limitations, and results

### Task 2 — Notes

- Sensor extrinsics are hardcoded approximations (no calibration data in bag).
- Camera support is projection/timestamp validation, not a learned image detector.
- lidar_left and lidar_right see track barriers only; they produce no vehicle detections.
- Radar detections at >10 m lateral are barrier reflections; they fall outside the
  Mahalanobis association gate and are correctly excluded from fused output.

---

## Task 3: Continuous Target Tracking

Adds a multi-object tracker on top of Task 2's per-frame sensor pipeline.
Outputs persistent track IDs that survive short occlusions (coasting) and are
never reused after deletion.

Key features:

- Constant-velocity EKF predict step with **ego-motion compensation** in world frame
- Two-stage Mahalanobis-gated Hungarian assignment (confirmed→LiDAR, then confirmed→radar)
- Track lifecycle: TENTATIVE → CONFIRMED → COASTING → DELETED
- Barrier rejection: new seeds with `|y_ego| > 8 m` are never promoted
- `--simulate-occlusion N` flag to demonstrate coasting recovery

### Task 3 — Run

```bash
uv run python scripts/run_task3.py --bag-dir data --max-frames 60 --output-dir outputs/task3
```

Occlusion recovery demo (5-frame blank at frame 20):

```bash
uv run python scripts/run_task3.py --bag-dir data --max-frames 60 \
    --simulate-occlusion 5 --occlusion-start 20
```

Motion model unit tests:

```bash
uv run python -m src.motion_model
```

### Task 3 — Outputs

- `outputs/task3/tracks.csv` — one row per (track_id, timestamp); persistent IDs
- `outputs/task3/tracks_summary.csv` — per-track: lifetime, path length, mean/max speed
- `outputs/task3/tracking_bev_NNN.png` — BEV plots with trails, oriented boxes, velocity arrows
- `outputs/task3/tracking_timeline.png` — Gantt chart: observed (solid) vs coasting (hatched)
- `outputs/task3/task3_report.md` — motion-model derivation, gating choices, metrics, limitations

### Task 3 — Notes

- Ego-motion compensation uses `ego.mat` via `EgoStateLog`; falls back to zero ego velocity if absent.
- LiDAR provides position; radar provides Doppler velocity. Cross-sensor EKF update is architecture-ready
  but radar positions don't spatially match LiDAR vehicle clusters in this bag (barrier returns).
- Tasks 1 and 2 are unchanged; `scripts/run_task1.py` and `scripts/run_task2.py` work as before.

---

## Task 4: Environment Mapping (Bonus)

Builds a probabilistic occupancy grid from the three LiDAR sensors,
accumulated in the world frame with dynamic-object masking from Task 3.

### Task 4 — Run

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 60 \
    --output-dir outputs/task4
```

Full-bag run (240 s, needs larger grid):

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 9999 \
    --grid-extent 700 --output-dir outputs/task4_full
```

Skip animation for speed: `--no-animation`

### Task 4 — Outputs

- `outputs/task4/world_map_static.png` — final occupancy map with boundary polylines
- `outputs/task4/world_map_with_trajectory.png` — same + ego world-frame trajectory
- `outputs/task4/track_boundaries.geojson` — inner + outer barrier polylines, world frame
- `outputs/task4/ego_view_NNN.png` — ego-centred 100×60 m views at 8 evenly-spaced timestamps
- `outputs/task4/map_animation.gif` — map filling in as ego drives (30 frames, 0.2 MB)
- `outputs/task4/task4_report.md` — methodology, parameters, and quantitative results

### Task 4 — Notes

- Default grid: ±300 m, 0.25 m/cell → 2 400² cells (22 MB). Use `--grid-extent 700` for full-bag.
- Dynamic-object masking requires `outputs/task3/tracks.csv` from Task 3. Skip with `--no-mask-tracks`.
- Full-bag trajectory spans 671 m × 1 122 m; the 60-frame run covers ~240 m of the circuit.

---

## Task 5: Trajectory Prediction (Bonus)

Predicts future opponent trajectories from Task 3 confirmed tracks at
horizons of 0.5, 1.0, and 2.0 s. Two motion models — Constant Velocity (CV)
and Constant Turn Rate + Velocity (CTRV) — with full covariance propagation
and a self-evaluation harness that computes FDE, lateral/longitudinal error,
and NLL against actual future track positions.

### Task 5 — Run

```bash
uv run python scripts/run_task5.py \
    --tracks outputs/task3/tracks.csv \
    --bag-dir data \
    --output-dir outputs/task5
```

Skip ground-truth evaluation (useful for tracks without future frames):

```bash
uv run python scripts/run_task5.py --tracks outputs/task3/tracks.csv --bag-dir data --no-eval
```

### Task 5 — Outputs

- `outputs/task5/predictions.csv` — every `(source_ts, track_id, horizon, model)` prediction
- `outputs/task5/prediction_errors.csv` — per-prediction FDE, lateral, longitudinal, NLL
- `outputs/task5/prediction_summary.csv` — aggregated mean/median/p95 per (model, horizon)
- `outputs/task5/prediction_bev_NNN.png` — BEV frames with fading trajectories + uncertainty ellipses
- `outputs/task5/prediction_error_hist.png` — FDE histograms CV vs CTRV per horizon
- `outputs/task5/manoeuvre_case_study.png` — zoom on most curved track
- `outputs/task5/task5_report.md` — methodology, math, comparison table, limitations

---

## Task 6: Behaviour Prediction

Classifies tracked opponent behaviour using Task 3 track histories, Task 4
boundary context, and Task 5 prediction evidence. The classifier is
rule-based and explainable because no labelled behaviour dataset is provided.
It assigns `lane_keeping`, `braking`, `overtaking_attempt`,
`trajectory_recovery`, or `unknown`, with confidence scores and per-frame
explanations.

### Task 6 — Run

```bash
uv run python src/task6_behavior_prediction.py \
    --tracks outputs/task3/tracks.csv \
    --predictions outputs/task5/predictions.csv \
    --prediction-errors outputs/task5/prediction_errors.csv \
    --map-dir outputs/task4 \
    --output-dir outputs/task6
```

### Task 6 — Outputs

- `outputs/task6/behaviour_predictions.csv` — per-frame behaviour labels, scores, confidence, and explanation
- `outputs/task6/track_behaviour_segments.csv` — consecutive same-behaviour intervals per track
- `outputs/task6/behaviour_summary.csv` — class counts, percentages, confidence, and input notes
- `outputs/task6/behaviour_timeline.png` — behaviour labels over time by track
- `outputs/task6/behaviour_bev.png` — BEV behaviour visualization
- `outputs/task6/track_*_behaviour_signals.png` — signal plots for interesting tracks
- `outputs/task6/README.md` — method report and limitations

### Task 6 — Notes

- Temporal hysteresis reduces frame-to-frame label flicker.
- Task 5 prediction disagreement contributes evidence for manoeuvre-like behaviour.
- Unknown and low-confidence tracks are intentionally conservative so Task 7 can use larger safety margins.

---

## Task 7: Overtake Strategy / Planning

Implements a hybrid overtake planner for collision avoidance, evasive
trajectories, racing-line adaptation, and explainable strategy selection. The
core planner is deterministic and safety-constrained; it samples short-horizon
candidate manoeuvres and rejects unsafe options with hard constraints. A
lightweight learned risk model then ranks safe candidates using pseudo-labels
generated from deterministic risk metrics.

Candidate strategies include:

- `keep_racing_line`
- `follow_opponent`
- `brake`
- `evasive_left`
- `evasive_right`
- `overtake_left`
- `overtake_right`
- `abort_overtake`
- `emergency_brake`

The innovation layer uses a `RandomForestRegressor` when scikit-learn is
available. It does not replace the safety planner: unsafe candidates are never
selected because of a learned score.

### Task 7 — Run

Default run:

```bash
uv run python src/task7_overtake_planning.py
```

Explicit run:

```bash
uv run python src/task7_overtake_planning.py \
    --tracks outputs/task3/tracks.csv \
    --predictions outputs/task5/predictions.csv \
    --behaviour outputs/task6/behaviour_predictions.csv \
    --map-dir outputs/task4 \
    --output-dir outputs/task7
```

Disable the learned ranking layer for a deterministic-only comparison:

```bash
uv run python src/task7_overtake_planning.py --no-learning
```

### Task 7 — Outputs

- `outputs/task7/planning_decisions.csv` — selected strategy, confidence, safety margin, risk scores, and explanation per frame
- `outputs/task7/planned_trajectories.csv` — sampled candidate and selected ego trajectories over the planning horizon
- `outputs/task7/risk_assessment.csv` — deterministic and learned risk metrics for every candidate
- `outputs/task7/planning_summary.csv` — frame counts, strategy distribution, safety margins, and input notes
- `outputs/task7/strategy_timeline.png` — selected strategy over time
- `outputs/task7/planning_bev.png` — BEV plot of selected plans, opponent tracks, and corridor context
- `outputs/task7/overtake_scenarios.png` — representative following, evasive, overtake, abort, and emergency scenarios
- `outputs/task7/README.md` — planner method report
- `outputs/task7/learned_risk_training_data.csv` — pseudo-labelled candidate table
- `outputs/task7/learned_risk_scores.csv` — learned risk scores and final costs
- `outputs/task7/learning_model_summary.csv` — model type, feature count, scores, fallback reason, and top features
- `outputs/task7/learning_feature_importance.png` — feature importance plot when supported

### Task 7 — Latest Generated Summary

The current generated `outputs/task7` run used Task 3-6 outputs plus
`data/ego.mat` and loaded 197 planning frames:

- 1,773 candidate risk assessments
- 21 selected overtake attempts
- 4 aborted overtakes
- 31 emergency-brake selections
- Mean selected safety margin: 4.15 m
- Minimum selected safety margin: -4.91 m in emergency/fallback cases
- Learned model: `RandomForestRegressor`
- Learning validation score: 0.987
- Top learned risk feature: `collision_risk_score`

### Task 7 — Notes

- Task 3 and Task 5 outputs are ego-frame, while Task 4 boundaries are world-frame. The planner therefore uses an inferred local racing corridor for hard constraints and uses Task 4 GeoJSON boundaries as visual/context evidence.
- Behaviour labels from Task 6 adjust margins: `unknown` and `trajectory_recovery` block overtaking, while `lane_keeping` and clear side clearance can unlock overtake candidates.
- A clean-overtake bonus rewards safe side passes only after hard safety checks are satisfied.
- If Task 5 predictions are missing, Task 7 falls back to constant-velocity predictions from Task 3 tracks.
- If scikit-learn is unavailable or the pseudo-labelled table is too small, Task 7 writes the learning files and runs deterministic scoring only.

---

## Verification

Task 7 focused tests:

```bash
uv run --with pytest python -m pytest tests/test_task7_overtake_planning.py -q
```

Current result: `6 passed`.

Full test suite:

```bash
uv run --with pytest python -m pytest -q
```

At the time Task 7 was added, the full suite had one unrelated Task 2 camera
validation assertion failure in `tests/test_task2_fusion.py`; Task 7 tests pass.

---

## Demo Video: Tasks 1 to 7

The demo video pipeline composites the existing Task 1-7 PNG/CSV outputs into
a recorded MP4 without rerunning or rewriting the core algorithms. It adds
timestamps, task labels, detection/fusion/tracking summaries, behaviour labels,
predicted opponent trajectories, and the selected Task 7 overtake plan.

Default run:

```bash
uv run python demo/generate_demo_video.py
```

Equivalent explicit run:

```bash
uv run python demo/generate_demo_video.py \
    --outputs-dir outputs \
    --bag data \
    --output demo/demo_task1_to_7.mp4
```

Optional raw camera panel from the ROS bag:

```bash
uv run python demo/generate_demo_video.py \
    --raw-source bag \
    --bag data \
    --camera-topic /sensor/camera/camera_fl/image/compressed
```

Useful controls:

```bash
uv run python demo/generate_demo_video.py --frames 80 --fps 8
uv run python demo/generate_demo_video.py --width 1280 --height 720
uv run python demo/generate_demo_video.py --output demo/custom_demo.mp4
```

Expected output:

- `demo/demo_task1_to_7.mp4`

The script is robust to missing artifacts: if a camera stream, CSV, rendered
task image, or planner output is unavailable, the affected panel shows a clear
placeholder or warning while the rest of the demo continues to render.
