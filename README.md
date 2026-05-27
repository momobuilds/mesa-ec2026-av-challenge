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
