# MESA EC2026 AV Challenge

## Task 1: Opponent Vehicle Detection

This repository includes a first-pass Task 1 pipeline that inspects the ROS 2
bag metadata, reads a limited number of standard `sensor_msgs/msg/PointCloud2`
messages directly from the sqlite3 `.db3` files, clusters BEV points with
DBSCAN, and writes candidate opponent detections.

The code is designed to run without a full ROS 2 installation.

### Task 1 — Run

```powershell
uv run python scripts/run_task1.py --bag-dir data --max-frames 20 --output-dir outputs/task1
```

To inspect the bag without running detection:

```powershell
uv run python scripts/run_task1.py --bag-dir data --inspect-only
```

To force a specific point cloud topic:

```powershell
uv run python scripts/run_task1.py --bag-dir data --topic /sensor/radar_front/points --max-frames 20 --output-dir outputs/task1
```

### Task 1 — Outputs

The default run writes:

- `outputs/task1/detections.csv`
- `outputs/task1/evaluation_summary.csv`
- `outputs/task1/frame_000.png`, `frame_001.png`, ...
- `outputs/task1/task1_report.md`

The CSV columns include timestamp, detection id, BEV center position, axis-aligned
bounding box limits, dimensions, point count, and a simple confidence score.

The evaluation summary maps the generated outputs to the challenge criteria:
accuracy, robustness, and false positive reduction. True precision/recall,
position error, and IoU require ground-truth opponent annotations; until those
are available, the report includes proxy run metrics and the filtering checks
used to control false positives.

### Task 1 — Notes

- The default topic is `/sensor/lidar_front/points` when available.
- Ground-plane filtering is on by default; disable with `--no-ground-filter`.
- If a future bag uses custom message types, add a parser for the topic/message
  type listed in `outputs/task1/task1_report.md`.

---

## Task 2: Multi-Sensor Fusion

Extends Task 1 across all 7 PointCloud2 sensors (3 LiDARs + 4 radars) with:

- Ego-frame coordinate unification via hardcoded sensor extrinsics
- Oriented minimum-area bounding boxes (Shapely)
- Radar Doppler velocity extraction with ego-motion correction
- EKF + Hungarian-matching fusion (master clock: lidar_front)
- BEV plots and fused detections CSV

### Task 2 — Run

```bash
uv run python scripts/run_task2.py --bag-dir data --max-frames 20 --output-dir outputs/task2
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
- `outputs/task2/task2_report.md` — full assumptions, limitations, and results

### Task 2 — Notes

- Sensor extrinsics are hardcoded approximations (no calibration data in bag).
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
