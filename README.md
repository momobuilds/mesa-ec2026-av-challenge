# MESA EC2026 AV Challenge

## Task 1: Opponent Vehicle Detection

This repository includes a first-pass Task 1 pipeline that inspects the ROS 2
bag metadata, reads a limited number of standard `sensor_msgs/msg/PointCloud2`
messages directly from the sqlite3 `.db3` files, clusters BEV points with
DBSCAN, and writes candidate opponent detections.

The code is designed to run without a full ROS 2 installation.

### Run

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

### Outputs

The default run writes:

- `outputs/task1/detections.csv`
- `outputs/task1/frame_000.png`, `frame_001.png`, ...
- `outputs/task1/task1_report.md`

The CSV columns include timestamp, detection id, BEV center position, axis-aligned
bounding box limits, dimensions, point count, and a simple confidence score.

### Notes

- The default topic is `/sensor/lidar_front/points` when available.
- The current detector processes one `PointCloud2` topic at a time.
- Multi-sensor fusion and ego-frame extrinsic calibration are left as future
  improvements.
- If a future bag uses custom message types, add a parser for the topic/message
  type listed in `outputs/task1/task1_report.md`.
