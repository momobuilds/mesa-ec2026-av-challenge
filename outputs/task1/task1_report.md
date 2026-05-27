# Task 1 Opponent Vehicle Detection Report

## Method

Task 1 reads standard `sensor_msgs/msg/PointCloud2` messages directly from the ROS 2 sqlite
bag, transforms LiDAR points into the URDF ego frame (`center_of_gravity`), removes ground/self
points, clusters the BEV points with DBSCAN, and retains clusters with vehicle-like support and
dimensions.

### Sensor Extrinsics (updated)

Sensor → ego transforms are now loaded from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, which runs automatically at module import. This replaces
the previous hardcoded approximations and resolves the sensor orientation uncertainty discussed
with the professor.

**URDF-derived sensor positions (relative to `center_of_gravity`):**

| Sensor | x (fwd, m) | y (lat, m) | z (up, m) | Mounting yaw |
|---|---:|---:|---:|---:|
| lidar_front | +0.662 | 0.000 | +0.610 | −179.5° |
| lidar_left | +0.002 | +0.135 | +0.674 | −59.4° |
| lidar_right | +0.002 | −0.135 | +0.674 | +60.0° |
| radar_front | +0.408 | 0.000 | +0.720 | −89.5° |
| radar_back | −2.550 | 0.000 | +0.248 | +91.7° |
| radar_left | +0.294 | +0.230 | +0.644 | 0.0° |
| radar_right | +0.294 | −0.230 | +0.644 | −180.0° |

The `lidar_left` and `lidar_right` mounting yaws of ≈±60° (not ±90° as previously assumed)
explain why these sensors saw limited forward coverage: they are positioned at the back of
the sensor ring and angle rearward, consistent with both the URDF and the professor's
clarification about the `front_left`/`back_right` naming ambiguity.

### Ego Pose in Map Frame (updated)

The bag's `/tf` topic (45 080 messages, ~200 Hz) publishes `map → center_of_gravity` transforms
directly. `BagTFReader` reads these and provides drift-free world-frame pose at any timestamp.
For Task 1 the ego pose is not needed during detection (all detections are in the ego frame),
but it is used by Tasks 3–5 for world-frame tracking and map building.

`ego.mat` velocity data (`EgoStateLog`) is retained **only for radar Doppler correction**:
removing the ego-motion component from each radar point's radial velocity.

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

Topics:
- /sensor/camera/camera_cl/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/camera/camera_cr/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4776 | serialization=cdr
- /sensor/camera/camera_fl/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/camera/camera_fr/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4775 | serialization=cdr
- /sensor/camera/camera_r/image/compressed  | type=sensor_msgs/msg/CompressedImage | messages=4643 | serialization=cdr
- /sensor/camera/camera_rl/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/camera/camera_rr/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/lidar_front/points  | type=sensor_msgs/msg/PointCloud2 | messages=4800 | serialization=cdr
- /sensor/lidar_left/points   | type=sensor_msgs/msg/PointCloud2 | messages=4799 | serialization=cdr
- /sensor/lidar_right/points  | type=sensor_msgs/msg/PointCloud2 | messages=4800 | serialization=cdr
- /sensor/radar_back/points   | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /sensor/radar_front/points  | type=sensor_msgs/msg/PointCloud2 | messages=3999 | serialization=cdr
- /sensor/radar_left/points   | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /sensor/radar_right/points  | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /tf        | type=tf2_msgs/msg/TFMessage | messages=45080 | serialization=cdr  ← map→CoG pose
- /tf_static | type=tf2_msgs/msg/TFMessage | messages=0     | serialization=cdr
```

## Detection Run

- Selected topic: `/sensor/lidar_front/points`
- Message type: `sensor_msgs/msg/PointCloud2`
- Frames processed: 200
- Detections written: 643
- URDF path: `data/eva.urdf`
- Base frame: `center_of_gravity`
- Extrinsic source: **URDF** (auto-loaded by `tf_utils._auto_init()`)
- Ego safety box defaults: length=5.00 m, width=2.20 m, height=1.60 m, margin=0.25 m

## URDF Use

The URDF provides the fixed transform from the LiDAR frame into the ego/base frame. BEV boxes
are reported in the base (`center_of_gravity`) frame using `center_x_base`, `center_y_base`,
and base-frame min/max columns.

- `center_of_gravity ← lidar_front` translation: [+0.662, +0.000, +0.610] m

The full kinematic chain in `eva.urdf` is:
```
center_of_gravity
  └─ front_axle_chassis  (+1.654 m fwd, −0.056 m up)
       └─ ring_center     (−1.322 m fwd, +0.666 m up)  ← all sensors attach here or to front_axle
            ├─ lidar_front  (+0.33 m fwd)  rpy=[0, −1.566, 3.15]
            ├─ lidar_left   (−0.33 m fwd, +0.135 m lat, +0.064 m up) rpy=[−π, −1.594, 2.105]
            ├─ lidar_right  (−0.33 m fwd, −0.135 m lat, +0.064 m up) rpy=[−π, −1.594, −2.094]
            ├─ radar_front  (+0.076 m fwd,  0.110 m up)  rpy=[0, 0, −1.561]
            ├─ radar_left   (−0.038 m fwd, +0.230 m lat, +0.034 m up) rpy=[0, 0, 0]
            └─ radar_right  (−0.038 m fwd, −0.230 m lat, +0.034 m up) rpy=[0, 0, π]
  └─ front_axle_chassis
       ├─ radar_back  (−4.204 m fwd from FA, +0.304 m up) rpy=[0, 0, 1.600]
       └─ camera_r   (−4.179 m fwd from FA, +0.390 m up) rpy=[0.023, −0.026, −3.133]
```

## Filter Evidence

- Raw DBSCAN clusters before filters: 10854
- Clusters after ground filter: 3515
- Clusters after ego/self filter: 3515
- Final detections after all filters: 643
- Overall raw-cluster to final-detection reduction: 94.08%
- Ground-filter point rejects: 2290710
- Ego/self point rejects: 0
- Vehicle-size rejected clusters: 1027
- Point-count rejected clusters: 1845

## Temporal Robustness

- Detection counts are tracked for every processed frame and plotted in `detection_count_over_time.png`.
- Sudden jumps/dropouts detected: 20
- Zero-detection frames: 51

## Evaluation Criteria

The challenge criteria are accuracy, robustness, and false positive reduction. This run
includes the measurable parts below. True accuracy metrics still require labelled opponent
positions/boxes.

| Criterion | Metric | Value | Interpretation |
| --- | --- | --- | --- |
| Accuracy | ground_truth_available | false | No labelled opponent positions/boxes are present in the project yet, so precision, recall, center error, and IoU cannot be computed honestly. |
| Accuracy | position_and_box_output | true | detections.csv contains target center_x/center_y plus BEV min/max bounds, width, length, and confidence for each candidate. |
| Accuracy | detections_total | 643 | Candidate opponent detections generated by the current run. |
| Robustness | frames_processed | 200 | Frame coverage for this run; --max-frames all removes the old 20-frame cap. |
| Robustness | frames_with_detections | 149/200 | Frames with at least one retained candidate after filtering. |
| Robustness | mean_detections_per_frame | 3.21 | Useful stability proxy before labelled temporal evaluation exists. |
| Robustness | max_detections_in_frame | 11 | Flags possible over-clustering bursts for manual plot review. |
| Robustness | deserialization_failures | 0 | PointCloud2 frames skipped due to parsing/conversion failures. |
| False positive reduction | spatial_crop | forward=3.0..95.0 m; lateral=-18.0..18.0 m | Suppresses obvious off-track or far-field clutter before clustering. |
| False positive reduction | cluster_filter | width=0.8..4.8 m; length=1.2..9.0 m; min_lidar_points=15 | Retains clusters with plausible vehicle footprint and support. |
| False positive reduction | ground_filter_enabled | true | Removes points close to the estimated ground band when enabled. |
| False positive reduction | confidence_distribution | mean=0.699; median=0.710; low_confidence_lt_0.50=4 | Confidence is a heuristic ranking score for manual review/tuning. |
| False positive reduction | mean_candidate_shape | width=2.32 m; length=3.14 m; points=37.4 | Shape/support sanity check for retained candidate clusters. |

## Diagnostics

- Loaded URDF geometry: data/eva.urdf
- URDF base/ego frame: center_of_gravity
- URDF extrinsics auto-loaded by `tf_utils._auto_init()` at import time
- URDF mesh geometry is referenced but not dimensioned in the URDF; using configurable ego_length/ego_width/ego_height safety box.
- Selected default point cloud topic: /sensor/lidar_front/points
- URDF transform lidar_front→center_of_gravity: translation=[+0.662, +0.000, +0.610] m

## Outputs

- CSV: `outputs/task1/detections.csv`
- Evaluation summary: `outputs/task1/evaluation_summary.csv`
- Evaluation JSON: `outputs/task1/evaluation_summary.json`
- Detection-count plot: `outputs/task1/detection_count_over_time.png`
- BEV plots: `outputs/task1/frame_*.png`
- Representative BEV plots: `outputs/task1/representative_frame_*_*.png`

## Reproduce

```bash
uv run python src/task1_detection.py --bag-path data --urdf-path data/eva.urdf \
    --output-dir outputs/task1 --max-frames all
```

## Current Limitations

- This first-pass detector processes one PointCloud2 topic at a time.
- Label-based accuracy scoring needs ground-truth opponent positions or boxes; no precision,
  recall, or IoU is reported here.
- Cluster boxes are axis-aligned in BEV and are intended as candidates for evaluation and tuning.
- If a future bag uses custom point cloud message types, add their definitions to the deserialiser
  or implement a topic-specific parser.
- ~~Hardcoded extrinsics~~ — resolved: sensor positions and orientations are now read from
  `data/eva.urdf` via the self-contained minimal URDF parser in `tf_utils`.
