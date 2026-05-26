# Task 1 Opponent Vehicle Detection Report

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
- /sensor/camera/camera_r/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4643 | serialization=cdr
- /sensor/camera/camera_rl/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/camera/camera_rr/image/compressed | type=sensor_msgs/msg/CompressedImage | messages=4800 | serialization=cdr
- /sensor/lidar_front/points | type=sensor_msgs/msg/PointCloud2 | messages=4800 | serialization=cdr
- /sensor/lidar_left/points | type=sensor_msgs/msg/PointCloud2 | messages=4799 | serialization=cdr
- /sensor/lidar_right/points | type=sensor_msgs/msg/PointCloud2 | messages=4800 | serialization=cdr
- /sensor/radar_back/points | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /sensor/radar_front/points | type=sensor_msgs/msg/PointCloud2 | messages=3999 | serialization=cdr
- /sensor/radar_left/points | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /sensor/radar_right/points | type=sensor_msgs/msg/PointCloud2 | messages=4000 | serialization=cdr
- /tf | type=tf2_msgs/msg/TFMessage | messages=45080 | serialization=cdr
- /tf_static | type=tf2_msgs/msg/TFMessage | messages=0 | serialization=cdr
```

## Detection Run

- Selected topic: `/sensor/lidar_front/points`
- Message type: `sensor_msgs/msg/PointCloud2`
- Frames processed: 20
- Detections written: 99
- Axis mapping: forward=`z`, lateral=`x`, vertical=`y`
- Axis mapping reason: LiDAR point clouds in this bag use z as range and x as lateral

## Diagnostics

- Selected default point cloud topic: /sensor/lidar_front/points
- Axis mapping: forward=z, lateral=x, vertical=y (LiDAR point clouds in this bag use z as range and x as lateral)

## Outputs

- CSV: `outputs\task1\detections.csv`
- BEV plots: `outputs\task1\frame_*.png`

## Current Limitations

- This first-pass detector processes one PointCloud2 topic at a time.
- Sensor extrinsic calibration and multi-LiDAR fusion are not applied yet.
- Cluster boxes are axis-aligned in BEV and are intended as candidates for evaluation and tuning.
- If a future bag uses custom point cloud message types, add their definitions to the deserializer or implement a topic-specific parser.
