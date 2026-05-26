"""CLI entry point for Engineering Challenge 2026 Task 2 (multi-sensor fusion)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import format_bag_summary, parse_bag_summary, pointcloud_topics
from src.task1_detection import DetectionConfig
from src.sensor_detectors import detect_all_sensors
from src.oriented_bbox import minimum_area_bbox
from src.radar_doppler import cluster_velocity, extract_cluster_points
from src.task1_detection import SQLitePointCloudReader, pointcloud2_to_structured_array
from src.task2_fusion import run_fusion, plot_fused_bev, FUSED_COLUMNS
from src.tf_utils import EgoStateLog, transform_points


_MASTER_TOPIC = "/sensor/lidar_front/points"

_DEFAULT_SENSORS = [
    "/sensor/lidar_front/points",
    "/sensor/lidar_left/points",
    "/sensor/lidar_right/points",
    "/sensor/radar_front/points",
    "/sensor/radar_back/points",
    "/sensor/radar_left/points",
    "/sensor/radar_right/points",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 2 multi-sensor fusion on ROS 2 bag data."
    )
    parser.add_argument(
        "--bag-dir",
        default="data",
        help="Directory containing metadata.yaml and ROS 2 sqlite3 bag files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of point cloud frames to process per sensor.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/task2",
        help="Directory for fused_detections.csv and BEV plots.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="Subset of PointCloud2 topics to process. Defaults to all 7 sensors.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print bag metadata summary and exit.",
    )
    parser.add_argument(
        "--no-ground-filter",
        action="store_true",
        help="Disable ground-plane filtering.",
    )
    parser.add_argument(
        "--bev-frames",
        type=int,
        default=3,
        help="Number of BEV fusion plots to save.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame ground-filter statistics.",
    )
    return parser.parse_args()


def _build_oriented_boxes(
    all_frames: dict,
    config: DetectionConfig,
) -> dict:
    """Compute oriented bounding boxes for every lidar detection."""
    oriented_boxes: dict = {}
    for topic, frames in all_frames.items():
        if "radar" in topic:
            continue
        topic_obbmap: dict = {}
        for fr in frames:
            frame_map: dict = {}
            for det in fr.detections:
                # Select only the BEV points that belong to this cluster
                # using the detection's axis-aligned bounds (with 0.5m margin).
                pts = fr.bev_points
                if len(pts) == 0:
                    continue
                margin = 0.5
                mask = (
                    (pts[:, 0] >= float(det["min_x"]) - margin)
                    & (pts[:, 0] <= float(det["max_x"]) + margin)
                    & (pts[:, 1] >= float(det["min_y"]) - margin)
                    & (pts[:, 1] <= float(det["max_y"]) + margin)
                )
                cluster_pts = pts[mask]
                if len(cluster_pts) < 3:
                    continue
                obb = minimum_area_bbox(cluster_pts)
                frame_map[det["detection_id"]] = obb
            if frame_map:
                topic_obbmap[fr.frame_index] = frame_map
        oriented_boxes[topic] = topic_obbmap
    return oriented_boxes


def _build_doppler_vel(
    all_frames: dict,
    summary,
    config: DetectionConfig,
    ego_log: EgoStateLog | None,
) -> dict:
    """Extract Doppler velocity for each radar cluster."""
    doppler_vel: dict = {}
    reader = SQLitePointCloudReader(summary)

    for topic, frames in all_frames.items():
        if "radar" not in topic:
            continue
        topic_velmap: dict = {}

        # Build timestamp → structured points map from the raw messages
        raw_by_ts: dict = {}
        for msg_dict in reader.iter_pointcloud_messages(topic, max_frames=len(frames) + 5):
            try:
                structured = pointcloud2_to_structured_array(msg_dict["msg"])
            except Exception:  # noqa: BLE001
                continue
            raw_by_ts[msg_dict["timestamp_ns"]] = (structured, msg_dict["source_frame"])

        for fr in frames:
            if fr.timestamp_ns not in raw_by_ts:
                continue
            structured, sensor_frame = raw_by_ts[fr.timestamp_ns]

            ego_vx, ego_vy = 0.0, 0.0
            if ego_log is not None:
                try:
                    ego_state = ego_log.interpolate(fr.timestamp_ns)
                    ego_vx = ego_state.get("vx", 0.0)
                    ego_vy = ego_state.get("vy", 0.0)
                except Exception:  # noqa: BLE001
                    pass

            vel_list: list[tuple[float, float]] = []
            for det in fr.detections:
                # cluster_points is not stored in FrameResult; use BEV mask to
                # approximate which structured points belong to this detection
                # by proximity to the detection centre.
                cx = float(det.get("center_x", 0))
                cy = float(det.get("center_y", 0))
                # Transform sensor-frame points to ego frame for proximity check.
                if "x" in structured.dtype.names and "y" in structured.dtype.names:
                    xs = structured["x"].astype(float)
                    ys = structured["y"].astype(float)
                    zs = structured["z"].astype(float) if "z" in structured.dtype.names else np.zeros_like(xs)
                    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                    if finite.any():
                        ego_xyz = transform_points(
                            np.column_stack([xs[finite], ys[finite], zs[finite]]),
                            sensor_frame,
                        )
                        dx = ego_xyz[:, 0] - cx
                        dy = ego_xyz[:, 1] - cy
                        close_mask = dx**2 + dy**2 < 25.0
                        orig_idx = finite.nonzero()[0][close_mask]
                        if len(orig_idx) > 0:
                            cluster_pts = extract_cluster_points(structured, orig_idx)
                            vx, vy = cluster_velocity(
                                cluster_pts,
                                sensor_frame,
                                ego_vx=ego_vx,
                                ego_vy=ego_vy,
                            )
                            vel_list.append((vx, vy))
                        else:
                            vel_list.append((0.0, 0.0))
                    else:
                        vel_list.append((0.0, 0.0))
                else:
                    vel_list.append((0.0, 0.0))

            topic_velmap[fr.frame_index] = vel_list
        doppler_vel[topic] = topic_velmap
    return doppler_vel


def main() -> None:
    args = parse_args()
    summary = parse_bag_summary(args.bag_dir)
    print(format_bag_summary(summary))

    if args.inspect_only:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DetectionConfig(
        ground_band_half_height=0.30 if not args.no_ground_filter else 0.0,
    )

    topic_names = args.sensors or _DEFAULT_SENSORS

    print("\n[Task 2] Running per-sensor detection …")
    all_frames = detect_all_sensors(
        summary,
        config,
        max_frames=args.max_frames,
        topic_names=topic_names,
        verbose=args.verbose,
    )

    total_dets = sum(
        sum(len(fr.detections) for fr in frames)
        for frames in all_frames.values()
    )
    print(f"\n[Task 2] {total_dets} raw detections across all sensors")

    # Load ego velocity log if available
    ego_log: EgoStateLog | None = None
    ego_mat_path = Path(args.bag_dir) / "ego.mat"
    if not ego_mat_path.exists():
        for candidate in Path(args.bag_dir).parent.glob("**/ego.mat"):
            ego_mat_path = candidate
            break
    if ego_mat_path.exists():
        try:
            ego_log = EgoStateLog(str(ego_mat_path), bag_start_ns=summary.starting_time_ns or 0)
            print(f"[Task 2] Loaded ego state log from {ego_mat_path.as_posix()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Task 2] ego.mat load failed: {exc}")

    print("[Task 2] Computing oriented bounding boxes …")
    oriented_boxes = _build_oriented_boxes(all_frames, config)

    print("[Task 2] Extracting radar Doppler velocities …")
    doppler_vel = _build_doppler_vel(all_frames, summary, config, ego_log)

    if _MASTER_TOPIC not in all_frames or not any(
        len(fr.detections) > 0 for fr in all_frames.get(_MASTER_TOPIC, [])
    ):
        print(
            f"\n[Task 2] Warning: master topic '{_MASTER_TOPIC}' has no detections. "
            "Skipping fusion. Re-run with a different --sensors list or check data."
        )
        return

    print("[Task 2] Running fusion …")
    df = run_fusion(
        all_frames,
        master_topic=_MASTER_TOPIC,
        oriented_boxes=oriented_boxes,
        doppler_vel=doppler_vel,
    )

    csv_path = output_dir / "fused_detections.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Task 2] Fused detections saved → {csv_path.as_posix()}")
    print(f"         {len(df)} rows | columns: {list(df.columns)}")

    # BEV plots
    master_frames = all_frames.get(_MASTER_TOPIC, [])
    n_plots = min(args.bev_frames, len(master_frames))
    if n_plots > 0:
        print(f"[Task 2] Saving {n_plots} BEV fusion plots …")
        for idx in range(n_plots):
            fr = master_frames[idx]
            ts = fr.timestamp_ns
            fused_slice = df[df["timestamp_ns"] == ts].to_dict("records")

            sensor_frs: dict = {}
            for topic, frames in all_frames.items():
                matched = next((f for f in frames if f.timestamp_ns == ts), None)
                if matched is None and frames:
                    matched = frames[min(idx, len(frames) - 1)]
                sensor_frs[topic] = matched

            plot_path = output_dir / f"fusion_bev_{idx:03d}.png"
            plot_fused_bev(fr, fused_slice, sensor_frs, plot_path)
            print(f"  {plot_path.as_posix()}")

    print("\n[Task 2] Done.")
    print(f"  Sensors processed : {list(all_frames.keys())}")
    print(f"  Fused detections  : {len(df)}")
    print(f"  Outputs           : {output_dir.as_posix()}/")


if __name__ == "__main__":
    main()
