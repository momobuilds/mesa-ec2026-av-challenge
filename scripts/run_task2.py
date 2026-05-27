"""CLI entry point for Engineering Challenge 2026 Task 2 (multi-sensor fusion)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import format_bag_summary, parse_bag_summary
from src.task1_detection import DetectionConfig
from src.sensor_detectors import detect_all_sensors
from src.oriented_bbox import minimum_area_bbox
from src.radar_doppler import cluster_velocity, extract_cluster_points
from src.task1_detection import SQLitePointCloudReader, pointcloud2_to_structured_array
from src.camera_projection import read_compressed_image
from src.task2_fusion import ASSOC_WINDOW_NS, run_fusion, plot_fused_bev
from src.tf_utils import EgoStateLog, transform_points

from rosbags.typesys import Stores, get_typestore


_MASTER_TOPIC = "/sensor/lidar_front/points"
_DEFAULT_CAMERA_TOPIC = "/sensor/camera/camera_fl/image/compressed"
_COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"

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
        default=200,
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
        "--camera-topic",
        default=_DEFAULT_CAMERA_TOPIC,
        help="CompressedImage topic used for projection-based camera validation.",
    )
    parser.add_argument(
        "--no-camera-validation",
        action="store_true",
        help="Disable camera projection validation.",
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

        # Build timestamp-to-structured-points map from the raw messages.
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


def _db_paths(summary) -> list[Path]:
    paths = [summary.bag_dir / item.path for item in summary.storage_files]
    existing = [path for path in paths if path.exists()]
    if existing:
        return existing
    return sorted(summary.bag_dir.glob("*.db3"))


def _build_camera_index(
    summary,
    topic_name: str,
    *,
    max_messages: int,
) -> dict[str, Any] | None:
    """Collect camera timestamps for projection-based validation."""
    topic = next((item for item in summary.topics if item.name == topic_name), None)
    if topic is None or topic.message_type != _COMPRESSED_IMAGE_TYPE:
        return None

    timestamps: list[int] = []
    remaining = max_messages
    for db_path in _db_paths(summary):
        if remaining <= 0:
            break
        with sqlite3.connect(str(db_path)) as connection:
            row = connection.execute(
                "select id from topics where name = ? and type = ?",
                (topic_name, _COMPRESSED_IMAGE_TYPE),
            ).fetchone()
            if row is None:
                continue
            rows = connection.execute(
                "select timestamp from messages "
                "where topic_id = ? order by timestamp limit ?",
                (int(row[0]), remaining),
            )
            for (timestamp_ns,) in rows:
                timestamps.append(int(timestamp_ns))
                remaining -= 1

    if not timestamps:
        return None
    return {
        "topic": topic_name,
        "timestamps_ns": np.array(sorted(timestamps), dtype=np.int64),
    }


def _read_nearest_compressed_image(
    summary,
    topic_name: str,
    target_ts: int,
    *,
    window_ns: int = int(250e6),
) -> tuple[int, np.ndarray] | None:
    typestore = get_typestore(Stores.ROS2_JAZZY)
    best: tuple[int, int, bytes] | None = None
    lo, hi = target_ts - window_ns, target_ts + window_ns

    for db_path in _db_paths(summary):
        with sqlite3.connect(str(db_path)) as connection:
            row = connection.execute(
                "select id from topics where name = ? and type = ?",
                (topic_name, _COMPRESSED_IMAGE_TYPE),
            ).fetchone()
            if row is None:
                continue
            rows = connection.execute(
                "select timestamp, data from messages "
                "where topic_id = ? and timestamp between ? and ? "
                "order by timestamp",
                (int(row[0]), lo, hi),
            )
            for timestamp_ns, raw_data in rows:
                delta = abs(int(timestamp_ns) - target_ts)
                if best is None or delta < best[0]:
                    best = (delta, int(timestamp_ns), raw_data)

    if best is None:
        return None
    _, timestamp_ns, raw_data = best
    msg = typestore.deserialize_cdr(raw_data, _COMPRESSED_IMAGE_TYPE)
    image = read_compressed_image(bytes(msg.data))
    if image is None:
        return None
    return timestamp_ns, image


def _save_lidar_overlay(all_frames: dict, output_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_dir / "verify_lidar_overlay_000.png"
    fig, ax = plt.subplots(figsize=(7, 11))
    colors = {
        "/sensor/lidar_front/points": "#1f77b4",
        "/sensor/lidar_left/points": "#2ca02c",
        "/sensor/lidar_right/points": "#d62728",
    }
    for topic, color in colors.items():
        frames = all_frames.get(topic, [])
        if not frames:
            continue
        points = frames[0].bev_points
        if len(points) == 0:
            continue
        ax.scatter(points[:, 1], points[:, 0], s=0.5, alpha=0.30, c=color, label=topic.split("/")[2])

    ax.set_title("Task 2 LiDAR ego-frame overlay | frame 000")
    ax.set_xlabel("lateral y_ego (m)")
    ax.set_ylabel("forward x_ego (m)")
    ax.set_xlim((-20, 20))
    ax.set_ylim((-5, 95))
    ax.set_aspect("equal")
    ax.grid(True, color="#ddd", lw=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _select_camera_projection_frames(
    df,
    *,
    max_images: int = 5,
    image_width: int = 1928,
    image_height: int = 500,
) -> list[int]:
    """Select timestamps with the strongest in-image observed fusion evidence."""
    if df.empty:
        return []

    candidates = df.copy()
    for column in ("camera_u", "camera_v", "center_x", "confidence"):
        if column in candidates.columns:
            candidates[column] = np.asarray(
                candidates[column].apply(lambda value: _safe_float(value, np.nan)),
                dtype=float,
            )

    if "is_predicted" in candidates.columns:
        candidates = candidates[
            candidates["is_predicted"].astype(str).str.lower() != "true"
        ]
    if "camera_validated" in candidates.columns:
        candidates = candidates[
            candidates["camera_validated"].astype(str).str.lower() == "true"
        ]

    candidates = candidates[
        candidates["camera_u"].between(0, image_width - 1)
        & candidates["camera_v"].between(0, image_height - 1)
        & candidates["center_x"].between(5.0, 95.0)
    ].copy()
    if candidates.empty:
        return []

    # Prefer frames where several projected fusion tracks land in the plausible
    # road/actor band. This is still projection-only, not image detection.
    candidates["visible_score"] = (
        (candidates["camera_v"] - 245.0).clip(lower=0.0) / 40.0
        + (95.0 - candidates["center_x"]).clip(lower=0.0) / 90.0
        + candidates["confidence"].clip(lower=0.0)
    )
    grouped = (
        candidates.groupby("timestamp_ns")
        .agg(
            best_visible_score=("visible_score", "max"),
            total_visible_score=("visible_score", "sum"),
            projected_rows=("timestamp_ns", "count"),
            mean_confidence=("confidence", "mean"),
            nearest_distance=("center_x", "min"),
            lowest_image_point=("camera_v", "max"),
        )
        .sort_values(
            [
                "best_visible_score",
                "lowest_image_point",
                "nearest_distance",
                "projected_rows",
                "total_visible_score",
            ],
            ascending=[False, False, True, False, False],
        )
    )
    return [int(ts) for ts in grouped.head(max_images).index]


def _format_projection_label(row: dict[str, Any]) -> str:
    """Compact projected-center label for camera verification images."""
    track_id = row.get("track_id", "?")
    distance = _safe_float(row.get("center_x"), 0.0)
    timestamp_ns = row.get("timestamp_ns", "?")
    sensors = row.get("sensors", "")
    return f"T{track_id} {distance:.1f}m t={timestamp_ns} {sensors}".strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _save_camera_projection(
    summary,
    df,
    output_dir: Path,
    camera_topic: str,
) -> list[Path]:
    import cv2

    if df.empty:
        output_path = output_dir / "verify_camera_projection_000.png"
        image = np.full((500, 1928, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "No fused detections available for camera projection",
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), image)
        return [output_path]

    target_timestamps = _select_camera_projection_frames(df, max_images=5)
    if not target_timestamps:
        target_timestamps = [int(df["timestamp_ns"].astype(int).iloc[0])]

    output_paths: list[Path] = []
    for image_index, target_ts in enumerate(target_timestamps):
        output_path = output_dir / f"verify_camera_projection_{image_index:03d}.png"
        image_result = _read_nearest_compressed_image(summary, camera_topic, target_ts)
        if image_result is None:
            image = np.full((500, 1928, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                f"Camera image unavailable: {camera_topic}",
                (40, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            image_ts, image = image_result
            detections_df = _camera_rows_for_timestamp(df, target_ts)
            detections = detections_df.to_dict("records")
            _draw_projection_markers(
                image,
                detections,
                target_ts=target_ts,
                image_ts=image_ts,
                camera_topic=camera_topic,
            )

        cv2.imwrite(str(output_path), image)
        output_paths.append(output_path)
    return output_paths


def _camera_rows_for_timestamp(df, timestamp_ns: int):
    detections_df = df[df["timestamp_ns"].astype(int) == int(timestamp_ns)].copy()
    if "is_predicted" in detections_df.columns:
        detections_df = detections_df[
            detections_df["is_predicted"].astype(str).str.lower() != "true"
        ]
    if "camera_score" in detections_df.columns:
        detections_df = detections_df[
            detections_df["camera_score"].apply(lambda value: _safe_float(value)) > 0.0
        ]
    detections_df = detections_df[
        detections_df["camera_u"].apply(lambda value: 0 <= _safe_float(value, -1) < 1928)
        & detections_df["camera_v"].apply(lambda value: 0 <= _safe_float(value, -1) < 500)
    ]
    return detections_df.sort_values(
        ["camera_v", "confidence", "center_x"],
        ascending=[False, False, True],
    ).head(5)


def _draw_projection_markers(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    target_ts: int,
    image_ts: int,
    camera_topic: str,
) -> None:
    import cv2

    label_lines: list[str] = []
    for marker_index, det in enumerate(detections, start=1):
        u = _safe_float(det.get("camera_u"), -1)
        v = _safe_float(det.get("camera_v"), -1)
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            continue
        center = (int(round(u)), int(round(v)))
        color = (0, 180, 255)
        cv2.circle(image, center, 6, color, 2)
        cv2.drawMarker(
            image,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=1,
        )

        short_label = f"{marker_index}:T{det.get('track_id', '?')}"
        label_origin = (min(center[0] + 8, image.shape[1] - 80), max(20, center[1] - 8))
        cv2.putText(
            image,
            short_label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        label_lines.append(f"{marker_index}. {_format_projection_label(det)}")

    header = (
        "Projection-only camera check; no image detector or 2D boxes in bag | "
        f"fusion_ts={target_ts} | image_ts={image_ts}"
    )
    cv2.putText(
        image,
        header,
        (35, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    if label_lines:
        panel_height = min(30 + 24 * len(label_lines), 155)
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (20, image.shape[0] - panel_height - 15),
            (image.shape[1] - 20, image.shape[0] - 15),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
        y = image.shape[0] - panel_height + 10
        for line in label_lines:
            cv2.putText(
                image,
                line,
                (35, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )
            y += 24


def _write_task2_report(
    summary,
    all_frames: dict,
    df,
    output_dir: Path,
    generated_files: list[Path],
    *,
    camera_index: dict[str, Any] | None,
    args: argparse.Namespace,
) -> Path:
    report_path = output_dir / "task2_report.md"
    per_sensor = [
        (
            topic.split("/")[2] if "/" in topic else topic,
            len(frames),
            sum(len(frame.detections) for frame in frames),
        )
        for topic, frames in all_frames.items()
    ]
    raw_total = sum(count for _, _, count in per_sensor)
    side_lidar_zero = {
        sensor: count
        for sensor, _, count in per_sensor
        if sensor in {"lidar_left", "lidar_right"} and count == 0
    }

    if df.empty:
        timestamps = low_conf = nonzero_velocity = 0
        radar_rows = camera_projection_rows = predicted_rows = max_sources = 0
        velocity_sources = {}
    else:
        timestamps = int(df["timestamp_ns"].nunique())
        low_conf = int((df["confidence"].astype(float) < 0.50).sum())
        nonzero_velocity = int(
            (
                (df["vx"].astype(float).abs() > 1e-3)
                | (df["vy"].astype(float).abs() > 1e-3)
            ).sum()
        )
        radar_rows = int((df["n_radar"].astype(int) > 0).sum())
        camera_projection_rows = int(
            df["camera_validated"].astype(str).str.lower().eq("true").sum()
        )
        predicted_rows = int(df["is_predicted"].astype(bool).sum())
        velocity_sources = df["velocity_source"].value_counts().to_dict()
        max_sources = int(
            (
                df["n_lidar"].astype(int)
                + df["n_radar"].astype(int)
            ).max()
        )

    lines = [
        "# Task 2 Multi-Sensor Fusion Report",
        "",
        "## Bag Inspection",
        "",
        f"- Bag directory: `{summary.bag_dir.as_posix()}`",
        f"- Duration: {summary.duration_s:.3f} s",
        f"- Start timestamp: {summary.starting_time_ns}",
        "",
        "## Implementation Summary",
        "",
        "- LiDAR clusters provide geometric position and oriented bounding-box evidence.",
        "- Radar clusters are processed and can contribute Doppler velocity when spatial association is plausible.",
        "- Camera images are used only for projection/timestamp sanity checks because no camera detector or camera_info topics are available.",
        "- A lightweight temporal filter assigns internal `track_id` values, estimates fallback velocity, and predicts through short missing-data gaps.",
        "- The CSV is generated from this LiDAR/radar/temporal fusion pipeline; it does not read bag-provided object labels or image detections.",
        "",
        "## Per-Sensor Detection",
        "",
        "| Sensor | Frames | Raw detections |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {sensor} | {frames} | {count} |" for sensor, frames, count in per_sensor)
    lines.extend(
        [
            f"| **Total** |  | **{raw_total}** |",
            "",
            "## Output Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| fused_rows | {len(df)} |",
            f"| frames_with_rows | {timestamps} |",
            f"| nonzero_velocity_rows | {nonzero_velocity} |",
            f"| radar_associated_rows | {radar_rows} |",
            f"| camera_projection_rows | {camera_projection_rows} |",
            f"| temporal_prediction_rows | {predicted_rows} |",
            f"| max_sources_per_row | {max_sources} |",
            f"| low_confidence_lt_0.50 | {low_conf} |",
            "",
            "## Velocity Sources",
            "",
            "| Source | Rows |",
            "| --- | ---: |",
        ]
    )
    if velocity_sources:
        lines.extend(f"| {source} | {count} |" for source, count in velocity_sources.items())
    else:
        lines.append("| none | 0 |")

    camera_topic = camera_index["topic"] if camera_index else args.camera_topic
    camera_samples = len(camera_index["timestamps_ns"]) if camera_index else 0
    lines.extend(
        [
            "",
            "## Sensor Association Notes",
            "",
            f"- Radar returns are not forced into tracks. In this {args.max_frames}-frame run, radar clusters that fail the spatial gate are rejected, so `n_radar=0` rows are honest non-associations.",
            f"- Camera topic: `{camera_topic}`",
            f"- Indexed camera timestamps: {camera_samples}",
            "- Camera validation is projection gating only; it is not counted in `sensors` or `n_camera` because no image-object detector is available.",
            "- Camera verification images are selected from timestamps with multiple observed, in-image projected fusion rows and label each marker with internal track ID, distance, source, and timestamp.",
            "",
            "## Known Limitations",
            "",
            "- Radar fusion depends on spatially plausible radar detections; current sample data does not provide radar matches to the LiDAR vehicle clusters.",
            "- Camera projection uses approximate calibration and is reported only as a visual/timestamp sanity check; no 2-D camera boxes or ground-truth actor labels were found in the bag.",
            "- Velocity is ego-frame relative velocity; radar Doppler is preferred when matched, otherwise temporal frame differences are used.",
            "",
            "## Outputs",
            "",
            f"- CSV: `{(output_dir / 'fused_detections.csv').as_posix()}`",
        ]
    )
    if side_lidar_zero:
        insert_at = lines.index(f"- Camera topic: `{camera_topic}`")
        lines.insert(
            insert_at,
            "- Side LiDAR returns are accepted only when clusters pass the same vehicle-shape filters; long guardrail-like strips are rejected instead of being forced into false vehicle detections.",
        )
    lines.extend(f"- Visualisation: `{path.as_posix()}`" for path in generated_files)
    lines.extend(
        [
            "",
            "## Reproducing Results",
            "",
            "```bash",
            (
                "uv run python scripts/run_task2.py "
                f"--bag-dir {args.bag_dir} "
                f"--max-frames {args.max_frames} "
                f"--bev-frames {args.bev_frames} "
                f"--output-dir {args.output_dir}"
            ),
            f"python3 scripts/validate_task2_outputs.py --output-dir {args.output_dir}",
            "```",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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

    print("\n[Task 2] Running per-sensor detection ...")
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

    print("[Task 2] Computing oriented bounding boxes ...")
    oriented_boxes = _build_oriented_boxes(all_frames, config)

    print("[Task 2] Extracting radar Doppler velocities ...")
    doppler_vel = _build_doppler_vel(all_frames, summary, config, ego_log)

    camera_index = None
    if args.no_camera_validation:
        print("[Task 2] Camera validation disabled.")
    else:
        camera_index = _build_camera_index(
            summary,
            args.camera_topic,
            max_messages=max(args.max_frames * 4, args.max_frames + 10),
        )
        if camera_index is None:
            print(
                "[Task 2] Camera validation unavailable: "
                f"{args.camera_topic} was not found or had no indexed messages."
            )
        else:
            print(
                "[Task 2] Camera validation index: "
                f"{camera_index['topic']} | {len(camera_index['timestamps_ns'])} timestamps"
            )

    if _MASTER_TOPIC not in all_frames or not any(
        len(fr.detections) > 0 for fr in all_frames.get(_MASTER_TOPIC, [])
    ):
        print(
            f"\n[Task 2] Warning: master topic '{_MASTER_TOPIC}' has no detections. "
            "Skipping fusion. Re-run with a different --sensors list or check data."
        )
        return

    print("[Task 2] Running fusion ...")
    df = run_fusion(
        all_frames,
        master_topic=_MASTER_TOPIC,
        oriented_boxes=oriented_boxes,
        doppler_vel=doppler_vel,
        camera_index=camera_index,
    )

    csv_path = output_dir / "fused_detections.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Task 2] Fused detections saved -> {csv_path.as_posix()}")
    print(f"         {len(df)} rows | columns: {list(df.columns)}")

    # BEV plots
    master_frames = all_frames.get(_MASTER_TOPIC, [])
    n_plots = min(args.bev_frames, len(master_frames))
    generated_files: list[Path] = []
    if n_plots > 0:
        print(f"[Task 2] Saving {n_plots} BEV fusion plots ...")
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
            generated_files.append(plot_path)
            print(f"  {plot_path.as_posix()}")

    print("[Task 2] Regenerating verification visualisations ...")
    lidar_overlay = _save_lidar_overlay(all_frames, output_dir)
    generated_files.append(lidar_overlay)
    camera_projection = _save_camera_projection(
        summary,
        df,
        output_dir,
        args.camera_topic,
    )
    generated_files.extend(camera_projection)
    report_path = _write_task2_report(
        summary,
        all_frames,
        df,
        output_dir,
        generated_files,
        camera_index=camera_index,
        args=args,
    )
    print(f"  {lidar_overlay.as_posix()}")
    for path in camera_projection:
        print(f"  {path.as_posix()}")
    print(f"[Task 2] Report saved -> {report_path.as_posix()}")

    print("\n[Task 2] Done.")
    print(f"  Sensors processed : {list(all_frames.keys())}")
    print(f"  Fused detections  : {len(df)}")
    print(f"  Outputs           : {output_dir.as_posix()}/")


if __name__ == "__main__":
    main()
