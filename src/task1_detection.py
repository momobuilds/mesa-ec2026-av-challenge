"""Task 1 opponent vehicle detection from ROS 2 bag point clouds.

The implementation deliberately avoids a full ROS 2 installation. It reads the
sqlite3 storage files directly and uses ``rosbags`` only for standard CDR
deserialization of ``sensor_msgs/msg/PointCloud2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.bag_inspector import (
    BagSummary,
    TopicSummary,
    format_bag_summary,
    parse_bag_summary,
    pointcloud_topics,
    select_default_pointcloud_topic,
)
from src.visualization import plot_bev_frame

try:
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise RuntimeError(
        "The rosbags package is required to deserialize ROS 2 PointCloud2 data. "
        "Install project dependencies with: uv sync"
    ) from exc


POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
DETECTION_COLUMNS = [
    "timestamp",
    "topic",
    "source_frame",
    "frame_index",
    "detection_id",
    "center_x",
    "center_y",
    "width",
    "length",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "point_count",
    "confidence",
]


@dataclass(frozen=True)
class AxisMapping:
    """Mapping from raw PointCloud2 field names to vehicle-style BEV axes."""

    forward: str
    lateral: str
    vertical: str | None
    reason: str


@dataclass
class DetectionConfig:
    """Clustering and filtering parameters for the first-pass detector."""

    max_frames: int = 20
    output_dir: Path = Path("outputs/task1")
    forward_range: tuple[float, float] = (3.0, 95.0)
    lateral_range: tuple[float, float] = (-18.0, 18.0)
    dbscan_eps_lidar: float = 0.45
    dbscan_eps_radar: float = 2.75
    dbscan_min_samples_lidar: int = 6
    dbscan_min_samples_radar: int = 3
    min_points_lidar: int = 15
    min_points_radar: int = 3
    min_width: float = 0.8
    max_width: float = 4.8
    min_length: float = 1.2
    max_length: float = 9.0
    max_plots: int = 20
    ground_band_half_height: float = 0.30


@dataclass
class FrameResult:
    """Detection result for one input point cloud frame."""

    frame_index: int
    timestamp_ns: int
    topic: str
    source_frame: str
    bev_points: np.ndarray
    detections: list[dict[str, Any]]


@dataclass
class PipelineResult:
    """Artifacts and diagnostics produced by the Task 1 pipeline."""

    summary: BagSummary
    selected_topic: TopicSummary | None
    axis_mapping: AxisMapping | None
    frames: list[FrameResult] = field(default_factory=list)
    detections: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: list[str] = field(default_factory=list)
    output_dir: Path = Path("outputs/task1")


def run_task1_pipeline(
    bag_dir: str | Path,
    *,
    max_frames: int = 20,
    output_dir: str | Path = "outputs/task1",
    topic_name: str | None = None,
    ground_filter: bool = True,
) -> PipelineResult:
    """Run metadata inspection, point-cloud detection, CSV export, and plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = parse_bag_summary(bag_dir)
    config = DetectionConfig(
        max_frames=max_frames,
        output_dir=output_path,
        ground_band_half_height=0.30 if ground_filter else 0.0,
    )
    diagnostics: list[str] = []

    selected_topic = _resolve_topic(summary, topic_name, diagnostics)
    if selected_topic is None:
        diagnostics.append(
            "No sensor_msgs/msg/PointCloud2 LiDAR or radar topics were found. "
            "Task 1 detection needs a supported point cloud topic or a custom "
            "message parser."
        )
        result = PipelineResult(
            summary=summary,
            selected_topic=None,
            axis_mapping=None,
            diagnostics=diagnostics,
            output_dir=output_path,
        )
        _write_outputs(result, config)
        return result

    reader = SQLitePointCloudReader(summary)
    frames: list[FrameResult] = []
    all_detections: list[dict[str, Any]] = []
    axis_mapping: AxisMapping | None = None

    for frame_index, message in enumerate(
        reader.iter_pointcloud_messages(selected_topic.name, max_frames=max_frames)
    ):
        try:
            structured_points = pointcloud2_to_structured_array(message["msg"])
        except Exception as exc:  # noqa: BLE001 - keep diagnostic explicit for challenge data.
            diagnostics.append(
                "PointCloud2 conversion failed for "
                f"{selected_topic.name} at timestamp {message['timestamp_ns']}: {exc}"
            )
            continue

        if axis_mapping is None:
            axis_mapping = infer_axis_mapping(
                selected_topic.name,
                structured_points,
                message["msg"].fields,
            )
            diagnostics.append(
                "Axis mapping: "
                f"forward={axis_mapping.forward}, lateral={axis_mapping.lateral}, "
                f"vertical={axis_mapping.vertical or 'none'} ({axis_mapping.reason})"
            )

        bev_points = extract_bev_points(
            structured_points, axis_mapping, config, frame_index=frame_index
        )
        sensor_kind = "radar" if "radar" in selected_topic.name.lower() else "lidar"
        detections = detect_opponents_in_frame(
            bev_points,
            sensor_kind=sensor_kind,
            config=config,
            timestamp_ns=message["timestamp_ns"],
            topic=selected_topic.name,
            source_frame=message["source_frame"],
            frame_index=frame_index,
        )

        frames.append(
            FrameResult(
                frame_index=frame_index,
                timestamp_ns=message["timestamp_ns"],
                topic=selected_topic.name,
                source_frame=message["source_frame"],
                bev_points=bev_points,
                detections=detections,
            )
        )
        all_detections.extend(detections)

    if not frames:
        diagnostics.append(
            "No frames were processed. If the topic exists but deserialization failed, "
            "inspect the topic type and message definition in the SQLite "
            "message_definitions table."
        )

    detections_df = pd.DataFrame(all_detections, columns=DETECTION_COLUMNS)
    result = PipelineResult(
        summary=summary,
        selected_topic=selected_topic,
        axis_mapping=axis_mapping,
        frames=frames,
        detections=detections_df,
        diagnostics=diagnostics,
        output_dir=output_path,
    )
    _write_outputs(result, config)
    return result


class SQLitePointCloudReader:
    """Minimal ROS 2 sqlite3 bag reader for standard PointCloud2 messages."""

    def __init__(self, summary: BagSummary) -> None:
        self.summary = summary
        self.typestore = get_typestore(Stores.ROS2_JAZZY)

    def iter_pointcloud_messages(
        self,
        topic_name: str,
        *,
        max_frames: int,
    ) -> Iterable[dict[str, Any]]:
        """Yield at most ``max_frames`` deserialized point cloud messages."""

        yielded = 0
        for db_path in self._db_paths():
            if yielded >= max_frames:
                break
            with sqlite3.connect(str(db_path)) as connection:
                topic_row = connection.execute(
                    "select id, type from topics where name = ?",
                    (topic_name,),
                ).fetchone()
                if topic_row is None:
                    continue

                topic_id, message_type = topic_row
                if message_type != POINTCLOUD2_TYPE:
                    raise ValueError(
                        f"Topic {topic_name} has type {message_type}; "
                        f"expected {POINTCLOUD2_TYPE}"
                    )

                remaining = max_frames - yielded
                rows = connection.execute(
                    "select timestamp, data from messages where topic_id = ? limit ?",
                    (topic_id, remaining),
                )
                for timestamp_ns, raw_data in rows:
                    msg = self.typestore.deserialize_cdr(raw_data, POINTCLOUD2_TYPE)
                    yielded += 1
                    yield {
                        "timestamp_ns": int(timestamp_ns),
                        "msg": msg,
                        "source_frame": getattr(msg.header, "frame_id", ""),
                    }

    def _db_paths(self) -> list[Path]:
        paths = [self.summary.bag_dir / item.path for item in self.summary.storage_files]
        existing = [path for path in paths if path.exists()]
        if existing:
            return existing
        return sorted(self.summary.bag_dir.glob("*.db3"))


def pointcloud2_to_structured_array(msg: Any) -> np.ndarray:
    """Convert a ``sensor_msgs/msg/PointCloud2`` object into a structured array."""

    endian = ">" if bool(msg.is_bigendian) else "<"
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []

    for field in msg.fields:
        if field.datatype not in _POINT_FIELD_DTYPES:
            continue
        names.append(field.name)
        offsets.append(int(field.offset))
        field_dtype = np.dtype(endian + _POINT_FIELD_DTYPES[field.datatype])
        if int(field.count) > 1:
            formats.append((field_dtype, (int(field.count),)))
        else:
            formats.append(field_dtype)

    if not names:
        raise ValueError("PointCloud2 message has no supported numeric fields")

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        }
    )
    raw = bytes(msg.data)
    width = int(msg.width)
    height = int(msg.height)
    point_step = int(msg.point_step)
    row_step = int(msg.row_step)

    if height > 1 and row_step != width * point_step:
        rows = []
        for row_index in range(height):
            start = row_index * row_step
            stop = start + width * point_step
            rows.append(np.frombuffer(raw[start:stop], dtype=dtype, count=width))
        return np.concatenate(rows)

    count = min(width * height, len(raw) // point_step)
    return np.frombuffer(raw, dtype=dtype, count=count)


def infer_axis_mapping(
    topic_name: str,
    points: np.ndarray,
    fields: Iterable[Any],
) -> AxisMapping:
    """Infer BEV axes from topic naming and observed point ranges."""

    field_names = {field.name for field in fields}
    lower_topic = topic_name.lower()

    if {"x", "y", "z"}.issubset(field_names) and "lidar" in lower_topic:
        return AxisMapping(
            forward="z",
            lateral="x",
            vertical="y",
            reason="LiDAR point clouds in this bag use z as range and x as lateral",
        )

    if {"x", "y", "z"}.issubset(field_names) and "radar" in lower_topic:
        return AxisMapping(
            forward="y",
            lateral="x",
            vertical="z",
            reason="radar point clouds in this bag use y as range and x as lateral",
        )

    numeric_axes = [axis for axis in ("x", "y", "z") if axis in points.dtype.names]
    if len(numeric_axes) < 2:
        raise ValueError(
            "Point cloud needs at least two of x/y/z fields for BEV clustering; "
            f"available fields: {points.dtype.names}"
        )

    axis_scores: dict[str, float] = {}
    for axis in numeric_axes:
        values = points[axis].astype(np.float64, copy=False)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            axis_scores[axis] = float("-inf")
            continue
        q50, q99 = np.quantile(values, [0.50, 0.99])
        axis_scores[axis] = float(q99 + max(q50, 0.0))

    forward = max(axis_scores, key=axis_scores.get)
    remaining = [axis for axis in numeric_axes if axis != forward]
    lateral = max(remaining, key=lambda axis: _axis_spread(points[axis]))
    vertical_candidates = [axis for axis in remaining if axis != lateral]
    vertical = vertical_candidates[0] if vertical_candidates else None
    return AxisMapping(
        forward=forward,
        lateral=lateral,
        vertical=vertical,
        reason="inferred from coordinate ranges",
    )


def extract_bev_points(
    structured_points: np.ndarray,
    axis_mapping: AxisMapping,
    config: DetectionConfig,
    *,
    frame_index: int = -1,
) -> np.ndarray:
    """Return cropped BEV points as ``[forward_m, lateral_m]``."""

    forward = structured_points[axis_mapping.forward].astype(np.float64, copy=False)
    lateral = structured_points[axis_mapping.lateral].astype(np.float64, copy=False)
    mask = np.isfinite(forward) & np.isfinite(lateral)
    mask &= forward >= config.forward_range[0]
    mask &= forward <= config.forward_range[1]
    mask &= lateral >= config.lateral_range[0]
    mask &= lateral <= config.lateral_range[1]

    vertical: np.ndarray | None = None
    if axis_mapping.vertical and axis_mapping.vertical in structured_points.dtype.names:
        vertical = structured_points[axis_mapping.vertical].astype(np.float64, copy=False)
        mask &= np.isfinite(vertical)
        # Trim only extreme vertical outliers. The LiDAR frame is not ego-aligned,
        # so a fixed ground-height threshold would be too brittle here.
        valid_vertical = vertical[mask]
        if len(valid_vertical) > 30:
            low, high = np.quantile(valid_vertical, [0.005, 0.995])
            mask &= vertical >= low
            mask &= vertical <= high

    if vertical is not None and config.ground_band_half_height > 0:
        total = int(mask.sum())
        close_mask = mask & (forward >= 5.0) & (forward <= 25.0)
        close_vertical = vertical[close_mask]
        if len(close_vertical) > 10:
            ground_median = float(np.median(close_vertical))
            mask &= np.abs(vertical - ground_median) > config.ground_band_half_height
        kept = int(mask.sum())
        print(f"frame {frame_index}: kept {kept}/{total} points after ground filter")

    return np.column_stack((forward[mask], lateral[mask])).astype(np.float32, copy=False)


def detect_opponents_in_frame(
    bev_points: np.ndarray,
    *,
    sensor_kind: str,
    config: DetectionConfig,
    timestamp_ns: int,
    topic: str,
    source_frame: str,
    frame_index: int,
) -> list[dict[str, Any]]:
    """Cluster BEV points and filter clusters by plausible vehicle dimensions.

    Args:
        bev_points: shape (N, 2), columns = [forward_m, lateral_m]
        sensor_kind: ``"lidar"`` or ``"radar"`` — selects DBSCAN presets
    """

    if len(bev_points) == 0:
        return []

    is_radar = sensor_kind == "radar"
    eps = config.dbscan_eps_radar if is_radar else config.dbscan_eps_lidar
    min_samples = (
        config.dbscan_min_samples_radar if is_radar else config.dbscan_min_samples_lidar
    )
    min_points = config.min_points_radar if is_radar else config.min_points_lidar

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(bev_points)
    detections: list[dict[str, Any]] = []
    detection_id = 0

    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster = bev_points[labels == label]
        point_count = len(cluster)
        if point_count < min_points:
            continue

        min_forward, min_lateral = cluster.min(axis=0)
        max_forward, max_lateral = cluster.max(axis=0)
        length = float(max_forward - min_forward)
        width = float(max_lateral - min_lateral)

        if not (
            config.min_width <= width <= config.max_width
            and config.min_length <= length <= config.max_length
        ):
            continue

        center_forward = float((min_forward + max_forward) / 2.0)
        center_lateral = float((min_lateral + max_lateral) / 2.0)
        confidence = _cluster_confidence(
            point_count=point_count,
            width=width,
            length=length,
            center_forward=center_forward,
            config=config,
            is_radar=is_radar,
        )

        detections.append(
            {
                "timestamp": timestamp_ns,
                "topic": topic,
                "source_frame": source_frame,
                "frame_index": frame_index,
                "detection_id": f"{frame_index:04d}_{detection_id:02d}",
                "center_x": center_forward,
                "center_y": center_lateral,
                "width": width,
                "length": length,
                "min_x": float(min_forward),
                "max_x": float(max_forward),
                "min_y": float(min_lateral),
                "max_y": float(max_lateral),
                "point_count": point_count,
                "confidence": confidence,
            }
        )
        detection_id += 1

    return detections


def detect_vehicle_candidates(
    bev_points: np.ndarray,
    *,
    timestamp_ns: int,
    topic: str,
    source_frame: str,
    frame_index: int,
    config: DetectionConfig,
) -> list[dict[str, Any]]:
    """Cluster BEV points and filter clusters by plausible vehicle dimensions.

    Thin wrapper around :func:`detect_opponents_in_frame` that infers
    ``sensor_kind`` from the topic name for backward compatibility.
    """
    sensor_kind = "radar" if "radar" in topic.lower() else "lidar"
    return detect_opponents_in_frame(
        bev_points,
        sensor_kind=sensor_kind,
        config=config,
        timestamp_ns=timestamp_ns,
        topic=topic,
        source_frame=source_frame,
        frame_index=frame_index,
    )


def _resolve_topic(
    summary: BagSummary,
    topic_name: str | None,
    diagnostics: list[str],
) -> TopicSummary | None:
    if topic_name:
        by_name = {topic.name: topic for topic in summary.topics}
        topic = by_name.get(topic_name)
        if topic is None:
            available = ", ".join(topic.name for topic in pointcloud_topics(summary))
            raise ValueError(
                f"Requested topic {topic_name!r} was not found. "
                f"Available PointCloud2 topics: {available}"
            )
        if topic.message_type != POINTCLOUD2_TYPE:
            raise ValueError(
                f"Requested topic {topic_name!r} has type {topic.message_type}; "
                f"expected {POINTCLOUD2_TYPE}"
            )
        diagnostics.append(f"Using requested point cloud topic: {topic.name}")
        return topic

    topic = select_default_pointcloud_topic(summary)
    if topic:
        diagnostics.append(f"Selected default point cloud topic: {topic.name}")
    return topic


def _write_outputs(result: PipelineResult, config: DetectionConfig) -> None:
    detections_csv = result.output_dir / "detections.csv"
    report_md = result.output_dir / "task1_report.md"

    if result.detections.empty:
        pd.DataFrame(columns=DETECTION_COLUMNS).to_csv(detections_csv, index=False)
    else:
        result.detections.to_csv(detections_csv, index=False)

    for frame in result.frames[: config.max_plots]:
        plot_bev_frame(
            frame.bev_points,
            frame.detections,
            result.output_dir / f"frame_{frame.frame_index:03d}.png",
            title=f"Task 1 BEV detections | frame {frame.frame_index:03d}",
            forward_range=config.forward_range,
            lateral_range=config.lateral_range,
        )

    report_md.write_text(_build_report(result), encoding="utf-8")


def _build_report(result: PipelineResult) -> str:
    lines = [
        "# Task 1 Opponent Vehicle Detection Report",
        "",
        "## Bag Inspection",
        "",
        "```text",
        format_bag_summary(result.summary),
        "```",
        "",
        "## Detection Run",
        "",
    ]

    if result.selected_topic is None:
        lines.append("- Selected topic: none")
    else:
        lines.append(f"- Selected topic: `{result.selected_topic.name}`")
        lines.append(f"- Message type: `{result.selected_topic.message_type}`")
    lines.append(f"- Frames processed: {len(result.frames)}")
    lines.append(f"- Detections written: {len(result.detections)}")

    if result.axis_mapping:
        lines.append(
            "- Axis mapping: "
            f"forward=`{result.axis_mapping.forward}`, "
            f"lateral=`{result.axis_mapping.lateral}`, "
            f"vertical=`{result.axis_mapping.vertical or 'none'}`"
        )
        lines.append(f"- Axis mapping reason: {result.axis_mapping.reason}")

    lines.extend(["", "## Diagnostics", ""])
    if result.diagnostics:
        lines.extend(f"- {item}" for item in result.diagnostics)
    else:
        lines.append("- No diagnostics.")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- CSV: `{(result.output_dir / 'detections.csv').as_posix()}`",
            f"- BEV plots: `{(result.output_dir / 'frame_*.png').as_posix()}`",
            "",
            "## Current Limitations",
            "",
            "- This first-pass detector processes one PointCloud2 topic at a time.",
            "- Sensor extrinsic calibration and multi-LiDAR fusion are not applied yet.",
            "- Cluster boxes are axis-aligned in BEV and are intended as candidates for evaluation and tuning.",
            "- If a future bag uses custom point cloud message types, add their definitions to the deserializer or implement a topic-specific parser.",
        ]
    )
    return "\n".join(lines) + "\n"


def _axis_spread(values: np.ndarray) -> float:
    finite = values.astype(np.float64, copy=False)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    q01, q99 = np.quantile(finite, [0.01, 0.99])
    return float(q99 - q01)


def _cluster_confidence(
    *,
    point_count: int,
    width: float,
    length: float,
    center_forward: float,
    config: DetectionConfig,
    is_radar: bool,
) -> float:
    ideal_width = 2.0
    ideal_length = 4.5
    width_score = max(0.0, 1.0 - abs(width - ideal_width) / 4.0)
    length_score = max(0.0, 1.0 - abs(length - ideal_length) / 8.0)
    density_scale = 15.0 if is_radar else 90.0
    density_score = min(1.0, np.log1p(point_count) / np.log1p(density_scale))
    range_score = max(
        0.0,
        1.0
        - (center_forward - config.forward_range[0])
        / (config.forward_range[1] - config.forward_range[0]),
    )
    score = 0.35 * density_score + 0.35 * ((width_score + length_score) / 2.0)
    score += 0.30 * range_score
    return round(float(np.clip(score, 0.0, 1.0)), 3)


_POINT_FIELD_DTYPES = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}
