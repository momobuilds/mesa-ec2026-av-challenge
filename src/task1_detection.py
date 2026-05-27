"""Task 1 opponent vehicle detection from ROS 2 bag point clouds.

The implementation deliberately avoids a full ROS 2 installation. It reads the
sqlite3 storage files directly and uses ``rosbags`` only for standard CDR
deserialization of ``sensor_msgs/msg/PointCloud2``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
DEFAULT_MAX_FRAMES: int | None = None
DEFAULT_MAX_PLOTS = 200
DETECTION_COLUMNS = [
    "timestamp",
    "timestamp_ns",
    "topic",
    "source_frame",
    "sensor_frame",
    "base_frame",
    "frame_index",
    "frame_idx",
    "detection_id",
    "center_x",
    "center_y",
    "center_x_base",
    "center_y_base",
    "width",
    "length",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "min_x_base",
    "max_x_base",
    "min_y_base",
    "max_y_base",
    "point_count",
    "confidence",
]
EVALUATION_COLUMNS = [
    "criterion",
    "metric",
    "value",
    "interpretation",
]


@dataclass(frozen=True)
class AxisMapping:
    """Mapping from raw PointCloud2 field names to vehicle-style BEV axes."""

    forward: str
    lateral: str
    vertical: str | None
    reason: str


@dataclass(frozen=True)
class UrdfJoint:
    """One fixed URDF joint represented as a parent<-child transform."""

    name: str
    parent: str
    child: str
    transform_parent_child: np.ndarray


@dataclass
class UrdfGeometry:
    """Small URDF frame graph used to locate sensors relative to the ego frame."""

    path: Path
    links: set[str]
    joints: list[UrdfJoint]
    mesh_filenames: list[str] = field(default_factory=list)

    def select_base_frame(self, requested: str | None = None) -> str:
        """Return the requested base frame or the best ego-frame available."""

        if requested:
            frame = _canonical_frame_name(requested)
            if frame not in self.links:
                raise ValueError(
                    f"URDF base frame {requested!r} was not found. "
                    f"Available links include: {', '.join(sorted(self.links)[:12])}"
                )
            return frame

        for candidate in ("base_link", "ego", "center_of_gravity", "rear_axle_middle_ground"):
            if candidate in self.links:
                return candidate
        raise ValueError("URDF does not contain a recognizable base/ego frame")

    def transform_sensor_to_base(self, sensor_frame: str, base_frame: str) -> np.ndarray:
        """Return a 4x4 matrix T where ``p_base = T @ p_sensor``."""

        sensor = _canonical_frame_name(sensor_frame)
        base = self.select_base_frame(base_frame)
        if sensor not in self.links:
            raise ValueError(
                f"URDF sensor frame {sensor_frame!r} was not found. "
                f"Available links include: {', '.join(sorted(self.links)[:12])}"
            )

        adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
        for joint in self.joints:
            # transform_parent_child maps child coordinates into parent coordinates.
            adjacency.setdefault(joint.parent, []).append(
                (joint.child, joint.transform_parent_child)
            )
            adjacency.setdefault(joint.child, []).append(
                (joint.parent, np.linalg.inv(joint.transform_parent_child))
            )

        queue: list[tuple[str, np.ndarray]] = [(base, np.eye(4, dtype=np.float64))]
        visited = {base}
        while queue:
            current, transform_base_current = queue.pop(0)
            if current == sensor:
                return transform_base_current
            for neighbor, transform_current_neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(
                    (neighbor, transform_base_current @ transform_current_neighbor)
                )

        raise ValueError(
            f"No fixed-joint transform path found in {self.path} from "
            f"sensor frame {sensor!r} to base frame {base!r}."
        )


@dataclass
class DetectionConfig:
    """Clustering and filtering parameters for the first-pass detector."""

    max_frames: int | None = DEFAULT_MAX_FRAMES
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
    max_plots: int = DEFAULT_MAX_PLOTS
    ground_band_half_height: float = 0.30
    ego_length: float = 5.0
    ego_width: float = 2.2
    ego_height: float = 1.6
    safety_margin: float = 0.25


@dataclass
class FilteredPointCloud:
    """Point cloud snapshots and counters from the Task 1 filter chain."""

    raw_points: np.ndarray
    after_ground_points: np.ndarray
    after_ego_points: np.ndarray
    final_points: np.ndarray
    raw_bev_points: np.ndarray
    after_ground_bev_points: np.ndarray
    after_ego_bev_points: np.ndarray
    final_bev_points: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Detection result for one input point cloud frame."""

    frame_index: int
    timestamp_ns: int
    topic: str
    source_frame: str
    bev_points: np.ndarray
    detections: list[dict[str, Any]]
    sensor_frame: str = ""
    base_frame: str = ""
    raw_bev_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    after_ego_bev_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    filter_stats: dict[str, Any] = field(default_factory=dict)
    cluster_stats: dict[str, Any] = field(default_factory=dict)


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
    urdf_path: Path | None = None
    base_frame: str | None = None
    sensor_to_base_transforms: dict[str, np.ndarray] = field(default_factory=dict)


def parse_max_frames_arg(value: str | int | None) -> int | None:
    """Parse CLI/API frame limits where ``all`` means no artificial limit."""

    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise argparse.ArgumentTypeError("max_frames must be positive or 'all'")
        return value
    text = str(value).strip().lower()
    if text == "all":
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid max frame value {value!r}; use a positive integer or 'all'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max_frames must be positive or 'all'")
    return parsed


def load_urdf_geometry(urdf_path: str | Path) -> UrdfGeometry:
    """Parse fixed links/joints from a URDF without requiring ROS."""

    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"URDF file not found: {path}")

    root = ET.parse(path).getroot()
    links = {
        _canonical_frame_name(link.attrib["name"])
        for link in root.findall("link")
        if "name" in link.attrib
    }
    mesh_filenames = [
        str(mesh.attrib.get("filename", ""))
        for mesh in root.findall(".//mesh")
        if mesh.attrib.get("filename")
    ]
    joints: list[UrdfJoint] = []

    for joint_xml in root.findall("joint"):
        joint_type = str(joint_xml.attrib.get("type", ""))
        if joint_type and joint_type != "fixed":
            continue
        parent_xml = joint_xml.find("parent")
        child_xml = joint_xml.find("child")
        if parent_xml is None or child_xml is None:
            continue
        parent = _canonical_frame_name(str(parent_xml.attrib.get("link", "")))
        child = _canonical_frame_name(str(child_xml.attrib.get("link", "")))
        if not parent or not child:
            continue

        origin = joint_xml.find("origin")
        xyz = _parse_float_triplet(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _parse_float_triplet(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        joints.append(
            UrdfJoint(
                name=str(joint_xml.attrib.get("name", f"{parent}_to_{child}")),
                parent=parent,
                child=child,
                transform_parent_child=_homogeneous_from_xyz_rpy(xyz, rpy),
            )
        )
        links.update({parent, child})

    return UrdfGeometry(path=path, links=links, joints=joints, mesh_filenames=mesh_filenames)


def transform_structured_points_to_base(
    structured_points: np.ndarray,
    transform_sensor_to_base: np.ndarray,
) -> np.ndarray:
    """Extract raw x/y/z fields and transform them into the URDF base frame."""

    for field_name in ("x", "y", "z"):
        if field_name not in structured_points.dtype.names:
            raise ValueError(
                "URDF projection requires PointCloud2 fields x, y, z; "
                f"available fields: {structured_points.dtype.names}"
            )

    x_s = structured_points["x"].astype(np.float64, copy=False)
    y_s = structured_points["y"].astype(np.float64, copy=False)
    z_s = structured_points["z"].astype(np.float64, copy=False)
    finite = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(z_s)
    if not finite.any():
        return np.zeros((0, 3), dtype=np.float64)

    sensor_xyz = np.column_stack((x_s[finite], y_s[finite], z_s[finite]))
    ones = np.ones((len(sensor_xyz), 1), dtype=np.float64)
    homogeneous = np.concatenate((sensor_xyz, ones), axis=1)
    return (transform_sensor_to_base @ homogeneous.T).T[:, :3]


def filter_base_points_for_detection(
    base_xyz: np.ndarray,
    config: DetectionConfig,
) -> FilteredPointCloud:
    """Apply ground, ego-self, and spatial filters in URDF base coordinates.

    URDF gives the detector a physical ego frame. That makes the self-filter
    meaningful: points inside the configurable vehicle safety box are removed
    before BEV clustering instead of being treated as opponent candidates.
    """

    if len(base_xyz) == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        empty_bev = np.zeros((0, 2), dtype=np.float32)
        return FilteredPointCloud(
            raw_points=empty_xyz,
            after_ground_points=empty_xyz,
            after_ego_points=empty_xyz,
            final_points=empty_xyz,
            raw_bev_points=empty_bev,
            after_ground_bev_points=empty_bev,
            after_ego_bev_points=empty_bev,
            final_bev_points=empty_bev,
            stats={
                "raw_points": 0,
                "ground_filter_points_rejected": 0,
                "ego_self_points_rejected": 0,
                "spatial_crop_points_rejected": 0,
            },
        )

    points = np.asarray(base_xyz, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    raw_points = points[finite]

    ground_mask = np.zeros(len(raw_points), dtype=bool)
    ground_z: float | None = None
    if config.ground_band_half_height > 0 and len(raw_points) > 10:
        horizontal = np.hypot(raw_points[:, 0], raw_points[:, 1])
        close = (horizontal >= 2.0) & (horizontal <= 25.0)
        if int(close.sum()) > 10:
            ground_z = float(np.median(raw_points[close, 2]))
            ground_mask = np.abs(raw_points[:, 2] - ground_z) <= config.ground_band_half_height
    after_ground = raw_points[~ground_mask]

    half_length = config.ego_length / 2.0 + config.safety_margin
    half_width = config.ego_width / 2.0 + config.safety_margin
    half_height = config.ego_height / 2.0 + config.safety_margin
    ego_mask = (
        (np.abs(after_ground[:, 0]) <= half_length)
        & (np.abs(after_ground[:, 1]) <= half_width)
        & (np.abs(after_ground[:, 2]) <= half_height)
    )
    after_ego = after_ground[~ego_mask]

    spatial_mask = _spatial_mask(after_ego, config)
    final_points = after_ego[spatial_mask]

    stats = {
        "input_points": int(len(points)),
        "finite_points": int(len(raw_points)),
        "raw_points": int(len(raw_points)),
        "ground_filter_enabled": bool(config.ground_band_half_height > 0),
        "ground_z_estimate": ground_z,
        "ground_filter_points_rejected": int(ground_mask.sum()),
        "ego_self_points_rejected": int(ego_mask.sum()),
        "spatial_crop_points_rejected": int((~spatial_mask).sum()),
        "final_points": int(len(final_points)),
    }

    return FilteredPointCloud(
        raw_points=raw_points,
        after_ground_points=after_ground,
        after_ego_points=after_ego,
        final_points=final_points,
        raw_bev_points=_bev_from_points(raw_points[_spatial_mask(raw_points, config)]),
        after_ground_bev_points=_bev_from_points(
            after_ground[_spatial_mask(after_ground, config)]
        ),
        after_ego_bev_points=_bev_from_points(after_ego[_spatial_mask(after_ego, config)]),
        final_bev_points=_bev_from_points(final_points),
        stats=stats,
    )


def run_task1_pipeline(
    bag_dir: str | Path,
    *,
    max_frames: int | str | None = DEFAULT_MAX_FRAMES,
    max_plots: int = DEFAULT_MAX_PLOTS,
    output_dir: str | Path = "outputs/task1",
    topic_name: str | None = None,
    ground_filter: bool = True,
    urdf_path: str | Path | None = None,
    base_frame: str | None = None,
    allow_identity_transform: bool = False,
    ego_length: float = 5.0,
    ego_width: float = 2.2,
    ego_height: float = 1.6,
    safety_margin: float = 0.25,
) -> PipelineResult:
    """Run metadata inspection, point-cloud detection, CSV export, and plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = parse_bag_summary(bag_dir)
    effective_max_frames = parse_max_frames_arg(max_frames)
    config = DetectionConfig(
        max_frames=effective_max_frames,
        max_plots=max_plots,
        output_dir=output_path,
        ground_band_half_height=0.30 if ground_filter else 0.0,
        ego_length=ego_length,
        ego_width=ego_width,
        ego_height=ego_height,
        safety_margin=safety_margin,
    )
    diagnostics: list[str] = []
    urdf_geometry, resolved_base_frame = _load_optional_urdf(
        summary.bag_dir,
        urdf_path,
        base_frame,
        diagnostics,
    )

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
            urdf_path=urdf_geometry.path if urdf_geometry else None,
            base_frame=resolved_base_frame,
        )
        _write_outputs(result, config)
        return result

    reader = SQLitePointCloudReader(summary)
    frames: list[FrameResult] = []
    all_detections: list[dict[str, Any]] = []
    axis_mapping: AxisMapping | None = None
    transform_cache: dict[str, np.ndarray] = {}

    for frame_index, message in enumerate(
        reader.iter_pointcloud_messages(
            selected_topic.name,
            max_frames=effective_max_frames,
        )
    ):
        try:
            structured_points = pointcloud2_to_structured_array(message["msg"])
        except Exception as exc:  # noqa: BLE001 - keep diagnostic explicit for challenge data.
            diagnostics.append(
                "PointCloud2 conversion failed for "
                f"{selected_topic.name} at timestamp {message['timestamp_ns']}: {exc}"
            )
            continue

        sensor_frame = _message_sensor_frame(message["source_frame"], selected_topic.name)
        sensor_kind = "radar" if "radar" in selected_topic.name.lower() else "lidar"

        if urdf_geometry is not None and resolved_base_frame is not None:
            transform = _resolve_sensor_transform(
                urdf_geometry,
                sensor_frame,
                resolved_base_frame,
                allow_identity_transform=allow_identity_transform,
                diagnostics=diagnostics,
                cache=transform_cache,
            )
            base_xyz = transform_structured_points_to_base(structured_points, transform)
            filtered = filter_base_points_for_detection(base_xyz, config)
            _add_filter_cluster_diagnostics(filtered, sensor_kind, config)
            bev_points = filtered.final_bev_points
        else:
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
                diagnostics.append(
                    "No URDF transform is active; BEV projection uses legacy "
                    "axis inference. Pass --urdf-path for geometry-backed evidence."
                )

            bev_points = extract_bev_points(
                structured_points, axis_mapping, config, frame_index=frame_index
            )
            filtered = _legacy_filtered_point_cloud(bev_points)
            _add_filter_cluster_diagnostics(filtered, sensor_kind, config)

        detections, cluster_stats = detect_opponents_with_diagnostics(
            bev_points,
            sensor_kind=sensor_kind,
            config=config,
            timestamp_ns=message["timestamp_ns"],
            topic=selected_topic.name,
            source_frame=message["source_frame"],
            sensor_frame=sensor_frame,
            base_frame=resolved_base_frame or "legacy_inferred_bev",
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
                sensor_frame=sensor_frame,
                base_frame=resolved_base_frame or "legacy_inferred_bev",
                raw_bev_points=filtered.raw_bev_points,
                after_ego_bev_points=filtered.after_ego_bev_points,
                filter_stats=filtered.stats,
                cluster_stats=cluster_stats,
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
        urdf_path=urdf_geometry.path if urdf_geometry else None,
        base_frame=resolved_base_frame,
        sensor_to_base_transforms=transform_cache,
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
        max_frames: int | None,
    ) -> Iterable[dict[str, Any]]:
        """Yield deserialized point cloud messages, optionally capped."""

        yielded = 0
        for db_path in self._db_paths():
            if max_frames is not None and yielded >= max_frames:
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

                if max_frames is None:
                    rows = connection.execute(
                        "select timestamp, data from messages where topic_id = ?",
                        (topic_id,),
                    )
                else:
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
        close_mask = mask & (forward >= 5.0) & (forward <= 25.0)
        close_vertical = vertical[close_mask]
        if len(close_vertical) > 10:
            ground_median = float(np.median(close_vertical))
            mask &= np.abs(vertical - ground_median) > config.ground_band_half_height

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


def detect_opponents_with_diagnostics(
    bev_points: np.ndarray,
    *,
    sensor_kind: str,
    config: DetectionConfig,
    timestamp_ns: int,
    topic: str,
    source_frame: str,
    sensor_frame: str,
    base_frame: str,
    frame_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cluster BEV points and return retained detections plus rejection counts."""

    if len(bev_points) == 0:
        return [], _empty_cluster_stats()

    is_radar = sensor_kind == "radar"
    eps = config.dbscan_eps_radar if is_radar else config.dbscan_eps_lidar
    min_samples = (
        config.dbscan_min_samples_radar if is_radar else config.dbscan_min_samples_lidar
    )
    min_points = config.min_points_radar if is_radar else config.min_points_lidar

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(bev_points)
    detections: list[dict[str, Any]] = []
    detection_id = 0
    stats = _empty_cluster_stats()
    stats["dbscan_cluster_count"] = int(len(set(labels)) - (1 if -1 in labels else 0))
    stats["dbscan_noise_points"] = int((labels == -1).sum())

    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster = bev_points[labels == label]
        point_count = len(cluster)
        stats["all_cluster_point_counts"].append(int(point_count))
        if point_count < min_points:
            stats["point_count_filter_rejected_clusters"] += 1
            continue

        min_forward, min_lateral = cluster.min(axis=0)
        max_forward, max_lateral = cluster.max(axis=0)
        length = float(max_forward - min_forward)
        width = float(max_lateral - min_lateral)

        if not (
            config.min_width <= width <= config.max_width
            and config.min_length <= length <= config.max_length
        ):
            stats["vehicle_size_filter_rejected_clusters"] += 1
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
                "timestamp_ns": timestamp_ns,
                "topic": topic,
                "source_frame": source_frame,
                "sensor_frame": sensor_frame,
                "base_frame": base_frame,
                "frame_index": frame_index,
                "frame_idx": frame_index,
                "detection_id": f"{frame_index:04d}_{detection_id:02d}",
                "center_x": center_forward,
                "center_y": center_lateral,
                "center_x_base": center_forward,
                "center_y_base": center_lateral,
                "width": width,
                "length": length,
                "min_x": float(min_forward),
                "max_x": float(max_forward),
                "min_y": float(min_lateral),
                "max_y": float(max_lateral),
                "min_x_base": float(min_forward),
                "max_x_base": float(max_forward),
                "min_y_base": float(min_lateral),
                "max_y_base": float(max_lateral),
                "point_count": point_count,
                "confidence": confidence,
            }
        )
        stats["retained_clusters"] += 1
        stats["retained_cluster_point_counts"].append(int(point_count))
        stats["retained_widths"].append(width)
        stats["retained_lengths"].append(length)
        detection_id += 1

    return detections, stats


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


def _load_optional_urdf(
    bag_dir: Path,
    urdf_path: str | Path | None,
    base_frame: str | None,
    diagnostics: list[str],
) -> tuple[UrdfGeometry | None, str | None]:
    candidate: Path | None = Path(urdf_path) if urdf_path is not None else None
    if candidate is None:
        default_candidate = bag_dir / "eva.urdf"
        if default_candidate.exists():
            candidate = default_candidate

    if candidate is None:
        diagnostics.append(
            "No URDF path was provided and no eva.urdf was found beside the bag; "
            "using legacy axis-inferred BEV projection."
        )
        return None, None

    geometry = load_urdf_geometry(candidate)
    selected_base = geometry.select_base_frame(base_frame)
    diagnostics.append(f"Loaded URDF geometry: {geometry.path.as_posix()}")
    diagnostics.append(f"URDF base/ego frame: {selected_base}")
    if geometry.mesh_filenames:
        diagnostics.append(
            "URDF mesh geometry is referenced but not dimensioned in the URDF; "
            "using configurable ego_length/ego_width/ego_height safety box."
        )
    return geometry, selected_base


def _resolve_sensor_transform(
    geometry: UrdfGeometry,
    sensor_frame: str,
    base_frame: str,
    *,
    allow_identity_transform: bool,
    diagnostics: list[str],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cache_key = f"{base_frame}<-{_canonical_frame_name(sensor_frame)}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        transform = geometry.transform_sensor_to_base(sensor_frame, base_frame)
    except ValueError as exc:
        if not allow_identity_transform:
            raise ValueError(
                f"Could not resolve URDF transform from sensor frame {sensor_frame!r} "
                f"to base frame {base_frame!r}: {exc} "
                "Pass --allow-identity-transform only for debugging."
            ) from exc
        transform = np.eye(4, dtype=np.float64)
        diagnostics.append(
            f"WARNING: using identity transform for {sensor_frame}->{base_frame} "
            "because --allow-identity-transform was passed."
        )
    cache[cache_key] = transform
    t = transform[:3, 3]
    diagnostics.append(
        f"URDF transform {sensor_frame}->{base_frame}: "
        f"translation=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
    )
    return transform


def _message_sensor_frame(source_frame: str, topic_name: str) -> str:
    frame = _canonical_frame_name(source_frame)
    if frame:
        return frame
    parts = [part for part in topic_name.split("/") if part]
    if len(parts) >= 2 and parts[0] == "sensor":
        return _canonical_frame_name(parts[1])
    return _canonical_frame_name(topic_name)


def _legacy_filtered_point_cloud(bev_points: np.ndarray) -> FilteredPointCloud:
    if len(bev_points) == 0:
        points = np.zeros((0, 3), dtype=np.float64)
        bev = np.zeros((0, 2), dtype=np.float32)
    else:
        bev = np.asarray(bev_points, dtype=np.float32)
        points = np.column_stack(
            (
                bev[:, 0].astype(np.float64),
                bev[:, 1].astype(np.float64),
                np.zeros(len(bev)),
            )
        )
    return FilteredPointCloud(
        raw_points=points,
        after_ground_points=points,
        after_ego_points=points,
        final_points=points,
        raw_bev_points=bev,
        after_ground_bev_points=bev,
        after_ego_bev_points=bev,
        final_bev_points=bev,
        stats={
            "input_points": int(len(points)),
            "finite_points": int(len(points)),
            "raw_points": int(len(points)),
            "ground_filter_points_rejected": 0,
            "ego_self_points_rejected": 0,
            "spatial_crop_points_rejected": 0,
            "final_points": int(len(points)),
            "legacy_axis_inference": True,
        },
    )


def _add_filter_cluster_diagnostics(
    filtered: FilteredPointCloud,
    sensor_kind: str,
    config: DetectionConfig,
) -> None:
    raw_clusters = _count_dbscan_clusters(filtered.raw_bev_points, sensor_kind, config)
    after_ground_clusters = _count_dbscan_clusters(
        filtered.after_ground_bev_points,
        sensor_kind,
        config,
    )
    after_ego_clusters = _count_dbscan_clusters(
        filtered.after_ego_bev_points,
        sensor_kind,
        config,
    )
    filtered.stats.update(
        {
            "raw_cluster_count": raw_clusters,
            "after_ground_cluster_count": after_ground_clusters,
            "after_ego_cluster_count": after_ego_clusters,
            "ground_filter_clusters_reduced": max(0, raw_clusters - after_ground_clusters),
            "ego_self_filter_clusters_reduced": max(
                0,
                after_ground_clusters - after_ego_clusters,
            ),
        }
    )


def _count_dbscan_clusters(
    bev_points: np.ndarray,
    sensor_kind: str,
    config: DetectionConfig,
) -> int:
    if len(bev_points) == 0:
        return 0
    is_radar = sensor_kind == "radar"
    eps = config.dbscan_eps_radar if is_radar else config.dbscan_eps_lidar
    min_samples = (
        config.dbscan_min_samples_radar if is_radar else config.dbscan_min_samples_lidar
    )
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(bev_points)
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def _empty_cluster_stats() -> dict[str, Any]:
    return {
        "dbscan_cluster_count": 0,
        "dbscan_noise_points": 0,
        "point_count_filter_rejected_clusters": 0,
        "vehicle_size_filter_rejected_clusters": 0,
        "retained_clusters": 0,
        "all_cluster_point_counts": [],
        "retained_cluster_point_counts": [],
        "retained_widths": [],
        "retained_lengths": [],
    }


def _write_outputs(result: PipelineResult, config: DetectionConfig) -> None:
    detections_csv = result.output_dir / "detections.csv"
    evaluation_csv = result.output_dir / "evaluation_summary.csv"
    evaluation_json = result.output_dir / "evaluation_summary.json"
    report_md = result.output_dir / "task1_report.md"
    evaluation_metrics = build_evaluation_metrics(result, config)
    evaluation_rows = build_evaluation_summary(result, config)

    if result.detections.empty:
        pd.DataFrame(columns=DETECTION_COLUMNS).to_csv(detections_csv, index=False)
    else:
        result.detections.to_csv(detections_csv, index=False)

    pd.DataFrame(evaluation_rows, columns=EVALUATION_COLUMNS).to_csv(
        evaluation_csv,
        index=False,
    )
    evaluation_json.write_text(
        json.dumps(evaluation_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_limit = len(result.frames) if config.max_plots is None else config.max_plots
    for frame in result.frames[:plot_limit]:
        plot_bev_frame(
            frame.bev_points,
            frame.detections,
            result.output_dir / f"frame_{frame.frame_index:03d}.png",
            title=f"Task 1 BEV detections | frame {frame.frame_index:03d}",
            forward_range=config.forward_range,
            lateral_range=config.lateral_range,
        )

    if result.frames:
        _plot_detection_count_over_time(
            result.frames,
            result.output_dir / "detection_count_over_time.png",
        )
        _write_representative_bev_plots(result, config)

    report_md.write_text(
        _build_report(result, evaluation_rows, evaluation_metrics, config),
        encoding="utf-8",
    )


def build_evaluation_metrics(
    result: PipelineResult,
    config: DetectionConfig,
) -> dict[str, Any]:
    """Build label-free evaluation evidence for robustness and filtering."""

    detections = result.detections
    detections_per_frame = [len(frame.detections) for frame in result.frames]
    frames_processed = len(result.frames)
    frames_with_detections = int(sum(count > 0 for count in detections_per_frame))

    filter_totals = _sum_frame_stats(frame.filter_stats for frame in result.frames)
    cluster_totals = _sum_frame_stats(frame.cluster_stats for frame in result.frames)
    all_cluster_sizes = _flatten_stat_lists(result.frames, "all_cluster_point_counts")
    retained_cluster_sizes = _flatten_stat_lists(result.frames, "retained_cluster_point_counts")
    temporal = _temporal_robustness_metrics(detections_per_frame)

    raw_clusters = int(filter_totals.get("raw_cluster_count", 0))
    after_ground = int(filter_totals.get("after_ground_cluster_count", 0))
    after_ego = int(filter_totals.get("after_ego_cluster_count", 0))
    final_detections = int(len(detections))

    metrics: dict[str, Any] = {
        "labelled_metrics_available": False,
        "labelled_metrics_note": (
            "No labelled opponent boxes/positions are available, so precision, "
            "recall, center error, and IoU cannot be computed honestly."
        ),
        "frames": {
            "total_frames_processed": frames_processed,
            "max_frames_argument": "all" if config.max_frames is None else config.max_frames,
            "frames_with_detections": frames_with_detections,
            "frames_with_detections_percent": _percent(
                frames_with_detections,
                frames_processed,
            ),
        },
        "detections": {
            "total_detections": final_detections,
            "detections_per_frame": _numeric_stats(detections_per_frame),
            "cluster_size_points_all_dbscan_clusters": _numeric_stats(all_cluster_sizes),
            "cluster_size_points_retained": _numeric_stats(retained_cluster_sizes),
            "bbox_width_m": _series_stats(detections, "width"),
            "bbox_length_m": _series_stats(detections, "length"),
        },
        "filters": {
            "ground_filter_points_rejected": int(
                filter_totals.get("ground_filter_points_rejected", 0)
            ),
            "ground_filter_clusters_reduced": int(
                filter_totals.get("ground_filter_clusters_reduced", 0)
            ),
            "ego_self_points_rejected": int(filter_totals.get("ego_self_points_rejected", 0)),
            "ego_self_clusters_reduced": int(
                filter_totals.get("ego_self_filter_clusters_reduced", 0)
            ),
            "point_count_filter_rejected_clusters": int(
                cluster_totals.get("point_count_filter_rejected_clusters", 0)
            ),
            "vehicle_size_filter_rejected_clusters": int(
                cluster_totals.get("vehicle_size_filter_rejected_clusters", 0)
            ),
            "raw_clusters_before_filters": raw_clusters,
            "clusters_after_ground_filter": after_ground,
            "clusters_after_ego_filter": after_ego,
            "final_detections_after_all_filters": final_detections,
            "overall_cluster_to_detection_reduction_percent": _percent(
                max(raw_clusters - final_detections, 0),
                raw_clusters,
            ),
            "ground_filter_cluster_reduction_percent": _percent(
                int(filter_totals.get("ground_filter_clusters_reduced", 0)),
                raw_clusters,
            ),
            "ego_filter_cluster_reduction_percent": _percent(
                int(filter_totals.get("ego_self_filter_clusters_reduced", 0)),
                after_ground,
            ),
            "point_count_cluster_rejection_percent": _percent(
                int(cluster_totals.get("point_count_filter_rejected_clusters", 0)),
                after_ego,
            ),
            "vehicle_size_cluster_rejection_percent": _percent(
                int(cluster_totals.get("vehicle_size_filter_rejected_clusters", 0)),
                after_ego,
            ),
            "ego_box_defaults": {
                "ego_length_m": config.ego_length,
                "ego_width_m": config.ego_width,
                "ego_height_m": config.ego_height,
                "safety_margin_m": config.safety_margin,
            },
        },
        "temporal_robustness": temporal,
        "urdf": {
            "urdf_path": result.urdf_path.as_posix() if result.urdf_path else None,
            "base_frame": result.base_frame,
            "sensor_frames": sorted({frame.sensor_frame for frame in result.frames if frame.sensor_frame}),
            "transforms": {
                key: _matrix_to_list(value)
                for key, value in sorted(result.sensor_to_base_transforms.items())
            },
        },
        "outputs": {
            "detections_csv": (result.output_dir / "detections.csv").as_posix(),
            "evaluation_summary_json": (
                result.output_dir / "evaluation_summary.json"
            ).as_posix(),
            "task1_report_md": (result.output_dir / "task1_report.md").as_posix(),
            "detection_count_plot": (
                result.output_dir / "detection_count_over_time.png"
            ).as_posix(),
        },
    }
    return _json_safe(metrics)


def build_evaluation_summary(
    result: PipelineResult,
    config: DetectionConfig,
) -> list[dict[str, str]]:
    """Build challenge-criteria evaluation rows for report and CSV output.

    This is not a labelled benchmark because the repository does not currently
    include ground-truth opponent boxes/positions. The rows distinguish true
    label-dependent metrics from run-time proxy checks that are available now.
    """

    detections = result.detections
    detections_per_frame = [len(frame.detections) for frame in result.frames]
    frames_processed = len(result.frames)
    frames_with_detections = sum(count > 0 for count in detections_per_frame)
    low_confidence_count = 0
    mean_confidence = "n/a"
    median_confidence = "n/a"
    mean_points = "n/a"
    mean_width = "n/a"
    mean_length = "n/a"

    if not detections.empty:
        low_confidence_count = int((detections["confidence"] < 0.50).sum())
        mean_confidence = f"{float(detections['confidence'].mean()):.3f}"
        median_confidence = f"{float(detections['confidence'].median()):.3f}"
        mean_points = f"{float(detections['point_count'].mean()):.1f}"
        mean_width = f"{float(detections['width'].mean()):.2f} m"
        mean_length = f"{float(detections['length'].mean()):.2f} m"

    mean_detections = (
        f"{float(np.mean(detections_per_frame)):.2f}" if detections_per_frame else "0.00"
    )
    max_detections = str(max(detections_per_frame) if detections_per_frame else 0)
    deserialization_failures = sum(
        "failed" in diagnostic.lower() for diagnostic in result.diagnostics
    )

    return [
        {
            "criterion": "Accuracy",
            "metric": "ground_truth_available",
            "value": "false",
            "interpretation": (
                "No labelled opponent positions/boxes are present in the project yet, "
                "so precision, recall, center error, and IoU cannot be computed honestly."
            ),
        },
        {
            "criterion": "Accuracy",
            "metric": "position_and_box_output",
            "value": "true",
            "interpretation": (
                "detections.csv contains target center_x/center_y plus BEV min/max "
                "bounds, width, length, and confidence for each candidate."
            ),
        },
        {
            "criterion": "Accuracy",
            "metric": "detections_total",
            "value": str(len(detections)),
            "interpretation": "Candidate opponent detections generated by the current run.",
        },
        {
            "criterion": "Robustness",
            "metric": "frames_processed",
            "value": str(frames_processed),
            "interpretation": (
                "Frame coverage for this run; --max-frames all removes the old "
                "20-frame cap."
            ),
        },
        {
            "criterion": "Robustness",
            "metric": "frames_with_detections",
            "value": f"{frames_with_detections}/{frames_processed}",
            "interpretation": "Frames with at least one retained candidate after filtering.",
        },
        {
            "criterion": "Robustness",
            "metric": "mean_detections_per_frame",
            "value": mean_detections,
            "interpretation": "Useful stability proxy before labelled temporal evaluation exists.",
        },
        {
            "criterion": "Robustness",
            "metric": "max_detections_in_frame",
            "value": max_detections,
            "interpretation": "Flags possible over-clustering bursts for manual plot review.",
        },
        {
            "criterion": "Robustness",
            "metric": "deserialization_failures",
            "value": str(deserialization_failures),
            "interpretation": "PointCloud2 frames skipped due to parsing/conversion failures.",
        },
        {
            "criterion": "False positive reduction",
            "metric": "spatial_crop",
            "value": (
                f"forward={config.forward_range[0]:.1f}..{config.forward_range[1]:.1f} m; "
                f"lateral={config.lateral_range[0]:.1f}..{config.lateral_range[1]:.1f} m"
            ),
            "interpretation": "Suppresses obvious off-track or far-field clutter before clustering.",
        },
        {
            "criterion": "False positive reduction",
            "metric": "cluster_filter",
            "value": (
                f"width={config.min_width:.1f}..{config.max_width:.1f} m; "
                f"length={config.min_length:.1f}..{config.max_length:.1f} m; "
                f"min_lidar_points={config.min_points_lidar}"
            ),
            "interpretation": "Retains clusters with plausible vehicle footprint and support.",
        },
        {
            "criterion": "False positive reduction",
            "metric": "ground_filter_enabled",
            "value": str(config.ground_band_half_height > 0).lower(),
            "interpretation": "Removes points close to the estimated ground band when enabled.",
        },
        {
            "criterion": "False positive reduction",
            "metric": "confidence_distribution",
            "value": (
                f"mean={mean_confidence}; median={median_confidence}; "
                f"low_confidence_lt_0.50={low_confidence_count}"
            ),
            "interpretation": "Confidence is a heuristic ranking score for manual review/tuning.",
        },
        {
            "criterion": "False positive reduction",
            "metric": "mean_candidate_shape",
            "value": (
                f"width={mean_width}; length={mean_length}; points={mean_points}"
            ),
            "interpretation": "Shape/support sanity check for retained candidate clusters.",
        },
    ]


def _build_report(
    result: PipelineResult,
    evaluation_rows: list[dict[str, str]],
    evaluation_metrics: dict[str, Any],
    config: DetectionConfig,
) -> str:
    filters = evaluation_metrics["filters"]
    temporal = evaluation_metrics["temporal_robustness"]
    lines = [
        "# Task 1 Opponent Vehicle Detection Report",
        "",
        "## Method",
        "",
        "Task 1 reads standard `sensor_msgs/msg/PointCloud2` messages directly "
        "from the ROS 2 sqlite bag, transforms LiDAR points into the URDF ego "
        "frame when a URDF is available, removes ground/self points, clusters "
        "the BEV points with DBSCAN, and retains clusters with vehicle-like "
        "support and dimensions.",
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
    lines.append(f"- URDF path: `{result.urdf_path.as_posix() if result.urdf_path else 'not used'}`")
    lines.append(f"- Base frame: `{result.base_frame or 'legacy_inferred_bev'}`")
    lines.append(
        "- Ego safety box defaults: "
        f"length={config.ego_length:.2f} m, width={config.ego_width:.2f} m, "
        f"height={config.ego_height:.2f} m, margin={config.safety_margin:.2f} m"
    )

    if result.axis_mapping:
        lines.append(
            "- Axis mapping: "
            f"forward=`{result.axis_mapping.forward}`, "
            f"lateral=`{result.axis_mapping.lateral}`, "
            f"vertical=`{result.axis_mapping.vertical or 'none'}`"
        )
        lines.append(f"- Axis mapping reason: {result.axis_mapping.reason}")

    if result.sensor_to_base_transforms:
        lines.extend(["", "## URDF Use", ""])
        lines.append(
            "The URDF provides the fixed transform from the LiDAR frame into the "
            "ego/base frame. BEV boxes are reported in that base frame using "
            "`center_x_base`, `center_y_base`, and base-frame min/max columns."
        )
        for key, transform in sorted(result.sensor_to_base_transforms.items()):
            t = transform[:3, 3]
            lines.append(
                f"- `{key}` translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
            )

    lines.extend(
        [
            "",
            "## Filter Evidence",
            "",
            f"- Raw DBSCAN clusters before filters: {filters['raw_clusters_before_filters']}",
            f"- Clusters after ground filter: {filters['clusters_after_ground_filter']}",
            f"- Clusters after ego/self filter: {filters['clusters_after_ego_filter']}",
            f"- Final detections after all filters: {filters['final_detections_after_all_filters']}",
            "- Overall raw-cluster to final-detection reduction: "
            f"{filters['overall_cluster_to_detection_reduction_percent']:.2f}%",
            f"- Ground-filter point rejects: {filters['ground_filter_points_rejected']}",
            f"- Ego/self point rejects: {filters['ego_self_points_rejected']}",
            "- Vehicle-size rejected clusters: "
            f"{filters['vehicle_size_filter_rejected_clusters']}",
            "- Point-count rejected clusters: "
            f"{filters['point_count_filter_rejected_clusters']}",
        ]
    )

    lines.extend(
        [
            "",
            "## Temporal Robustness",
            "",
            "- Detection counts are tracked for every processed frame and plotted in "
            "`detection_count_over_time.png`.",
            "- Sudden jumps/dropouts detected: "
            f"{len(temporal['sudden_jumps_or_dropouts'])}",
            f"- Zero-detection frames: {temporal['zero_detection_frames']}",
        ]
    )

    lines.extend(
        [
            "",
            "## Evaluation Criteria",
            "",
            "The challenge criteria are accuracy, robustness, and false positive reduction. "
            "This run includes the measurable parts below. True accuracy metrics still "
            "require labelled opponent positions/boxes.",
            "",
            "| Criterion | Metric | Value | Interpretation |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in evaluation_rows:
        lines.append(
            "| "
            f"{_markdown_cell(row['criterion'])} | "
            f"{_markdown_cell(row['metric'])} | "
            f"{_markdown_cell(row['value'])} | "
            f"{_markdown_cell(row['interpretation'])} |"
        )

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
            f"- Evaluation summary: `{(result.output_dir / 'evaluation_summary.csv').as_posix()}`",
            f"- Evaluation JSON: `{(result.output_dir / 'evaluation_summary.json').as_posix()}`",
            f"- Detection-count plot: `{(result.output_dir / 'detection_count_over_time.png').as_posix()}`",
            f"- BEV plots: `{(result.output_dir / 'frame_*.png').as_posix()}`",
            f"- Representative BEV plots: `{(result.output_dir / 'representative_frame_*_*.png').as_posix()}`",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "uv run python src/task1_detection.py --bag-path data --urdf-path data/eva.urdf --output-dir outputs/task1 --max-frames all",
            "```",
            "",
            "## Current Limitations",
            "",
            "- This first-pass detector processes one PointCloud2 topic at a time.",
            "- Label-based accuracy scoring needs ground-truth opponent positions or boxes; no precision, recall, or IoU is reported here.",
            "- Cluster boxes are axis-aligned in BEV and are intended as candidates for evaluation and tuning.",
            "- If a future bag uses custom point cloud message types, add their definitions to the deserializer or implement a topic-specific parser.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _sum_frame_stats(stats_iter: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for stats in stats_iter:
        for key, value in stats.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value is not None and np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _flatten_stat_lists(frames: list[FrameResult], key: str) -> list[float]:
    values: list[float] = []
    for frame in frames:
        item = frame.cluster_stats.get(key, [])
        if isinstance(item, list):
            values.extend(float(value) for value in item)
    return values


def _numeric_stats(values: Iterable[float | int]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "p05": float(np.percentile(data, 5)),
        "p95": float(np.percentile(data, 95)),
    }


def _series_stats(dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    if dataframe.empty or column not in dataframe.columns:
        return _numeric_stats([])
    return _numeric_stats(dataframe[column].astype(float).tolist())


def _percent(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 2)


def _temporal_robustness_metrics(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {
            "sudden_change_threshold": 0,
            "sudden_jumps_or_dropouts": [],
            "zero_detection_frames": 0,
        }
    deltas = np.diff(np.asarray(counts, dtype=np.float64))
    if len(deltas) == 0:
        threshold = 1
    else:
        abs_deltas = np.abs(deltas)
        threshold = int(max(3, math.ceil(float(abs_deltas.mean() + 2.0 * abs_deltas.std()))))

    events: list[dict[str, Any]] = []
    for index, delta in enumerate(deltas, start=1):
        previous_count = counts[index - 1]
        current_count = counts[index]
        is_dropout = previous_count > 0 and current_count == 0
        if abs(delta) >= threshold or is_dropout:
            events.append(
                {
                    "frame_idx": index,
                    "previous_count": int(previous_count),
                    "current_count": int(current_count),
                    "delta": int(delta),
                    "dropout_to_zero": bool(is_dropout),
                }
            )
    return {
        "detection_counts": [int(value) for value in counts],
        "detection_count_delta_stats": _numeric_stats(deltas.tolist() if len(deltas) else []),
        "sudden_change_threshold": threshold,
        "sudden_jumps_or_dropouts": events,
        "zero_detection_frames": int(sum(count == 0 for count in counts)),
    }


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plot_detection_count_over_time(
    frames: list[FrameResult],
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = [frame.frame_index for frame in frames]
    counts = [len(frame.detections) for frame in frames]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_indices, counts, color="#1f77b4", linewidth=1.8, marker="o", markersize=2.5)
    ax.set_title("Task 1 detection count over time")
    ax.set_xlabel("frame index")
    ax.set_ylabel("detections")
    ax.grid(True, color="#d8d8d8", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_representative_bev_plots(
    result: PipelineResult,
    config: DetectionConfig,
) -> None:
    for frame in _representative_frames(result.frames):
        prefix = result.output_dir / f"representative_frame_{frame.frame_index:03d}"
        plot_bev_frame(
            frame.raw_bev_points,
            [],
            f"{prefix}_raw.png",
            title=f"Task 1 raw URDF-base points | frame {frame.frame_index:03d}",
            forward_range=config.forward_range,
            lateral_range=config.lateral_range,
        )
        plot_bev_frame(
            frame.after_ego_bev_points,
            [],
            f"{prefix}_after_ego.png",
            title=f"Task 1 after ground + ego filtering | frame {frame.frame_index:03d}",
            forward_range=config.forward_range,
            lateral_range=config.lateral_range,
        )
        plot_bev_frame(
            frame.bev_points,
            frame.detections,
            f"{prefix}_final.png",
            title=f"Task 1 final clusters/detections | frame {frame.frame_index:03d}",
            forward_range=config.forward_range,
            lateral_range=config.lateral_range,
        )


def _representative_frames(frames: list[FrameResult]) -> list[FrameResult]:
    if not frames:
        return []
    counts = np.asarray([len(frame.detections) for frame in frames], dtype=np.float64)
    candidate_indices = {
        0,
        len(frames) - 1,
        int(np.argmax(counts)),
        int(np.argmin(np.abs(counts - np.median(counts)))),
    }
    return [frames[index] for index in sorted(candidate_indices)]


def _canonical_frame_name(frame_name: str | None) -> str:
    return str(frame_name or "").strip().strip("/")


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [float(item) for item in str(text).replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected three float values in URDF origin, got {text!r}")
    return parts[0], parts[1], parts[2]


def _homogeneous_from_xyz_rpy(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _spatial_mask(points_xyz: np.ndarray, config: DetectionConfig) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.zeros(0, dtype=bool)
    return (
        (points_xyz[:, 0] >= config.forward_range[0])
        & (points_xyz[:, 0] <= config.forward_range[1])
        & (points_xyz[:, 1] >= config.lateral_range[0])
        & (points_xyz[:, 1] <= config.lateral_range[1])
    )


def _bev_from_points(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points_xyz[:, :2].astype(np.float32, copy=False)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 1 opponent vehicle detection on ROS 2 bag point clouds."
    )
    parser.add_argument(
        "--bag-path",
        "--bag-dir",
        dest="bag_dir",
        default="data",
        help="Directory containing metadata.yaml and ROS 2 sqlite3 bag files.",
    )
    parser.add_argument(
        "--urdf-path",
        default=None,
        help="Vehicle URDF used to transform sensor points into the ego/base frame.",
    )
    parser.add_argument(
        "--base-frame",
        default=None,
        help="URDF base/ego frame. Defaults to base_link, ego, or center_of_gravity.",
    )
    parser.add_argument(
        "--allow-identity-transform",
        action="store_true",
        help="Debug fallback only: use identity if the URDF sensor transform is missing.",
    )
    parser.add_argument(
        "--max-frames",
        type=parse_max_frames_arg,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum point cloud frames to process, or 'all' for no limit.",
    )
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=None,
        help="Quick-test frame cap that overrides --max-frames when provided.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=DEFAULT_MAX_PLOTS,
        help="Maximum number of per-frame BEV plots to write.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/task1",
        help="Directory for detections, metrics, reports, and visualizations.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Optional PointCloud2 topic to process. If omitted, front LiDAR is preferred.",
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
    parser.add_argument("--ego-length", type=float, default=5.0)
    parser.add_argument("--ego-width", type=float, default=2.2)
    parser.add_argument("--ego-height", type=float, default=1.6)
    parser.add_argument("--safety-margin", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = parse_bag_summary(args.bag_dir)
    print(format_bag_summary(summary))

    if args.inspect_only:
        return

    effective_max_frames = (
        args.debug_max_frames if args.debug_max_frames is not None else args.max_frames
    )
    result = run_task1_pipeline(
        args.bag_dir,
        max_frames=effective_max_frames,
        max_plots=args.max_plots,
        output_dir=args.output_dir,
        topic_name=args.topic,
        ground_filter=not args.no_ground_filter,
        urdf_path=args.urdf_path,
        base_frame=args.base_frame,
        allow_identity_transform=args.allow_identity_transform,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
        ego_height=args.ego_height,
        safety_margin=args.safety_margin,
    )

    print("")
    print("Task 1 pipeline complete")
    print(f"- selected topic: {result.selected_topic.name if result.selected_topic else 'none'}")
    print(f"- frames processed: {len(result.frames)}")
    print(f"- detections: {len(result.detections)}")
    print(f"- outputs: {Path(args.output_dir)}")
    for diagnostic in result.diagnostics:
        print(f"- diagnostic: {diagnostic}")


if __name__ == "__main__":
    main()
