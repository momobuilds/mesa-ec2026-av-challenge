"""Utilities for inspecting the EC2026 ROS 2 bag metadata.

This module intentionally reads the YAML metadata first. The bag files are
large, so the default inspection path avoids scanning the SQLite messages table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicSummary:
    """A topic listed by the ROS 2 bag metadata."""

    name: str
    message_type: str
    message_count: int
    serialization_format: str


@dataclass(frozen=True)
class StorageFileSummary:
    """A storage file listed by the ROS 2 bag metadata."""

    path: str
    duration_s: float
    message_count: int
    starting_time_ns: int | None
    size_bytes: int | None


@dataclass(frozen=True)
class BagSummary:
    """Readable summary of a ROS 2 sqlite3 bag."""

    bag_dir: Path
    storage_identifier: str
    ros_distro: str | None
    duration_s: float
    starting_time_ns: int | None
    message_count: int
    topics: list[TopicSummary]
    storage_files: list[StorageFileSummary]


def load_metadata(bag_dir: str | Path) -> dict[str, Any]:
    """Load ``metadata.yaml`` from a ROS 2 bag directory."""

    bag_path = Path(bag_dir)
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag directory: {bag_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "rosbag2_bagfile_information" not in data:
        raise ValueError(f"{metadata_path} is not a ROS 2 bag metadata file")
    return data


def parse_bag_summary(bag_dir: str | Path) -> BagSummary:
    """Parse the bag metadata into a small dataclass."""

    bag_path = Path(bag_dir)
    info = load_metadata(bag_path)["rosbag2_bagfile_information"]

    topics = []
    for item in info.get("topics_with_message_count", []):
        topic_metadata = item.get("topic_metadata", {})
        topics.append(
            TopicSummary(
                name=str(topic_metadata.get("name", "")),
                message_type=str(topic_metadata.get("type", "")),
                message_count=int(item.get("message_count", 0)),
                serialization_format=str(topic_metadata.get("serialization_format", "")),
            )
        )

    storage_files = []
    for file_info in info.get("files", []):
        rel_path = str(file_info.get("path", ""))
        abs_path = bag_path / rel_path
        storage_files.append(
            StorageFileSummary(
                path=rel_path,
                duration_s=_nanoseconds_to_seconds(file_info.get("duration", {})),
                message_count=int(file_info.get("message_count", 0)),
                starting_time_ns=file_info.get("starting_time", {}).get("nanoseconds_since_epoch"),
                size_bytes=abs_path.stat().st_size if abs_path.exists() else None,
            )
        )

    if not storage_files:
        for rel_path in info.get("relative_file_paths", []):
            abs_path = bag_path / rel_path
            storage_files.append(
                StorageFileSummary(
                    path=str(rel_path),
                    duration_s=0.0,
                    message_count=0,
                    starting_time_ns=None,
                    size_bytes=abs_path.stat().st_size if abs_path.exists() else None,
                )
            )

    return BagSummary(
        bag_dir=bag_path,
        storage_identifier=str(info.get("storage_identifier", "")),
        ros_distro=info.get("ros_distro"),
        duration_s=_nanoseconds_to_seconds(info.get("duration", {})),
        starting_time_ns=info.get("starting_time", {}).get("nanoseconds_since_epoch"),
        message_count=int(info.get("message_count", 0)),
        topics=topics,
        storage_files=storage_files,
    )


def pointcloud_topics(summary: BagSummary) -> list[TopicSummary]:
    """Return all topics that look like standard ROS ``PointCloud2`` streams."""

    return [
        topic
        for topic in summary.topics
        if topic.message_type == "sensor_msgs/msg/PointCloud2"
        or topic.message_type.endswith("/PointCloud2")
    ]


def select_default_pointcloud_topic(summary: BagSummary) -> TopicSummary | None:
    """Pick a conservative default topic for single-sensor Task 1 detection."""

    pointclouds = pointcloud_topics(summary)
    if not pointclouds:
        return None

    priority_names = [
        "/sensor/lidar_front/points",
        "/sensor/lidar_left/points",
        "/sensor/lidar_right/points",
        "/sensor/radar_front/points",
        "/sensor/radar_left/points",
        "/sensor/radar_right/points",
        "/sensor/radar_back/points",
    ]
    by_name = {topic.name: topic for topic in pointclouds}
    for name in priority_names:
        if name in by_name:
            return by_name[name]

    lidar_topics = [topic for topic in pointclouds if "lidar" in topic.name.lower()]
    if lidar_topics:
        return sorted(lidar_topics, key=lambda topic: topic.name)[0]

    radar_topics = [topic for topic in pointclouds if "radar" in topic.name.lower()]
    if radar_topics:
        return sorted(radar_topics, key=lambda topic: topic.name)[0]

    return sorted(pointclouds, key=lambda topic: topic.name)[0]


def format_bag_summary(summary: BagSummary) -> str:
    """Format a readable bag summary for stdout and markdown reports."""

    lines = [
        "ROS 2 bag summary",
        f"- Bag directory: {summary.bag_dir}",
        f"- Storage: {summary.storage_identifier}",
        f"- ROS distro: {summary.ros_distro or 'unknown'}",
        f"- Duration: {summary.duration_s:.3f} s",
        f"- Start time: {_format_time(summary.starting_time_ns)}",
        f"- Metadata message count: {summary.message_count}",
        "",
        "Storage files:",
    ]

    for file_info in summary.storage_files:
        size = _format_size(file_info.size_bytes)
        start = _format_time(file_info.starting_time_ns)
        lines.append(
            "- "
            f"{file_info.path} | duration={file_info.duration_s:.3f}s | "
            f"messages={file_info.message_count} | size={size} | start={start}"
        )

    lines.extend(["", "Topics:"])
    for topic in sorted(summary.topics, key=lambda item: item.name):
        lines.append(
            "- "
            f"{topic.name} | type={topic.message_type} | "
            f"messages={topic.message_count} | serialization={topic.serialization_format}"
        )

    return "\n".join(lines)


def _nanoseconds_to_seconds(value: dict[str, Any]) -> float:
    nanoseconds = int(value.get("nanoseconds", 0))
    return nanoseconds / 1_000_000_000


def _format_time(timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return "unknown"
    timestamp_s = timestamp_ns / 1_000_000_000
    return datetime.fromtimestamp(timestamp_s, tz=timezone.utc).isoformat()


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "missing"
    gib = size_bytes / (1024**3)
    return f"{gib:.2f} GiB"


if __name__ == "__main__":
    print(format_bag_summary(parse_bag_summary("data")))
