"""Per-sensor detection in the ego (centre_of_gravity) frame for Task 2.

Unlike Task 1, raw 3-D points are transformed to the ego frame using the
hardcoded extrinsics from ``tf_utils`` *before* BEV projection and DBSCAN
clustering, so every sensor's output shares a common coordinate system.
Task 1's DBSCAN + dimension filter (``detect_opponents_in_frame``) is
reused unchanged.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from src.bag_inspector import BagSummary, TopicSummary, pointcloud_topics
from src.task1_detection import (
    DetectionConfig,
    FrameResult,
    SQLitePointCloudReader,
    detect_opponents_in_frame,
    pointcloud2_to_structured_array,
)
from src.tf_utils import transform_points

# Ego-frame spatial bounds applied after the sensor→ego transform.
# These are wider than Task 1's sensor-frame bounds to accommodate
# side- and rear-mounted sensors.
EGO_FORWARD_RANGE: tuple[float, float] = (-5.0, 95.0)   # x_ego (m)
EGO_LATERAL_RANGE: tuple[float, float] = (-20.0, 20.0)  # y_ego (m)


def iter_ego_frame_points(
    summary: BagSummary,
    topic: str,
    *,
    max_frames: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (timestamp_ns, ego_xyz) for each frame of *topic*.

    Returns full 3D ego-frame (N, 3) points without ground filtering.
    The existing API (ego_bev_points, detect_topic_frames, detect_all_sensors)
    is untouched.
    """
    reader = SQLitePointCloudReader(summary)
    for msg_dict in reader.iter_pointcloud_messages(topic, max_frames=max_frames):
        try:
            structured = pointcloud2_to_structured_array(msg_dict["msg"])
        except Exception:  # noqa: BLE001
            continue
        sensor_frame = msg_dict["source_frame"]
        for fld in ("x", "y", "z"):
            if fld not in structured.dtype.names:
                import warnings
                warnings.warn(
                    f"iter_ego_frame_points: field '{fld}' missing in {topic} "
                    f"frame {msg_dict.get('timestamp_ns')}; skipping",
                    stacklevel=2,
                )
                break
        else:
            x_s = structured["x"].astype(np.float64)
            y_s = structured["y"].astype(np.float64)
            z_s = structured["z"].astype(np.float64)
            finite = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(z_s)
            if finite.any():
                ego_xyz = transform_points(
                    np.column_stack([x_s[finite], y_s[finite], z_s[finite]]),
                    sensor_frame,
                )
                yield msg_dict["timestamp_ns"], ego_xyz


def ego_bev_points(
    structured_points: np.ndarray,
    sensor_frame: str,
    config: DetectionConfig,
    *,
    frame_index: int = -1,
    verbose: bool = False,
) -> np.ndarray:
    """Return ego-frame BEV points ``(x_ego, y_ego)`` as a float32 (N, 2) array.

    Steps:
      1. Extract raw sensor-frame (x, y, z) from the structured point cloud.
      2. Transform to ego frame via the hardcoded extrinsic for *sensor_frame*.
      3. Apply ego-frame spatial bounds.
      4. Apply ground-band filter using z_ego (if ``config.ground_band_half_height > 0``).
    """
    for field in ("x", "y", "z"):
        if field not in structured_points.dtype.names:
            return np.zeros((0, 2), dtype=np.float32)

    x_s = structured_points["x"].astype(np.float64)
    y_s = structured_points["y"].astype(np.float64)
    z_s = structured_points["z"].astype(np.float64)

    finite = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(z_s)
    if not finite.any():
        return np.zeros((0, 2), dtype=np.float32)

    ego_xyz = transform_points(
        np.column_stack([x_s[finite], y_s[finite], z_s[finite]]),
        sensor_frame,
    )
    x_e, y_e, z_e = ego_xyz[:, 0], ego_xyz[:, 1], ego_xyz[:, 2]

    mask = (
        np.isfinite(x_e) & np.isfinite(y_e) & np.isfinite(z_e)
        & (x_e >= EGO_FORWARD_RANGE[0]) & (x_e <= EGO_FORWARD_RANGE[1])
        & (y_e >= EGO_LATERAL_RANGE[0]) & (y_e <= EGO_LATERAL_RANGE[1])
    )

    if config.ground_band_half_height > 0 and mask.sum() > 0:
        total = int(mask.sum())
        # Use horizontal distance from ego origin so the close-range band
        # works for front-, side-, and rear-facing sensors alike.
        horiz = np.sqrt(x_e**2 + y_e**2)
        close = mask & (horiz >= 2.0) & (horiz <= 25.0)
        if close.sum() > 10:
            ground_z = float(np.median(z_e[close]))
            mask &= np.abs(z_e - ground_z) > config.ground_band_half_height
        if verbose:
            print(
                f"  frame {frame_index:3d} [{sensor_frame}]:"
                f" kept {mask.sum()}/{total} after ground filter"
            )

    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([x_e[mask], y_e[mask]]).astype(np.float32)


def detect_topic_frames(
    summary: BagSummary,
    topic: TopicSummary,
    config: DetectionConfig,
    *,
    max_frames: int,
    verbose: bool = False,
) -> list[FrameResult]:
    """Run ego-frame detection for all processed frames of one PointCloud2 topic."""
    reader = SQLitePointCloudReader(summary)
    sensor_kind = "radar" if "radar" in topic.name.lower() else "lidar"
    results: list[FrameResult] = []

    for frame_index, msg_dict in enumerate(
        reader.iter_pointcloud_messages(topic.name, max_frames=max_frames)
    ):
        try:
            structured = pointcloud2_to_structured_array(msg_dict["msg"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] deserialization failed {topic.name} frame {frame_index}: {exc}")
            continue

        sensor_frame = msg_dict["source_frame"]
        bev = ego_bev_points(
            structured, sensor_frame, config,
            frame_index=frame_index, verbose=verbose,
        )
        dets = detect_opponents_in_frame(
            bev,
            sensor_kind=sensor_kind,
            config=config,
            timestamp_ns=msg_dict["timestamp_ns"],
            topic=topic.name,
            source_frame=sensor_frame,
            frame_index=frame_index,
        )
        results.append(
            FrameResult(
                frame_index=frame_index,
                timestamp_ns=msg_dict["timestamp_ns"],
                topic=topic.name,
                source_frame=sensor_frame,
                bev_points=bev,
                detections=dets,
            )
        )

    return results


def detect_all_sensors(
    summary: BagSummary,
    config: DetectionConfig,
    *,
    max_frames: int,
    topic_names: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, list[FrameResult]]:
    """Run ego-frame detection for every requested PointCloud2 topic.

    Returns ``{topic_name: [FrameResult, ...]}``.  When *topic_names* is
    ``None`` all PointCloud2 topics in the bag are processed.
    """
    pc_topics = pointcloud_topics(summary)
    if topic_names is not None:
        name_set = set(topic_names)
        pc_topics = [t for t in pc_topics if t.name in name_set]

    results: dict[str, list[FrameResult]] = {}
    for topic in pc_topics:
        print(f"[sensor_detectors] {topic.name} …")
        frames = detect_topic_frames(
            summary, topic, config, max_frames=max_frames, verbose=verbose
        )
        results[topic.name] = frames
        n_dets = sum(len(fr.detections) for fr in frames)
        print(f"  {len(frames)} frames, {n_dets} detections")

    return results
