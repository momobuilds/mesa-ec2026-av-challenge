"""Full multi-sensor tracking pipeline for Task 3.

This module is the bridge between the per-sensor detector infrastructure
(Task 2) and the multi-object tracker (Task 3).  It:

  1. Runs per-sensor DBSCAN detection + oriented-bbox + Doppler (reusing
     everything from Task 2's run_task2.py pipeline).
  2. Builds a flat SensorDetection index for fast windowed lookup.
  3. Drives the MultiObjectTracker frame-by-frame over the master-clock
     (lidar_front) timestamps.
  4. Returns the tracker's full row log and all track objects for reporting.

Occlusion simulation
--------------------
Pass ``simulate_occlusion=(track_id, first_frame, last_frame)`` to
artificially suppress all detections for the given frame range.
The tracker must coast through the gap and re-associate on the far side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.bag_inspector import BagSummary
from src.sensor_detectors import detect_all_sensors
from src.task1_detection import (
    DetectionConfig, SQLitePointCloudReader, pointcloud2_to_structured_array,
)
from src.oriented_bbox import minimum_area_bbox
from src.radar_doppler import cluster_velocity, extract_cluster_points
from src.task2_fusion import (
    SensorDetection, ASSOC_WINDOW_NS,
    build_detection_index, gather_detections,
)
from src.tf_utils import EgoStateLog, transform_points
from src.multi_object_tracker import MultiObjectTracker
from src.track_state import Track, TrackStatus, TRACKS_CSV_COLUMNS

_MASTER_TOPIC = "/sensor/lidar_front/points"


def run_tracking_pipeline(
    summary: BagSummary,
    *,
    max_frames: int = 60,
    topic_names: list[str] | None = None,
    ego_log: EgoStateLog | None = None,
    n_init: int = 3,
    max_age: int = 5,
    window_ns: int = ASSOC_WINDOW_NS,
    no_ground_filter: bool = False,
    simulate_occlusion: tuple[int, int, int] | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[Track]]:
    """End-to-end tracking pipeline.

    Parameters
    ----------
    summary           : parsed BagSummary.
    max_frames        : frames per sensor to process.
    topic_names       : sensor topics (defaults to all 7).
    ego_log           : EgoStateLog for ego-motion compensation.
    n_init            : consecutive hits to confirm a track.
    max_age           : missed frames before deletion.
    window_ns         : ±ns detection gather window.
    no_ground_filter  : disable ground-plane filtering.
    simulate_occlusion: (chosen_track_id, first_dropped_frame, last_dropped_frame).
                        Drops all detections in that frame range for the occlusion demo.
    verbose           : print per-frame stats.

    Returns
    -------
    rows   : list of dicts matching TRACKS_CSV_COLUMNS.
    tracks : all Track objects (active + deleted) for summary reporting.
    """
    config = DetectionConfig(
        ground_band_half_height=0.0 if no_ground_filter else 0.30,
    )

    _default_sensors = [
        "/sensor/lidar_front/points",
        "/sensor/lidar_left/points",
        "/sensor/lidar_right/points",
        "/sensor/radar_front/points",
        "/sensor/radar_back/points",
        "/sensor/radar_left/points",
        "/sensor/radar_right/points",
    ]
    topics = topic_names or _default_sensors

    print("[Task 3] Running per-sensor detection …")
    all_frames = detect_all_sensors(
        summary, config, max_frames=max_frames,
        topic_names=topics, verbose=verbose,
    )

    # ── Oriented bbox ──────────────────────────────────────────────────────────
    print("[Task 3] Computing oriented bounding boxes …")
    oriented_boxes = _build_oriented_boxes(all_frames)

    # ── Doppler velocities ─────────────────────────────────────────────────────
    print("[Task 3] Extracting radar Doppler velocities …")
    doppler_vel = _build_doppler_vel(all_frames, summary, ego_log)

    # ── Build flat detection index ─────────────────────────────────────────────
    all_dets, ts_arr = build_detection_index(
        all_frames,
        oriented_boxes=oriented_boxes,
        doppler_vel=doppler_vel,
    )
    print(f"[Task 3] Detection index: {len(all_dets)} detections")

    # ── Master-clock timestamps ───────────────────────────────────────────────
    master_frames = all_frames.get(_MASTER_TOPIC, [])
    if not master_frames:
        print("[Task 3] Warning: no lidar_front frames; aborting.")
        return [], []
    master_ts_list = sorted(fr.timestamp_ns for fr in master_frames)

    # ── Occlusion sim: determine which frame indices to suppress ──────────────
    _occ_first, _occ_last = -1, -1
    if simulate_occlusion:
        _, _occ_first, _occ_last = simulate_occlusion

    # ── Run tracker ───────────────────────────────────────────────────────────
    tracker = MultiObjectTracker(
        ego_log=ego_log, n_init=n_init, max_age=max_age, window_ns=window_ns,
    )
    print(f"[Task 3] Tracking {len(master_ts_list)} master-clock frames …")

    for frame_idx, ts in enumerate(master_ts_list):
        dets = gather_detections(all_dets, ts_arr, ts, window_ns=window_ns)

        if _occ_first <= frame_idx <= _occ_last:
            if verbose:
                print(f"  frame {frame_idx}: occluded (simulated), dropping {len(dets)} detections")
            dets = []

        rows = tracker.step(ts, list(dets))

        if verbose:
            n_conf = sum(1 for r in rows if r["status"] == "CONFIRMED")
            n_coast = sum(1 for r in rows if r["status"] == "COASTING")
            print(f"  frame {frame_idx:3d}  ts={ts}  dets={len(dets)}"
                  f"  confirmed={n_conf}  coasting={n_coast}")

    all_tracks = tracker.all_tracks
    rows = tracker.all_rows
    n_conf = sum(1 for t in all_tracks if t.status != TrackStatus.DELETED or t.hits >= 3)
    print(f"[Task 3] Done. {len(rows)} track-rows, {len(all_tracks)} total tracks.")
    return rows, all_tracks


# ── Detection helpers (same logic as run_task2.py, kept local) ─────────────────

def _build_oriented_boxes(all_frames: dict) -> dict:
    oriented_boxes: dict = {}
    for topic, frames in all_frames.items():
        if "radar" in topic:
            continue
        topic_map: dict = {}
        for fr in frames:
            frame_map: dict = {}
            for det in fr.detections:
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
                topic_map[fr.frame_index] = frame_map
        oriented_boxes[topic] = topic_map
    return oriented_boxes


def _build_doppler_vel(
    all_frames: dict,
    summary: BagSummary,
    ego_log: EgoStateLog | None,
) -> dict:
    doppler_vel: dict = {}
    reader = SQLitePointCloudReader(summary)

    for topic, frames in all_frames.items():
        if "radar" not in topic:
            continue
        topic_velmap: dict = {}

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
                    es = ego_log.interpolate(fr.timestamp_ns)
                    ego_vx, ego_vy = es.get("vx", 0.0), es.get("vy", 0.0)
                except Exception:  # noqa: BLE001
                    pass

            vel_list: list[tuple[float, float]] = []
            for det in fr.detections:
                cx, cy = float(det.get("center_x", 0)), float(det.get("center_y", 0))
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
                        close = dx**2 + dy**2 < 25.0
                        orig_idx = finite.nonzero()[0][close]
                        if len(orig_idx) > 0:
                            cluster_pts = extract_cluster_points(structured, orig_idx)
                            vx, vy = cluster_velocity(cluster_pts, sensor_frame, ego_vx=ego_vx, ego_vy=ego_vy)
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
