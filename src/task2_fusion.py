"""Multi-sensor fusion pipeline for Task 2.

Architecture
------------
1. **Master clock** - LiDAR-front timestamps.
2. **Gather** - for each master timestamp, collect detections from every
   other sensor within +/-50 ms.
3. **Association** - Hungarian matching (``scipy.optimize.linear_sum_assignment``)
   on Mahalanobis distance with per-sensor position covariance.
4. **EKF update** - single correction per matched sensor group, state vector
   ``[x, y, vx, vy, yaw, length, width]``.
5. **Temporal filter** - assign track IDs, estimate fallback velocity, and
   predict through short missing-data gaps.
6. **Camera validation** - projection-gate fused candidates against synced
   compressed camera frames when no image detector is available.
7. **Output** - one row per fused detection/prediction per master timestamp.

Sensor covariance assumptions
------------------------------
LiDAR: small position uncertainty, large velocity uncertainty.
Radar: large position uncertainty, small velocity uncertainty (Doppler).

All coordinates are in the ego (centre_of_gravity) frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.camera_projection import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    bbox_3d_corners,
    project_points,
)
from src.task1_detection import FrameResult

# ── State vector indices ──────────────────────────────────────────────────────
_X, _Y, _VX, _VY, _YAW, _L, _W = range(7)
STATE_DIM = 7

# ── Per-sensor measurement noise (diagonal R matrix) ─────────────────────────
# [x_pos (m²), y_pos (m²), vx (m²/s²), vy (m²/s²)]
_R_LIDAR = np.diag([0.5, 0.5, 9.0, 9.0]).astype(np.float64)   # good pos, poor vel
_R_RADAR = np.diag([4.0, 4.0, 0.4, 0.4]).astype(np.float64)   # poor pos, good vel

# Maximum Mahalanobis distance for a valid match
_MAX_MAHAL = 5.0

# Association time window (nanoseconds)
ASSOC_WINDOW_NS = int(50e6)  # 50 ms

# Fused detections CSV columns
FUSED_COLUMNS = [
    "timestamp_ns", "timestamp", "master_topic", "detection_id",
    "center_x", "center_y", "x", "y", "vx", "vy", "speed",
    "yaw", "length", "width",
    "cov_xx", "cov_yy", "cov_vxvx", "cov_vyvy",
    "confidence", "sensors",
    "n_lidar", "n_radar",
    "n_camera", "camera_topic", "camera_validated",
    "camera_u", "camera_v", "camera_score",
    "track_id", "age", "missed_frames", "is_predicted",
    "velocity_source",
]


# ── Detection-level data class ────────────────────────────────────────────────

@dataclass
class SensorDetection:
    """One DBSCAN detection from a single sensor, ego frame."""

    timestamp_ns: int
    topic: str
    source_frame: str
    frame_index: int
    detection_id: str
    center_x: float
    center_y: float
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    length: float = 4.5
    width: float = 2.0
    confidence: float = 0.5
    cluster_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def is_radar(self) -> bool:
        return "radar" in self.topic.lower()

    def position_cov(self) -> np.ndarray:
        """2×2 position covariance (diagonal) appropriate for this sensor."""
        R = _R_RADAR if self.is_radar else _R_LIDAR
        return R[:2, :2]

    def state_vector(self) -> np.ndarray:
        return np.array(
            [self.center_x, self.center_y, self.vx, self.vy,
             self.yaw, self.length, self.width],
            dtype=np.float64,
        )

    def full_R(self) -> np.ndarray:
        return _R_RADAR.copy() if self.is_radar else _R_LIDAR.copy()


@dataclass
class _FusionTrack:
    """Small Task 2 track used for velocity and short missing-data gaps."""

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    length: float
    width: float
    confidence: float
    last_ts: int
    age: int = 1
    missed_frames: int = 0

    def predicted_xy(self, timestamp_ns: int) -> tuple[float, float]:
        dt = max((timestamp_ns - self.last_ts) / 1_000_000_000.0, 0.0)
        return self.x + self.vx * dt, self.y + self.vy * dt


# ── EKF update (single step, constant-velocity model) ────────────────────────

def _ekf_update(
    state: np.ndarray,        # (7,)
    P: np.ndarray,            # (7, 7) covariance
    measurement: np.ndarray,  # (4,) = [x, y, vx, vy]
    R: np.ndarray,            # (4, 4) measurement noise
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman correction step for measurement model z = Hx + noise."""
    H = np.zeros((4, STATE_DIM), dtype=np.float64)
    H[0, _X], H[1, _Y], H[2, _VX], H[3, _VY] = 1, 1, 1, 1

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    z = measurement
    z_hat = H @ state
    state_new = state + K @ (z - z_hat)
    P_new = (np.eye(STATE_DIM) - K @ H) @ P
    return state_new, P_new


def _init_state(det: SensorDetection) -> tuple[np.ndarray, np.ndarray]:
    """Initialise EKF state from a single detection."""
    state = det.state_vector()
    R = det.full_R()
    P = np.eye(STATE_DIM, dtype=np.float64)
    P[_X, _X] = R[0, 0]
    P[_Y, _Y] = R[1, 1]
    P[_VX, _VX] = R[2, 2]
    P[_VY, _VY] = R[3, 3]
    P[_YAW, _YAW] = (math.pi / 4) ** 2
    P[_L, _L] = 1.5 ** 2
    P[_W, _W] = 0.5 ** 2
    return state, P


# ── Association ───────────────────────────────────────────────────────────────

def _mahalanobis(
    det_a: SensorDetection,
    det_b: SensorDetection,
) -> float:
    diff = np.array([det_a.center_x - det_b.center_x,
                     det_a.center_y - det_b.center_y], dtype=np.float64)
    S = det_a.position_cov() + det_b.position_cov()
    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("inf")
    d2 = float(diff @ inv_S @ diff)
    return math.sqrt(max(d2, 0.0))


def _associate(
    anchors: list[SensorDetection],
    candidates: list[SensorDetection],
) -> list[tuple[int, int]]:
    """Return matched (anchor_idx, candidate_idx) pairs below the distance gate."""
    if not anchors or not candidates:
        return []

    cost = np.full((len(anchors), len(candidates)), _MAX_MAHAL * 10, dtype=np.float64)
    for i, a in enumerate(anchors):
        for j, b in enumerate(candidates):
            d = _mahalanobis(a, b)
            if d < _MAX_MAHAL:
                cost[i, j] = d

    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if cost[r, c] < _MAX_MAHAL
    ]


# ── Per-frame fusion ──────────────────────────────────────────────────────────

def fuse_frame(
    master_ts: int,
    master_dets: list[SensorDetection],
    other_dets: list[SensorDetection],
    master_topic: str,
) -> list[dict[str, Any]]:
    """Fuse detections from the master sensor with nearby other-sensor detections.

    For each master detection, collect matching other-sensor detections within
    the association gate and apply EKF correction for each match.

    Returns a list of fused detection dicts with FUSED_COLUMNS fields.
    """
    results: list[dict[str, Any]] = []

    for track_id, anchor in enumerate(master_dets):
        state, P = _init_state(anchor)
        sensors: list[str] = [anchor.topic]
        n_lidar = 1 if not anchor.is_radar else 0
        n_radar = 1 if anchor.is_radar else 0

        # Try to match with every contributing other-sensor detection
        matched_others = _associate([anchor], other_dets)
        for _, j in matched_others:
            companion = other_dets[j]
            meas = np.array(
                [companion.center_x, companion.center_y,
                 companion.vx, companion.vy],
                dtype=np.float64,
            )
            state, P = _ekf_update(state, P, meas, companion.full_R())
            sensors.append(companion.topic)
            if companion.is_radar:
                n_radar += 1
            else:
                n_lidar += 1

        # Reconcile yaw with velocity if we have a good velocity estimate
        state[_YAW] = _reconcile_yaw(
            float(state[_YAW]), float(state[_VX]), float(state[_VY])
        )

        fused_conf = float(np.clip(
            anchor.confidence * (1 + 0.1 * (len(sensors) - 1)), 0.0, 1.0
        ))

        results.append({
            "timestamp_ns": master_ts,
            "timestamp": master_ts,
            "master_topic": master_topic,
            "detection_id": f"{master_ts}_{track_id:02d}",
            "center_x": float(state[_X]),
            "center_y": float(state[_Y]),
            "x": float(state[_X]),
            "y": float(state[_Y]),
            "vx": float(state[_VX]),
            "vy": float(state[_VY]),
            "speed": float(math.hypot(state[_VX], state[_VY])),
            "yaw": float(state[_YAW]),
            "length": float(state[_L]),
            "width": float(state[_W]),
            "cov_xx": float(P[_X, _X]),
            "cov_yy": float(P[_Y, _Y]),
            "cov_vxvx": float(P[_VX, _VX]),
            "cov_vyvy": float(P[_VY, _VY]),
            "confidence": fused_conf,
            "sensors": "+".join(sorted(set(
                t.split("/")[2] if "/" in t else t for t in sensors
            ))),
            "n_lidar": n_lidar,
            "n_radar": n_radar,
            "n_camera": 0,
            "camera_topic": "",
            "camera_validated": False,
            "camera_u": "",
            "camera_v": "",
            "camera_score": 0.0,
            "track_id": -1,
            "age": 1,
            "missed_frames": 0,
            "is_predicted": False,
            "velocity_source": (
                "radar" if n_radar > 0 and math.hypot(state[_VX], state[_VY]) > 0.1
                else "none"
            ),
        })

    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def _to_sensor_detections(
    frames: list[FrameResult],
    *,
    oriented_boxes: dict[str, dict[int, dict[str, Any]]] | None = None,
    doppler_vel: dict[str, dict[int, list[tuple[float, float]]]] | None = None,
) -> dict[int, list[SensorDetection]]:
    """Convert FrameResult lists → {timestamp_ns: [SensorDetection]} mapping."""
    by_ts: dict[int, list[SensorDetection]] = {}
    for fr in frames:
        ts = fr.timestamp_ns
        for det in fr.detections:
            # Oriented bbox override
            yaw = 0.0
            length = float(det.get("length", 4.5))
            width = float(det.get("width", 2.0))
            if oriented_boxes and fr.topic in oriented_boxes:
                obb = oriented_boxes[fr.topic].get(fr.frame_index, {}).get(
                    det["detection_id"]
                )
                if obb:
                    yaw = float(obb.get("yaw", 0.0))
                    length = float(obb.get("length", length))
                    width = float(obb.get("width", width))

            # Doppler velocity override
            vx, vy = 0.0, 0.0
            if doppler_vel and fr.topic in doppler_vel:
                vel_list = doppler_vel[fr.topic].get(fr.frame_index, [])
                det_idx = int(det["detection_id"].split("_")[-1]) if "_" in str(det["detection_id"]) else 0
                if det_idx < len(vel_list):
                    vx, vy = vel_list[det_idx]

            sd = SensorDetection(
                timestamp_ns=ts,
                topic=fr.topic,
                source_frame=fr.source_frame,
                frame_index=fr.frame_index,
                detection_id=str(det["detection_id"]),
                center_x=float(det["center_x"]),
                center_y=float(det["center_y"]),
                vx=vx,
                vy=vy,
                yaw=yaw,
                length=length,
                width=width,
                confidence=float(det.get("confidence", 0.5)),
            )
            by_ts.setdefault(ts, []).append(sd)
    return by_ts


def build_detection_index(
    all_frames: dict[str, list[FrameResult]],
    *,
    oriented_boxes: dict | None = None,
    doppler_vel: dict | None = None,
) -> tuple[list[SensorDetection], np.ndarray]:
    """Convert all FrameResult lists into a flat sorted list of SensorDetections.

    Returns
    -------
    all_dets : sorted by timestamp_ns
    ts_arr   : int64 numpy array of timestamps (same order) for searchsorted
    """
    all_dets: list[SensorDetection] = []
    for topic, frames in all_frames.items():
        by_ts = _to_sensor_detections(frames, oriented_boxes=oriented_boxes, doppler_vel=doppler_vel)
        for dets in by_ts.values():
            all_dets.extend(dets)
    all_dets.sort(key=lambda d: d.timestamp_ns)
    ts_arr = np.array([d.timestamp_ns for d in all_dets], dtype=np.int64)
    return all_dets, ts_arr


def gather_detections(
    all_dets: list[SensorDetection],
    ts_arr: np.ndarray,
    timestamp_ns: int,
    window_ns: int = ASSOC_WINDOW_NS,
) -> list[SensorDetection]:
    """Return all detections within ±window_ns of timestamp_ns."""
    lo, hi = timestamp_ns - window_ns, timestamp_ns + window_ns
    idx = np.searchsorted(ts_arr, [lo, hi])
    return all_dets[idx[0]: idx[1]]


def run_fusion(
    all_frames: dict[str, list[FrameResult]],
    *,
    master_topic: str = "/sensor/lidar_front/points",
    window_ns: int = ASSOC_WINDOW_NS,
    oriented_boxes: dict | None = None,
    doppler_vel: dict | None = None,
    camera_index: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the full fusion pipeline and return a fused-detections DataFrame.

    Parameters
    ----------
    all_frames : {topic_name: [FrameResult]} from ``detect_all_sensors``.
    master_topic : topic whose timestamps drive the fusion clock.
    window_ns : ±ns window for gathering other-sensor detections.
    oriented_boxes : optional {topic: {frame_idx: {det_id: obb_dict}}}.
    doppler_vel : optional {topic: {frame_idx: [(vx, vy), ...]}}.
    camera_index : optional {"topic": str, "timestamps_ns": np.ndarray}
        used for projection-gating camera validation.
    """
    if master_topic not in all_frames:
        raise ValueError(
            f"Master topic '{master_topic}' not in all_frames; "
            f"available: {list(all_frames)}"
        )

    # Build per-topic detection maps
    by_topic_ts: dict[str, dict[int, list[SensorDetection]]] = {}
    for topic, frames in all_frames.items():
        by_topic_ts[topic] = _to_sensor_detections(
            frames, oriented_boxes=oriented_boxes, doppler_vel=doppler_vel
        )

    master_ts_list = sorted(by_topic_ts[master_topic].keys())
    other_topics = [t for t in all_frames if t != master_topic]

    # Flatten other-sensor detections for quick window lookup
    other_all: list[SensorDetection] = []
    for topic in other_topics:
        for dets in by_topic_ts[topic].values():
            other_all.extend(dets)
    other_all.sort(key=lambda d: d.timestamp_ns)
    other_ts = np.array([d.timestamp_ns for d in other_all], dtype=np.int64)

    all_fused: list[dict[str, Any]] = []
    for ts in master_ts_list:
        master_dets = by_topic_ts[master_topic].get(ts, [])
        if not master_dets:
            continue

        # Gather other-sensor detections within window
        lo, hi = ts - window_ns, ts + window_ns
        idx = np.searchsorted(other_ts, [lo, hi])
        window_dets = other_all[idx[0]: idx[1]]

        fused = fuse_frame(ts, master_dets, window_dets, master_topic)
        all_fused.extend(fused)

    if not all_fused:
        return pd.DataFrame(columns=FUSED_COLUMNS)

    tracked = _apply_temporal_filter(
        all_fused,
        camera_index=camera_index,
        window_ns=window_ns,
        master_topic=master_topic,
    )
    return pd.DataFrame(tracked, columns=FUSED_COLUMNS)


def _apply_temporal_filter(
    rows: list[dict[str, Any]],
    *,
    camera_index: dict[str, Any] | None,
    window_ns: int,
    master_topic: str,
) -> list[dict[str, Any]]:
    """Attach track IDs, estimate fallback velocity, and emit short predictions."""

    by_ts: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_ts.setdefault(int(row["timestamp_ns"]), []).append(dict(row))

    tracks: dict[int, _FusionTrack] = {}
    next_track_id = 1
    output: list[dict[str, Any]] = []
    max_missed = 2
    association_gate_m = 10.0

    for timestamp_ns in sorted(by_ts):
        detections = by_ts[timestamp_ns]
        track_ids = [
            track_id for track_id, track in tracks.items()
            if track.missed_frames <= max_missed
        ]
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        if track_ids and detections:
            cost = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float64)
            for row_index, track_id in enumerate(track_ids):
                pred_x, pred_y = tracks[track_id].predicted_xy(timestamp_ns)
                for col_index, det in enumerate(detections):
                    cost[row_index, col_index] = math.hypot(
                        float(det["center_x"]) - pred_x,
                        float(det["center_y"]) - pred_y,
                    )
            row_idx, col_idx = linear_sum_assignment(cost)
            for row_index, col_index in zip(row_idx, col_idx):
                if cost[row_index, col_index] <= association_gate_m:
                    track_id = track_ids[int(row_index)]
                    det_index = int(col_index)
                    matches.append((track_id, det_index))
                    unmatched_tracks.discard(track_id)
                    unmatched_detections.discard(det_index)

        for track_id, det_index in matches:
            row = _update_track_with_detection(tracks[track_id], detections[det_index])
            _annotate_camera(row, camera_index, window_ns)
            output.append(row)

        for det_index in sorted(unmatched_detections):
            row = detections[det_index]
            radar_speed = math.hypot(float(row["vx"]), float(row["vy"]))
            velocity_source = (
                "radar"
                if int(row.get("n_radar", 0)) > 0 and radar_speed > 0.1
                else "initialized"
            )
            if radar_speed > 1.0:
                row["yaw"] = math.atan2(float(row["vy"]), float(row["vx"]))

            track = _FusionTrack(
                track_id=next_track_id,
                x=float(row["center_x"]),
                y=float(row["center_y"]),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                yaw=float(row["yaw"]),
                length=float(row["length"]),
                width=float(row["width"]),
                confidence=float(row["confidence"]),
                last_ts=timestamp_ns,
            )
            tracks[next_track_id] = track
            row.update({
                "timestamp": int(row["timestamp_ns"]),
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "speed": radar_speed,
                "track_id": next_track_id,
                "age": track.age,
                "missed_frames": track.missed_frames,
                "is_predicted": False,
                "velocity_source": velocity_source,
            })
            _annotate_camera(row, camera_index, window_ns)
            output.append(row)
            next_track_id += 1

        for track_id in sorted(unmatched_tracks):
            track = tracks[track_id]
            track.missed_frames += 1
            if track.age < 2 or track.missed_frames > max_missed:
                continue
            row = _predicted_track_row(
                track,
                timestamp_ns=timestamp_ns,
                master_topic=master_topic,
            )
            _annotate_camera(row, camera_index, window_ns)
            output.append(row)

        tracks = {
            track_id: track
            for track_id, track in tracks.items()
            if track.missed_frames <= max_missed
        }

    output.sort(key=lambda row: (int(row["timestamp_ns"]), int(row["track_id"])))
    return output


def _update_track_with_detection(
    track: _FusionTrack,
    row: dict[str, Any],
) -> dict[str, Any]:
    timestamp_ns = int(row["timestamp_ns"])
    dt = max((timestamp_ns - track.last_ts) / 1_000_000_000.0, 1e-3)
    det_x = float(row["center_x"])
    det_y = float(row["center_y"])
    pred_x, pred_y = track.predicted_xy(timestamp_ns)

    temporal_vx, temporal_vy = _limit_velocity_pair(
        (det_x - track.x) / dt,
        (det_y - track.y) / dt,
    )
    radar_speed = math.hypot(float(row["vx"]), float(row["vy"]))

    if int(row.get("n_radar", 0)) > 0 and radar_speed > 0.1:
        measured_vx = float(row["vx"])
        measured_vy = float(row["vy"])
        velocity_source = "radar"
    else:
        measured_vx = temporal_vx
        measured_vy = temporal_vy
        velocity_source = "temporal"

    alpha = 0.65 if track.age > 1 else 0.85
    vx, vy = _limit_velocity_pair(
        alpha * measured_vx + (1.0 - alpha) * track.vx,
        alpha * measured_vy + (1.0 - alpha) * track.vy,
    )
    speed = math.hypot(vx, vy)
    yaw = math.atan2(vy, vx) if speed > 1.0 else float(row["yaw"])

    smoothed_x = 0.85 * det_x + 0.15 * pred_x
    smoothed_y = 0.85 * det_y + 0.15 * pred_y

    track.x = smoothed_x
    track.y = smoothed_y
    track.vx = vx
    track.vy = vy
    track.yaw = yaw
    track.length = float(row["length"])
    track.width = float(row["width"])
    track.confidence = float(row["confidence"])
    track.last_ts = timestamp_ns
    track.age += 1
    track.missed_frames = 0

    row.update({
        "timestamp": timestamp_ns,
        "center_x": smoothed_x,
        "center_y": smoothed_y,
        "x": smoothed_x,
        "y": smoothed_y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "yaw": yaw,
        "cov_vxvx": min(
            float(row["cov_vxvx"]),
            4.0 if velocity_source == "temporal" else float(row["cov_vxvx"]),
        ),
        "cov_vyvy": min(
            float(row["cov_vyvy"]),
            4.0 if velocity_source == "temporal" else float(row["cov_vyvy"]),
        ),
        "confidence": float(np.clip(float(row["confidence"]) + min(track.age, 5) * 0.01, 0.0, 1.0)),
        "track_id": track.track_id,
        "age": track.age,
        "missed_frames": 0,
        "is_predicted": False,
        "velocity_source": velocity_source,
    })
    return row


def _predicted_track_row(
    track: _FusionTrack,
    *,
    timestamp_ns: int,
    master_topic: str,
) -> dict[str, Any]:
    pred_x, pred_y = track.predicted_xy(timestamp_ns)
    speed = math.hypot(track.vx, track.vy)
    yaw = math.atan2(track.vy, track.vx) if speed > 1.0 else track.yaw
    return {
        "timestamp_ns": timestamp_ns,
        "timestamp": timestamp_ns,
        "master_topic": master_topic,
        "detection_id": f"{timestamp_ns}_track{track.track_id:03d}_pred",
        "center_x": pred_x,
        "center_y": pred_y,
        "x": pred_x,
        "y": pred_y,
        "vx": track.vx,
        "vy": track.vy,
        "speed": speed,
        "yaw": yaw,
        "length": track.length,
        "width": track.width,
        "cov_xx": 2.0 + track.missed_frames,
        "cov_yy": 2.0 + track.missed_frames,
        "cov_vxvx": 6.0 + track.missed_frames,
        "cov_vyvy": 6.0 + track.missed_frames,
        "confidence": float(np.clip(track.confidence * (0.7 ** track.missed_frames), 0.0, 1.0)),
        "sensors": "temporal_prediction",
        "n_lidar": 0,
        "n_radar": 0,
        "n_camera": 0,
        "camera_topic": "",
        "camera_validated": False,
        "camera_u": "",
        "camera_v": "",
        "camera_score": 0.0,
        "track_id": track.track_id,
        "age": track.age,
        "missed_frames": track.missed_frames,
        "is_predicted": True,
        "velocity_source": "prediction",
    }


def _annotate_camera(
    row: dict[str, Any],
    camera_index: dict[str, Any] | None,
    window_ns: int,
) -> None:
    topic = str(camera_index.get("topic", "")) if camera_index else ""
    timestamps = (
        np.asarray(camera_index.get("timestamps_ns", []), dtype=np.int64)
        if camera_index
        else np.zeros(0, dtype=np.int64)
    )
    synced = False
    if len(timestamps):
        ts = int(row["timestamp_ns"])
        idx = int(np.searchsorted(timestamps, ts))
        candidates: list[int] = []
        if idx < len(timestamps):
            candidates.append(int(timestamps[idx]))
        if idx > 0:
            candidates.append(int(timestamps[idx - 1]))
        if candidates:
            nearest = min(candidates, key=lambda value: abs(value - ts))
            synced = abs(nearest - ts) <= window_ns

    camera_u, camera_v, visible_score = _camera_projection_score(row)
    validated = bool(synced and visible_score > 0.0)
    row["camera_topic"] = topic if synced else ""
    row["camera_u"] = "" if camera_u is None else round(camera_u, 2)
    row["camera_v"] = "" if camera_v is None else round(camera_v, 2)
    row["camera_score"] = float(visible_score if synced else 0.0)
    row["camera_validated"] = validated
    row["n_camera"] = 1 if validated else 0

    if validated:
        sensors = {
            item for item in str(row.get("sensors", "")).split("+")
            if item
        }
        sensors.add(_sensor_label(topic))
        row["sensors"] = "+".join(sorted(sensors))
        row["confidence"] = float(np.clip(float(row["confidence"]) + 0.05, 0.0, 1.0))


def _camera_projection_score(row: dict[str, Any]) -> tuple[float | None, float | None, float]:
    cx = float(row["center_x"])
    cy = float(row["center_y"])
    yaw = float(row["yaw"])
    length = max(float(row["length"]), 0.1)
    width = max(float(row["width"]), 0.1)
    center = np.array([[cx, cy, 0.0]], dtype=np.float64)
    center_uv, center_front = project_points(center)
    center_in_image = bool(
        center_front[0]
        and 0.0 <= center_uv[0, 0] < IMAGE_WIDTH
        and 0.0 <= center_uv[0, 1] < IMAGE_HEIGHT
    )
    corners = bbox_3d_corners(cx, cy, 0.0, length, width, 1.5, yaw)
    corner_uv, corner_front = project_points(corners)
    corner_in_image = (
        corner_front
        & (corner_uv[:, 0] >= 0.0)
        & (corner_uv[:, 0] < IMAGE_WIDTH)
        & (corner_uv[:, 1] >= 0.0)
        & (corner_uv[:, 1] < IMAGE_HEIGHT)
    )
    if center_in_image:
        return float(center_uv[0, 0]), float(center_uv[0, 1]), 1.0
    if bool(corner_in_image.any()):
        pts = corner_uv[corner_in_image]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean()), 0.5
    return (
        float(center_uv[0, 0]) if center_front[0] else None,
        float(center_uv[0, 1]) if center_front[0] else None,
        0.0,
    )


def _sensor_label(topic: str) -> str:
    parts = [part for part in topic.split("/") if part]
    if len(parts) >= 3 and parts[0] == "sensor":
        return parts[1]
    return topic.replace("/", "_").strip("_") or "camera"


def _limit_velocity_pair(
    vx: float,
    vy: float,
    *,
    max_speed: float = 60.0,
) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed <= max_speed or speed <= 1e-9:
        return float(vx), float(vy)
    scale = max_speed / speed
    return float(vx * scale), float(vy * scale)


# ── BEV plot ─────────────────────────────────────────────────────────────────

def plot_fused_bev(
    frame_result: FrameResult,
    fused_dets: list[dict[str, Any]],
    all_sensor_results: dict[str, FrameResult | None],
    output_path: str | Path,
    *,
    forward_range: tuple[float, float] = (-5.0, 95.0),
    lateral_range: tuple[float, float] = (-20.0, 20.0),
) -> None:
    """Save a BEV plot: raw points per sensor + per-sensor boxes + fused box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(10, 11))

    sensor_colors = {
        "lidar_front": "#2196F3",
        "lidar_left":  "#4CAF50",
        "lidar_right": "#FF5722",
        "radar_front": "#9C27B0",
        "radar_back":  "#607D8B",
        "radar_left":  "#FFEB3B",
        "radar_right": "#F44336",
    }

    # Raw points per sensor (faded)
    for topic, fr in all_sensor_results.items():
        if fr is None or len(fr.bev_points) == 0:
            continue
        key = topic.split("/")[2] if "/" in topic else topic
        c = sensor_colors.get(key, "#999")
        pts = fr.bev_points
        ax.scatter(pts[:, 1], pts[:, 0], s=0.3, c=c, alpha=0.25, linewidths=0)

    # Fused detections (bold)
    for det in fused_dets:
        cx, cy = float(det["center_x"]), float(det["center_y"])
        yaw = float(det.get("yaw", 0.0))
        l, w = float(det.get("length", 4.0)), float(det.get("width", 2.0))

        # Oriented box corners
        hl, hw = l / 2, w / 2
        c_ang, s_ang = math.cos(yaw), math.sin(yaw)
        corners_local = np.array([
            [ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]
        ])
        R2 = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
        corners = (R2 @ corners_local.T).T + np.array([cx, cy])
        poly = Polygon(corners[:, [1, 0]], closed=True,
                       fill=False, edgecolor="#E91E63", linewidth=2.5)
        ax.add_patch(poly)

        # Velocity arrow
        vx, vy = float(det.get("vx", 0)), float(det.get("vy", 0))
        if math.hypot(vx, vy) > 0.5:
            ax.annotate("", xy=(cy + vy * 2, cx + vx * 2), xytext=(cy, cx),
                        arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5))

        # Heading arrow (yaw direction)
        hx = math.cos(yaw) * 3
        hy = math.sin(yaw) * 3
        ax.annotate("", xy=(cy + hy, cx + hx), xytext=(cy, cx),
                    arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5))

        label = f"{det['detection_id']}\n{det['sensors']}"
        ax.text(cy, cx, label, fontsize=6, ha="center", va="bottom", color="#E91E63",
                bbox={"facecolor": "white", "alpha": 0.6, "pad": 1, "edgecolor": "none"})

    ax.set_xlabel("lateral y_ego (m)")
    ax.set_ylabel("forward x_ego (m)")
    ax.set_xlim(lateral_range)
    ax.set_ylim(forward_range)
    ax.set_aspect("equal")
    ax.grid(True, color="#ddd", lw=0.6)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#E91E63", lw=2, label="fused detection"),
        Line2D([0], [0], color="#FF9800", lw=1.5, ls="-", label="heading"),
        Line2D([0], [0], color="#E91E63", lw=1.5, ls="-", label="velocity×2"),
    ]
    for topic, c in sensor_colors.items():
        legend_els.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=6, label=topic.replace("/sensor/", "").replace("/points", ""))
        )
    ax.legend(handles=legend_els, loc="upper right", fontsize=7)

    ts_ns = frame_result.timestamp_ns
    ax.set_title(f"Task 2 fusion | t={ts_ns} ns | {len(fused_dets)} fused detections")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _reconcile_yaw(yaw: float, vx: float, vy: float) -> float:
    speed = math.hypot(vx, vy)
    if speed < 1.0:
        return yaw
    vel_angle = math.atan2(vy, vx)
    diff = (yaw - vel_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) > math.pi / 2:
        yaw = (yaw + math.pi + math.pi) % (2 * math.pi) - math.pi
    return yaw
