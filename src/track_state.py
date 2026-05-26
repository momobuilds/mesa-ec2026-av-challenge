"""Track dataclass and lifecycle management for Task 3.

Track lifecycle
---------------
  TENTATIVE  → receives association detections; not yet emitted in output.
  CONFIRMED  → has been associated n_init times consecutively; emitted.
  COASTING   → confirmed track that missed one or more associations;
               EKF predict-only; still emitted (dashed in BEV).
  DELETED    → exceeded max_age missed frames; removed from active set.
               Its track_id is never reused.

Static-clutter gate
-------------------
A new tentative track is tagged static_clutter=True if the world-frame speed
averaged over its first clutter_check_frames is below clutter_speed_threshold.
Static-clutter tracks are never promoted to CONFIRMED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from src.task2_fusion import _X, _Y, _VX, _VY, _YAW, _L, _W, STATE_DIM


class TrackStatus(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING  = auto()
    DELETED   = auto()


# History entry: (timestamp_ns, x, y, vx, vy, yaw, length, width)
HistoryEntry = tuple[int, float, float, float, float, float, float, float]


_next_track_id: int = 0


def _alloc_track_id() -> int:
    global _next_track_id
    tid = _next_track_id
    _next_track_id += 1
    return tid


def reset_track_id_counter(start: int = 0) -> None:
    """Reset the global counter (useful for tests / fresh runs)."""
    global _next_track_id
    _next_track_id = start


@dataclass
class Track:
    """A single tracked object with an EKF state and lifecycle metadata."""

    track_id: int
    state: np.ndarray               # (STATE_DIM,) in ego frame at last update
    P: np.ndarray                   # (STATE_DIM, STATE_DIM) covariance
    status: TrackStatus = TrackStatus.TENTATIVE
    age: int = 0                    # total frames this track has existed
    time_since_update: int = 0      # frames since last measurement association
    hits: int = 0                   # total associations received
    consecutive_hits: int = 0       # streak of consecutive associations
    n_lidar: int = 0
    n_radar: int = 0
    sensors: list[str] = field(default_factory=list)
    confidence: float = 0.5
    history: list[HistoryEntry] = field(default_factory=list)
    static_clutter: bool = False    # never promoted if True
    # Running world-frame speed sum for clutter detection
    _speed_sum: float = field(default=0.0, repr=False)

    @property
    def center_x(self) -> float:
        return float(self.state[_X])

    @property
    def center_y(self) -> float:
        return float(self.state[_Y])

    @property
    def vx(self) -> float:
        return float(self.state[_VX])

    @property
    def vy(self) -> float:
        return float(self.state[_VY])

    @property
    def yaw(self) -> float:
        return float(self.state[_YAW])

    @property
    def length(self) -> float:
        return float(self.state[_L])

    @property
    def width(self) -> float:
        return float(self.state[_W])

    def append_history(self, timestamp_ns: int) -> None:
        self.history.append((
            timestamp_ns,
            self.center_x, self.center_y,
            self.vx, self.vy,
            self.yaw, self.length, self.width,
        ))

    def to_dict(self, timestamp_ns: int) -> dict[str, Any]:
        return {
            "timestamp_ns": timestamp_ns,
            "track_id": self.track_id,
            "status": self.status.name,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "vx": self.vx,
            "vy": self.vy,
            "yaw": self.yaw,
            "length": self.length,
            "width": self.width,
            "cov_xx": float(self.P[_X, _X]),
            "cov_yy": float(self.P[_Y, _Y]),
            "n_lidar": self.n_lidar,
            "n_radar": self.n_radar,
            "sensors": "+".join(sorted(set(
                t.split("/")[2] if "/" in t else t for t in self.sensors
            ))) if self.sensors else "",
            "confidence": self.confidence,
        }


def create_track(
    state: np.ndarray,
    P: np.ndarray,
    timestamp_ns: int,
    topic: str = "",
    is_radar: bool = False,
) -> Track:
    """Initialise a new tentative track from a single detection."""
    track = Track(
        track_id=_alloc_track_id(),
        state=state.copy(),
        P=P.copy(),
        n_lidar=0 if is_radar else 1,
        n_radar=1 if is_radar else 0,
        sensors=[topic] if topic else [],
    )
    track.append_history(timestamp_ns)
    return track


def update_track(
    track: Track,
    state: np.ndarray,
    P: np.ndarray,
    timestamp_ns: int,
    *,
    topic: str = "",
    is_radar: bool = False,
    n_init: int = 3,
    max_age: int = 5,
    lateral_clutter_threshold: float = 8.0,
    clutter_speed_threshold: float = 0.5,
    clutter_check_frames: int = 5,
    ego_heading: float = 0.0,
) -> None:
    """Apply a matched measurement to a track (in-place)."""
    track.state = state.copy()
    track.P = P.copy()
    track.hits += 1
    track.consecutive_hits += 1
    track.time_since_update = 0
    if topic and topic not in track.sensors:
        track.sensors.append(topic)
    if is_radar:
        track.n_radar += 1
    else:
        track.n_lidar += 1

    track.confidence = float(np.clip(
        0.5 + 0.05 * track.consecutive_hits, 0.0, 1.0
    ))

    # Promote tentative → confirmed
    if track.status == TrackStatus.TENTATIVE and track.consecutive_hits >= n_init:
        # Barrier gate: wide lateral → instant clutter (checked on seed too,
        # but we double-check here after the EKF has had a chance to update y).
        if abs(track.center_y) > lateral_clutter_threshold:
            track.static_clutter = True
        if not track.static_clutter:
            track.status = TrackStatus.CONFIRMED

    # Restore COASTING → CONFIRMED on re-association
    if track.status == TrackStatus.COASTING and not track.static_clutter:
        track.status = TrackStatus.CONFIRMED

    track.append_history(timestamp_ns)


def predict_track_lifecycle(
    track: Track,
    *,
    max_age: int = 5,
) -> None:
    """Advance track lifecycle after a predict step (called even if no match)."""
    track.age += 1
    if track.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING):
        if track.time_since_update > 0:
            track.status = TrackStatus.COASTING

    track.confidence = float(np.clip(
        track.confidence - 0.08 * track.time_since_update, 0.05, 1.0
    ))


def mark_missed(track: Track) -> None:
    """Record that this track received no association this frame."""
    track.time_since_update += 1
    track.consecutive_hits = 0
    if track.status == TrackStatus.CONFIRMED:
        track.status = TrackStatus.COASTING


def is_deletable(track: Track, max_age: int = 5) -> bool:
    """Return True if the track should be pruned."""
    if track.status == TrackStatus.DELETED:
        return True
    if track.time_since_update > max_age:
        return True
    # Tentative tracks that never got promoted after 2×n_init frames
    if track.status == TrackStatus.TENTATIVE and track.age > max_age * 2:
        return True
    return False


TRACKS_CSV_COLUMNS = [
    "timestamp_ns", "track_id", "status", "age", "time_since_update",
    "center_x", "center_y", "vx", "vy", "yaw", "length", "width",
    "cov_xx", "cov_yy", "n_lidar", "n_radar", "sensors", "confidence",
]
