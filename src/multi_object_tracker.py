"""Multi-Object Tracker (MOT) for Task 3.

Orchestrates: predict → gather detections → associate → update → lifecycle.

The tracker operates on the master-clock timeline (LiDAR-front timestamps)
and pulls in all-sensor detections within ±50 ms of each master tick.

Two-stage association
----------------------
Stage 1: Confirmed tracks  ↔  LiDAR detections   (Mahalanobis gate d < 4.0)
Stage 2: Confirmed tracks  ↔  Radar detections    (gate d < 6.0 — wider, Doppler)
Stage 3: Unmatched detections → seed new tentative tracks

Barrier rejection
-----------------
New tentative track seeds with |center_y| > 8 m are tagged static_clutter
immediately, preventing barrier returns from ever being promoted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.motion_model import predict_state
from src.task2_fusion import (
    _X, _Y, _VX, _VY, _YAW, _L, _W, STATE_DIM,
    _R_LIDAR, _R_RADAR, ASSOC_WINDOW_NS,
    SensorDetection, _ekf_update, _init_state,
)
from src.track_state import (
    Track, TrackStatus,
    create_track, update_track, mark_missed,
    predict_track_lifecycle, is_deletable, TRACKS_CSV_COLUMNS,
)
from src.tf_utils import EgoStateLog


# Association gates
_GATE_LIDAR: float = 4.0
_GATE_RADAR: float = 6.0

# Lateral barrier threshold for instant clutter tagging on seed
_BARRIER_Y: float = 8.0


def _mahalanobis(
    state: np.ndarray,
    P: np.ndarray,
    det: SensorDetection,
) -> float:
    """Mahalanobis distance between track position and detection position."""
    diff = np.array([
        det.center_x - float(state[_X]),
        det.center_y - float(state[_Y]),
    ], dtype=np.float64)
    S = P[np.ix_([_X, _Y], [_X, _Y])] + det.position_cov()
    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("inf")
    d2 = float(diff @ inv_S @ diff)
    return math.sqrt(max(d2, 0.0))


def _hungarian(
    tracks: list[Track],
    dets: list[SensorDetection],
    gate: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Run Mahalanobis-gated Hungarian assignment.

    Returns
    -------
    matches         : list of (track_idx, det_idx) pairs
    unmatched_tracks: set of unmatched track indices
    unmatched_dets  : set of unmatched detection indices
    """
    if not tracks or not dets:
        return [], set(range(len(tracks))), set(range(len(dets)))

    INF_COST = gate * 10
    cost = np.full((len(tracks), len(dets)), INF_COST, dtype=np.float64)
    for ti, trk in enumerate(tracks):
        for di, det in enumerate(dets):
            d = _mahalanobis(trk.state, trk.P, det)
            if d < gate:
                cost[ti, di] = d

    rows, cols = linear_sum_assignment(cost)
    matches, unmatched_tracks, unmatched_dets = [], set(range(len(tracks))), set(range(len(dets)))
    for r, c in zip(rows, cols):
        if cost[r, c] < gate:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)
    return matches, unmatched_tracks, unmatched_dets


class MultiObjectTracker:
    """Ego-motion-compensated KF-based multi-object tracker.

    Parameters
    ----------
    ego_log      : EgoStateLog providing (vx, vy, heading) per timestamp.
    n_init       : consecutive hits before tentative → confirmed.
    max_age      : missed frames before confirmed → deleted.
    window_ns    : ±ns detection gather window around master timestamp.
    """

    def __init__(
        self,
        ego_log: EgoStateLog | None = None,
        *,
        n_init: int = 3,
        max_age: int = 5,
        window_ns: int = ASSOC_WINDOW_NS,
    ) -> None:
        self._ego_log = ego_log
        self._n_init = n_init
        self._max_age = max_age
        self._window_ns = window_ns

        self._tracks: list[Track] = []
        self._deleted_tracks: list[Track] = []
        self._prev_ts: int | None = None
        self._prev_ego: dict[str, float] | None = None

        # All rows emitted (one per active track per timestamp)
        self.all_rows: list[dict[str, Any]] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def step(
        self,
        timestamp_ns: int,
        detections: list[SensorDetection],
    ) -> list[dict[str, Any]]:
        """Process one master-clock frame.

        Parameters
        ----------
        timestamp_ns : ROS timestamp (ns) for this frame.
        detections   : all SensorDetection objects from all sensors gathered
                       within ±window_ns of timestamp_ns (already windowed).

        Returns
        -------
        List of track-state dicts (TRACKS_CSV_COLUMNS) for active tracks
        at this timestamp.
        """
        # ── 1. Get ego state at this timestamp ───────────────────────────────
        ego_now = self._get_ego(timestamp_ns)

        # ── 2. Predict all active tracks ─────────────────────────────────────
        if self._prev_ts is not None and self._prev_ego is not None:
            dt = (timestamp_ns - self._prev_ts) * 1e-9
            for trk in self._tracks:
                trk.state, trk.P = predict_state(
                    trk.state, trk.P,
                    self._prev_ego, ego_now,
                    dt,
                )
        for trk in self._tracks:
            predict_track_lifecycle(trk, max_age=self._max_age)

        # ── 3. Split detections by sensor type ───────────────────────────────
        lidar_dets = [d for d in detections if not d.is_radar]
        radar_dets = [d for d in detections if d.is_radar]

        confirmed = [t for t in self._tracks
                     if t.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]
        tentative = [t for t in self._tracks if t.status == TrackStatus.TENTATIVE]

        # ── 4a. Stage 1: confirmed ↔ LiDAR ───────────────────────────────────
        m1, unm_conf, unm_lidar = _hungarian(confirmed, lidar_dets, _GATE_LIDAR)
        used_conf: set[int] = set()

        for ti, di in m1:
            trk = confirmed[ti]
            det = lidar_dets[di]
            meas = np.array([det.center_x, det.center_y, det.vx, det.vy])
            new_state, new_P = _ekf_update(trk.state, trk.P, meas, _R_LIDAR)
            update_track(
                trk, new_state, new_P, timestamp_ns,
                topic=det.topic, is_radar=False,
                n_init=self._n_init, max_age=self._max_age,
                ego_heading=ego_now["heading"],
            )
            used_conf.add(ti)

        # ── 4b. Stage 2: unmatched confirmed ↔ Radar ─────────────────────────
        unm_conf_tracks = [confirmed[i] for i in sorted(unm_conf)]
        m2, unm_conf2, unm_radar = _hungarian(unm_conf_tracks, radar_dets, _GATE_RADAR)

        for ti, di in m2:
            trk = unm_conf_tracks[ti]
            det = radar_dets[di]
            meas = np.array([det.center_x, det.center_y, det.vx, det.vy])
            new_state, new_P = _ekf_update(trk.state, trk.P, meas, _R_RADAR)
            update_track(
                trk, new_state, new_P, timestamp_ns,
                topic=det.topic, is_radar=True,
                n_init=self._n_init, max_age=self._max_age,
                ego_heading=ego_now["heading"],
            )
            used_conf.add(confirmed.index(trk))

        # ── 4c. Tentative tracks get all remaining detections ─────────────────
        remaining_lidar = [lidar_dets[i] for i in sorted(unm_lidar)]
        remaining_radar = [radar_dets[i] for i in sorted(unm_radar)]
        remaining_dets = remaining_lidar + remaining_radar

        m3, _, unm_remaining = _hungarian(tentative, remaining_dets, _GATE_LIDAR)
        for ti, di in m3:
            trk = tentative[ti]
            det = remaining_dets[di]
            meas = np.array([det.center_x, det.center_y, det.vx, det.vy])
            R = _R_RADAR if det.is_radar else _R_LIDAR
            new_state, new_P = _ekf_update(trk.state, trk.P, meas, R)
            update_track(
                trk, new_state, new_P, timestamp_ns,
                topic=det.topic, is_radar=det.is_radar,
                n_init=self._n_init, max_age=self._max_age,
                ego_heading=ego_now["heading"],
            )

        # ── 5. Mark missed tracks ─────────────────────────────────────────────
        matched_conf_ids = {confirmed[ti].track_id for ti, _ in m1}
        matched_conf_ids |= {unm_conf_tracks[ti].track_id for ti, _ in m2}
        matched_tent_ids = {tentative[ti].track_id for ti, _ in m3}

        for trk in self._tracks:
            if trk.track_id not in matched_conf_ids | matched_tent_ids:
                mark_missed(trk)
                if trk.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING):
                    trk.append_history(timestamp_ns)

        # ── 6. Seed new tentative tracks from unmatched detections ────────────
        seeded_dets = [remaining_dets[i] for i in sorted(unm_remaining)]
        for det in seeded_dets:
            sd = det  # SensorDetection
            init_state, init_P = _init_state(sd)
            new_trk = create_track(
                init_state, init_P, timestamp_ns,
                topic=sd.topic, is_radar=sd.is_radar,
            )
            # Instant barrier tag
            if abs(sd.center_y) > _BARRIER_Y:
                new_trk.static_clutter = True
            self._tracks.append(new_trk)

        # ── 7. Delete expired tracks ──────────────────────────────────────────
        alive, dead = [], []
        for trk in self._tracks:
            if is_deletable(trk, max_age=self._max_age):
                trk.status = TrackStatus.DELETED
                dead.append(trk)
            else:
                alive.append(trk)
        self._tracks = alive
        self._deleted_tracks.extend(dead)

        # ── 8. Emit rows for all active tracks ────────────────────────────────
        rows = []
        for trk in self._tracks:
            if trk.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING):
                row = trk.to_dict(timestamp_ns)
                rows.append(row)
                self.all_rows.append(row)

        # ── 9. Advance clock ──────────────────────────────────────────────────
        self._prev_ts = timestamp_ns
        self._prev_ego = ego_now
        return rows

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks) + list(self._deleted_tracks)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_ego(self, timestamp_ns: int) -> dict[str, float]:
        if self._ego_log is None:
            return {"vx": 0.0, "vy": 0.0, "heading": 0.0}
        try:
            return self._ego_log.interpolate(timestamp_ns)
        except Exception:
            return {"vx": 0.0, "vy": 0.0, "heading": 0.0}
