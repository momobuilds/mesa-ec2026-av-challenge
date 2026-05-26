"""Sensor extrinsics and ego-state utilities for Task 2.

The bag's /tf topic contains only ``map → center_of_gravity`` transforms;
no ``sensor_* → center_of_gravity`` static frames exist.  All sensor
extrinsics below are *hardcoded approximations* derived from the axis
mappings observed in Task 1 and typical autonomous-racing car geometry
(IndyCar-class chassis, ~3 m wheelbase, COG ≈ 40 % from front axle).

Ego frame: ``center_of_gravity`` (x forward, y left, z up).
Extrinsic convention: p_ego = T @ p_sensor_homogeneous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ── Axis-mapping rotation bases ───────────────────────────────────────────────
#
# Task 1 determined (from data statistics) that:
#   LiDARs: z = range/forward, x = lateral, y = vertical
#   Radars: y = range/forward, x = lateral, z = vertical
#
# _R_AXES maps raw sensor xyz → intermediate (forward, left, up) frame:
#   ego_x ← sensor range axis
#   ego_y ← −sensor lateral axis  (convention: positive raw x = car-right)
#   ego_z ← sensor vertical axis

_R_AXES_LIDAR = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

_R_AXES_RADAR = np.array(
    [[0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _yaw_rot(deg: float) -> np.ndarray:
    """3×3 CCW rotation matrix around z for mounting yaw."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_extrinsic(
    r_axes: np.ndarray, yaw_deg: float, t: list[float]
) -> np.ndarray:
    """Build a 4×4 sensor→ego transform (axis-map rotation × mounting yaw + translation)."""
    R = _yaw_rot(yaw_deg) @ r_axes
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── Hardcoded extrinsics ──────────────────────────────────────────────────────
#
# Translation: [x_forward, y_left, z_up] metres from centre-of-gravity.
# Yaw: mounting angle CCW from car-forward (e.g. left sensor → +90°).
# All values are approximate; no calibration data is available in the bag.

_SENSOR_SPECS: dict[str, tuple[np.ndarray, float, list[float]]] = {
    # frame_id       : (R_axes,       yaw_deg,  translation)
    "lidar_front": (_R_AXES_LIDAR,   0.0, [+1.8,  0.0, +0.9]),
    "lidar_left":  (_R_AXES_LIDAR,  90.0, [+0.3, +0.8, +0.8]),
    "lidar_right": (_R_AXES_LIDAR, -90.0, [+0.3, -0.8, +0.8]),
    "radar_front": (_R_AXES_RADAR,   0.0, [+2.0,  0.0, +0.4]),
    "radar_back":  (_R_AXES_RADAR, 180.0, [-1.8,  0.0, +0.4]),
    "radar_left":  (_R_AXES_RADAR,  90.0, [ 0.0, +0.8, +0.4]),
    "radar_right": (_R_AXES_RADAR, -90.0, [ 0.0, -0.8, +0.4]),
}

_EXTRINSICS: dict[str, np.ndarray] = {
    frame: _make_extrinsic(r, yaw, t)
    for frame, (r, yaw, t) in _SENSOR_SPECS.items()
}

# Mounting yaw in degrees (CCW from car-forward) — used by radar_doppler.
MOUNTING_YAW_DEG: dict[str, float] = {
    frame: yaw for frame, (_, yaw, _) in _SENSOR_SPECS.items()
}


def get_extrinsic(sensor_frame: str) -> np.ndarray:
    """Return the 4×4 sensor→ego extrinsic for *sensor_frame*.

    Falls back to identity (with a warning) for unknown frames.
    """
    if sensor_frame not in _EXTRINSICS:
        import warnings

        warnings.warn(
            f"No hardcoded extrinsic for '{sensor_frame}'; using identity.",
            stacklevel=2,
        )
        return np.eye(4, dtype=np.float64)
    return _EXTRINSICS[sensor_frame].copy()


def transform_points(points_xyz: np.ndarray, sensor_frame: str) -> np.ndarray:
    """Transform an (N, 3) float array from sensor frame to ego frame."""
    if len(points_xyz) == 0:
        return points_xyz
    T = get_extrinsic(sensor_frame)
    ones = np.ones((len(points_xyz), 1), dtype=np.float64)
    homog = np.concatenate([points_xyz.astype(np.float64), ones], axis=1)
    return (T @ homog.T).T[:, :3]


def print_extrinsic_tree() -> None:
    """Print a human-readable summary of all sensor→ego extrinsics."""
    print("Sensor extrinsic tree  (hardcoded approximations → centre_of_gravity):")
    print("  [!] /tf in bag contains ONLY 'map → center_of_gravity' — no sensor frames.")
    for frame in sorted(_SENSOR_SPECS):
        _, yaw_deg, t = _SENSOR_SPECS[frame]
        print(
            f"  {frame:<20s}  t=[{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}] m"
            f"  mounting_yaw={yaw_deg:+.1f}°"
        )


# ── EgoStateLog ───────────────────────────────────────────────────────────────

class EgoStateLog:
    """Time-indexed ego vehicle state read from *ego.mat*.

    The ``stamp__tot`` field in ego.mat is in seconds elapsed since the
    start of the log, which aligns with the bag's starting_time_ns.
    Absolute ROS timestamps are reconstructed as::

        ros_ns = bag_start_ns + stamp_s * 1e9

    Provides linear interpolation of ``ekf_vx``, ``ekf_vy``, and
    ``ekf_heading`` (radians) by ROS timestamp in nanoseconds.
    """

    def __init__(self, mat_path: str | Path, bag_start_ns: int) -> None:
        import h5py

        self._bag_start_ns = int(bag_start_ns)
        mat_path = Path(mat_path)
        with h5py.File(mat_path, "r") as f:
            stamps_s = f["log/estimation/stamp__tot"][0, :]
            self._timestamps_ns = (
                self._bag_start_ns + stamps_s * 1e9
            ).astype(np.float64)
            self._vx = f["log/estimation/ekf_vx"][0, :].astype(np.float64)
            self._vy = f["log/estimation/ekf_vy"][0, :].astype(np.float64)
            self._heading = f["log/estimation/ekf_heading"][0, :].astype(np.float64)

        self.duration_s = float(stamps_s[-1] - stamps_s[0])
        self.n_samples = len(self._timestamps_ns)

    def interpolate(self, timestamp_ns: int) -> dict[str, float]:
        """Return interpolated (vx, vy, heading) for the given ROS timestamp."""
        t = float(timestamp_ns)
        t_arr = self._timestamps_ns
        idx = int(np.searchsorted(t_arr, t))
        idx = max(1, min(idx, len(t_arr) - 1))
        alpha = (t - t_arr[idx - 1]) / max(t_arr[idx] - t_arr[idx - 1], 1.0)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        def lerp(arr: np.ndarray) -> float:
            return float(arr[idx - 1] + alpha * (arr[idx] - arr[idx - 1]))

        return {
            "vx": lerp(self._vx),
            "vy": lerp(self._vy),
            "heading": lerp(self._heading),
        }

    def __repr__(self) -> str:
        return (
            f"EgoStateLog(n={self.n_samples}, duration={self.duration_s:.1f}s, "
            f"vx_range=[{self._vx.min():.1f},{self._vx.max():.1f}] m/s)"
        )
