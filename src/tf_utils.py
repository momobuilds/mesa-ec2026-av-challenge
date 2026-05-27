"""Sensor extrinsics and ego-state utilities.

Sensor extrinsic strategy (professor-confirmed approach)
---------------------------------------------------------
The bag's ``/tf`` topic publishes ``map → center_of_gravity`` transforms at
~200 Hz.  The URDF file (``data/eva.urdf``) defines all sensor frames as
fixed joints relative to ``center_of_gravity``.

Together they give:
    sensor → center_of_gravity   (from URDF)
    map    → center_of_gravity   (from /tf topic in the bag)

The ego.mat velocity data (EgoStateLog) is **only** needed for correcting
the Doppler radial velocity from radar — NOT for ego pose in map frame.

BagTFReader
-----------
Use :class:`BagTFReader` to obtain the ego pose (x, y, heading) in the map
frame at any timestamp.  This is more accurate than integrating velocities
from ego.mat.

Sensor extrinsics
-----------------
:func:`get_extrinsic` returns the 4×4 ``sensor → center_of_gravity`` matrix.
When a URDF path is supplied at module init via :func:`init_extrinsics_from_urdf`,
the transforms are taken from the URDF (recommended).  The module auto-loads
``data/eva.urdf`` relative to this file's parent on first use if present.
The hardcoded values below serve only as a fallback.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


# ── Hardcoded fallback extrinsics ─────────────────────────────────────────────
#
# These are approximate values derived from axis-mapping observations.
# They are replaced by URDF-derived values when the URDF is available.
# See _auto_init() at the bottom of this file.
#
# Axis-mapping rotation bases:
#   LiDARs: z = range/forward, x = lateral, y = vertical
#   Radars: y = range/forward, x = lateral, z = vertical
#
# _R_AXES maps raw sensor xyz → intermediate (forward, left, up) frame.

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


_FALLBACK_SENSOR_SPECS: dict[str, tuple[np.ndarray, float, list[float]]] = {
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
    for frame, (r, yaw, t) in _FALLBACK_SENSOR_SPECS.items()
}

# Mounting yaw in degrees (CCW from car-forward) — used by radar_doppler.
MOUNTING_YAW_DEG: dict[str, float] = {
    frame: yaw for frame, (_, yaw, _) in _FALLBACK_SENSOR_SPECS.items()
}

# Track whether URDF-derived extrinsics have been loaded
_URDF_EXTRINSICS_LOADED: bool = False


# ── URDF-based extrinsic loading ──────────────────────────────────────────────

def init_extrinsics_from_urdf(urdf_path: str | Path) -> bool:
    """Replace the module-level extrinsic tables with URDF-derived transforms.

    Returns True on success, False if the URDF cannot be loaded.
    The URDF defines sensor frames as fixed joints relative to
    ``center_of_gravity``, giving accurate sensor→ego transforms.

    Uses a self-contained minimal URDF parser (stdlib + numpy only) so that
    this module stays importable even when heavy dependencies (rosbags, pandas,
    sklearn) are absent.

    This is the professor-recommended approach:
    *"Questo, insieme al urdf file dovrebbe bastare."*
    (Together with the URDF file this should suffice.)
    """
    global _EXTRINSICS, MOUNTING_YAW_DEG, _URDF_EXTRINSICS_LOADED

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        return False

    try:
        extrinsics, mounting_yaw = _parse_urdf_extrinsics(
            urdf_path,
            base_frame="center_of_gravity",
            sensor_frames=list(_FALLBACK_SENSOR_SPECS.keys()),
        )
    except Exception:  # noqa: BLE001
        return False

    # Merge: keep fallback for any sensor not found in URDF
    merged_ext = dict(_EXTRINSICS)
    merged_yaw = dict(MOUNTING_YAW_DEG)
    merged_ext.update(extrinsics)
    merged_yaw.update(mounting_yaw)

    _EXTRINSICS = merged_ext
    MOUNTING_YAW_DEG = merged_yaw
    _URDF_EXTRINSICS_LOADED = True
    return True


# ── Minimal self-contained URDF parser (stdlib + numpy only) ─────────────────

def _parse_urdf_extrinsics(
    urdf_path: Path,
    base_frame: str,
    sensor_frames: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Parse fixed-joint transforms from a URDF file.

    Returns (extrinsics, mounting_yaw_deg) where extrinsics[frame] is the
    4×4 matrix T satisfying ``p_base = T @ p_sensor_homogeneous``.

    This parser is intentionally minimal — it only needs stdlib + numpy so
    tf_utils stays importable without pandas/sklearn/rosbags.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()

    # Build link set and joint adjacency (parent→child, parent-from-child 4×4)
    links: set[str] = {
        e.attrib["name"].strip().strip("/")
        for e in root.findall("link")
        if "name" in e.attrib
    }
    # adjacency[frame] = list of (neighbor, T_frame_from_neighbor)
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}

    for joint_xml in root.findall("joint"):
        if str(joint_xml.attrib.get("type", "")) not in ("fixed", ""):
            continue
        parent_el = joint_xml.find("parent")
        child_el = joint_xml.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = str(parent_el.attrib.get("link", "")).strip().strip("/")
        child = str(child_el.attrib.get("link", "")).strip().strip("/")
        if not parent or not child:
            continue

        origin = joint_xml.find("origin")
        xyz_str = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy_str = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = _parse3(xyz_str)
        rpy = _parse3(rpy_str)
        T_parent_child = _rpy_xyz_to_matrix(rpy, xyz)

        links.update({parent, child})
        adjacency.setdefault(parent, []).append((child, T_parent_child))
        # Inverse: child→parent
        adjacency.setdefault(child, []).append(
            (parent, _invert_rigid(T_parent_child))
        )

    base = base_frame.strip().strip("/")
    extrinsics: dict[str, np.ndarray] = {}
    mounting_yaw: dict[str, float] = {}

    for sensor_frame in sensor_frames:
        sensor = sensor_frame.strip().strip("/")
        if sensor not in links:
            continue
        # BFS from base to sensor
        T = _bfs_transform(base, sensor, adjacency)
        if T is None:
            continue
        extrinsics[sensor_frame] = T
        mounting_yaw[sensor_frame] = math.degrees(
            math.atan2(float(T[1, 0]), float(T[0, 0]))
        )

    return extrinsics, mounting_yaw


def _parse3(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return parts[0], parts[1], parts[2]


def _rpy_xyz_to_matrix(
    rpy: tuple[float, float, float],
    xyz: tuple[float, float, float],
) -> np.ndarray:
    """Build a 4×4 rigid transform from roll-pitch-yaw (intrinsic x→y→z) and translation."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4×4 transform without np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def _bfs_transform(
    start: str,
    goal: str,
    adjacency: dict[str, list[tuple[str, np.ndarray]]],
) -> np.ndarray | None:
    """BFS from start to goal; returns T_start_from_goal (or None)."""
    from collections import deque
    queue: deque[tuple[str, np.ndarray]] = deque([(start, np.eye(4, dtype=np.float64))])
    visited = {start}
    while queue:
        current, T_start_current = queue.popleft()
        if current == goal:
            return T_start_current
        for neighbor, T_current_neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, T_start_current @ T_current_neighbor))
    return None


def _auto_init() -> None:
    """Try to load URDF on first import from the default data/ location."""
    default_candidates = [
        Path(__file__).parent.parent / "data" / "eva.urdf",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            init_extrinsics_from_urdf(candidate)
            return


# ── Public extrinsic API ──────────────────────────────────────────────────────

def get_extrinsic(sensor_frame: str) -> np.ndarray:
    """Return the 4×4 sensor→ego (center_of_gravity) extrinsic for *sensor_frame*.

    Uses URDF-derived values when available; falls back to hardcoded
    approximations for unknown frames.
    """
    clean = sensor_frame.strip().strip("/")
    if clean not in _EXTRINSICS:
        import warnings
        warnings.warn(
            f"No extrinsic for '{clean}'; using identity.",
            stacklevel=2,
        )
        return np.eye(4, dtype=np.float64)
    return _EXTRINSICS[clean].copy()


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
    source = "URDF (data/eva.urdf)" if _URDF_EXTRINSICS_LOADED else "hardcoded approximations"
    print(f"Sensor extrinsic tree  ({source} → centre_of_gravity):")
    print("  /tf in bag provides: map → center_of_gravity  (use BagTFReader)")
    print("  URDF provides:       sensor → center_of_gravity")
    for frame in sorted(_EXTRINSICS):
        T = _EXTRINSICS[frame]
        t = T[:3, 3]
        yaw = MOUNTING_YAW_DEG.get(frame, 0.0)
        print(
            f"  {frame:<20s}  t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m"
            f"  mounting_yaw={yaw:+.1f}°"
        )


# ── BagTFReader ───────────────────────────────────────────────────────────────

class BagTFReader:
    """Read ``map → center_of_gravity`` transforms from the ``/tf`` bag topic.

    The ``/tf`` topic in this bag publishes the ego vehicle pose in the map
    frame at ~200 Hz.  This is the **recommended** source for world-frame ego
    pose (professor-confirmed).  Use :attr:`pose_at` to interpolate (x, y,
    heading) at any ROS timestamp.

    The ego.mat velocity data (:class:`EgoStateLog`) is kept separately and
    used **only** for Doppler velocity correction in radar processing.

    Example
    -------
    >>> reader = BagTFReader("data")
    >>> x, y, heading = reader.pose_at(timestamp_ns)

    Parameters
    ----------
    bag_dir:
        Directory containing the ``*.db3`` bag files and ``metadata.yaml``.
    child_frame:
        The child frame of the desired transform (default: ``center_of_gravity``).
    parent_frame:
        The parent frame (default: ``map``).
    """

    TF_TOPIC = "/tf"
    TF_MSG_TYPE = "tf2_msgs/msg/TFMessage"

    def __init__(
        self,
        bag_dir: str | Path,
        *,
        child_frame: str = "center_of_gravity",
        parent_frame: str = "map",
    ) -> None:
        self._child_frame = child_frame
        self._parent_frame = parent_frame

        try:
            from rosbags.typesys import Stores, get_typestore
            typestore = get_typestore(Stores.ROS2_JAZZY)
        except ImportError as exc:
            raise RuntimeError(
                "rosbags package required for BagTFReader. Install with: uv sync"
            ) from exc

        bag_dir = Path(bag_dir)
        db_paths = sorted(bag_dir.glob("*.db3"))
        if not db_paths:
            raise FileNotFoundError(f"No *.db3 files found in {bag_dir}")

        timestamps: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        headings: list[float] = []

        for db_path in db_paths:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "select id, type from topics where name = ?",
                    (self.TF_TOPIC,),
                ).fetchone()
                if row is None:
                    continue
                topic_id, msg_type = row
                for ts, data in conn.execute(
                    "select timestamp, data from messages where topic_id = ?",
                    (topic_id,),
                ):
                    try:
                        msg = typestore.deserialize_cdr(data, msg_type)
                    except Exception:  # noqa: BLE001
                        continue
                    for tf_stamped in msg.transforms:
                        if tf_stamped.child_frame_id != child_frame:
                            continue
                        if tf_stamped.header.frame_id != parent_frame:
                            continue
                        t = tf_stamped.transform.translation
                        r = tf_stamped.transform.rotation
                        heading = _quat_to_yaw(r.x, r.y, r.z, r.w)
                        timestamps.append(int(ts))
                        xs.append(float(t.x))
                        ys.append(float(t.y))
                        headings.append(float(heading))

        if not timestamps:
            raise RuntimeError(
                f"No '{parent_frame}→{child_frame}' transforms found in {bag_dir}. "
                f"Check that the /tf topic exists and contains the expected frames."
            )

        order = np.argsort(timestamps)
        self._timestamps_ns = np.asarray(timestamps, dtype=np.float64)[order]
        self._x = np.asarray(xs, dtype=np.float64)[order]
        self._y = np.asarray(ys, dtype=np.float64)[order]
        self._heading = np.asarray(headings, dtype=np.float64)[order]

        self.n_samples: int = len(self._timestamps_ns)
        self.duration_s: float = float(
            (self._timestamps_ns[-1] - self._timestamps_ns[0]) / 1e9
        )

    # ── public interface ──────────────────────────────────────────────────────

    def pose_at(self, timestamp_ns: int) -> tuple[float, float, float]:
        """Return linearly interpolated ``(x_map, y_map, heading_rad)`` at *timestamp_ns*.

        Heading is the yaw angle of the ego vehicle in the map frame (rad).
        """
        t = float(timestamp_ns)
        idx = int(np.searchsorted(self._timestamps_ns, t))
        idx = max(1, min(idx, len(self._timestamps_ns) - 1))
        t0, t1 = self._timestamps_ns[idx - 1], self._timestamps_ns[idx]
        alpha = float(np.clip((t - t0) / max(t1 - t0, 1.0), 0.0, 1.0))

        def lerp(arr: np.ndarray) -> float:
            return float(arr[idx - 1] + alpha * (arr[idx] - arr[idx - 1]))

        return lerp(self._x), lerp(self._y), lerp(self._heading)

    def ego_to_world(self, points_ego_xy: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Transform (N, 2) ego-frame XY points into world/map frame at *timestamp_ns*."""
        x_w, y_w, heading = self.pose_at(timestamp_ns)
        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        out = (R @ points_ego_xy.astype(np.float64).T).T.copy()
        out[:, 0] += x_w
        out[:, 1] += y_w
        return out

    def trajectory_xy(self) -> np.ndarray:
        """Return (N, 2) float64 array of all map-frame ego positions."""
        return np.column_stack([self._x, self._y])

    def world_extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) over all loaded samples."""
        return (
            float(self._x.min()), float(self._x.max()),
            float(self._y.min()), float(self._y.max()),
        )

    @property
    def timestamps_ns(self) -> np.ndarray:
        return self._timestamps_ns

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def heading(self) -> np.ndarray:
        return self._heading

    def __repr__(self) -> str:
        return (
            f"BagTFReader(n={self.n_samples}, duration={self.duration_s:.1f}s, "
            f"'{self._parent_frame}→{self._child_frame}')"
        )


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation around z-axis) from a unit quaternion."""
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


# ── EgoStateLog ───────────────────────────────────────────────────────────────

class EgoStateLog:
    """Time-indexed ego vehicle state read from *ego.mat*.

    **Usage note (professor-confirmed):**
    The velocity fields (``ekf_vx``, ``ekf_vy``) from ego.mat are used
    *only* for correcting the Doppler radial velocity from radar sensors.
    For ego pose in the map frame, use :class:`BagTFReader` instead, which
    reads the ``map → center_of_gravity`` transforms directly from the bag's
    ``/tf`` topic.

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
        """Return interpolated (vx, vy, heading) for the given ROS timestamp.

        These velocities are in the ego frame and are used for Doppler
        velocity correction.  Do **not** integrate these for ego pose —
        use :class:`BagTFReader` for that.
        """
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


# ── Auto-init: load URDF extrinsics at import time ───────────────────────────
_auto_init()
