"""World-frame ego pose trajectory for the challenge pipeline.

Two sources are supported (professor-confirmed strategy):

BagTFReader  ←  **preferred for ego pose in map frame**
    The ``/tf`` topic in the bag publishes ``map → center_of_gravity`` at
    ~200 Hz.  :meth:`EgoTrajectory.from_bag_tf` wraps this source and is
    the recommended way to obtain world-frame pose.

EgoStateLog (ego.mat)  ←  **used only for Doppler velocity correction**
    The EKF outputs (ekf_vx, ekf_vy) are ego-frame velocities needed to
    remove the ego-motion component from radar Doppler measurements.
    :meth:`EgoTrajectory.from_ego_log` integrates them into a world-frame
    trajectory as a fallback when the bag is unavailable, but the TF-based
    approach is preferred.

Integration note for from_ego_log
----------------------------------
The (ekf_vx, ekf_vy) in ego.mat are in the ego frame (longitudinal/lateral).
To integrate into world-frame displacement, each sample is rotated by the
instantaneous heading::

    dx_w = vx · cos(h) − vy · sin(h)
    dy_w = vx · sin(h) + vy · cos(h)

heading in ego.mat is a monotonically increasing odometer value (never
wrapped to [−π, π]).  cos/sin are 2π-periodic, so no unwrapping is needed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _rot2d(heading_rad: float) -> np.ndarray:
    """2×2 rotation matrix: ego-frame → world frame."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class EgoTrajectory:
    """Continuous world-frame pose trajectory for the ego vehicle.

    Construct via:
    - :meth:`from_bag_tf`  — TF-based (preferred, uses ``/tf`` from the bag)
    - :meth:`from_ego_log` — velocity-integration fallback (uses ego.mat)
    """

    def __init__(
        self,
        timestamps_ns: np.ndarray,
        x_w: np.ndarray,
        y_w: np.ndarray,
        heading: np.ndarray,
    ) -> None:
        self._timestamps_ns = np.asarray(timestamps_ns, dtype=np.float64)
        self._x_w = np.asarray(x_w, dtype=np.float64)
        self._y_w = np.asarray(y_w, dtype=np.float64)
        self._heading = np.asarray(heading, dtype=np.float64)

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_bag_tf(cls, bag_dir: str | Path) -> "EgoTrajectory":
        """Build EgoTrajectory from the ``/tf`` topic in the bag  (preferred).

        Reads every ``map → center_of_gravity`` transform from the bag's
        ``/tf`` topic and stores them as a time-indexed trajectory.  This
        avoids drift caused by velocity integration and is the
        professor-recommended source for world-frame ego pose.

        The ego.mat velocity data is still required separately for Doppler
        velocity correction (via :class:`~src.tf_utils.EgoStateLog`).

        Parameters
        ----------
        bag_dir:
            Directory containing the ``*.db3`` bag files.
        """
        from src.tf_utils import BagTFReader
        reader = BagTFReader(bag_dir)
        return cls(
            timestamps_ns=reader.timestamps_ns,
            x_w=reader.x,
            y_w=reader.y,
            heading=reader.heading,
        )

    @classmethod
    def from_ego_log(cls, ego_log: "Any") -> "EgoTrajectory":
        """Integrate all EgoStateLog samples into a world-frame trajectory.

        Fallback for when the bag is unavailable.  Prefer
        :meth:`from_bag_tf` for accuracy.

        **Do not use these integrated positions for Doppler correction** —
        that requires only the instantaneous ego velocity from
        :meth:`~src.tf_utils.EgoStateLog.interpolate`.
        """
        t = ego_log._timestamps_ns
        vx = ego_log._vx
        vy = ego_log._vy
        h = ego_log._heading

        # Rotate ego-frame (longitudinal, lateral) → world-frame (x, y)
        vx_w = vx * np.cos(h) - vy * np.sin(h)
        vy_w = vx * np.sin(h) + vy * np.cos(h)

        dt = np.diff(t) / 1e9  # seconds
        dx = 0.5 * (vx_w[:-1] + vx_w[1:]) * dt
        dy = 0.5 * (vy_w[:-1] + vy_w[1:]) * dt

        x_w = np.concatenate([[0.0], np.cumsum(dx)])
        y_w = np.concatenate([[0.0], np.cumsum(dy)])

        return cls(timestamps_ns=t, x_w=x_w, y_w=y_w, heading=h)

    # ── public interface ──────────────────────────────────────────────────────

    def pose_at(self, t_ns: int) -> tuple[float, float, float]:
        """Interpolate (x_w, y_w, heading) at ROS timestamp *t_ns*."""
        t = float(t_ns)
        idx = int(np.searchsorted(self._timestamps_ns, t))
        idx = max(1, min(idx, len(self._timestamps_ns) - 1))
        t0, t1 = self._timestamps_ns[idx - 1], self._timestamps_ns[idx]
        alpha = float(np.clip((t - t0) / max(t1 - t0, 1.0), 0.0, 1.0))

        def lerp(a: np.ndarray) -> float:
            return float(a[idx - 1] + alpha * (a[idx] - a[idx - 1]))

        return lerp(self._x_w), lerp(self._y_w), lerp(self._heading)

    def ego_to_world(self, points_ego: np.ndarray, t_ns: int) -> np.ndarray:
        """Transform (N, 2) ego-frame XY to world frame at time *t_ns*."""
        x_w, y_w, heading = self.pose_at(t_ns)
        R = _rot2d(heading)
        out = (R @ points_ego.T).T.copy()
        out[:, 0] += x_w
        out[:, 1] += y_w
        return out

    def world_extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) over all trajectory samples."""
        return (
            float(self._x_w.min()), float(self._x_w.max()),
            float(self._y_w.min()), float(self._y_w.max()),
        )

    def trajectory_xy(self) -> np.ndarray:
        """Return (N, 2) float64 array of all world-frame positions."""
        return np.column_stack([self._x_w, self._y_w])


# Allow `from src.tf_utils import EgoStateLog` in type hints without circular import
from typing import Any
