"""World-frame ego pose trajectory by integrating EgoStateLog velocities.

The EKF outputs (ekf_vx, ekf_vy) in ego.mat are ego-frame velocities
(longitudinal / lateral), NOT world-frame. To integrate into world-frame
displacement, rotate each sample by the instantaneous heading:

    dx_w = vx · cos(h) − vy · sin(h)
    dy_w = vx · sin(h) + vy · cos(h)

heading in ego.mat is a monotonically increasing odometer value (never
wrapped to [−π, π]). cos/sin are 2π-periodic, so no unwrapping is needed.

predict_state() in motion_model.py skips this rotation because Δt ≈ 0.2 s
gives Δheading ≈ 0.4° (rotation ≈ identity for a single step). Here we
integrate the full trajectory so the rotation is essential.

Origin: bag start → (x=0, y=0), heading=h[0].
"""
from __future__ import annotations

import math

import numpy as np

from src.tf_utils import EgoStateLog


def _rot2d(heading_rad: float) -> np.ndarray:
    """2×2 rotation matrix: ego-frame → world frame."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class EgoTrajectory:
    """Continuous world-frame pose trajectory integrated from EgoStateLog."""

    def __init__(
        self,
        timestamps_ns: np.ndarray,
        x_w: np.ndarray,
        y_w: np.ndarray,
        heading: np.ndarray,
    ) -> None:
        self._timestamps_ns = timestamps_ns
        self._x_w = x_w
        self._y_w = y_w
        self._heading = heading

    @classmethod
    def from_ego_log(cls, ego_log: EgoStateLog) -> "EgoTrajectory":
        """Integrate all EgoStateLog samples into a world-frame trajectory."""
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
        """Return (x_min, x_max, y_min, y_max) over all integrated samples."""
        return (
            float(self._x_w.min()), float(self._x_w.max()),
            float(self._y_w.min()), float(self._y_w.max()),
        )

    def trajectory_xy(self) -> np.ndarray:
        """Return (N, 2) float64 array of all world-frame positions."""
        return np.column_stack([self._x_w, self._y_w])
