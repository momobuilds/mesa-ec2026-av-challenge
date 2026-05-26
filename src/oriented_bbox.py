"""Oriented bounding-box utilities for Task 2.

Replaces Task 1's axis-aligned box with a minimum-area rectangle computed
from the cluster's 2-D BEV convex hull (via Shapely).  Heading is taken as
the long-axis yaw of the rectangle, then optionally sanity-checked against a
supplied velocity direction.

All coordinates are in the ego (centre_of_gravity) frame:
  x = forward, y = left.
Yaw is CCW-positive from the ego-x axis (car-forward), in radians ∈ (−π, π].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from shapely.geometry import MultiPoint
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("shapely is required for oriented_bbox; run: uv sync") from exc


def minimum_area_bbox(points_2d: np.ndarray) -> dict[str, Any]:
    """Compute the minimum-area oriented bounding box for a 2-D point set.

    Parameters
    ----------
    points_2d:
        (N, 2) float array, columns = [x_ego, y_ego].

    Returns
    -------
    dict with keys:
        center_x, center_y  – box centre (m, ego frame)
        length              – extent along the long axis (m)
        width               – extent along the short axis (m)
        yaw                 – long-axis heading CCW from ego-x (rad, −π..π]
        corners             – (4, 2) array of corner points
    """
    if len(points_2d) < 3:
        return _degenerate_bbox(points_2d)

    hull = MultiPoint(points_2d).convex_hull
    if hull.is_empty or hull.geom_type == "Point":
        return _degenerate_bbox(points_2d)

    mrr = hull.minimum_rotated_rectangle
    if mrr is None or mrr.is_empty:
        return _degenerate_bbox(points_2d)

    corners = np.array(mrr.exterior.coords[:-1])  # 4 corners, drop repeated first
    if len(corners) != 4:
        return _degenerate_bbox(points_2d)

    # Edge vectors and lengths
    e0 = corners[1] - corners[0]
    e1 = corners[2] - corners[1]
    l0, l1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))

    if l0 >= l1:
        long_vec, length, width = e0, l0, l1
    else:
        long_vec, length, width = e1, l1, l0

    yaw = float(math.atan2(float(long_vec[1]), float(long_vec[0])))
    center = corners.mean(axis=0)

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "length": length,
        "width": width,
        "yaw": yaw,
        "corners": corners,
    }


def oriented_bbox_from_detection(
    detection: dict[str, Any],
    cluster_points: np.ndarray,
) -> dict[str, Any]:
    """Augment a Task 1/2 detection dict with oriented bbox fields.

    If the cluster has < 3 points or Shapely fails, falls back to the
    axis-aligned bbox already in *detection* with yaw = 0.

    Added / updated keys: ``center_x``, ``center_y``, ``length``,
    ``width``, ``yaw``, ``corners``.
    """
    obb = minimum_area_bbox(cluster_points)
    out = dict(detection)
    out.update(obb)
    return out


def reconcile_yaw_with_velocity(
    yaw: float,
    vx: float,
    vy: float,
    *,
    speed_threshold: float = 1.0,
) -> float:
    """Flip *yaw* by π if it disagrees with the velocity direction by > 90°.

    Heading from a min-area rectangle is ambiguous by ±π (both long-axis
    directions are equally valid geometrically).  When a reliable velocity
    vector is available, we pick the half that agrees with motion direction.

    Returns the reconciled yaw in (−π, π].
    """
    speed = math.hypot(vx, vy)
    if speed < speed_threshold:
        return yaw

    vel_angle = math.atan2(vy, vx)
    diff = _angle_diff(yaw, vel_angle)
    if abs(diff) > math.pi / 2:
        yaw = _wrap_angle(yaw + math.pi)
    return yaw


# ── helpers ───────────────────────────────────────────────────────────────────

def _degenerate_bbox(points_2d: np.ndarray) -> dict[str, Any]:
    """Axis-aligned fallback when a proper MRR cannot be computed."""
    if len(points_2d) == 0:
        return {
            "center_x": 0.0, "center_y": 0.0,
            "length": 0.0, "width": 0.0, "yaw": 0.0,
            "corners": np.zeros((4, 2)),
        }
    mn = points_2d.min(axis=0)
    mx = points_2d.max(axis=0)
    cx, cy = float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2)
    lx, ly = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    length = max(lx, ly, 1e-3)
    width = min(lx, ly, length)
    corners = np.array([
        [mn[0], mn[1]], [mx[0], mn[1]],
        [mx[0], mx[1]], [mn[0], mx[1]],
    ])
    return {
        "center_x": cx, "center_y": cy,
        "length": length, "width": width, "yaw": 0.0,
        "corners": corners,
    }


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(a - b)


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)
