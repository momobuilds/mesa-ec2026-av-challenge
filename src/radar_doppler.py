"""Radar Doppler velocity extraction for Task 2.

The radar ``PointCloud2`` messages in this bag contain a scalar ``velocity``
field that is the *radial* range-rate of each target relative to the sensor
(negative = target approaching = stationary object when ego moves forward).

Bag field schema (confirmed from live data)::

    x, y, z, velocity, snr, rcs, confidence, velocity_interval

Coordinate convention: y = range/forward, x = lateral, z = vertical
(same as the Task 1 axis mapping for radar topics).

Algorithm per cluster
---------------------
1. For each point *p* in the cluster (sensor frame):
   a. Compute the line-of-sight unit vector from sensor origin to *p*.
   b. Transform it to ego frame via the sensor extrinsic rotation.
   c. Correct for ego-motion contribution:
        v_target_radial = v_measured + (v_ego_ego · los_hat_ego)
      where ``v_ego_ego`` = [ekf_vx, ekf_vy, 0] from EgoStateLog.
   d. Project radial onto ego x/y:  dv = v_target_radial * los_hat_ego[:2]
2. Cluster (vx, vy) = weighted mean of per-point dv, weighted by confidence.

If no ``velocity`` field is present, or all weights are zero, returns (0, 0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.tf_utils import get_extrinsic


def cluster_velocity(
    cluster_pts_sensor: np.ndarray,
    sensor_frame: str,
    *,
    ego_vx: float = 0.0,
    ego_vy: float = 0.0,
    confidence_weights: np.ndarray | None = None,
    min_confidence: float = 0.1,
) -> tuple[float, float]:
    """Estimate (vx, vy) of a radar cluster in the ego frame.

    Parameters
    ----------
    cluster_pts_sensor:
        Structured numpy array of the cluster's points, still in sensor frame.
        Must include fields ``x``, ``y``, ``z``, and ``velocity``.
    sensor_frame:
        ROS frame_id of the radar sensor (e.g. ``'radar_front'``).
    ego_vx, ego_vy:
        Ego velocity in ego frame at this timestamp, from EgoStateLog.
    confidence_weights:
        Per-point confidence values (same length as cluster).  If None,
        the ``confidence`` field from the structured array is used when
        present, otherwise uniform weights.
    min_confidence:
        Points below this confidence are excluded.

    Returns
    -------
    (vx_ego, vy_ego) in m/s (ego frame); (0.0, 0.0) on failure.
    """
    if "velocity" not in cluster_pts_sensor.dtype.names:
        return 0.0, 0.0

    x_s = cluster_pts_sensor["x"].astype(np.float64)
    y_s = cluster_pts_sensor["y"].astype(np.float64)
    z_s = cluster_pts_sensor["z"].astype(np.float64)
    v_meas = cluster_pts_sensor["velocity"].astype(np.float64)

    # Confidence weights
    if confidence_weights is not None:
        w = confidence_weights.astype(np.float64)
    elif "confidence" in cluster_pts_sensor.dtype.names:
        w = cluster_pts_sensor["confidence"].astype(np.float64)
    else:
        w = np.ones(len(x_s), dtype=np.float64)

    # Filter low-confidence and degenerate points
    dist = np.sqrt(x_s**2 + y_s**2 + z_s**2)
    valid = (w >= min_confidence) & (dist > 0.1) & np.isfinite(v_meas)
    if not valid.any():
        return 0.0, 0.0

    x_s, y_s, z_s = x_s[valid], y_s[valid], z_s[valid]
    v_meas, w = v_meas[valid], w[valid]
    dist = dist[valid]

    # Line-of-sight unit vectors in sensor frame
    los_sensor = np.column_stack([x_s, y_s, z_s]) / dist[:, None]  # (N, 3)

    # Rotate LOS to ego frame (translation irrelevant for direction)
    R_ego = get_extrinsic(sensor_frame)[:3, :3]
    los_ego = (R_ego @ los_sensor.T).T  # (N, 3)

    # Ego-motion radial contribution at each point
    ego_vel_ego = np.array([ego_vx, ego_vy, 0.0])
    v_ego_radial = los_ego @ ego_vel_ego  # (N,) dot product per point

    # Target radial velocity in ego frame (correct for ego motion)
    v_target_radial = v_meas + v_ego_radial  # (N,)

    # Project onto ego x/y plane
    vx_pts = v_target_radial * los_ego[:, 0]  # (N,)
    vy_pts = v_target_radial * los_ego[:, 1]  # (N,)

    total_w = w.sum()
    if total_w < 1e-9:
        return 0.0, 0.0

    vx_ego = float((w * vx_pts).sum() / total_w)
    vy_ego = float((w * vy_pts).sum() / total_w)
    return vx_ego, vy_ego


def extract_cluster_points(
    structured_points: np.ndarray,
    bev_mask_indices: np.ndarray,
) -> np.ndarray:
    """Extract the raw structured rows for points that belong to a cluster.

    *bev_mask_indices* are the original row indices within
    ``structured_points`` that survived the BEV projection step.
    """
    return structured_points[bev_mask_indices]


def print_radar_field_schema(structured_points: np.ndarray, topic: str) -> None:
    """Print field names and statistics for one radar frame (diagnostic)."""
    print(f"Radar field schema — {topic}:")
    print(f"  Fields: {structured_points.dtype.names}")
    for name in structured_points.dtype.names:
        v = structured_points[name].astype(float)
        fin = v[np.isfinite(v)]
        if len(fin):
            print(
                f"  {name:22s}  n={len(fin)}  "
                f"min={fin.min():.3f}  max={fin.max():.3f}  mean={fin.mean():.3f}"
            )
