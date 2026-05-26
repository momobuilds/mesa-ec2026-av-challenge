"""Unit tests for EgoTrajectory."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

EGO_MAT = Path("data/ego.mat")
pytestmark = pytest.mark.skipif(not EGO_MAT.exists(), reason="data/ego.mat not found")

from src.tf_utils import EgoStateLog
from src.ego_trajectory import EgoTrajectory


def _make_traj() -> tuple[EgoTrajectory, EgoStateLog]:
    meta = yaml.safe_load(Path("data/metadata.yaml").read_text())
    bag_start = int(
        meta["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]
    )
    log = EgoStateLog("data/ego.mat", bag_start_ns=bag_start)
    return EgoTrajectory.from_ego_log(log), log


def test_pose_at_start_is_origin():
    traj, log = _make_traj()
    t_start = int(log._timestamps_ns[0])
    x, y, _ = traj.pose_at(t_start)
    assert abs(x) < 0.01 and abs(y) < 0.01, (
        f"Start pose should be (0,0), got ({x:.4f},{y:.4f})"
    )


def test_rotation_heading_zero():
    """With heading=0, forward motion (vx=10) should produce x displacement."""
    from unittest.mock import MagicMock

    # Synthetic EgoStateLog: 3 samples, heading=0, vx=10, vy=0
    # dt = 1.0 s each → total time 2s → x_w = 10*2 = 20 m, y_w = 0
    mock_log = MagicMock()
    mock_log._timestamps_ns = np.array([0.0, 1e9, 2e9])
    mock_log._vx = np.array([10.0, 10.0, 10.0])
    mock_log._vy = np.array([0.0, 0.0, 0.0])
    mock_log._heading = np.array([0.0, 0.0, 0.0])

    traj = EgoTrajectory.from_ego_log(mock_log)
    x, y, _ = traj.pose_at(int(2e9))
    assert abs(x - 20.0) < 0.01, f"Expected x≈20, got {x:.4f}"
    assert abs(y) < 0.01, f"Expected y≈0, got {y:.4f}"


def test_rotation_heading_pi_over_2():
    """With heading=π/2 (pointing left/north), forward motion produces y displacement."""
    from unittest.mock import MagicMock

    h = math.pi / 2
    mock_log = MagicMock()
    mock_log._timestamps_ns = np.array([0.0, 1e9, 2e9])
    mock_log._vx = np.array([10.0, 10.0, 10.0])
    mock_log._vy = np.array([0.0, 0.0, 0.0])
    mock_log._heading = np.array([h, h, h])

    traj = EgoTrajectory.from_ego_log(mock_log)
    x, y, _ = traj.pose_at(int(2e9))
    # vx_w = 10*cos(π/2) = 0, vy_w = 10*sin(π/2) = 10
    assert abs(x) < 0.01, f"Expected x≈0, got {x:.4f}"
    assert abs(y - 20.0) < 0.01, f"Expected y≈20, got {y:.4f}"


def test_rotation_lateral_velocity():
    """Lateral velocity (vy) at heading=0 should produce y displacement."""
    from unittest.mock import MagicMock

    # vy=5 at heading=0, vx=0:
    # vx_w = 0*cos(0) - 5*sin(0) = 0
    # vy_w = 0*sin(0) + 5*cos(0) = 5
    # total over 2s: x=0, y=10
    mock_log = MagicMock()
    mock_log._timestamps_ns = np.array([0.0, 1e9, 2e9])
    mock_log._vx = np.array([0.0, 0.0, 0.0])
    mock_log._vy = np.array([5.0, 5.0, 5.0])
    mock_log._heading = np.array([0.0, 0.0, 0.0])

    traj = EgoTrajectory.from_ego_log(mock_log)
    x, y, _ = traj.pose_at(int(2e9))
    assert abs(x) < 0.01, f"Expected x≈0, got {x:.4f}"
    assert abs(y - 10.0) < 0.01, f"Expected y≈10, got {y:.4f}"


def test_ego_to_world_at_start_applies_heading_rotation():
    traj, log = _make_traj()
    t_start = int(log._timestamps_ns[0])
    pts_ego = np.array([[10.0, 0.0], [0.0, 5.0]])
    pts_world = traj.ego_to_world(pts_ego, t_start)

    x0, y0, h0 = traj.pose_at(t_start)
    R = np.array([[math.cos(h0), -math.sin(h0)], [math.sin(h0), math.cos(h0)]])
    expected = (R @ pts_ego.T).T + np.array([x0, y0])
    np.testing.assert_allclose(pts_world, expected, atol=1e-6)


def test_world_extent_is_plausible():
    traj, _ = _make_traj()
    x_min, x_max, y_min, y_max = traj.world_extent()
    # Full bag: ~670 m × ~1120 m circuit
    assert x_max - x_min > 100, f"Track x-span {x_max - x_min:.0f} m suspiciously small"
    assert y_max - y_min > 100, f"Track y-span {y_max - y_min:.0f} m suspiciously small"
