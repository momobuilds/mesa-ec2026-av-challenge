"""Constant-velocity motion model with ego-motion compensation for Task 3.

Why ego-motion compensation matters
-------------------------------------
Sensor detections are expressed in the *ego* (centre_of_gravity) frame at the
moment they are measured.  Between two frames the ego vehicle may have moved
10 m forward and rotated several degrees.  A naive EKF predict step that just
adds v·Δt in ego coordinates treats the ego frame as inertial — it isn't.
Every stationary object then appears to travel −v_ego per step, causing every
track to immediately lose association with its next detection.

The correct approach is to predict in the **world** frame and then re-express
the result in the new ego frame:

    Let R1 = rot2D(h1)  (ego-frame → world rotation at t1)
        R2 = rot2D(h2)  (ego-frame → world rotation at t2)
        Δe = world-frame ego displacement ≈ ½(v_ego1 + v_ego2) · Δt

    Predicted position in ego2 frame:
        p_ego2 = R2ᵀ @ R1 @ (p_ego1 + v_ego1·Δt) − R2ᵀ @ Δe
               = R2ᵀ @ (R1 @ p_ego1 + R1 @ v_ego1·Δt − Δe)

    Predicted velocity in ego2 frame:
        v_ego2 = R2ᵀ @ R1 @ v_ego1

State vector: [x, y, vx, vy, yaw, length, width]  (indices from task2_fusion)
Process noise Q is per-second; it is scaled by Δt inside predict_state().
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Reuse state-vector index constants from task2_fusion to stay consistent.
from src.task2_fusion import _X, _Y, _VX, _VY, _YAW, _L, _W, STATE_DIM

# ── Process noise (per second, diagonal variances) ────────────────────────────
# These are 1-σ² values; multiply by Δt when building Q.
#   position: 0.5 m²/s  — opponent lane-keeping is fairly constrained
#   velocity: 2.0 m²/s² — acceleration / braking bursts
#   yaw:      (15°)²/s  — heading can change quickly in corners
#   dims:     0.1 m²/s  — dimensions should be nearly constant
_Q_DIAG_PER_SEC = np.array(
    [0.5, 0.5, 2.0, 2.0, (math.radians(15)) ** 2, 0.1, 0.1],
    dtype=np.float64,
)


def _rot2d(heading_rad: float) -> np.ndarray:
    """2×2 rotation matrix: ego-frame → world frame for given heading (rad)."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def predict_state(
    state: np.ndarray,          # (7,) in ego frame at t1
    P: np.ndarray,              # (7, 7) covariance at t1
    ego_state_t1: dict[str, float],  # from EgoStateLog.interpolate(t1)
    ego_state_t2: dict[str, float],  # from EgoStateLog.interpolate(t2)
    dt: float,                  # (t2 - t1) in seconds
) -> tuple[np.ndarray, np.ndarray]:
    """Predict track state from t1 to t2, compensating for ego motion.

    Parameters
    ----------
    state        : [x, y, vx, vy, yaw, length, width] in ego frame at t1.
    P            : 7×7 state covariance at t1.
    ego_state_t1 : dict with keys 'vx', 'vy', 'heading' at t1.
    ego_state_t2 : dict with keys 'vx', 'vy', 'heading' at t2.
    dt           : time step in seconds (positive).

    Returns
    -------
    state_pred : (7,) predicted state in ego frame at t2.
    P_pred     : (7, 7) predicted covariance at t2.
    """
    if dt <= 0.0:
        return state.copy(), P.copy()

    # ── 1. Build rotations and ego displacement ──
    h1 = float(ego_state_t1["heading"])
    h2 = float(ego_state_t2["heading"])
    R1 = _rot2d(h1)  # ego1 → world
    R2 = _rot2d(h2)  # ego2 → world

    vx_ego1 = float(ego_state_t1["vx"])
    vy_ego1 = float(ego_state_t1["vy"])
    vx_ego2 = float(ego_state_t2["vx"])
    vy_ego2 = float(ego_state_t2["vy"])

    # Trapezoidal integration of ego world-frame displacement
    delta_e = np.array(
        [0.5 * (vx_ego1 + vx_ego2) * dt,
         0.5 * (vy_ego1 + vy_ego2) * dt],
        dtype=np.float64,
    )

    # ── 2. Ego-motion-compensated CV position prediction ──
    p_ego1 = state[[_X, _Y]]          # (2,) position in ego1
    v_ego1 = state[[_VX, _VY]]        # (2,) velocity in ego1

    # Position in ego2 frame (derivation in module docstring):
    #   p_ego2 = R2ᵀ @ (R1 @ p_ego1 + R1 @ v_ego1 · Δt − Δe)
    p_world_t2 = R1 @ p_ego1 + R1 @ v_ego1 * dt - delta_e
    p_ego2 = R2.T @ p_world_t2

    # Velocity in ego2 frame:
    #   v_ego2 = R2ᵀ @ R1 @ v_ego1
    v_ego2 = R2.T @ R1 @ v_ego1

    # Yaw: rotate by the heading change so it stays relative to ego forward.
    delta_h = h2 - h1
    yaw_pred = _wrap_angle(float(state[_YAW]) - delta_h)

    # Dimensions don't change.
    length_pred = float(state[_L])
    width_pred = float(state[_W])

    state_pred = np.array(
        [p_ego2[0], p_ego2[1], v_ego2[0], v_ego2[1],
         yaw_pred, length_pred, width_pred],
        dtype=np.float64,
    )

    # ── 3. Linearised state transition Jacobian F ──────────────────────────────
    # For [x, y, vx, vy, yaw, l, w] the full Jacobian in the ego-motion-
    # compensated CV model is:
    #   F_pos_pos = R2ᵀ @ R1                 (2×2 block)
    #   F_pos_vel = R2ᵀ @ R1 * Δt            (2×2 block)
    #   F_vel_vel = R2ᵀ @ R1                 (2×2 block)
    #   yaw, l, w: identity (scalar propagation)
    R2T_R1 = R2.T @ R1

    F = np.eye(STATE_DIM, dtype=np.float64)
    F[np.ix_([_X, _Y], [_X, _Y])]   = R2T_R1
    F[np.ix_([_X, _Y], [_VX, _VY])] = R2T_R1 * dt
    F[np.ix_([_VX, _VY], [_VX, _VY])] = R2T_R1

    # ── 4. Process noise Q (scaled by Δt) ──────────────────────────────────────
    Q = np.diag(_Q_DIAG_PER_SEC * dt)

    # ── 5. Predicted covariance ──────────────────────────────────────────────
    P_pred = F @ P @ F.T + Q

    return state_pred, P_pred


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


# ── Standalone unit test (run directly: python -m src.motion_model) ───────────

def _unit_test_stationary_opponent() -> None:
    """A stationary opponent must stay near the origin in every ego frame.

    Scenario: ego drives 10 m forward at constant 20 m/s heading=0.
    Opponent at ego1-frame position (50, 0) with zero velocity.
    After predict, opponent should still be at ~(40, 0) in ego2 frame
    (it moved from 50 m ahead to 40 m ahead as ego closed the gap).
    """
    import math

    # Ego moves forward at 20 m/s heading=0 for 0.5 s → Δe = (10, 0)
    ego1 = {"vx": 20.0, "vy": 0.0, "heading": 0.0}
    ego2 = {"vx": 20.0, "vy": 0.0, "heading": 0.0}
    dt = 0.5

    # Stationary opponent at (50, 2) in ego1 frame, zero velocity
    state = np.zeros(STATE_DIM)
    state[_X] = 50.0
    state[_Y] = 2.0
    state[_L] = 4.5
    state[_W] = 2.0
    P = np.eye(STATE_DIM) * 0.1

    pred, _ = predict_state(state, P, ego1, ego2, dt)

    expected_x = 40.0   # ego closed 10 m
    expected_y = 2.0    # no lateral motion
    pos_err = math.hypot(pred[_X] - expected_x, pred[_Y] - expected_y)
    assert pos_err < 0.01, (
        f"Stationary-opponent test FAILED: pred=({pred[_X]:.3f},{pred[_Y]:.3f}), "
        f"expected=({expected_x},{expected_y}), err={pos_err:.4f} m"
    )
    print(f"[PASS] stationary opponent: pred=({pred[_X]:.3f}, {pred[_Y]:.3f}), "
          f"err={pos_err:.4f} m  (expected ~{expected_x}, {expected_y})")


def _unit_test_moving_opponent() -> None:
    """Moving opponent should predict correctly through ego-frame change.

    Scenario: ego drives 20 m/s forward heading=0 for 0.5 s.
    Opponent drives at 25 m/s forward (same direction) 50 m ahead.
    After 0.5 s the opponent should be at ~(52.5, 0) in ego2 frame
    (opponent gained 2.5 m on ego: 5 m opp travel - 10 m ego travel → 50+2.5).
    """
    ego1 = {"vx": 20.0, "vy": 0.0, "heading": 0.0}
    ego2 = {"vx": 20.0, "vy": 0.0, "heading": 0.0}
    dt = 0.5

    state = np.zeros(STATE_DIM)
    state[_X] = 50.0
    state[_Y] = 0.0
    state[_VX] = 25.0   # opponent goes 25 m/s in world → 5 m/s rel to ego
    state[_VY] = 0.0
    state[_L] = 4.5
    state[_W] = 2.0
    P = np.eye(STATE_DIM) * 0.1

    pred, _ = predict_state(state, P, ego1, ego2, dt)

    # In ego1 frame, opponent is at vx=25 (world) but ego goes 20 → apparent vx_rel=5 m/s
    # After 0.5 s: 50 + 5*0.5 = 52.5
    expected_x = 52.5
    pos_err = abs(pred[_X] - expected_x)
    assert pos_err < 0.01, (
        f"Moving-opponent test FAILED: pred_x={pred[_X]:.3f}, "
        f"expected={expected_x}, err={pos_err:.4f} m"
    )
    print(f"[PASS] moving opponent: pred_x={pred[_X]:.3f}, err={pos_err:.4f} m "
          f"(expected ~{expected_x})")


def _unit_test_turning_ego() -> None:
    """When ego turns, a forward opponent should stay roughly forward.

    Scenario: ego turns 30° left (heading 0→π/6) over 0.5 s at 20 m/s.
    Opponent is directly ahead at (50, 0) in ego1 frame with zero velocity.
    In ego2 frame the opponent should be ~50 m away, rotated right by ~30°.
    """
    import math as _math

    heading2 = _math.pi / 6  # 30° turn
    ego1 = {"vx": 20.0, "vy": 0.0, "heading": 0.0}
    # After turning, ego heading is 30°; ego velocity direction also changes.
    # For simplicity, keep speed the same but in new direction.
    ego2 = {"vx": 20.0 * _math.cos(heading2), "vy": 20.0 * _math.sin(heading2), "heading": heading2}
    dt = 0.5

    state = np.zeros(STATE_DIM)
    state[_X] = 50.0
    state[_Y] = 0.0
    state[_L] = 4.5
    state[_W] = 2.0
    P = np.eye(STATE_DIM) * 0.1

    pred, _ = predict_state(state, P, ego1, ego2, dt)

    dist = _math.hypot(pred[_X], pred[_Y])
    # Distance should be < 50 (ego gained ~10 m), and opponent should be
    # to the right in ego2 frame (negative y) due to left turn.
    dist_ok = 35.0 < dist < 50.0
    lateral_ok = pred[_Y] < -1.0   # opponent appears right after left turn
    assert dist_ok and lateral_ok, (
        f"Turning-ego test FAILED: pred=({pred[_X]:.2f},{pred[_Y]:.2f}) dist={dist:.2f}"
    )
    print(f"[PASS] turning ego: pred=({pred[_X]:.2f}, {pred[_Y]:.2f}), "
          f"dist={dist:.2f} m, y<0 (right)={pred[_Y]:.2f}")


if __name__ == "__main__":
    print("Running motion_model unit tests …")
    _unit_test_stationary_opponent()
    _unit_test_moving_opponent()
    _unit_test_turning_ego()
    print("All tests passed.")
