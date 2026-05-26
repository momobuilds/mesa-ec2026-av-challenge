"""Camera projection utilities for Task 2 (sanity-check visualisation).

No ``camera_info`` topics exist in the bag and all ``frame_id`` fields in the
compressed-image messages are empty.  Intrinsics and extrinsics used here are
therefore *placeholder approximations*.

Intrinsics (camera_fl, 1928 × 500 px)
--------------------------------------
Assumed 90° horizontal field-of-view →
    fx = fy = width / (2 · tan(45°)) = 964 px
    cx = 964,  cy = 250

Extrinsic (ego → camera_fl optical frame)
------------------------------------------
Camera mounted at  t = [+1.5, +0.4, +0.8] m  (fwd, left, up from COG).
Forward-facing (mounting yaw = 0°).

Camera optical frame convention: x right, y down, z into scene (forward).
    R_ego_to_optical = [[0, −1, 0],
                        [0,  0, −1],
                        [1,  0,  0]]

These assumptions are documented as limitations in task2_report.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ── Placeholder intrinsics ────────────────────────────────────────────────────

IMAGE_WIDTH = 1928
IMAGE_HEIGHT = 500

# 90° HFOV → fx = width / (2·tan(45°))
_FX = float(IMAGE_WIDTH / 2.0)
_FY = _FX
_CX = float(IMAGE_WIDTH / 2.0)
_CY = float(IMAGE_HEIGHT / 2.0)

K = np.array(
    [[_FX, 0.0, _CX],
     [0.0, _FY, _CY],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# ── Placeholder extrinsic (ego → camera_fl optical) ──────────────────────────

# Camera position in ego frame (x fwd, y left, z up).
_T_CAM_EGO = np.array([1.5, 0.4, 0.8], dtype=np.float64)

# Ego frame (x fwd, y left, z up) → camera optical (x right, y down, z fwd).
_R_EGO_TO_OPTICAL = np.array(
    [[0.0, -1.0, 0.0],
     [0.0,  0.0, -1.0],
     [1.0,  0.0,  0.0]],
    dtype=np.float64,
)


def project_points(
    points_ego: np.ndarray,
    *,
    K_mat: np.ndarray = K,
    R_ego_to_cam: np.ndarray = _R_EGO_TO_OPTICAL,
    t_cam_in_ego: np.ndarray = _T_CAM_EGO,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame 3-D points onto the camera image plane.

    Parameters
    ----------
    points_ego : (N, 3) float array in ego frame.

    Returns
    -------
    uv : (N, 2) pixel coordinates (may be outside image bounds).
    in_front : (N,) bool mask — True where z_cam > 0.
    """
    if len(points_ego) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    p_cam = (R_ego_to_cam @ (points_ego.astype(np.float64) - t_cam_in_ego).T).T
    in_front = p_cam[:, 2] > 0.1
    z = np.where(in_front, p_cam[:, 2], 1.0)
    u = K_mat[0, 0] * p_cam[:, 0] / z + K_mat[0, 2]
    v = K_mat[1, 1] * p_cam[:, 1] / z + K_mat[1, 2]
    return np.column_stack([u, v]), in_front


def bbox_3d_corners(
    cx: float, cy: float, cz: float,
    length: float, width: float, height: float,
    yaw: float,
) -> np.ndarray:
    """Return the 8 corners of a 3-D oriented bounding box in ego frame.

    The box is specified in BEV (cx, cy ego frame) with *cz* as the vertical
    centre, *height* = 1.5 m (fixed vehicle assumption), *yaw* CCW from ego-x.
    """
    half_l, half_w, half_h = length / 2, width / 2, height / 2
    c, s = np.cos(yaw), np.sin(yaw)
    # Corners relative to centre in BEV (l, w), then lift to 3-D
    corners_bev = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l,  half_w],
        [-half_l, -half_w],
    ])
    rot = np.array([[c, -s], [s, c]])
    rotated = (rot @ corners_bev.T).T  # (4, 2)
    corners_3d = []
    for dx, dy in rotated:
        for dz in (-half_h, half_h):
            corners_3d.append([cx + dx, cy + dy, cz + dz])
    return np.array(corners_3d, dtype=np.float64)  # (8, 3)


def draw_projected_boxes(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    vehicle_height: float = 1.5,
    cz_ego: float = 0.0,
) -> np.ndarray:
    """Draw projected 3-D bounding boxes on *image* (in-place copy).

    Uses ego-frame fields ``center_x``, ``center_y``, ``length``, ``width``,
    and ``yaw`` from each detection dict.
    """
    img = image.copy()
    for det in detections:
        cx = float(det.get("center_x", det.get("center_x", 0)))
        cy = float(det.get("center_y", det.get("center_y", 0)))
        length = float(det.get("length", 4.0))
        width = float(det.get("width", 2.0))
        yaw = float(det.get("yaw", 0.0))

        corners = bbox_3d_corners(cx, cy, cz_ego, length, width, vehicle_height, yaw)
        uv, in_front = project_points(corners)

        if not in_front.any():
            continue

        # Draw lines connecting the 8 corners (bottom 4 = indices 0-3, top 4 = 4-7)
        pairs = [
            (0,1),(2,3),(0,2),(1,3),   # bottom face
            (4,5),(6,7),(4,6),(5,7),   # top face
            (0,4),(1,5),(2,6),(3,7),   # verticals
        ]
        for i, j in pairs:
            if in_front[i] and in_front[j]:
                pt1 = (int(uv[i, 0]), int(uv[i, 1]))
                pt2 = (int(uv[j, 0]), int(uv[j, 1]))
                cv2.line(img, pt1, pt2, color, thickness)

        # Label
        front_pts = uv[in_front]
        if len(front_pts):
            cx_px = int(front_pts[:, 0].mean())
            cy_px = int(front_pts[:, 1].mean())
            label = det.get("detection_id", "?")
            cv2.putText(img, str(label), (cx_px, cy_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def read_compressed_image(raw_data: bytes) -> np.ndarray:
    """Decode a JPEG/PNG CompressedImage payload to a BGR numpy array."""
    arr = np.frombuffer(raw_data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
