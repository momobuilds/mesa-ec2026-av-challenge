"""Track barrier polyline extraction from the accumulated occupancy grid.

Pipeline
--------
1. Threshold log-odds grid to binary occupied image.
2. Morphological closing (3×3 kernel) to bridge gaps in the barriers.
3. cv2.findContours with RETR_CCOMP retrieves both outer contour (outer
   barrier edge) and inner hole contours (inner barrier edge of a closed
   loop track).
4. Filter: discard contours whose arc length < min_perimeter_cells cells
   (default 200 cells = 50 m at 0.25 m/cell).
5. Simplify with Douglas-Peucker (cv2.approxPolyDP) at ε ~ 0.5 m.
6. Convert OpenCV pixel [col, row] → world [x, y] coordinates.
7. Serialise as GeoJSON LineString features.
"""
from __future__ import annotations

import json

import cv2
import numpy as np


def extract_boundaries(
    occupied_mask: np.ndarray,         # (H, W) bool
    origin_x: float,
    origin_y: float,
    resolution: float,
    min_perimeter_cells: int = 200,    # 200 × 0.25 m = 50 m minimum
    dp_epsilon_m: float = 0.5,
) -> list[np.ndarray]:
    """Extract track barrier polylines from the occupancy mask.

    Parameters
    ----------
    occupied_mask      : Boolean occupancy array (True = occupied).
    origin_x, origin_y : World coordinates of the grid's SW corner.
    resolution         : Metres per cell.
    min_perimeter_cells: Contours shorter than this (cells) are discarded.
    dp_epsilon_m       : Douglas-Peucker simplification tolerance (metres).

    Returns
    -------
    List of (M, 2) float64 arrays; each row is (x_world, y_world).
    """
    u8 = occupied_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP: level-0 = outer contours, level-1 = inner hole contours
    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    dp_eps = dp_epsilon_m / resolution
    boundaries: list[np.ndarray] = []

    for contour in contours:
        if cv2.arcLength(contour, closed=True) < min_perimeter_cells:
            continue
        approx = cv2.approxPolyDP(contour, dp_eps, closed=True)
        pts = approx.reshape(-1, 2)   # (M, 2)  [col, row] OpenCV convention
        x_w = pts[:, 0].astype(np.float64) * resolution + origin_x
        y_w = pts[:, 1].astype(np.float64) * resolution + origin_y
        boundaries.append(np.column_stack([x_w, y_w]))

    return boundaries


def save_geojson(boundaries: list[np.ndarray], path: str) -> None:
    """Serialise boundary polylines as a GeoJSON FeatureCollection of LineStrings."""
    features = []
    for i, pts in enumerate(boundaries):
        if len(pts) < 2:
            continue
        length_m = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        features.append({
            "type": "Feature",
            "properties": {"id": i, "length_m": round(length_m, 2)},
            "geometry": {"type": "LineString", "coordinates": pts.tolist()},
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
