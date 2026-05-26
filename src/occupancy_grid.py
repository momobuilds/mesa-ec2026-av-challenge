"""Probabilistic occupancy grid using the log-odds inverse sensor model.

Log-odds formulation:
    l(x) = log( P(occ|z) / P(free|z) )

    Hit  (LiDAR return at cell x): l(x) += L_HIT  ≈ +0.85  (→ ~70 % occupied)
    Miss (ray passes through x):   l(x) += L_MISS ≈ -0.40  (→ ~40 % occupied)
    Clipped to [L_MIN, L_MAX] = [-5, 5] so no cell is permanently locked.

Recovering probability: p = sigmoid(l) = 1 / (1 + exp(-l))

Grid axes: columns = world-x (east), rows = world-y (north).
Origin (0,0) of the grid array is the SW corner at (origin_x, origin_y).
"""
from __future__ import annotations

import math

import numpy as np

from src.raycasting import cast_rays_batch

L_HIT: float = 0.85     # log-odds increment per LiDAR return
L_MISS: float = -0.40   # log-odds decrement per free-space traversal
L_MAX: float = 5.0
L_MIN: float = -5.0


class ProbabilisticGrid:
    """Probabilistic 2D occupancy grid in the world frame."""

    def __init__(
        self,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        extent_m: float = 300.0,
        resolution: float = 0.25,
    ) -> None:
        self.resolution = float(resolution)
        # SW corner coordinates
        self.origin_x = float(origin_xy[0]) - extent_m
        self.origin_y = float(origin_xy[1]) - extent_m
        n = int(round(2.0 * extent_m / resolution))
        self._log_odds = np.zeros((n, n), dtype=np.float32)
        self._shape: tuple[int, int] = (n, n)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) world XY → (N, 2) int32 [row, col]."""
        col = ((xy[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        row = ((xy[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        return np.column_stack([row, col])

    def _origin_cell(self, xy: np.ndarray) -> np.ndarray:
        """(2,) world XY → (2,) int32 [row, col]."""
        col = int((float(xy[0]) - self.origin_x) / self.resolution)
        row = int((float(xy[1]) - self.origin_y) / self.resolution)
        return np.array([row, col], dtype=np.int32)

    # ── update ────────────────────────────────────────────────────────────────

    def update_from_scan(
        self,
        sensor_origin_world: np.ndarray,   # (2,) world XY
        hits_world: np.ndarray,             # (N, 2) world XY
        max_range_m: float = 80.0,
    ) -> None:
        """Update log-odds from one LiDAR scan.

        Free cells (ray traversal) receive L_MISS; hit endpoints receive L_HIT.
        """
        if len(hits_world) == 0:
            return

        origin_cell = self._origin_cell(sensor_origin_world)
        hit_cells = self._world_to_cell(hits_world)
        max_steps = int(max_range_m / self.resolution)

        free_cells, valid_hits = cast_rays_batch(
            origin_cell, hit_cells, self._shape, max_steps=max_steps
        )

        if len(free_cells):
            np.add.at(self._log_odds, (free_cells[:, 0], free_cells[:, 1]), L_MISS)
        if len(valid_hits):
            np.add.at(self._log_odds, (valid_hits[:, 0], valid_hits[:, 1]), L_HIT)

        np.clip(self._log_odds, L_MIN, L_MAX, out=self._log_odds)

    # ── dynamic object masking ────────────────────────────────────────────────

    def mask_dynamic_objects(
        self,
        boxes_world: list[tuple[float, float, float, float, float]],
    ) -> None:
        """Reset log-odds to 0 inside each (cx, cy, yaw, length, width) box."""
        for cx, cy, yaw, length, width in boxes_world:
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            hl, hw = float(length) / 2.0, float(width) / 2.0
            corners = np.array(
                [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64
            )
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            corners_w = (R @ corners.T).T + np.array([cx, cy])
            cells = self._world_to_cell(corners_w)
            r0 = max(0, int(cells[:, 0].min()))
            r1 = min(self._shape[0] - 1, int(cells[:, 0].max()))
            c0 = max(0, int(cells[:, 1].min()))
            c1 = min(self._shape[1] - 1, int(cells[:, 1].max()))
            if r0 <= r1 and c0 <= c1:
                self._log_odds[r0 : r1 + 1, c0 : c1 + 1] = 0.0

    # ── query ─────────────────────────────────────────────────────────────────

    def occupied_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Bool array where P(occ) > threshold (l > log(p/(1-p)))."""
        l_thresh = math.log(threshold / (1.0 - threshold))
        return self._log_odds > l_thresh

    def free_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Bool array where P(occ) < threshold."""
        l_thresh = math.log(threshold / (1.0 - threshold))
        return self._log_odds < l_thresh

    def render(self) -> np.ndarray:
        """uint8 image: 0=occupied(black), 128=unknown(grey), 255=free(white)."""
        prob = 1.0 / (1.0 + np.exp(-self._log_odds.astype(np.float32)))
        return ((1.0 - prob) * 255.0).astype(np.uint8)

    def log_odds_copy(self) -> np.ndarray:
        """Snapshot of current log-odds array for animation frames."""
        return self._log_odds.copy()
