"""Vectorised DDA / Bresenham-style ray casting for the occupancy grid.

Algorithm (per ray from origin_cell to hit_cell[i]):
    steps_i = max(|dr_i|, |dc_i|, 1)          Chebyshev length of ray
    for s in 0 .. steps_i − 1:                 free-space cells
        row[s] = round(r0 + dr_i · s / steps_i)
        col[s] = round(c0 + dc_i · s / steps_i)
    hit cell  = hit_cell[i]

All N rays are processed simultaneously via broadcast numpy operations,
producing a (N, L_max) matrix of row/col values that is then masked and
flattened. Rays longer than *max_steps* are truncated.
"""
from __future__ import annotations

import numpy as np


def cast_rays_batch(
    origin_cell: np.ndarray,      # (2,) int  [row, col]
    hit_cells: np.ndarray,        # (N, 2) int  [row, col]
    grid_shape: tuple[int, int],
    max_steps: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast N rays from *origin_cell* to each row of *hit_cells*.

    Returns
    -------
    free_cells  : (M, 2) int32 — cells along rays (free space),
                  includes the origin cell, excludes hit endpoints.
    valid_hits  : (K, 2) int32 — hit_cells that fall within *grid_shape*.
    """
    if len(hit_cells) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )

    N = len(hit_cells)
    dr = (hit_cells[:, 0] - origin_cell[0]).astype(np.float32)  # (N,)
    dc = (hit_cells[:, 1] - origin_cell[1]).astype(np.float32)  # (N,)

    # Chebyshev length (at least 1 to avoid division by zero)
    lengths = np.maximum(
        np.maximum(np.abs(dr), np.abs(dc)).astype(np.int32), 1
    )
    lengths = np.minimum(lengths, max_steps)  # (N,)

    L = int(lengths.max())
    steps = np.arange(L, dtype=np.float32)  # (L,)

    # Parametric positions t[i, s] = s / lengths[i]
    t = steps[None, :] / lengths[:, None].astype(np.float32)  # (N, L)
    valid_mask = steps[None, :] < lengths[:, None]             # (N, L) bool

    rows = np.round(origin_cell[0] + dr[:, None] * t).astype(np.int32)  # (N, L)
    cols = np.round(origin_cell[1] + dc[:, None] * t).astype(np.int32)  # (N, L)

    H, W = grid_shape
    in_grid = (
        (rows >= 0) & (rows < H)
        & (cols >= 0) & (cols < W)
        & valid_mask
    )

    free_cells = np.column_stack([rows[in_grid], cols[in_grid]]).astype(np.int32)

    hit_in = (
        (hit_cells[:, 0] >= 0) & (hit_cells[:, 0] < H)
        & (hit_cells[:, 1] >= 0) & (hit_cells[:, 1] < W)
    )
    valid_hits = hit_cells[hit_in].astype(np.int32)

    return free_cells, valid_hits
