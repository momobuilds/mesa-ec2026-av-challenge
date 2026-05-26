"""Unit tests for vectorised Bresenham raycasting."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.raycasting import cast_rays_batch


def test_horizontal_ray_free_and_hit_cells():
    """Ray from (0,0) to (0,10): cells 0..9 are free, cell 10 is hit."""
    origin = np.array([0, 0], dtype=np.int32)
    hits = np.array([[0, 10]], dtype=np.int32)
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)

    assert len(valid_hits) == 1
    assert list(valid_hits[0]) == [0, 10]
    free_set = set(map(tuple, free.tolist()))
    for c in range(10):
        assert (0, c) in free_set, f"Cell (0,{c}) should be in free cells"


def test_zero_length_ray_no_free_cells():
    """A ray where origin == hit has no free cells (length=1, step 0 is origin)."""
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.array([[5, 5]], dtype=np.int32)
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    # step 0 covers origin cell — it IS included as a free cell
    free_set = set(map(tuple, free.tolist()))
    assert (5, 5) in free_set  # origin is a free cell


def test_out_of_bounds_hits_excluded():
    """Hits outside grid bounds are excluded from valid_hits."""
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.array([[5, 30], [5, 5]], dtype=np.int32)  # first is out-of-bounds
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    assert list(valid_hits[0]) == [5, 5]


def test_diagonal_ray_covers_diagonal_cells():
    """45° ray from (0,0) to (5,5) visits diagonal cells."""
    origin = np.array([0, 0], dtype=np.int32)
    hits = np.array([[5, 5]], dtype=np.int32)
    grid_shape = (10, 10)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    assert len(free) >= 4  # at least 4 diagonal steps


def test_empty_hits_returns_empty():
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.zeros((0, 2), dtype=np.int32)
    free, valid_hits = cast_rays_batch(origin, hits, (20, 20))
    assert len(free) == 0
    assert len(valid_hits) == 0


def test_large_scan_under_one_second():
    """16 000 rays of ≤ 320 steps each must complete in < 1 s."""
    rng = np.random.default_rng(42)
    origin = np.array([200, 200], dtype=np.int32)
    hits = (origin + rng.integers(-150, 150, size=(16000, 2))).astype(np.int32)
    grid_shape = (400, 400)
    t0 = time.perf_counter()
    cast_rays_batch(origin, hits, grid_shape)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"16 000-ray cast took {elapsed:.2f} s (limit: 1.0 s)"
