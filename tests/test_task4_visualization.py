"""Tests for Task 4 map visualisation helpers."""
from __future__ import annotations

import numpy as np

import src.map_visualization as map_viz


def test_semantic_animation_palette_maps_free_occupied_unknown_cells() -> None:
    semantic_rgb = getattr(map_viz, "_log_odds_to_semantic_rgb", None)
    assert semantic_rgb is not None, "animation should expose a semantic RGB palette helper"

    log_odds = np.array(
        [
            [-2.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    rgb = semantic_rgb(log_odds)

    assert tuple(rgb[0, 0]) == (49, 185, 107)   # free space: green
    assert tuple(rgb[0, 1]) == (99, 109, 121)   # unknown: slate grey
    assert tuple(rgb[0, 2]) == (8, 11, 16)      # occupied: near black
