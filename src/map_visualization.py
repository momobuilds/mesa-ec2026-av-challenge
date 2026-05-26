"""Visualisation for Task 4 environment mapping outputs.

Deliverables
------------
- world_map_static.png           : final occupancy grid + boundary polylines
- world_map_with_trajectory.png  : same + ego world-frame trajectory
- ego_view_NNN.png               : ego-centred 100×60 m rotated crop, with tracks
- map_animation.gif              : map filling in over time (uses Pillow)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ego_trajectory import EgoTrajectory
from src.occupancy_grid import ProbabilisticGrid


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_odds_to_uint8(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds array to uint8: 0=occupied, 128=unknown, 255=free."""
    prob = 1.0 / (1.0 + np.exp(-log_odds.astype(np.float32)))
    return ((1.0 - prob) * 255.0).astype(np.uint8)


def _grid_extent(grid: ProbabilisticGrid) -> list[float]:
    """[x_min, x_max, y_min, y_max] for matplotlib imshow extent."""
    return [
        grid.origin_x,
        grid.origin_x + grid._shape[1] * grid.resolution,
        grid.origin_y,
        grid.origin_y + grid._shape[0] * grid.resolution,
    ]


def _draw_oriented_box(
    ax: plt.Axes, cx: float, cy: float, yaw: float,
    length: float, width: float, track_id: int,
) -> None:
    """Draw a rotated bounding box on *ax* (ego frame coordinates)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
    R = np.array([[c, -s], [s, c]])
    pts = (R @ corners.T).T + np.array([cx, cy])
    ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.5)
    ax.text(cx, cy, str(track_id), fontsize=6, color="red", ha="center", va="center")


# ── public functions ──────────────────────────────────────────────────────────

def render_world_map_static(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    output_path: str,
) -> None:
    """Save final occupancy grid with boundary polylines overlaid (red)."""
    img = np.flipud(grid.render())  # flip so y increases upward
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=_grid_extent(grid), origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.0, alpha=0.8,
                label="boundary" if i == 0 else "")
    if boundaries:
        ax.legend(fontsize=8)
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Accumulated occupancy map (static environment)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_world_map_with_trajectory(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    traj_xy: np.ndarray,       # (N, 2) world positions
    output_path: str,
) -> None:
    """Save occupancy grid + ego trajectory + boundary polylines."""
    img = np.flipud(grid.render())
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=_grid_extent(grid), origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.0, alpha=0.8,
                label="boundary" if i == 0 else "")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "b-", lw=0.8, alpha=0.6,
            label="ego trajectory")
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "go", ms=6, label="start")
    ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "rs", ms=6, label="end")
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Occupancy map with ego trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_ego_view(
    grid: ProbabilisticGrid,
    ego_traj: EgoTrajectory,
    timestamp_ns: int,
    track_rows: list[dict],    # rows from tracks.csv at this timestamp (ego frame)
    output_path: str,
    window_x_m: float = 100.0,
    window_y_m: float = 60.0,
) -> None:
    """Render a heading-aligned, ego-centred crop of the world map.

    The window is rotated so ego-forward points upward in the image.
    Track boxes (from tracks.csv, ego frame) are drawn directly on the crop.
    """
    x_ego, y_ego, heading = ego_traj.pose_at(timestamp_ns)
    res = grid.resolution
    nx = int(window_x_m / res)
    ny = int(window_y_m / res)

    # Sample world-frame occupancy at points covering the ego window
    xs = (np.arange(nx) - nx // 2) * res   # ego-x axis
    ys = (np.arange(ny) - ny // 2) * res   # ego-y axis
    xs_g, ys_g = np.meshgrid(xs, ys)       # (ny, nx)

    c, s = math.cos(heading), math.sin(heading)
    xw = x_ego + c * xs_g - s * ys_g
    yw = y_ego + s * xs_g + c * ys_g

    col_f = (xw - grid.origin_x) / res
    row_f = (yw - grid.origin_y) / res
    col_i = np.clip(col_f.astype(np.int32), 0, grid._shape[1] - 1)
    row_i = np.clip(row_f.astype(np.int32), 0, grid._shape[0] - 1)
    in_bounds = (
        (col_f >= 0) & (col_f < grid._shape[1])
        & (row_f >= 0) & (row_f < grid._shape[0])
    )

    img_full = grid.render()
    crop = np.full((ny, nx), 128, dtype=np.uint8)
    crop[in_bounds] = img_full[row_i[in_bounds], col_i[in_bounds]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        np.flipud(crop), cmap="gray", vmin=0, vmax=255,
        extent=[-window_x_m / 2, window_x_m / 2,
                -window_y_m / 2, window_y_m / 2],
        origin="lower",
    )
    for row_d in track_rows:
        _draw_oriented_box(
            ax,
            float(row_d["center_x"]), float(row_d["center_y"]),
            float(row_d["yaw"]), float(row_d["length"]), float(row_d["width"]),
            int(row_d["track_id"]),
        )
    ax.plot(0, 0, "g^", ms=10, label="ego")
    ax.set_xlabel("ego-x (m, forward)")
    ax.set_ylabel("ego-y (m, left)")
    ax.set_title(f"Ego-frame view  t={timestamp_ns}")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def render_animation(
    grid_snapshots: list[np.ndarray],        # list of log_odds float32 copies
    ego_poses: list[tuple[float, float]],    # (x_w, y_w) per snapshot
    grid: ProbabilisticGrid,
    output_path: str,
) -> None:
    """Save animated GIF showing map filling in.

    Uses Pillow (PIL) directly — no imageio dependency required.
    """
    from PIL import Image

    frames_pil: list[Image.Image] = []
    extent = _grid_extent(grid)

    for i, (log_odds, (x_ego, y_ego)) in enumerate(zip(grid_snapshots, ego_poses)):
        img_arr = np.flipud(_log_odds_to_uint8(log_odds))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_arr, cmap="gray", vmin=0, vmax=255,
                  extent=extent, origin="lower")
        # Fading trail of last 20 poses
        trail_x = [p[0] for p in ego_poses[max(0, i - 20): i + 1]]
        trail_y = [p[1] for p in ego_poses[max(0, i - 20): i + 1]]
        if trail_x:
            ax.plot(trail_x, trail_y, "b-", lw=0.8, alpha=0.5)
        ax.plot(x_ego, y_ego, "go", ms=7)
        ax.set_title(f"Frame {i + 1}/{len(grid_snapshots)}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0.5)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgb = buf.reshape(h, w, 3)
        frames_pil.append(Image.fromarray(rgb))
        plt.close(fig)

    if frames_pil:
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            loop=0,
            duration=250,   # ms per frame
            optimize=True,
        )
