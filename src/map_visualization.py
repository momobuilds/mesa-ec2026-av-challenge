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
from matplotlib.patches import Patch, Rectangle
import numpy as np

from src.ego_trajectory import EgoTrajectory
from src.occupancy_grid import ProbabilisticGrid


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_odds_to_uint8(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds array to uint8: 0=occupied, 128=unknown, 255=free."""
    prob = 1.0 / (1.0 + np.exp(-log_odds.astype(np.float32)))
    return ((1.0 - prob) * 255.0).astype(np.uint8)


def _log_odds_to_semantic_rgb(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds to RGB colors for presentation-ready animation."""
    rgb = np.empty((*log_odds.shape, 3), dtype=np.uint8)
    rgb[:, :] = (99, 109, 121)  # unknown: slate grey

    free_thresh = math.log(0.3 / 0.7)
    free_mask = log_odds < free_thresh
    occ_mask = log_odds > 0.0

    rgb[free_mask] = (49, 185, 107)  # free: green
    rgb[occ_mask] = (8, 11, 16)      # occupied: near black
    return rgb


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


def _animation_view_extent(
    grid: ProbabilisticGrid,
    ego_poses: list[tuple[float, float]],
    margin_m: float = 85.0,
) -> list[float]:
    """Square view focused on the driven segment, clamped to the grid extent."""
    full = _grid_extent(grid)
    if not ego_poses:
        return full

    xs = np.array([p[0] for p in ego_poses], dtype=np.float64)
    ys = np.array([p[1] for p in ego_poses], dtype=np.float64)
    cx = float((xs.min() + xs.max()) / 2.0)
    cy = float((ys.min() + ys.max()) / 2.0)
    half = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
    half = max(half + margin_m, 70.0)

    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    dx = max(0.0, full[0] - x0) - max(0.0, x1 - full[1])
    dy = max(0.0, full[2] - y0) - max(0.0, y1 - full[3])
    x0 += dx
    x1 += dx
    y0 += dy
    y1 += dy
    return [max(full[0], x0), min(full[1], x1), max(full[2], y0), min(full[3], y1)]


def _canvas_rgb(fig: plt.Figure) -> np.ndarray:
    """Read a Matplotlib canvas into an RGB array across Matplotlib versions."""
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return buf.reshape(h, w, 4)[:, :, :3]


# ── public functions ──────────────────────────────────────────────────────────

def render_freespace_map(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    output_path: str,
) -> None:
    """Save a color-coded freespace map for visualisation clarity.

    Color scheme (evaluation-friendly):
      - Black  (RGB 0,0,0)      : Occupied  — barriers / walls
      - Green  (RGB 64,180,64)  : Free space — confirmed navigable area
      - Grey   (RGB 160,160,160): Unknown   — not yet observed
    Track boundary polylines are drawn in bright yellow for contrast.
    """
    lo = grid._log_odds  # (H, W) float32
    H, W = lo.shape

    # Build RGB image
    rgb = np.full((H, W, 3), 160, dtype=np.uint8)   # default: unknown grey

    free_thresh = math.log(0.3 / 0.7)    # l < this → free (P<0.3)
    occ_thresh  = math.log(0.5 / 0.5)   # l > this → occupied (P>0.5)

    free_mask = lo < free_thresh
    occ_mask  = lo > occ_thresh

    # Free: green
    rgb[free_mask, 0] = 64
    rgb[free_mask, 1] = 180
    rgb[free_mask, 2] = 64
    # Occupied: black
    rgb[occ_mask,  :] = 0

    # Flip so y increases upward
    img = np.flipud(rgb)

    extent = _grid_extent(grid)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, extent=extent, origin="lower", interpolation="nearest")

    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], color="#FFD700", lw=1.5, alpha=0.9,
                label="track boundary" if i == 0 else "")
    if boundaries:
        ax.legend(fontsize=9)

    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Free-space map  (green=free, black=occupied, grey=unknown)")
    ax.set_aspect("equal")

    # Scale bar (50 m)
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.04 * (extent[3] - extent[2])
    ax.plot([x0, x0 + 50], [y0, y0], "w-", lw=3)
    ax.text(x0 + 25, y0 + 2, "50 m", color="white", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_world_map_static(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    output_path: str,
) -> None:
    """Save final occupancy grid with boundary polylines overlaid (red)."""
    img = np.flipud(grid.render())  # flip so y increases upward
    extent = _grid_extent(grid)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=extent, origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.5, alpha=0.9,
                label=f"boundary {i}" if i < 5 else "_nolegend_")
    if boundaries:
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Accumulated occupancy map  (0=occupied, 255=free)")
    ax.set_aspect("equal")
    # Scale bar
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.04 * (extent[3] - extent[2])
    ax.plot([x0, x0 + 50], [y0, y0], "r-", lw=3)
    ax.text(x0 + 25, y0 + 2, "50 m", color="red", fontsize=8, ha="center")
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
    extent = _grid_extent(grid)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=extent, origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.5, alpha=0.9,
                label=f"boundary {i}" if i < 5 else "_nolegend_")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "b-", lw=0.8, alpha=0.7,
            label="ego trajectory")
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "go", ms=8, label="start")
    ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "rs", ms=8, label="end")
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Occupancy map with ego trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    # Scale bar
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.04 * (extent[3] - extent[2])
    ax.plot([x0, x0 + 50], [y0, y0], "r-", lw=3)
    ax.text(x0 + 25, y0 + 2, "50 m", color="red", fontsize=8, ha="center")
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
    view_extent = _animation_view_extent(grid, ego_poses)
    n_frames = min(len(grid_snapshots), len(ego_poses))

    legend_handles = [
        Patch(facecolor="#31B96B", edgecolor="none", label="free"),
        Patch(facecolor="#080B10", edgecolor="#7C8590", label="occupied"),
        Patch(facecolor="#636D79", edgecolor="none", label="unknown"),
        Patch(facecolor="#38BDF8", edgecolor="none", label="ego path"),
    ]

    for i, (log_odds, (x_ego, y_ego)) in enumerate(zip(grid_snapshots, ego_poses)):
        img_arr = np.flipud(_log_odds_to_semantic_rgb(log_odds))

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0B1020")
        ax.set_facecolor("#0B1020")
        ax.imshow(img_arr, extent=extent, origin="lower", interpolation="nearest")

        trail_x = [p[0] for p in ego_poses[: i + 1]]
        trail_y = [p[1] for p in ego_poses[: i + 1]]
        if trail_x:
            ax.plot(trail_x, trail_y, color="#38BDF8", lw=2.0, alpha=0.9)
            ax.plot(trail_x[0], trail_y[0], marker="o", ms=6,
                    color="#FACC15", mec="#111827", mew=0.8)
        ax.plot(x_ego, y_ego, marker="o", ms=9, color="#F8FAFC",
                mec="#22C55E", mew=2.0)

        ax.set_xlim(view_extent[0], view_extent[1])
        ax.set_ylim(view_extent[2], view_extent[3])
        ax.set_aspect("equal")
        ax.axis("off")

        title = f"Task 4 - Occupancy Map  |  Frame {i + 1}/{n_frames}"
        ax.text(0.02, 0.97, title, transform=ax.transAxes, ha="left", va="top",
                color="#F8FAFC", fontsize=11, fontweight="bold")
        ax.text(0.02, 0.925, "LiDAR log-odds map filling as ego drives",
                transform=ax.transAxes, ha="left", va="top",
                color="#CBD5E1", fontsize=8)

        progress = (i + 1) / max(n_frames, 1)
        ax.add_patch(Rectangle((0.02, 0.035), 0.38, 0.014,
                               transform=ax.transAxes, color="#1E293B",
                               zorder=5, clip_on=False))
        ax.add_patch(Rectangle((0.02, 0.035), 0.38 * progress, 0.014,
                               transform=ax.transAxes, color="#38BDF8",
                               zorder=6, clip_on=False))

        x0 = view_extent[0] + 0.06 * (view_extent[1] - view_extent[0])
        y0 = view_extent[2] + 0.08 * (view_extent[3] - view_extent[2])
        scale_len = 50.0 if (view_extent[1] - view_extent[0]) > 180 else 25.0
        ax.plot([x0, x0 + scale_len], [y0, y0], color="#F8FAFC", lw=3)
        ax.text(x0 + scale_len / 2.0, y0 + 3.0, f"{int(scale_len)} m",
                color="#F8FAFC", fontsize=8, ha="center")

        leg = ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
                        frameon=True, facecolor="#0F172A", edgecolor="#334155",
                        labelcolor="#E2E8F0", ncol=2)
        leg.get_frame().set_alpha(0.88)

        plt.tight_layout(pad=0.15)
        rgb = _canvas_rgb(fig)
        frames_pil.append(Image.fromarray(rgb))
        plt.close(fig)

    if frames_pil:
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            loop=0,
            duration=180,   # ms per frame
            optimize=True,
        )
