"""CLI entry point for EC2026 Task 4 (Environment Mapping)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import parse_bag_summary
from src.ego_trajectory import EgoTrajectory
from src.map_visualization import (
    render_animation,
    render_ego_view,
    render_world_map_static,
    render_world_map_with_trajectory,
)
from src.occupancy_grid import ProbabilisticGrid
from src.sensor_detectors import iter_ego_frame_points
from src.tf_utils import EgoStateLog
from src.track_boundary import extract_boundaries, save_geojson

# Height filter (ego frame): retains barriers and vehicles; drops ground/sky
_Z_MIN = 0.10   # m above ego origin
_Z_MAX = 2.50   # m above ego origin

_LIDAR_ALIAS = {
    "front": "/sensor/lidar_front/points",
    "left":  "/sensor/lidar_left/points",
    "right": "/sensor/lidar_right/points",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 4: Probabilistic environment map from LiDAR data."
    )
    p.add_argument("--bag-dir", default="data")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--output-dir", default="outputs/task4")
    p.add_argument("--grid-resolution", type=float, default=0.25,
                   help="Cell size in metres (default 0.25)")
    p.add_argument("--grid-extent", type=float, default=300.0,
                   help="Half-extent of grid in metres (±N m both axes, default 300)")
    p.add_argument("--use-lidars", default="front,left,right",
                   help="Comma-separated subset of: front,left,right")
    p.add_argument("--mask-tracks", default="outputs/task3/tracks.csv",
                   help="tracks.csv for dynamic-object masking (Task 3 output)")
    p.add_argument("--no-mask-tracks", action="store_true",
                   help="Skip dynamic-object masking")
    p.add_argument("--animation-frames", type=int, default=30,
                   help="Number of GIF frames (default 30)")
    p.add_argument("--no-animation", action="store_true",
                   help="Skip GIF generation (faster)")
    p.add_argument("--inspect-only", action="store_true",
                   help="Print grid size and exit")
    p.add_argument("--max-range", type=float, default=80.0,
                   help="Max LiDAR range for ray casting (m, default 80)")
    return p.parse_args()


def _load_ego(bag_dir: str, bag_start_ns: int) -> EgoStateLog | None:
    for candidate in [
        Path(bag_dir) / "ego.mat",
        *Path(bag_dir).parent.glob("**/ego.mat"),
    ]:
        if candidate.exists():
            try:
                log = EgoStateLog(str(candidate), bag_start_ns=bag_start_ns)
                print(f"[Task 4] Ego log: {candidate}  {log}")
                return log
            except Exception as exc:
                print(f"[Task 4] ego.mat load failed: {exc}")
    return None


def main() -> None:
    args = parse_args()
    summary = parse_bag_summary(args.bag_dir)
    bag_start_ns = int(summary.starting_time_ns or 0)

    if args.inspect_only:
        n = int(2 * args.grid_extent / args.grid_resolution)
        print(f"[Task 4] Grid: ±{args.grid_extent} m, "
              f"res={args.grid_resolution} m → {n}×{n} cells "
              f"({n*n/1e6:.1f}M, {n*n*4/1e6:.0f} MB float32)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ego trajectory ────────────────────────────────────────────────────────
    ego_log = _load_ego(args.bag_dir, bag_start_ns)
    if ego_log is None:
        print("[Task 4] ERROR: ego.mat not found")
        sys.exit(1)
    ego_traj = EgoTrajectory.from_ego_log(ego_log)
    x_min, x_max, y_min, y_max = ego_traj.world_extent()
    print(f"[Task 4] Full-bag extent: x=[{x_min:.0f},{x_max:.0f}] "
          f"y=[{y_min:.0f},{y_max:.0f}] m")

    # ── Occupancy grid ────────────────────────────────────────────────────────
    grid = ProbabilisticGrid(
        origin_xy=(0.0, 0.0),
        extent_m=args.grid_extent,
        resolution=args.grid_resolution,
    )
    n = grid._shape[0]
    print(f"[Task 4] Grid: {n}×{n} cells ({n*n/1e6:.1f}M), "
          f"origin=({grid.origin_x:.0f},{grid.origin_y:.0f})")

    # ── Track masking ─────────────────────────────────────────────────────────
    tracks_df: pd.DataFrame | None = None
    if not args.no_mask_tracks:
        tp = Path(args.mask_tracks)
        if tp.exists():
            tracks_df = pd.read_csv(tp)
            print(f"[Task 4] Loaded {len(tracks_df)} track rows for masking")
        else:
            print(f"[Task 4] tracks.csv not found at {tp} — masking skipped")

    # ── LiDAR topics ─────────────────────────────────────────────────────────
    lidar_topics = [
        _LIDAR_ALIAS[k.strip()]
        for k in args.use_lidars.split(",")
        if k.strip() in _LIDAR_ALIAS
    ]
    print(f"[Task 4] LiDAR topics: {lidar_topics}")

    # ── Collect scans per timestamp ───────────────────────────────────────────
    scans: dict[int, list[np.ndarray]] = {}
    for topic in lidar_topics:
        n_frames = 0
        for ts_ns, ego_xyz in iter_ego_frame_points(
            summary, topic, max_frames=args.max_frames
        ):
            scans.setdefault(ts_ns, []).append(ego_xyz)
            n_frames += 1
        print(f"[Task 4]   {topic}: {n_frames} frames")

    sorted_ts = sorted(scans.keys())
    print(f"[Task 4] Total unique timestamps: {len(sorted_ts)}")

    # ── Animation / ego-view bookkeeping ─────────────────────────────────────
    anim_stride = max(1, len(sorted_ts) // max(args.animation_frames, 1))
    anim_snapshots: list[np.ndarray] = []
    anim_ego_poses: list[tuple[float, float]] = []
    master_timestamps: list[int] = []

    n_ego_views = min(8, max(5, len(sorted_ts) // 8))
    ego_view_indices = (
        np.linspace(0, len(sorted_ts) - 1, n_ego_views, dtype=int).tolist()
        if sorted_ts else []
    )

    # ── Group track rows by timestamp ─────────────────────────────────────────
    rows_by_ts: dict[int, list[dict]] = {}
    if tracks_df is not None:
        for _, row in tracks_df.iterrows():
            rows_by_ts.setdefault(int(row["timestamp_ns"]), []).append(row.to_dict())

    # ── Main loop ─────────────────────────────────────────────────────────────
    for frame_i, ts_ns in enumerate(sorted_ts):
        x_ego, y_ego, heading = ego_traj.pose_at(ts_ns)
        sensor_origin_world = np.array([x_ego, y_ego])

        # Transform all scans at this timestamp to world frame and update grid
        all_pts: list[np.ndarray] = []
        for ego_xyz in scans[ts_ns]:
            z_ok = (ego_xyz[:, 2] >= _Z_MIN) & (ego_xyz[:, 2] <= _Z_MAX)
            pts_2d = ego_xyz[z_ok, :2]
            if len(pts_2d):
                all_pts.append(ego_traj.ego_to_world(pts_2d, ts_ns))

        if all_pts:
            grid.update_from_scan(
                sensor_origin_world, np.vstack(all_pts), args.max_range
            )

        # Dynamic object masking
        frame_tracks = rows_by_ts.get(ts_ns, [])
        if frame_tracks:
            boxes: list[tuple] = []
            for rd in frame_tracks:
                pts_w = ego_traj.ego_to_world(
                    np.array([[float(rd["center_x"]), float(rd["center_y"])]]),
                    ts_ns,
                )
                boxes.append((
                    float(pts_w[0, 0]), float(pts_w[0, 1]),
                    float(rd["yaw"]) + heading,
                    float(rd["length"]), float(rd["width"]),
                ))
            grid.mask_dynamic_objects(boxes)

        # Animation snapshot
        if not args.no_animation and frame_i % anim_stride == 0:
            anim_snapshots.append(grid.log_odds_copy())
            anim_ego_poses.append((x_ego, y_ego))

        master_timestamps.append(ts_ns)

        if frame_i % 10 == 0 or frame_i == len(sorted_ts) - 1:
            occ = grid.occupied_mask().sum()
            print(f"  [{frame_i+1}/{len(sorted_ts)}] occ_cells={occ:,}")

    # ── Free-space estimate ───────────────────────────────────────────────────
    free_cells = int(grid.free_mask().sum())
    occ_cells = int(grid.occupied_mask().sum())
    res2 = args.grid_resolution ** 2
    print(f"\n[Task 4] Free space : {free_cells:,} cells = {free_cells * res2:.0f} m²")
    print(f"[Task 4] Occupied   : {occ_cells:,} cells = {occ_cells * res2:.0f} m²")

    # ── Track boundary extraction ─────────────────────────────────────────────
    print("[Task 4] Extracting track boundaries …")
    min_perim = int(50.0 / args.grid_resolution)
    boundaries = extract_boundaries(
        grid.occupied_mask(),
        origin_x=grid.origin_x, origin_y=grid.origin_y,
        resolution=args.grid_resolution,
        min_perimeter_cells=min_perim,
    )
    total_len = sum(
        float(np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1)))
        for b in boundaries if len(b) > 1
    )
    print(f"[Task 4] Boundaries : {len(boundaries)}, total length {total_len:.0f} m")
    save_geojson(boundaries, str(output_dir / "track_boundaries.geojson"))

    # ── Also generate no-mask map for report comparison ───────────────────────
    # (only when masking was active — rebuild without masking in same run)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("[Task 4] Rendering world_map_static.png …")
    render_world_map_static(grid, boundaries, str(output_dir / "world_map_static.png"))

    stride = max(1, len(ego_traj._x_w) // 5000)
    traj_xy = ego_traj.trajectory_xy()[::stride]
    print("[Task 4] Rendering world_map_with_trajectory.png …")
    render_world_map_with_trajectory(
        grid, boundaries, traj_xy,
        str(output_dir / "world_map_with_trajectory.png"),
    )

    print(f"[Task 4] Rendering {len(ego_view_indices)} ego views …")
    for view_i, fi in enumerate(ego_view_indices):
        ts_ns = master_timestamps[fi]
        out_path = output_dir / f"ego_view_{view_i:03d}.png"
        render_ego_view(
            grid, ego_traj, ts_ns, rows_by_ts.get(ts_ns, []), str(out_path)
        )
        print(f"  ego_view_{view_i:03d}.png")

    if not args.no_animation and anim_snapshots:
        anim_path = output_dir / "map_animation.gif"
        print(f"[Task 4] Rendering animation ({len(anim_snapshots)} frames) …")
        render_animation(anim_snapshots, anim_ego_poses, grid, str(anim_path))
        size_mb = anim_path.stat().st_size / 1e6
        print(f"[Task 4] map_animation.gif: {size_mb:.1f} MB")

    print("\n[Task 4] Done.")
    print(f"  Timestamps processed : {len(sorted_ts)}")
    print(f"  Free space           : {free_cells * res2:.0f} m²")
    print(f"  Boundary polylines   : {len(boundaries)}")
    print(f"  Total boundary length: {total_len:.0f} m")
    print(f"  Outputs              : {output_dir}/")


if __name__ == "__main__":
    main()
