"""CLI entry point for Engineering Challenge 2026 Task 3 (continuous tracking)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import format_bag_summary, parse_bag_summary
from src.tf_utils import EgoStateLog
from src.track_association import run_tracking_pipeline, _MASTER_TOPIC
from src.track_metrics import (
    build_tracks_df, build_summary_df, detect_id_switches,
    print_metrics_summary, plot_tracking_bev, plot_tracking_timeline,
)
from src.track_state import TrackStatus, reset_track_id_counter
from src.sensor_detectors import detect_all_sensors
from src.task1_detection import DetectionConfig
from src.task2_fusion import ASSOC_WINDOW_NS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 3 continuous multi-object tracking on ROS 2 bag data."
    )
    parser.add_argument("--bag-dir", default="data",
                        help="Directory containing metadata.yaml and .db3 files.")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Maximum LiDAR-front frames to process.")
    parser.add_argument("--output-dir", default="outputs/task3",
                        help="Directory for tracks.csv, BEV plots, and report.")
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="Subset of PointCloud2 topics. Defaults to all 7.")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Consecutive hits before tentative→confirmed.")
    parser.add_argument("--max-age", type=int, default=5,
                        help="Missed frames before a confirmed track is deleted.")
    parser.add_argument("--bev-frames", type=int, default=5,
                        help="Number of BEV tracking plots to save.")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Print bag metadata and exit.")
    parser.add_argument("--no-ground-filter", action="store_true",
                        help="Disable ground-plane filtering.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame tracking statistics.")
    parser.add_argument(
        "--simulate-occlusion", type=int, default=None,
        metavar="N",
        help=(
            "Artificially drop all detections for N frames starting at "
            "frame --occlusion-start (default 20) to test coasting recovery."
        ),
    )
    parser.add_argument("--occlusion-start", type=int, default=20,
                        help="First frame index to blank for --simulate-occlusion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reset_track_id_counter(0)

    summary = parse_bag_summary(args.bag_dir)
    print(format_bag_summary(summary))

    if args.inspect_only:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ego state log ──────────────────────────────────────────────────────────
    ego_log: EgoStateLog | None = None
    for candidate in [
        Path(args.bag_dir) / "ego.mat",
        *Path(args.bag_dir).parent.glob("**/ego.mat"),
    ]:
        if candidate.exists():
            try:
                ego_log = EgoStateLog(
                    str(candidate),
                    bag_start_ns=summary.starting_time_ns or 0,
                )
                print(f"[Task 3] Ego state log: {candidate.as_posix()}")
                print(f"         {ego_log}")
            except Exception as exc:
                print(f"[Task 3] ego.mat load failed: {exc}")
            break

    # ── Occlusion simulation setup ─────────────────────────────────────────────
    simulate_occlusion = None
    if args.simulate_occlusion is not None:
        occ_start = args.occlusion_start
        occ_end = occ_start + args.simulate_occlusion - 1
        simulate_occlusion = (0, occ_start, occ_end)
        print(f"[Task 3] Simulating occlusion: frames {occ_start}–{occ_end} "
              f"({args.simulate_occlusion} frames blanked)")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    rows, all_tracks = run_tracking_pipeline(
        summary,
        max_frames=args.max_frames,
        topic_names=args.sensors,
        ego_log=ego_log,
        n_init=args.n_init,
        max_age=args.max_age,
        no_ground_filter=args.no_ground_filter,
        simulate_occlusion=simulate_occlusion,
        verbose=args.verbose,
    )

    # ── Save tracks.csv ───────────────────────────────────────────────────────
    df_tracks = build_tracks_df(rows)
    tracks_csv = output_dir / "tracks.csv"
    df_tracks.to_csv(tracks_csv, index=False)
    print(f"[Task 3] tracks.csv → {tracks_csv.as_posix()} ({len(df_tracks)} rows)")

    # ── Save tracks_summary.csv ───────────────────────────────────────────────
    df_summary = build_summary_df(all_tracks)
    summary_csv = output_dir / "tracks_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[Task 3] tracks_summary.csv → {summary_csv.as_posix()} "
          f"({len(df_summary)} tracks)")

    # ── Print metrics ─────────────────────────────────────────────────────────
    print_metrics_summary(rows, all_tracks)

    # ── BEV plots ─────────────────────────────────────────────────────────────
    if args.bev_frames > 0 and rows:
        print(f"[Task 3] Saving {args.bev_frames} BEV tracking plots …")

        # Re-run detection to get bev_points for each sensor frame
        # (lightweight: just reuse detect_all_sensors with same config)
        config = DetectionConfig(
            ground_band_half_height=0.0 if args.no_ground_filter else 0.30
        )
        _default_sensors = [
            "/sensor/lidar_front/points", "/sensor/lidar_left/points",
            "/sensor/lidar_right/points", "/sensor/radar_front/points",
            "/sensor/radar_back/points", "/sensor/radar_left/points",
            "/sensor/radar_right/points",
        ]
        all_frames = detect_all_sensors(
            summary, config,
            max_frames=args.max_frames,
            topic_names=args.sensors or _default_sensors,
            verbose=False,
        )
        master_frames = all_frames.get(_MASTER_TOPIC, [])
        master_ts_list = sorted(fr.timestamp_ns for fr in master_frames)

        # Group rows by timestamp
        rows_by_ts: dict[int, list[dict]] = {}
        for r in rows:
            rows_by_ts.setdefault(r["timestamp_ns"], []).append(r)

        n_plots = min(args.bev_frames, len(master_ts_list))
        # Pick evenly-spaced frames
        indices = np.linspace(0, len(master_ts_list) - 1, n_plots, dtype=int).tolist()

        for plot_i, fi in enumerate(indices):
            ts = master_ts_list[fi]
            frame_rows = rows_by_ts.get(ts, [])

            sensor_frs: dict = {}
            for topic, frames in all_frames.items():
                matched = next((f for f in frames if f.timestamp_ns == ts), None)
                if matched is None and frames:
                    matched = frames[min(fi, len(frames) - 1)]
                sensor_frs[topic] = matched

            plot_path = output_dir / f"tracking_bev_{plot_i:03d}.png"
            plot_tracking_bev(
                frame_rows, all_tracks, sensor_frs, str(plot_path),
                timestamp_ns=ts, frame_idx=fi,
            )
            print(f"  {plot_path.as_posix()}")

        # Timeline plot
        timeline_path = output_dir / "tracking_timeline.png"
        plot_tracking_timeline(all_tracks, master_ts_list, str(timeline_path))
        print(f"[Task 3] tracking_timeline.png → {timeline_path.as_posix()}")

    # ── Occlusion recovery verification ──────────────────────────────────────
    if simulate_occlusion is not None and not df_summary.empty:
        long_tracks = df_summary[
            df_summary["n_frames_observed"] > args.simulate_occlusion
        ]
        if not long_tracks.empty:
            best = long_tracks.iloc[0]
            print(
                f"\n[Task 3] Occlusion recovery: track T{int(best['track_id'])} "
                f"survived {args.simulate_occlusion}-frame gap "
                f"(n_frames_observed={int(best['n_frames_observed'])}, "
                f"n_frames_coasted={int(best['n_frames_coasted'])})"
            )
        else:
            print("[Task 3] Occlusion: no track survived the gap with the same ID.")

    # ── Final summary ─────────────────────────────────────────────────────────
    n_long = (df_summary["n_frames_observed"] > 30).sum() if not df_summary.empty else 0
    print("\n[Task 3] Done.")
    print(f"  Total tracks             : {len(all_tracks)}")
    print(f"  Confirmed tracks         : {sum(1 for t in all_tracks if t.hits >= 3)}")
    print(f"  Tracks with >30 obs frames: {n_long}")
    print(f"  Outputs                  : {output_dir.as_posix()}/")


if __name__ == "__main__":
    main()
