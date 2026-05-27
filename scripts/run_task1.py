"""CLI entry point for Engineering Challenge 2026 Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import format_bag_summary, parse_bag_summary
from src.task1_detection import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PLOTS,
    parse_max_frames_arg,
    run_task1_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 1 opponent vehicle detection on ROS 2 bag point clouds."
    )
    parser.add_argument(
        "--bag-path",
        "--bag-dir",
        dest="bag_dir",
        default="data",
        help="Directory containing metadata.yaml and ROS 2 sqlite3 bag files.",
    )
    parser.add_argument(
        "--max-frames",
        type=parse_max_frames_arg,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of point cloud frames to process, or 'all'.",
    )
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=None,
        help="Quick-test frame cap that overrides --max-frames when provided.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=DEFAULT_MAX_PLOTS,
        help="Maximum number of BEV frame plots to write.",
    )
    parser.add_argument(
        "--urdf-path",
        default=None,
        help="Vehicle URDF used to transform sensor points into the ego/base frame.",
    )
    parser.add_argument(
        "--base-frame",
        default=None,
        help="URDF base/ego frame. Defaults to base_link, ego, or center_of_gravity.",
    )
    parser.add_argument(
        "--allow-identity-transform",
        action="store_true",
        help="Debug fallback only: use identity if the URDF sensor transform is missing.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/task1",
        help="Directory for BEV plots, detections.csv, and task1_report.md.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Optional PointCloud2 topic to process. If omitted, front LiDAR is preferred.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print bag metadata summary and exit.",
    )
    parser.add_argument(
        "--no-ground-filter",
        action="store_true",
        help="Disable ground-plane filtering (useful for debugging raw point cloud layout).",
    )
    parser.add_argument("--ego-length", type=float, default=5.0)
    parser.add_argument("--ego-width", type=float, default=2.2)
    parser.add_argument("--ego-height", type=float, default=1.6)
    parser.add_argument("--safety-margin", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = parse_bag_summary(args.bag_dir)
    print(format_bag_summary(summary))

    if args.inspect_only:
        return

    effective_max_frames = (
        args.debug_max_frames if args.debug_max_frames is not None else args.max_frames
    )

    result = run_task1_pipeline(
        args.bag_dir,
        max_frames=effective_max_frames,
        max_plots=args.max_plots,
        output_dir=args.output_dir,
        topic_name=args.topic,
        ground_filter=not args.no_ground_filter,
        urdf_path=args.urdf_path,
        base_frame=args.base_frame,
        allow_identity_transform=args.allow_identity_transform,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
        ego_height=args.ego_height,
        safety_margin=args.safety_margin,
    )

    print("")
    print("Task 1 pipeline complete")
    print(f"- selected topic: {result.selected_topic.name if result.selected_topic else 'none'}")
    print(f"- frames processed: {len(result.frames)}")
    print(f"- detections: {len(result.detections)}")
    print(f"- outputs: {Path(args.output_dir)}")
    for diagnostic in result.diagnostics:
        print(f"- diagnostic: {diagnostic}")


if __name__ == "__main__":
    main()
