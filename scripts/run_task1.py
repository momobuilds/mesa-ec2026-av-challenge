"""CLI entry point for Engineering Challenge 2026 Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import format_bag_summary, parse_bag_summary
from src.task1_detection import run_task1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 1 opponent vehicle detection on ROS 2 bag point clouds."
    )
    parser.add_argument(
        "--bag-dir",
        default="data",
        help="Directory containing metadata.yaml and ROS 2 sqlite3 bag files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of point cloud frames to process.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = parse_bag_summary(args.bag_dir)
    print(format_bag_summary(summary))

    if args.inspect_only:
        return

    result = run_task1_pipeline(
        args.bag_dir,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        topic_name=args.topic,
        ground_filter=not args.no_ground_filter,
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
