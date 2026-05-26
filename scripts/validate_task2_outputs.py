"""Validate Task 2 fusion outputs without requiring pytest."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import sys


REQUIRED_COLUMNS = [
    "timestamp_ns",
    "master_topic",
    "detection_id",
    "center_x",
    "center_y",
    "vx",
    "vy",
    "yaw",
    "length",
    "width",
    "confidence",
    "sensors",
    "n_lidar",
    "n_radar",
    "n_camera",
    "camera_visible",
    "camera_validated",
    "track_id",
    "age",
    "missed_frames",
    "is_predicted",
    "velocity_source",
]

REQUIRED_VISUALS = [
    "fusion_bev_000.png",
    "verify_lidar_overlay_000.png",
    "verify_camera_projection_000.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Task 2 output artifacts.")
    parser.add_argument(
        "--output-dir",
        default="outputs/task2",
        help="Directory containing fused_detections.csv and Task 2 visualisations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "fused_detections.csv"
    report_path = output_dir / "task2_report.md"
    errors: list[str] = []
    warnings: list[str] = []

    if not csv_path.exists():
        errors.append(f"missing CSV: {csv_path}")
        rows: list[dict[str, str]] = []
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    if not rows:
        errors.append("fused_detections.csv has no rows")
        columns: list[str] = []
    else:
        columns = list(rows[0])
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            errors.append(f"missing required columns: {missing}")

    for visual_name in REQUIRED_VISUALS:
        path = output_dir / visual_name
        if not path.exists():
            errors.append(f"missing visualisation: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"empty visualisation file: {path}")

    if rows:
        timestamps = {row["timestamp_ns"] for row in rows}
        nonzero_velocity = [
            row for row in rows
            if abs(_float(row.get("vx"))) > 1e-3 or abs(_float(row.get("vy"))) > 1e-3
        ]
        if len(timestamps) > 1 and not nonzero_velocity:
            errors.append("all vx/vy values are zero despite multiple timestamps")

        low_conf = sum(_float(row.get("confidence")) < 0.50 for row in rows)
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            match = re.search(
                r"\|\s*low_confidence_lt_0\.50\s*\|\s*(\d+)\s*\|",
                report_text,
            )
            if match is None:
                errors.append("task2_report.md does not report low_confidence_lt_0.50")
            elif int(match.group(1)) != low_conf:
                errors.append(
                    "report low_confidence_lt_0.50 does not match CSV: "
                    f"report={match.group(1)} csv={low_conf}"
                )
        else:
            errors.append(f"missing report: {report_path}")

        max_sources = max(
            _int(row.get("n_lidar")) + _int(row.get("n_radar")) + _int(row.get("n_camera"))
            for row in rows
        )
        if max_sources < 2:
            errors.append(
                "no row has at least two source indicators "
                "(n_lidar + n_radar + n_camera)"
            )

        radar_rows = sum(_int(row.get("n_radar")) > 0 for row in rows)
        if radar_rows == 0:
            warnings.append(
                "no radar-associated rows; this can be valid if radar returns are "
                "rejected as clutter, but the report should state it clearly"
            )

        camera_rows = sum(_int(row.get("n_camera")) > 0 for row in rows)
        if camera_rows == 0:
            errors.append("no camera-validated rows found")

        invalid_yaw = [
            row["detection_id"] for row in rows
            if not -math.pi <= _float(row.get("yaw")) <= math.pi
        ]
        if invalid_yaw:
            errors.append(f"yaw outside [-pi, pi] for {len(invalid_yaw)} rows")

        print("Task 2 validation summary")
        print(f"- rows: {len(rows)}")
        print(f"- timestamps: {len(timestamps)}")
        print(f"- nonzero velocity rows: {len(nonzero_velocity)}")
        print(f"- max source indicators per row: {max_sources}")
        print(f"- radar-associated rows: {radar_rows}")
        print(f"- camera-validated rows: {camera_rows}")
        print(f"- low confidence rows (<0.50): {low_conf}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidation passed.")
    return 0


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
