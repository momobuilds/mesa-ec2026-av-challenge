"""Validate Task 2 fusion outputs without pytest or project imports."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import sys


REQUIRED_COLUMNS = [
    "timestamp",
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "confidence",
    "sensors",
    "n_lidar",
    "n_radar",
    "n_camera",
    "camera_validated",
    "track_id",
    "age",
    "missed_frames",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Task 2 output artifacts.")
    parser.add_argument(
        "--output-dir",
        default="outputs/task2",
        help="Directory containing fused_detections.csv, report, and PNGs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "fused_detections.csv"
    report_path = output_dir / "task2_report.md"
    errors: list[str] = []
    warnings: list[str] = []

    rows = _read_csv(csv_path, errors)
    if rows:
        columns = list(rows[0])
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            errors.append(f"missing required CSV columns: {missing}")

    pngs = sorted(output_dir.glob("*.png"))
    if not pngs:
        errors.append(f"no PNG visualisations found in {output_dir}")
    elif not any(_is_png(path) for path in pngs):
        errors.append("PNG files exist but none have a valid PNG signature")
    camera_projection_pngs = sorted(output_dir.glob("verify_camera_projection_*.png"))
    if len(camera_projection_pngs) < 3:
        errors.append(
            "expected at least 3 camera projection verification images "
            f"(found {len(camera_projection_pngs)})"
        )

    if rows:
        timestamps = {row["timestamp"] for row in rows}
        if len(timestamps) < 2:
            errors.append("expected multiple timestamps in fused_detections.csv")

        nonzero_velocity = [
            row for row in rows
            if abs(_float(row.get("vx"))) > 1e-3
            or abs(_float(row.get("vy"))) > 1e-3
        ]
        if len(timestamps) > 1 and not nonzero_velocity:
            errors.append("vx/vy are all zero despite multiple timestamps")

        low_conf = sum(_float(row.get("confidence")) < 0.50 for row in rows)
        report_low_conf = _report_low_confidence(report_path, errors)
        if report_low_conf is not None and report_low_conf != low_conf:
            errors.append(
                "report low_confidence_lt_0.50 does not match CSV: "
                f"report={report_low_conf} csv={low_conf}"
            )

        max_sources = max(
            _int(row.get("n_lidar"))
            + _int(row.get("n_radar"))
            for row in rows
        )
        if max_sources < 2:
            errors.append("no row has both LiDAR and radar source indicators")

        radar_rows = sum(_int(row.get("n_radar")) > 0 for row in rows)
        camera_projection_rows = sum(
            str(row.get("camera_validated", "")).lower() == "true"
            for row in rows
        )
        predicted_rows = sum(str(row.get("is_predicted", "")).lower() == "true" for row in rows)

        if camera_projection_rows == 0:
            errors.append("no camera projection-validated rows found")
        if radar_rows == 0:
            warnings.append(
                "no radar-associated rows; acceptable only if the report explains "
                "that radar returns were rejected as clutter or unavailable"
            )

        bad_sources = _source_flag_errors(rows)
        if bad_sources:
            errors.extend(bad_sources[:10])

        bad_yaw = [
            row.get("detection_id", "?")
            for row in rows
            if not -math.pi <= _float(row.get("yaw")) <= math.pi
        ]
        if bad_yaw:
            errors.append(f"yaw outside [-pi, pi] for {len(bad_yaw)} rows")

        print("Task 2 validation summary")
        print(f"- rows: {len(rows)}")
        print(f"- timestamps: {len(timestamps)}")
        print(f"- nonzero velocity rows: {len(nonzero_velocity)}")
        print(f"- max LiDAR/radar source indicators per row: {max_sources}")
        print(f"- radar-associated rows: {radar_rows}")
        print(f"- camera projection rows: {camera_projection_rows}")
        print(f"- temporal prediction rows: {predicted_rows}")
        print(f"- low confidence rows (<0.50): {low_conf}")
        print(f"- png visualisations: {len(pngs)}")
        print(f"- camera projection visualisations: {len(camera_projection_pngs)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPASS")
    return 0


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        errors.append(f"missing fused detections CSV: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        errors.append(f"{path} has no rows")
    return rows


def _report_low_confidence(path: Path, errors: list[str]) -> int | None:
    if not path.exists():
        errors.append(f"missing Task 2 report: {path}")
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\|\s*low_confidence_lt_0\.50\s*\|\s*(\d+)\s*\|", text)
    if match is None:
        errors.append("task2_report.md does not include low_confidence_lt_0.50")
        return None
    return int(match.group(1))


def _source_flag_errors(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        sensors = row.get("sensors", "")
        det_id = row.get("detection_id", "?")
        is_prediction = str(row.get("is_predicted", "")).lower() == "true"
        if _int(row.get("n_lidar")) > 0 and "lidar" not in sensors:
            errors.append(f"{det_id}: n_lidar>0 but sensors={sensors!r}")
        if _int(row.get("n_radar")) > 0 and "radar" not in sensors:
            errors.append(f"{det_id}: n_radar>0 but sensors={sensors!r}")
        if _int(row.get("n_camera")) > 0:
            errors.append(
                f"{det_id}: n_camera>0 even though camera projection is not sensor support"
            )
        if not is_prediction and _int(row.get("n_lidar")) + _int(row.get("n_radar")) == 0:
            errors.append(f"{det_id}: observed row has no LiDAR or radar source")
    return errors


def _is_png(path: Path) -> bool:
    try:
        return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


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
