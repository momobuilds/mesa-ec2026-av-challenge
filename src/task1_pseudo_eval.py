"""Pseudo-accuracy evaluation for Task 1 using Task 3 confirmed tracks.

This is intentionally labelled as pseudo-evaluation. Task 3 tracks are not
human ground truth; they are a stronger temporal consistency reference derived
from the challenge pipeline. The resulting precision/recall/IoU numbers are
useful evidence, not official labelled accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class PseudoEvaluationResult:
    """Pseudo-evaluation summary and matched detection/track pairs."""

    summary: dict[str, Any]
    matches: pd.DataFrame


def evaluate_task1_against_tracks(
    detections: pd.DataFrame,
    tracks: pd.DataFrame,
    *,
    timestamp_tolerance_ns: int = int(60e6),
    distance_gate_m: float = 5.0,
) -> PseudoEvaluationResult:
    """Compare Task 1 detections against nearest Task 3 confirmed tracks."""

    detections = _normalize_detections(detections)
    tracks = _normalize_tracks(tracks)

    if detections.empty or tracks.empty:
        summary = _build_summary([], len(detections), len(tracks), timestamp_tolerance_ns, distance_gate_m)
        return PseudoEvaluationResult(summary=summary, matches=pd.DataFrame())

    det_by_ts = {int(ts): group.copy() for ts, group in detections.groupby("timestamp_ns")}
    track_timestamps = np.array(sorted(int(ts) for ts in tracks["timestamp_ns"].unique()), dtype=np.int64)

    match_rows: list[dict[str, Any]] = []
    evaluated_track_keys: set[tuple[int, int]] = set()
    evaluated_detection_count = 0

    for det_ts, det_group in sorted(det_by_ts.items()):
        nearest_ts = _nearest_timestamp(det_ts, track_timestamps, timestamp_tolerance_ns)
        if nearest_ts is None:
            continue
        track_group = tracks[tracks["timestamp_ns"] == nearest_ts].copy()
        if track_group.empty:
            continue

        evaluated_detection_count += len(det_group)
        for track in track_group.itertuples(index=False):
            evaluated_track_keys.add((int(track.timestamp_ns), int(track.track_id)))

        matches = _match_frame(det_group, track_group, distance_gate_m)
        match_rows.extend(matches)

    summary = _build_summary(
        match_rows,
        evaluated_detection_count,
        len(evaluated_track_keys),
        timestamp_tolerance_ns,
        distance_gate_m,
    )
    return PseudoEvaluationResult(
        summary=summary,
        matches=pd.DataFrame(match_rows),
    )


def write_pseudo_evaluation(
    detections_csv: str | Path,
    tracks_csv: str | Path,
    output_dir: str | Path,
    *,
    timestamp_tolerance_ns: int = int(60e6),
    distance_gate_m: float = 5.0,
) -> PseudoEvaluationResult:
    detections = pd.read_csv(detections_csv)
    tracks = pd.read_csv(tracks_csv)
    result = evaluate_task1_against_tracks(
        detections,
        tracks,
        timestamp_tolerance_ns=timestamp_tolerance_ns,
        distance_gate_m=distance_gate_m,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.matches.to_csv(output / "pseudo_accuracy_matches.csv", index=False)
    (output / "pseudo_accuracy_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "pseudo_accuracy_report.md").write_text(
        _build_report(result.summary, title="Task 1 Pseudo-Accuracy Report"),
        encoding="utf-8",
    )
    temporal = evaluate_task1_temporal_consistency(detections)
    temporal.tracks.to_csv(output / "temporal_pseudo_tracks.csv", index=False)
    (output / "temporal_pseudo_accuracy_summary.json").write_text(
        json.dumps(temporal.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "temporal_pseudo_accuracy_report.md").write_text(
        _build_report(temporal.summary, title="Task 1 Temporal Pseudo-Accuracy Report"),
        encoding="utf-8",
    )
    return result


@dataclass
class TemporalPseudoEvaluationResult:
    """Temporal consistency pseudo-metrics and lightweight tracks."""

    summary: dict[str, Any]
    tracks: pd.DataFrame


def evaluate_task1_temporal_consistency(
    detections: pd.DataFrame,
    *,
    distance_gate_m: float = 6.0,
    min_confirmed_hits: int = 3,
    max_gap_frames: int = 2,
) -> TemporalPseudoEvaluationResult:
    """Estimate Task 1 reliability from repeated detections over time.

    A detection that can be associated to a short-lived temporal track is less
    likely to be one-frame clutter. This is still a pseudo metric, but it is a
    useful false-positive proxy when no labelled ground truth exists.
    """

    detections = _normalize_detections(detections)
    if detections.empty:
        return TemporalPseudoEvaluationResult(
            summary=_temporal_summary([], 0, distance_gate_m, min_confirmed_hits, max_gap_frames),
            tracks=pd.DataFrame(),
        )

    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    next_track_id = 1
    total_detections = 0

    timestamps = sorted(int(ts) for ts in detections["timestamp_ns"].unique())
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}

    for timestamp in timestamps:
        frame_index = timestamp_to_index[timestamp]
        frame = detections[detections["timestamp_ns"] == timestamp].reset_index(drop=True)
        total_detections += len(frame)

        active = _expire_tracks(active, finished, frame_index, max_gap_frames)
        if active and len(frame):
            cost = np.full((len(active), len(frame)), 1e6, dtype=np.float64)
            for row, track in enumerate(active):
                for col, det in frame.iterrows():
                    cost[row, col] = float(
                        np.hypot(
                            track["last_x"] - float(det["center_x_base"]),
                            track["last_y"] - float(det["center_y_base"]),
                        )
                    )
            rows, cols = linear_sum_assignment(cost)
        else:
            rows, cols = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, col in zip(rows, cols):
            distance = float(cost[row, col])
            if distance > distance_gate_m:
                continue
            det = frame.iloc[int(col)]
            track = active[int(row)]
            track["hits"] += 1
            track["last_frame_index"] = frame_index
            track["last_timestamp_ns"] = timestamp
            track["last_x"] = float(det["center_x_base"])
            track["last_y"] = float(det["center_y_base"])
            track["detection_ids"].append(str(det["detection_id"]))
            matched_tracks.add(int(row))
            matched_detections.add(int(col))

        for col, det in frame.iterrows():
            if int(col) in matched_detections:
                continue
            active.append(
                {
                    "temporal_track_id": next_track_id,
                    "first_frame_index": frame_index,
                    "last_frame_index": frame_index,
                    "first_timestamp_ns": timestamp,
                    "last_timestamp_ns": timestamp,
                    "last_x": float(det["center_x_base"]),
                    "last_y": float(det["center_y_base"]),
                    "hits": 1,
                    "detection_ids": [str(det["detection_id"])],
                }
            )
            next_track_id += 1

    finished.extend(active)
    summary = _temporal_summary(
        finished,
        total_detections,
        distance_gate_m,
        min_confirmed_hits,
        max_gap_frames,
    )
    return TemporalPseudoEvaluationResult(
        summary=summary,
        tracks=pd.DataFrame(_temporal_track_rows(finished, min_confirmed_hits)),
    )


def _normalize_detections(detections: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_ns", "center_x_base", "center_y_base", "length", "width"}
    missing = required - set(detections.columns)
    if missing:
        fallback = {"center_x_base": "center_x", "center_y_base": "center_y"}
        for target, source in fallback.items():
            if target in missing and source in detections.columns:
                detections[target] = detections[source]
                missing.remove(target)
    if missing:
        raise ValueError(f"detections CSV missing required columns: {sorted(missing)}")

    output = detections.copy()
    if "detection_id" not in output.columns:
        output["detection_id"] = [f"det_{idx}" for idx in range(len(output))]
    return output


def _normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_ns", "track_id", "center_x", "center_y", "length", "width"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks CSV missing required columns: {sorted(missing)}")

    output = tracks.copy()
    if "status" in output.columns:
        output = output[output["status"].astype(str).str.upper() == "CONFIRMED"].copy()
    return output


def _nearest_timestamp(
    timestamp_ns: int,
    candidates: np.ndarray,
    tolerance_ns: int,
) -> int | None:
    if len(candidates) == 0:
        return None
    idx = int(np.searchsorted(candidates, timestamp_ns))
    nearest: list[int] = []
    if idx < len(candidates):
        nearest.append(int(candidates[idx]))
    if idx > 0:
        nearest.append(int(candidates[idx - 1]))
    best = min(nearest, key=lambda value: abs(value - timestamp_ns))
    if abs(best - timestamp_ns) <= tolerance_ns:
        return best
    return None


def _match_frame(
    detections: pd.DataFrame,
    tracks: pd.DataFrame,
    distance_gate_m: float,
) -> list[dict[str, Any]]:
    det_xy = detections[["center_x_base", "center_y_base"]].to_numpy(dtype=np.float64)
    trk_xy = tracks[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    if len(det_xy) == 0 or len(trk_xy) == 0:
        return []

    diff = det_xy[:, None, :] - trk_xy[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    rows, cols = linear_sum_assignment(cost)

    match_rows: list[dict[str, Any]] = []
    det_records = detections.reset_index(drop=True)
    track_records = tracks.reset_index(drop=True)
    for row, col in zip(rows, cols):
        distance = float(cost[row, col])
        if distance > distance_gate_m:
            continue
        det = det_records.iloc[int(row)]
        track = track_records.iloc[int(col)]
        iou = _axis_aligned_iou(
            float(det["center_x_base"]),
            float(det["center_y_base"]),
            float(det["length"]),
            float(det["width"]),
            float(track["center_x"]),
            float(track["center_y"]),
            float(track["length"]),
            float(track["width"]),
        )
        match_rows.append(
            {
                "detection_timestamp_ns": int(det["timestamp_ns"]),
                "track_timestamp_ns": int(track["timestamp_ns"]),
                "timestamp_delta_ms": (int(det["timestamp_ns"]) - int(track["timestamp_ns"])) / 1e6,
                "detection_id": str(det["detection_id"]),
                "track_id": int(track["track_id"]),
                "center_error_m": distance,
                "axis_aligned_iou": iou,
                "detection_center_x": float(det["center_x_base"]),
                "detection_center_y": float(det["center_y_base"]),
                "track_center_x": float(track["center_x"]),
                "track_center_y": float(track["center_y"]),
            }
        )
    return match_rows


def _axis_aligned_iou(
    cx_a: float,
    cy_a: float,
    length_a: float,
    width_a: float,
    cx_b: float,
    cy_b: float,
    length_b: float,
    width_b: float,
) -> float:
    ax1, ax2 = cx_a - length_a / 2.0, cx_a + length_a / 2.0
    ay1, ay2 = cy_a - width_a / 2.0, cy_a + width_a / 2.0
    bx1, bx2 = cx_b - length_b / 2.0, cx_b + length_b / 2.0
    by1, by2 = cy_b - width_b / 2.0, cy_b + width_b / 2.0
    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_x * inter_y
    area_a = max(length_a, 0.0) * max(width_a, 0.0)
    area_b = max(length_b, 0.0) * max(width_b, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _build_summary(
    matches: list[dict[str, Any]],
    detections_evaluated: int,
    tracks_evaluated: int,
    timestamp_tolerance_ns: int,
    distance_gate_m: float,
) -> dict[str, Any]:
    matched = len(matches)
    center_errors = [row["center_error_m"] for row in matches]
    ious = [row["axis_aligned_iou"] for row in matches]
    return {
        "metric_type": "pseudo_accuracy_against_task3_tracks",
        "ground_truth_available": False,
        "important_note": (
            "These are pseudo metrics using Task 3 confirmed tracks as reference. "
            "They are not labelled benchmark accuracy and must not be reported as "
            "true precision/recall/IoU."
        ),
        "timestamp_tolerance_ms": timestamp_tolerance_ns / 1e6,
        "distance_gate_m": distance_gate_m,
        "detections_evaluated": int(detections_evaluated),
        "pseudo_tracks_evaluated": int(tracks_evaluated),
        "matched_pairs": int(matched),
        "unmatched_detections": int(max(detections_evaluated - matched, 0)),
        "unmatched_pseudo_tracks": int(max(tracks_evaluated - matched, 0)),
        "pseudo_precision": _ratio(matched, detections_evaluated),
        "pseudo_recall": _ratio(matched, tracks_evaluated),
        "mean_center_error_m": _mean(center_errors),
        "median_center_error_m": _median(center_errors),
        "mean_axis_aligned_iou": _mean(ious),
        "median_axis_aligned_iou": _median(ious),
    }


def _expire_tracks(
    active: list[dict[str, Any]],
    finished: list[dict[str, Any]],
    frame_index: int,
    max_gap_frames: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for track in active:
        if frame_index - int(track["last_frame_index"]) > max_gap_frames:
            finished.append(track)
        else:
            kept.append(track)
    return kept


def _temporal_track_rows(
    tracks: list[dict[str, Any]],
    min_confirmed_hits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in tracks:
        hits = int(track["hits"])
        rows.append(
            {
                "temporal_track_id": int(track["temporal_track_id"]),
                "confirmed": hits >= min_confirmed_hits,
                "hits": hits,
                "first_frame_index": int(track["first_frame_index"]),
                "last_frame_index": int(track["last_frame_index"]),
                "first_timestamp_ns": int(track["first_timestamp_ns"]),
                "last_timestamp_ns": int(track["last_timestamp_ns"]),
                "last_x": float(track["last_x"]),
                "last_y": float(track["last_y"]),
                "detection_ids": ";".join(track["detection_ids"]),
            }
        )
    return rows


def _temporal_summary(
    tracks: list[dict[str, Any]],
    total_detections: int,
    distance_gate_m: float,
    min_confirmed_hits: int,
    max_gap_frames: int,
) -> dict[str, Any]:
    confirmed = [track for track in tracks if int(track["hits"]) >= min_confirmed_hits]
    confirmed_hits = sum(int(track["hits"]) for track in confirmed)
    return {
        "metric_type": "temporal_self_consistency_pseudo_accuracy",
        "ground_truth_available": False,
        "important_note": (
            "This is not labelled accuracy. It estimates reliability by measuring "
            "what fraction of Task 1 detections persist in short temporal tracks."
        ),
        "distance_gate_m": distance_gate_m,
        "min_confirmed_hits": min_confirmed_hits,
        "max_gap_frames": max_gap_frames,
        "total_detections": int(total_detections),
        "temporal_tracks_total": int(len(tracks)),
        "confirmed_tracks": int(len(confirmed)),
        "detections_on_confirmed_tracks": int(confirmed_hits),
        "one_frame_tracks": int(sum(int(track["hits"]) == 1 for track in tracks)),
        "temporal_pseudo_precision": _ratio(confirmed_hits, total_detections),
        "mean_confirmed_track_hits": _mean([int(track["hits"]) for track in confirmed]),
        "median_confirmed_track_hits": _median([int(track["hits"]) for track in confirmed]),
    }


def _build_report(summary: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "These numbers are useful evidence, but they are not labelled ground-truth accuracy.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-evaluate Task 1 detections with Task 3 tracks.")
    parser.add_argument("--detections", default="outputs/task1/detections.csv")
    parser.add_argument("--tracks", default="outputs/task3/tracks.csv")
    parser.add_argument("--output-dir", default="outputs/task1")
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=60.0)
    parser.add_argument("--distance-gate", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_pseudo_evaluation(
        args.detections,
        args.tracks,
        args.output_dir,
        timestamp_tolerance_ns=int(args.timestamp_tolerance_ms * 1e6),
        distance_gate_m=args.distance_gate,
    )
    print("Task 1 pseudo-evaluation complete")
    print(f"- matched pairs: {result.summary['matched_pairs']}")
    print(f"- pseudo precision: {result.summary['pseudo_precision']}")
    print(f"- pseudo recall: {result.summary['pseudo_recall']}")
    print(f"- mean center error: {result.summary['mean_center_error_m']}")
    print(f"- mean axis-aligned IoU: {result.summary['mean_axis_aligned_iou']}")


if __name__ == "__main__":
    main()
