from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.task6_behavior_prediction import (
    build_behaviour_segments,
    classify_behaviours,
    prepare_track_features,
    run_behavior_prediction,
)


def _ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def _synthetic_tracks() -> pd.DataFrame:
    rows: list[dict] = []
    times = np.arange(0.0, 1.6, 0.2)

    for t in times:
        rows.append(
            {
                "timestamp_ns": _ns(t),
                "track_id": 1,
                "center_x": 5.0 * t,
                "center_y": 0.0,
                "vx": 5.0,
                "vy": 0.0,
                "yaw": 0.0,
                "status": "CONFIRMED",
                "confidence": 0.9,
            }
        )

    for t in times:
        speed = max(0.5, 6.0 - 3.0 * t)
        rows.append(
            {
                "timestamp_ns": _ns(t),
                "track_id": 2,
                "center_x": 6.0 * t - 1.5 * t * t,
                "center_y": 2.0,
                "vx": speed,
                "vy": 0.0,
                "yaw": 0.0,
                "status": "CONFIRMED",
                "confidence": 0.9,
            }
        )

    for t in times:
        rows.append(
            {
                "timestamp_ns": _ns(t),
                "track_id": 3,
                "center_x": 5.5 * t,
                "center_y": 2.2 * t,
                "vx": 5.5,
                "vy": 2.2,
                "yaw": 0.30 * t,
                "status": "CONFIRMED",
                "confidence": 0.9,
            }
        )

    recovery_yaw = [0.0, 0.15, 0.55, 0.65, 0.35, 0.12, 0.04, 0.02]
    recovery_y = [0.0, 0.4, 1.2, 1.7, 1.35, 0.8, 0.35, 0.12]
    for t, yaw, y in zip(times, recovery_yaw, recovery_y):
        rows.append(
            {
                "timestamp_ns": _ns(t),
                "track_id": 4,
                "center_x": 4.5 * t,
                "center_y": y,
                "vx": 4.5,
                "vy": 0.0,
                "yaw": yaw,
                "status": "CONFIRMED",
                "confidence": 0.9,
            }
        )

    rows.extend(
        [
            {
                "timestamp_ns": _ns(0.0),
                "track_id": 5,
                "center_x": 0.0,
                "center_y": 0.0,
                "confidence": 0.2,
            },
            {
                "timestamp_ns": _ns(0.2),
                "track_id": 5,
                "center_x": 0.2,
                "center_y": 0.0,
                "confidence": 0.2,
            },
        ]
    )

    return pd.DataFrame(rows)


def test_prepare_track_features_infers_motion_columns_when_missing() -> None:
    tracks = pd.DataFrame(
        [
            {"timestamp_ns": _ns(0.0), "track_id": 9, "center_x": 0.0, "center_y": 0.0},
            {"timestamp_ns": _ns(0.5), "track_id": 9, "center_x": 1.0, "center_y": 0.0},
            {"timestamp_ns": _ns(1.0), "track_id": 9, "center_x": 3.0, "center_y": 0.0},
        ]
    )

    features, notes = prepare_track_features(tracks, window_s=1.0)

    assert notes["used_velocity_columns"] is False
    assert features["speed"].iloc[-1] > features["speed"].iloc[1]
    assert features["acceleration"].iloc[-1] > 0.0
    assert abs(features["heading"].iloc[-1]) < 1e-6
    assert abs(features["yaw_rate"].fillna(0.0)).max() < 1e-6


def test_classifier_identifies_core_behaviour_patterns() -> None:
    features, _ = prepare_track_features(_synthetic_tracks(), window_s=1.0)
    predictions = classify_behaviours(features, min_points=4)

    by_track = {
        int(track_id): set(group["predicted_behaviour"])
        for track_id, group in predictions.groupby("track_id")
    }

    assert "lane_keeping" in by_track[1]
    assert "braking" in by_track[2]
    assert "overtaking_attempt" in by_track[3]
    assert "trajectory_recovery" in by_track[4]
    assert by_track[5] == {"unknown"}


def test_segments_group_consecutive_behaviours() -> None:
    features, _ = prepare_track_features(_synthetic_tracks(), window_s=1.0)
    predictions = classify_behaviours(features, min_points=4)

    segments = build_behaviour_segments(predictions)

    assert {"track_id", "behaviour", "start_time", "end_time", "duration"}.issubset(
        segments.columns
    )
    assert (segments["duration"] >= 0.0).all()
    assert any(segments["behaviour"] == "braking")


def test_run_behavior_prediction_writes_required_outputs(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.csv"
    output_dir = tmp_path / "task6"
    _synthetic_tracks().to_csv(tracks_path, index=False)

    result = run_behavior_prediction(
        tracks_path=tracks_path,
        predictions_path=tmp_path / "missing_predictions.csv",
        prediction_errors_path=tmp_path / "missing_prediction_errors.csv",
        map_dir=tmp_path / "missing_map",
        output_dir=output_dir,
        window_s=1.0,
        min_points=4,
    )

    assert not result.predictions.empty
    for name in [
        "behaviour_predictions.csv",
        "behaviour_summary.csv",
        "track_behaviour_segments.csv",
        "behaviour_timeline.png",
        "behaviour_bev.png",
        "README.md",
    ]:
        assert (output_dir / name).exists()

    written = pd.read_csv(output_dir / "behaviour_predictions.csv")
    required_columns = {
        "timestamp_ns",
        "track_id",
        "x",
        "y",
        "speed",
        "acceleration",
        "heading",
        "yaw_rate",
        "lateral_motion_score",
        "braking_score",
        "overtaking_score",
        "recovery_score",
        "lane_keeping_score",
        "predicted_behaviour",
        "confidence",
        "explanation",
    }
    assert required_columns.issubset(written.columns)
