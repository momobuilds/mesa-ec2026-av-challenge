from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.task7_overtake_planning import (
    STRATEGIES,
    CandidateTrajectory,
    EgoState,
    PlannerConfig,
    build_learning_dataset,
    generate_candidate_trajectories,
    run_overtake_planning,
    score_candidate_trajectory,
)


def _ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def _synthetic_tracks() -> pd.DataFrame:
    rows: list[dict] = []
    for idx, t in enumerate(np.arange(0.0, 1.2, 0.2)):
        rows.append(
            {
                "timestamp_ns": _ns(float(t)),
                "track_id": 11,
                "status": "CONFIRMED",
                "center_x": 14.0 + 0.6 * idx,
                "center_y": 0.1,
                "vx": 1.0,
                "vy": 0.0,
                "yaw": 0.0,
                "length": 4.8,
                "width": 1.8,
                "confidence": 0.85,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_behaviour(tracks: pd.DataFrame) -> pd.DataFrame:
    out = tracks[["timestamp_ns", "track_id", "center_x", "center_y"]].copy()
    out["time_s"] = (out["timestamp_ns"] - out["timestamp_ns"].min()) / 1e9
    out["x"] = out["center_x"]
    out["y"] = out["center_y"]
    out["speed"] = 1.0
    out["acceleration"] = -1.0
    out["heading"] = 0.0
    out["predicted_behaviour"] = "braking"
    out["confidence"] = 0.8
    out["explanation"] = "synthetic braking opponent"
    return out


def test_candidate_generation_includes_required_strategies() -> None:
    config = PlannerConfig(horizon_s=1.5, dt_s=0.5, track_half_width_m=5.0)
    ego = EgoState(timestamp_ns=_ns(0.0), time_s=0.0, x=0.0, y=0.0, speed=12.0, heading=0.0)

    candidates = generate_candidate_trajectories(ego, config)
    by_strategy = {candidate.strategy: candidate for candidate in candidates}

    assert set(STRATEGIES).issubset(by_strategy)
    assert by_strategy["overtake_left"].points["planned_y"].iloc[-1] > 1.0
    assert by_strategy["overtake_right"].points["planned_y"].iloc[-1] < -1.0
    assert by_strategy["emergency_brake"].points["planned_speed"].iloc[-1] < ego.speed


def test_scoring_rejects_candidate_that_collides_with_prediction() -> None:
    config = PlannerConfig(horizon_s=1.0, dt_s=0.5, min_safe_distance_m=4.0)
    ego = EgoState(timestamp_ns=_ns(0.0), time_s=0.0, x=0.0, y=0.0, speed=8.0, heading=0.0)
    points = pd.DataFrame(
        {
            "horizon_time": [0.5, 1.0],
            "step_index": [1, 2],
            "planned_x": [4.0, 8.0],
            "planned_y": [0.0, 0.0],
            "planned_speed": [8.0, 8.0],
            "planned_heading": [0.0, 0.0],
        }
    )
    candidate = CandidateTrajectory("keep_racing_line", points, target_track_id=7)
    predictions = {
        7: pd.DataFrame(
            {
                "horizon_s": [0.5, 1.0],
                "pred_x": [4.2, 8.1],
                "pred_y": [0.1, 0.0],
            }
        )
    }

    score = score_candidate_trajectory(
        candidate,
        ego,
        predictions,
        behaviour_by_track={7: ("lane_keeping", 0.9)},
        config=config,
    )

    assert score.is_safe is False
    assert score.min_distance_to_opponent < config.min_safe_distance_m
    assert score.collision_risk_score > 0.9


def test_scoring_handles_duplicate_prediction_index_from_horizon_alignment() -> None:
    config = PlannerConfig(horizon_s=1.0, dt_s=0.25, min_safe_distance_m=4.0)
    ego = EgoState(timestamp_ns=_ns(0.0), time_s=0.0, x=0.0, y=0.0, speed=5.0, heading=0.0)
    candidate = generate_candidate_trajectories(ego, config)[0]
    predictions = {
        7: pd.DataFrame(
            {
                "horizon_s": [0.25, 0.50, 0.75, 1.00],
                "pred_x": [10.0, 12.0, 14.0, 16.0],
                "pred_y": [0.0, 0.0, 0.0, 0.0],
            },
            index=[0, 0, 1, 1],
        )
    }

    score = score_candidate_trajectory(
        candidate,
        ego,
        predictions,
        behaviour_by_track={7: ("lane_keeping", 0.9)},
        config=config,
    )

    assert np.isfinite(score.min_distance_to_opponent)


def test_clean_overtake_gets_strategy_bonus_when_it_improves_clearance() -> None:
    config = PlannerConfig(horizon_s=1.0, dt_s=0.25, min_safe_distance_m=2.5)
    ego = EgoState(timestamp_ns=_ns(0.0), time_s=0.0, x=0.0, y=0.0, speed=6.0, heading=0.0)
    predictions = {
        7: pd.DataFrame(
            {
                "horizon_s": [0.25, 0.50, 0.75, 1.00],
                "pred_x": [6.5, 7.5, 8.5, 10.0],
                "pred_y": [0.0, 0.0, 0.0, 0.0],
            }
        )
    }
    behaviour = {7: ("lane_keeping", 0.85)}
    candidates = {c.strategy: c for c in generate_candidate_trajectories(ego, config, target_track_id=7)}

    keep = score_candidate_trajectory(candidates["keep_racing_line"], ego, predictions, behaviour, config)
    overtake = score_candidate_trajectory(candidates["overtake_right"], ego, predictions, behaviour, config)

    assert overtake.is_safe
    assert overtake.min_distance_to_opponent > keep.min_distance_to_opponent
    assert overtake.deterministic_cost < keep.deterministic_cost


def test_run_overtake_planning_writes_required_outputs(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.csv"
    behaviour_path = tmp_path / "behaviour_predictions.csv"
    output_dir = tmp_path / "task7"

    tracks = _synthetic_tracks()
    tracks.to_csv(tracks_path, index=False)
    _synthetic_behaviour(tracks).to_csv(behaviour_path, index=False)

    result = run_overtake_planning(
        tracks_path=tracks_path,
        predictions_path=tmp_path / "missing_predictions.csv",
        behaviour_path=behaviour_path,
        map_dir=tmp_path / "missing_map",
        output_dir=output_dir,
        horizon_s=1.0,
        dt_s=0.5,
        max_frames=4,
        train_learning=False,
    )

    assert not result.decisions.empty
    assert not result.risk_assessment.empty
    assert not result.planned_trajectories.empty
    assert not result.learning_training_data.empty
    assert result.decisions["selected_strategy"].isin(STRATEGIES + ["brake"]).all()

    for name in [
        "planning_decisions.csv",
        "planned_trajectories.csv",
        "risk_assessment.csv",
        "planning_summary.csv",
        "strategy_timeline.png",
        "planning_bev.png",
        "overtake_scenarios.png",
        "README.md",
        "learned_risk_training_data.csv",
        "learned_risk_scores.csv",
        "learning_model_summary.csv",
    ]:
        assert (output_dir / name).exists(), name

    decisions = pd.read_csv(output_dir / "planning_decisions.csv")
    required_columns = {
        "timestamp_ns",
        "ego_x",
        "ego_y",
        "ego_speed",
        "ego_heading",
        "selected_strategy",
        "confidence",
        "safety_margin",
        "min_predicted_distance",
        "deterministic_cost",
        "learned_risk_score",
        "final_cost",
        "explanation",
    }
    assert required_columns.issubset(decisions.columns)

    training = build_learning_dataset(result.risk_assessment)
    assert {"pseudo_label_safe", "pseudo_label_risk_score"}.issubset(training.columns)


@pytest.mark.skipif(not Path("data/ego.mat").exists(), reason="data/ego.mat not available")
def test_cli_script_mode_can_import_project_modules_for_ego_log(tmp_path: Path) -> None:
    output_dir = tmp_path / "task7_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "src/task7_overtake_planning.py",
            "--tracks",
            "outputs/task3/tracks.csv",
            "--predictions",
            "outputs/task5/predictions.csv",
            "--behaviour",
            "outputs/task6/behaviour_predictions.csv",
            "--map-dir",
            "outputs/task4",
            "--output-dir",
            str(output_dir),
            "--max-frames",
            "1",
            "--no-learning",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(output_dir / "planning_summary.csv")
    notes = " ".join(summary["notes"].fillna("").astype(str))
    assert "No module named 'src'" not in notes
    assert "Task 7 complete" in completed.stdout
