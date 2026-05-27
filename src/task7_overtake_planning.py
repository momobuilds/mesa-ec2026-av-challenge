"""Task 7: hybrid overtake strategy and collision-avoidance planner.

The planner is intentionally hybrid: a deterministic receding-horizon safety
planner generates and filters manoeuvre candidates, while an optional
lightweight learning model ranks risk from pseudo-labels. The learned model can
adjust costs, but hard safety constraints remain the final authority.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGIES = [
    "keep_racing_line",
    "follow_opponent",
    "brake",
    "evasive_left",
    "evasive_right",
    "overtake_left",
    "overtake_right",
    "abort_overtake",
    "emergency_brake",
]

STRATEGY_COLORS = {
    "keep_racing_line": "#2E7D32",
    "follow_opponent": "#546E7A",
    "brake": "#EF6C00",
    "evasive_left": "#00838F",
    "evasive_right": "#00ACC1",
    "overtake_left": "#1565C0",
    "overtake_right": "#5E35B1",
    "abort_overtake": "#8D6E63",
    "emergency_brake": "#C62828",
    "unknown": "#6D6D6D",
}

BEHAVIOUR_CODE = {
    "lane_keeping": 0,
    "braking": 1,
    "overtaking_attempt": 2,
    "trajectory_recovery": 3,
    "unknown": 4,
}


@dataclass
class PlannerConfig:
    horizon_s: float = 2.0
    dt_s: float = 0.25
    max_accel_mps2: float = 2.5
    max_brake_mps2: float = 6.5
    max_lateral_accel_mps2: float = 5.5
    max_yaw_rate_radps: float = 0.9
    min_safe_distance_m: float = 4.5
    collision_distance_m: float = 2.2
    boundary_margin_m: float = 0.8
    track_half_width_m: float = 6.0
    overtake_offset_m: float = 3.1
    evasive_offset_m: float = 2.4
    default_ego_speed_mps: float = 8.0
    learning_alpha: float = 2.5
    hysteresis_cost_margin: float = 0.12
    max_frames: int | None = None


@dataclass
class EgoState:
    timestamp_ns: int
    time_s: float
    x: float
    y: float
    speed: float
    heading: float
    yaw_rate: float = 0.0
    acceleration: float = 0.0


@dataclass
class CandidateTrajectory:
    strategy: str
    points: pd.DataFrame
    target_track_id: int | None = None
    side: str = ""


@dataclass
class CandidateScore:
    candidate_strategy: str
    target_track_id: int | None
    min_distance_to_opponent: float
    time_to_closest_approach: float
    collision_risk_score: float
    boundary_risk_score: float
    behaviour_uncertainty_score: float
    racing_line_cost: float
    progress_reward: float
    smoothness_cost: float
    deterministic_cost: float
    learned_risk_score: float
    final_cost: float
    is_safe: bool
    opponent_distance: float
    opponent_behaviour: str
    opponent_behaviour_confidence: float
    lateral_offset: float
    predicted_path_overlap: float
    safety_margin: float
    explanation: str


@dataclass
class TrackCorridor:
    half_width_m: float
    source: str
    boundaries: list[np.ndarray] = field(default_factory=list)


@dataclass
class LearningModel:
    model: Any | None
    feature_columns: list[str]
    model_type: str
    model_used: bool
    train_score: float | None
    validation_score: float | None
    fallback_reason: str
    top_features: list[tuple[str, float]]


@dataclass
class Task7Result:
    decisions: pd.DataFrame
    planned_trajectories: pd.DataFrame
    risk_assessment: pd.DataFrame
    summary: pd.DataFrame
    learning_training_data: pd.DataFrame
    learned_scores: pd.DataFrame
    learning_model_summary: pd.DataFrame
    notes: list[str]


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _safe_min(values: list[float], default: float) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    return min(clean) if clean else default


def _load_optional_csv(path: Path | None, name: str, notes: list[str]) -> pd.DataFrame | None:
    if path is None:
        notes.append(f"{name}: not requested")
        return None
    path = Path(path)
    if not path.exists():
        notes.append(f"{name}: missing at {path}; using fallback")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"{name}: failed to read {path}: {exc}; using fallback")
        return None
    notes.append(f"{name}: loaded {len(df)} rows from {path}")
    return df


def _normalise_tracks(tracks_df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if tracks_df is None or tracks_df.empty:
        notes.append("tracks: no rows available; running single-frame no-op planner")
        return pd.DataFrame(
            columns=[
                "timestamp_ns",
                "time_s",
                "track_id",
                "x",
                "y",
                "vx",
                "vy",
                "heading",
                "speed",
                "confidence",
                "status",
                "length",
                "width",
            ]
        )

    df = tracks_df.copy()
    ts_col = _first_existing(df.columns, ["timestamp_ns", "timestamp", "frame"])
    if ts_col is None:
        df["timestamp_ns"] = np.arange(len(df), dtype=np.int64)
        notes.append("tracks: missing timestamp column; synthetic frame timestamps used")
    else:
        df["timestamp_ns"] = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=["timestamp_ns"]).copy()
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)

    tid_col = _first_existing(df.columns, ["track_id", "id", "object_id"])
    if tid_col is None:
        df["track_id"] = 0
        notes.append("tracks: missing track_id; all rows assigned track_id=0")
    else:
        df["track_id"] = pd.to_numeric(df[tid_col], errors="coerce").fillna(-1).astype(int)

    x_col = _first_existing(df.columns, ["x", "center_x", "cx", "pos_x"])
    y_col = _first_existing(df.columns, ["y", "center_y", "cy", "pos_y"])
    if x_col is None or y_col is None:
        notes.append("tracks: missing position columns; opponent positions set to zero")
        df["x"] = 0.0
        df["y"] = 0.0
    else:
        df["x"] = _to_numeric(df[x_col])
        df["y"] = _to_numeric(df[y_col])

    vx_col = _first_existing(df.columns, ["vx", "velocity_x"])
    vy_col = _first_existing(df.columns, ["vy", "velocity_y"])
    if vx_col is not None and vy_col is not None:
        df["vx"] = _to_numeric(df[vx_col])
        df["vy"] = _to_numeric(df[vy_col])
    else:
        df = df.sort_values(["track_id", "timestamp_ns"]).copy()
        df["time_tmp"] = df.groupby("track_id")["timestamp_ns"].transform(
            lambda s: (s - s.iloc[0]) / 1e9
        )
        pieces: list[pd.DataFrame] = []
        for _, grp in df.groupby("track_id", sort=False):
            grp = grp.copy()
            dt = grp["time_tmp"].diff().replace(0.0, np.nan)
            grp["vx"] = grp["x"].diff() / dt
            grp["vy"] = grp["y"].diff() / dt
            grp[["vx", "vy"]] = grp[["vx", "vy"]].bfill().ffill().fillna(0.0)
            pieces.append(grp)
        df = pd.concat(pieces, ignore_index=True) if pieces else df
        df.drop(columns=["time_tmp"], inplace=True, errors="ignore")
        notes.append("tracks: velocity inferred from position differences")

    heading_col = _first_existing(df.columns, ["heading", "yaw", "theta"])
    if heading_col is not None:
        df["heading"] = _to_numeric(df[heading_col])
    else:
        df["heading"] = np.arctan2(df["vy"], df["vx"])
        notes.append("tracks: heading inferred from velocity")

    conf_col = _first_existing(df.columns, ["confidence", "score"])
    df["confidence"] = _to_numeric(df[conf_col], 0.5) if conf_col else 0.5
    status_col = _first_existing(df.columns, ["status", "state"])
    df["status"] = df[status_col].astype(str) if status_col else "UNKNOWN"
    len_col = _first_existing(df.columns, ["length", "l"])
    width_col = _first_existing(df.columns, ["width", "w"])
    df["length"] = _to_numeric(df[len_col], 4.5) if len_col else 4.5
    df["width"] = _to_numeric(df[width_col], 1.8) if width_col else 1.8
    df["speed"] = np.hypot(df["vx"], df["vy"])
    first_ts = int(df["timestamp_ns"].min()) if not df.empty else 0
    df["time_s"] = (df["timestamp_ns"] - first_ts) / 1e9

    cols = [
        "timestamp_ns",
        "time_s",
        "track_id",
        "x",
        "y",
        "vx",
        "vy",
        "heading",
        "speed",
        "confidence",
        "status",
        "length",
        "width",
    ]
    return df[cols].sort_values(["timestamp_ns", "track_id"]).reset_index(drop=True)


def _normalise_behaviour(behaviour_df: pd.DataFrame | None, notes: list[str]) -> pd.DataFrame:
    if behaviour_df is None or behaviour_df.empty:
        notes.append("behaviour: unavailable; all opponents treated as unknown")
        return pd.DataFrame(columns=["timestamp_ns", "track_id", "predicted_behaviour", "confidence"])
    df = behaviour_df.copy()
    if "timestamp_ns" not in df.columns or "track_id" not in df.columns:
        notes.append("behaviour: missing timestamp_ns/track_id; ignored")
        return pd.DataFrame(columns=["timestamp_ns", "track_id", "predicted_behaviour", "confidence"])
    label_col = _first_existing(df.columns, ["predicted_behaviour", "behaviour", "label"])
    if label_col is None:
        df["predicted_behaviour"] = "unknown"
    else:
        df["predicted_behaviour"] = df[label_col].fillna("unknown").astype(str)
    conf_col = _first_existing(df.columns, ["confidence", "behaviour_confidence"])
    df["confidence"] = _to_numeric(df[conf_col], 0.0) if conf_col else 0.0
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce").astype("Int64")
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["timestamp_ns", "track_id"]).copy()
    df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
    df["track_id"] = df["track_id"].astype(int)
    return df[["timestamp_ns", "track_id", "predicted_behaviour", "confidence"]]


def _normalise_predictions(predictions_df: pd.DataFrame | None, notes: list[str]) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        notes.append("predictions: unavailable; constant-velocity fallback enabled")
        return pd.DataFrame(
            columns=["track_id", "source_ts_ns", "horizon_s", "pred_x", "pred_y", "model"]
        )
    required = {"track_id", "source_ts_ns", "horizon_s", "pred_x", "pred_y"}
    if not required.issubset(predictions_df.columns):
        notes.append("predictions: missing required columns; constant-velocity fallback enabled")
        return pd.DataFrame(
            columns=["track_id", "source_ts_ns", "horizon_s", "pred_x", "pred_y", "model"]
        )
    df = predictions_df.copy()
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype("Int64")
    df["source_ts_ns"] = pd.to_numeric(df["source_ts_ns"], errors="coerce").astype("Int64")
    df["horizon_s"] = pd.to_numeric(df["horizon_s"], errors="coerce")
    df["pred_x"] = pd.to_numeric(df["pred_x"], errors="coerce")
    df["pred_y"] = pd.to_numeric(df["pred_y"], errors="coerce")
    df = df.dropna(subset=["track_id", "source_ts_ns", "horizon_s", "pred_x", "pred_y"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["source_ts_ns"] = df["source_ts_ns"].astype(np.int64)
    if "model" not in df.columns:
        df["model"] = "unknown"
    notes.append("predictions: Task 5 future positions used where source timestamps match")
    return df[["track_id", "source_ts_ns", "horizon_s", "pred_x", "pred_y", "model"]]


def load_inputs(
    tracks_path: Path | str = Path("outputs/task3/tracks.csv"),
    predictions_path: Path | str | None = Path("outputs/task5/predictions.csv"),
    behaviour_path: Path | str | None = Path("outputs/task6/behaviour_predictions.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and normalize Task 3, Task 5, and Task 6 inputs."""
    notes: list[str] = []
    tracks_path = Path(tracks_path)
    if tracks_path.exists():
        tracks_raw = pd.read_csv(tracks_path)
        notes.append(f"tracks: loaded {len(tracks_raw)} rows from {tracks_path}")
    else:
        tracks_raw = pd.DataFrame()
        notes.append(f"tracks: missing at {tracks_path}; degraded no-op planning")
    tracks = _normalise_tracks(tracks_raw, notes)
    predictions = _normalise_predictions(
        _load_optional_csv(Path(predictions_path) if predictions_path is not None else None, "predictions", notes),
        notes,
    )
    behaviour = _normalise_behaviour(
        _load_optional_csv(Path(behaviour_path) if behaviour_path is not None else None, "behaviour", notes),
        notes,
    )
    return tracks, predictions, behaviour, notes


def _load_geojson_boundaries(map_dir: Path | str | None, notes: list[str]) -> list[np.ndarray]:
    if map_dir is None:
        notes.append("map: not requested; using inferred local corridor")
        return []
    path = Path(map_dir) / "track_boundaries.geojson"
    if not path.exists():
        notes.append(f"map: boundaries missing at {path}; using inferred local corridor")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"map: failed to read {path}: {exc}; using inferred local corridor")
        return []
    boundaries: list[np.ndarray] = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        pts = np.asarray(geom.get("coordinates", []), dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
            boundaries.append(pts[:, :2])
    if boundaries:
        notes.append(f"map: loaded {len(boundaries)} GeoJSON boundary polylines for context")
    else:
        notes.append("map: GeoJSON contained no usable LineStrings")
    return boundaries


def load_or_infer_track_corridor(
    tracks_df: pd.DataFrame,
    map_dir: Path | str | None,
    config: PlannerConfig,
    notes: list[str],
) -> TrackCorridor:
    """Return local planning corridor and optional world-frame boundaries for plots."""
    boundaries = _load_geojson_boundaries(map_dir, notes)
    half_width = float(config.track_half_width_m)
    source = "default local ego-frame corridor"
    if tracks_df is not None and not tracks_df.empty and "y" in tracks_df.columns:
        observed = float(np.nanpercentile(np.abs(tracks_df["y"].to_numpy(dtype=float)), 95))
        inferred = float(np.clip(observed + 2.5, 4.0, 9.0))
        if inferred > half_width:
            half_width = inferred
            source = "inferred from Task 3 lateral track spread"
    if boundaries:
        source += " with Task 4 boundaries for visual context"
    notes.append(f"corridor: half_width={half_width:.2f} m ({source})")
    return TrackCorridor(half_width_m=half_width, source=source, boundaries=boundaries)


def _load_ego_log(notes: list[str]) -> Any | None:
    mat_path = Path("data/ego.mat")
    metadata_path = Path("data/metadata.yaml")
    if not mat_path.exists() or not metadata_path.exists():
        notes.append("ego: data/ego.mat or metadata.yaml missing; speed inferred from tracks")
        return None
    try:
        import yaml

        from src.tf_utils import EgoStateLog

        meta = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        bag_start_ns = int(
            meta["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]
        )
        log = EgoStateLog(mat_path, bag_start_ns=bag_start_ns)
        notes.append(f"ego: loaded ego.mat velocity log with {log.n_samples} samples")
        return log
    except Exception as exc:  # noqa: BLE001
        notes.append(f"ego: failed to load ego.mat ({exc}); speed inferred from tracks")
        return None


def estimate_ego_state(
    timestamp_ns: int,
    tracks_df: pd.DataFrame,
    config: PlannerConfig,
    notes: list[str] | None = None,
    ego_log: Any | None = None,
) -> EgoState:
    """Estimate local ego state for one planning timestamp.

    Task 3 tracks and Task 5 predictions are ego-frame outputs, so the planner
    keeps ego position at the local origin and heading at zero. When available,
    ego.mat contributes speed only.
    """
    first_ts = int(tracks_df["timestamp_ns"].min()) if tracks_df is not None and not tracks_df.empty else timestamp_ns
    time_s = (int(timestamp_ns) - first_ts) / 1e9
    speed = config.default_ego_speed_mps
    if ego_log is not None:
        try:
            ego = ego_log.interpolate(int(timestamp_ns))
            speed = max(0.0, float(math.hypot(ego["vx"], ego["vy"])))
        except Exception as exc:  # noqa: BLE001
            if notes is not None:
                notes.append(f"ego: interpolation failed at {timestamp_ns}: {exc}")
    elif tracks_df is not None and not tracks_df.empty and "speed" in tracks_df.columns:
        near = tracks_df[tracks_df["timestamp_ns"] == timestamp_ns]
        median_speed = float(near["speed"].median()) if not near.empty else float(tracks_df["speed"].median())
        if np.isfinite(median_speed) and median_speed > 0.1:
            speed = float(np.clip(median_speed + 3.0, 4.0, 18.0))
    return EgoState(
        timestamp_ns=int(timestamp_ns),
        time_s=float(time_s),
        x=0.0,
        y=0.0,
        speed=float(speed),
        heading=0.0,
    )


def _estimate_ego_states(
    timestamps: list[int],
    tracks_df: pd.DataFrame,
    config: PlannerConfig,
    notes: list[str],
) -> dict[int, EgoState]:
    ego_log = _load_ego_log(notes)
    states = {
        int(ts): estimate_ego_state(int(ts), tracks_df, config, notes=notes, ego_log=ego_log)
        for ts in timestamps
    }
    prev: EgoState | None = None
    for ts in timestamps:
        state = states[int(ts)]
        if prev is not None:
            dt = max((state.timestamp_ns - prev.timestamp_ns) / 1e9, 1e-6)
            state.acceleration = float(np.clip((state.speed - prev.speed) / dt, -config.max_brake_mps2, config.max_accel_mps2))
        prev = state
    return states


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def generate_candidate_trajectories(
    ego: EgoState,
    config: PlannerConfig,
    target_track_id: int | None = None,
) -> list[CandidateTrajectory]:
    """Generate physically plausible local ego-frame manoeuvre candidates."""
    steps = max(1, int(round(config.horizon_s / config.dt_s)))
    times = np.arange(1, steps + 1, dtype=float) * config.dt_s
    strategies = {
        "keep_racing_line": (0.0, 0.0),
        "follow_opponent": (-1.2, 0.0),
        "brake": (-3.0, 0.0),
        "evasive_left": (-0.8, config.evasive_offset_m),
        "evasive_right": (-0.8, -config.evasive_offset_m),
        "overtake_left": (1.0, config.overtake_offset_m),
        "overtake_right": (1.0, -config.overtake_offset_m),
        "abort_overtake": (-2.0, 0.0),
        "emergency_brake": (-config.max_brake_mps2, 0.0),
    }
    candidates: list[CandidateTrajectory] = []
    tangent = np.array([math.cos(ego.heading), math.sin(ego.heading)])
    normal = np.array([-math.sin(ego.heading), math.cos(ego.heading)])

    for strategy, (accel, target_y) in strategies.items():
        accel = float(np.clip(accel, -config.max_brake_mps2, config.max_accel_mps2))
        speed = np.maximum(0.0, ego.speed + accel * times)
        longitudinal = ego.speed * times + 0.5 * accel * times * times
        longitudinal = np.maximum.accumulate(np.maximum(0.0, longitudinal))
        if strategy == "abort_overtake":
            target_y = 0.0
        lateral = ego.y + (target_y - ego.y) * _smoothstep(times / max(config.horizon_s, config.dt_s))
        dlateral_dt = np.gradient(lateral, config.dt_s, edge_order=1) if len(lateral) > 1 else np.array([0.0])
        pts = np.array([ego.x, ego.y]) + np.outer(longitudinal, tangent) + np.outer(lateral - ego.y, normal)
        heading = ego.heading + np.arctan2(dlateral_dt, np.maximum(speed, 0.1))
        side = "left" if target_y > 0.2 else "right" if target_y < -0.2 else ""
        df = pd.DataFrame(
            {
                "horizon_time": times,
                "step_index": np.arange(1, len(times) + 1),
                "planned_x": pts[:, 0],
                "planned_y": pts[:, 1],
                "planned_speed": speed,
                "planned_heading": heading,
            }
        )
        candidates.append(
            CandidateTrajectory(
                strategy=strategy,
                points=df,
                target_track_id=target_track_id,
                side=side,
            )
        )
    return candidates


def _nearest_rows_by_horizon(pred_df: pd.DataFrame, horizons: np.ndarray) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    pred_df = pred_df.reset_index(drop=True)
    rows: list[dict[str, float]] = []
    for h in horizons:
        idx = (pred_df["horizon_s"] - h).abs().idxmin()
        row = pred_df.iloc[int(idx)]
        rows.append(
            {
                "horizon_s": float(h),
                "pred_x": float(row["pred_x"]),
                "pred_y": float(row["pred_y"]),
            }
        )
    return pd.DataFrame(rows)


def _predictions_from_task5(
    predictions_df: pd.DataFrame,
    timestamp_ns: int,
    horizons: np.ndarray,
    tolerance_ns: int = 120_000_000,
) -> dict[int, pd.DataFrame]:
    if predictions_df is None or predictions_df.empty:
        return {}
    source_ts = predictions_df["source_ts_ns"].to_numpy(dtype=np.int64)
    if source_ts.size == 0:
        return {}
    nearest = int(source_ts[np.argmin(np.abs(source_ts - int(timestamp_ns)))])
    if abs(nearest - int(timestamp_ns)) > tolerance_ns:
        return {}
    current = predictions_df[predictions_df["source_ts_ns"] == nearest].copy()
    out: dict[int, pd.DataFrame] = {}
    for tid, grp in current.groupby("track_id", sort=False):
        agg = (
            grp.groupby("horizon_s", as_index=False)
            .agg(pred_x=("pred_x", "mean"), pred_y=("pred_y", "mean"))
            .sort_values("horizon_s")
        )
        out[int(tid)] = _nearest_rows_by_horizon(agg, horizons)[["horizon_s", "pred_x", "pred_y"]]
    return out


def _constant_velocity_predictions(
    tracks_at_time: pd.DataFrame,
    horizons: np.ndarray,
) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for _, row in tracks_at_time.iterrows():
        tid = int(row["track_id"])
        out[tid] = pd.DataFrame(
            {
                "horizon_s": horizons,
                "pred_x": float(row["x"]) + float(row["vx"]) * horizons,
                "pred_y": float(row["y"]) + float(row["vy"]) * horizons,
            }
        )
    return out


def predict_opponents_if_needed(
    tracks_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    timestamp_ns: int,
    config: PlannerConfig,
) -> dict[int, pd.DataFrame]:
    """Use Task 5 predictions where possible; fill gaps with CV predictions."""
    steps = max(1, int(round(config.horizon_s / config.dt_s)))
    horizons = np.arange(1, steps + 1, dtype=float) * config.dt_s
    tracks_at_time = tracks_df[tracks_df["timestamp_ns"] == int(timestamp_ns)].copy()
    task5 = _predictions_from_task5(predictions_df, int(timestamp_ns), horizons)
    fallback = _constant_velocity_predictions(tracks_at_time, horizons)
    merged = dict(fallback)
    merged.update(task5)
    return merged


def _behaviour_at_time(
    behaviour_df: pd.DataFrame,
    timestamp_ns: int,
    track_ids: list[int],
    tolerance_ns: int = 120_000_000,
) -> dict[int, tuple[str, float]]:
    out = {int(tid): ("unknown", 0.0) for tid in track_ids}
    if behaviour_df is None or behaviour_df.empty:
        return out
    for tid in track_ids:
        grp = behaviour_df[behaviour_df["track_id"] == int(tid)]
        if grp.empty:
            continue
        idx = (grp["timestamp_ns"] - int(timestamp_ns)).abs().idxmin()
        row = grp.loc[idx]
        if abs(int(row["timestamp_ns"]) - int(timestamp_ns)) <= tolerance_ns:
            label = str(row.get("predicted_behaviour", "unknown"))
            conf = float(row.get("confidence", 0.0))
            if label not in BEHAVIOUR_CODE:
                label = "unknown"
            out[int(tid)] = (label, conf)
    return out


def _closest_opponent(
    predictions: dict[int, pd.DataFrame],
    ego: EgoState,
) -> tuple[int | None, float]:
    best_tid: int | None = None
    best_dist = float("inf")
    for tid, pred in predictions.items():
        if pred.empty:
            continue
        first = pred.iloc[0]
        dist = math.hypot(float(first["pred_x"]) - ego.x, float(first["pred_y"]) - ego.y)
        if dist < best_dist:
            best_dist = dist
            best_tid = int(tid)
    return best_tid, best_dist


def _trajectory_smoothness(candidate: CandidateTrajectory, config: PlannerConfig) -> tuple[float, bool]:
    pts = candidate.points
    if len(pts) < 2:
        return 0.0, True
    speed = pts["planned_speed"].to_numpy(dtype=float)
    heading = np.unwrap(pts["planned_heading"].to_numpy(dtype=float))
    accel = np.gradient(speed, config.dt_s, edge_order=1)
    yaw_rate = np.gradient(heading, config.dt_s, edge_order=1)
    lat_accel = speed * yaw_rate
    smoothness = float(
        0.5 * np.nanmean(np.abs(accel)) / max(config.max_brake_mps2, 1e-6)
        + 0.5 * np.nanmean(np.abs(lat_accel)) / max(config.max_lateral_accel_mps2, 1e-6)
    )
    ok = bool(
        np.nanmax(np.abs(lat_accel)) <= config.max_lateral_accel_mps2 * 1.15
        and np.nanmax(np.abs(yaw_rate)) <= config.max_yaw_rate_radps * 1.25
    )
    return float(np.clip(smoothness, 0.0, 3.0)), ok


def score_candidate_trajectory(
    candidate: CandidateTrajectory,
    ego: EgoState,
    opponent_predictions: dict[int, pd.DataFrame],
    behaviour_by_track: dict[int, tuple[str, float]],
    config: PlannerConfig,
    corridor: TrackCorridor | None = None,
) -> CandidateScore:
    """Score one candidate with deterministic safety and racing costs."""
    half_width = corridor.half_width_m if corridor is not None else config.track_half_width_m
    pts = candidate.points
    horizons = pts["horizon_time"].to_numpy(dtype=float)
    min_dist = float("inf")
    closest_time = float(horizons[-1]) if len(horizons) else 0.0
    target_tid: int | None = candidate.target_track_id
    overlap_hits = 0
    comparisons = 0
    opponent_distance = float("inf")

    for tid, pred in opponent_predictions.items():
        if pred.empty:
            continue
        aligned = _nearest_rows_by_horizon(pred, horizons)
        pred_xy = aligned[["pred_x", "pred_y"]].to_numpy(dtype=float)
        plan_xy = pts[["planned_x", "planned_y"]].to_numpy(dtype=float)
        n = min(len(pred_xy), len(plan_xy))
        if n == 0:
            continue
        distances = np.linalg.norm(plan_xy[:n] - pred_xy[:n], axis=1)
        first_distance = float(np.linalg.norm(np.array([ego.x, ego.y]) - pred_xy[0]))
        if first_distance < opponent_distance:
            opponent_distance = first_distance
        local_min_idx = int(np.nanargmin(distances))
        local_min = float(distances[local_min_idx])
        if local_min < min_dist:
            min_dist = local_min
            closest_time = float(horizons[min(local_min_idx, len(horizons) - 1)])
            target_tid = int(tid)
        overlap_hits += int(np.sum(distances < config.min_safe_distance_m * 1.25))
        comparisons += n

    if not np.isfinite(min_dist):
        min_dist = 999.0
    if not np.isfinite(opponent_distance):
        opponent_distance = 999.0

    behaviour, behaviour_conf = behaviour_by_track.get(int(target_tid), ("unknown", 0.0)) if target_tid is not None else ("unknown", 0.0)
    behaviour_uncertainty = {
        "lane_keeping": 0.10,
        "braking": 0.30,
        "overtaking_attempt": 0.40,
        "trajectory_recovery": 0.75,
        "unknown": 0.85,
    }.get(behaviour, 0.85)
    behaviour_uncertainty = float(np.clip(behaviour_uncertainty + 0.45 * (1.0 - behaviour_conf), 0.0, 1.0))
    margin_boost = 1.2 * behaviour_uncertainty
    safe_distance = config.min_safe_distance_m + margin_boost
    collision_risk = _clip01((safe_distance - min_dist) / max(safe_distance - config.collision_distance_m, 1e-6))
    if min_dist <= config.collision_distance_m:
        collision_risk = 1.0

    abs_y = np.abs(pts["planned_y"].to_numpy(dtype=float))
    max_abs_y = float(np.nanmax(abs_y)) if len(abs_y) else 0.0
    min_boundary_clearance = half_width - max_abs_y
    boundary_risk = _clip01((config.boundary_margin_m - min_boundary_clearance) / max(config.boundary_margin_m, 1e-6))
    outside_boundary = min_boundary_clearance < 0.0

    racing_line_cost = float(np.nanmean(abs_y) / max(half_width, 1e-6)) if len(abs_y) else 0.0
    final_x = float(pts["planned_x"].iloc[-1]) if not pts.empty else 0.0
    nominal_progress = max(ego.speed * config.horizon_s, 1.0)
    progress_reward = float(np.clip(final_x / nominal_progress, 0.0, 1.5))
    smoothness_cost, smooth_ok = _trajectory_smoothness(candidate, config)
    lateral_offset = float(pts["planned_y"].iloc[-1] - ego.y) if not pts.empty else 0.0
    predicted_overlap = float(overlap_hits / max(comparisons, 1))

    behaviour_penalty = 0.0
    hard_unsafe_reason: list[str] = []
    if min_dist < safe_distance:
        hard_unsafe_reason.append("clearance below safety margin")
    if outside_boundary:
        hard_unsafe_reason.append("outside inferred track corridor")
    if not smooth_ok:
        hard_unsafe_reason.append("smoothness limit exceeded")
    if candidate.strategy.startswith("overtake"):
        if behaviour in {"unknown", "trajectory_recovery"} or behaviour_conf < 0.45:
            hard_unsafe_reason.append("overtake blocked by uncertain opponent behaviour")
        if min_dist < safe_distance + 1.5 or predicted_overlap > 0.30:
            hard_unsafe_reason.append("overtake path overlaps opponent prediction")
        if behaviour == "overtaking_attempt" and target_tid in opponent_predictions:
            pred = opponent_predictions[target_tid]
            if not pred.empty:
                lateral_trend = float(pred["pred_y"].iloc[-1] - pred["pred_y"].iloc[0])
                if candidate.strategy.endswith("left") and lateral_trend > 0.4:
                    hard_unsafe_reason.append("opponent is drifting toward left overtake side")
                if candidate.strategy.endswith("right") and lateral_trend < -0.4:
                    hard_unsafe_reason.append("opponent is drifting toward right overtake side")
    if behaviour == "braking" and candidate.strategy == "keep_racing_line":
        behaviour_penalty += 1.2
    if behaviour == "trajectory_recovery":
        behaviour_penalty += 0.8
    if behaviour == "unknown":
        behaviour_penalty += 0.6
    if candidate.strategy == "emergency_brake":
        behaviour_penalty -= 0.6 if collision_risk > 0.4 else 0.0
    if candidate.strategy in {"follow_opponent", "brake"} and min_dist < safe_distance + 3.0:
        behaviour_penalty -= 0.3
    clean_overtake_bonus = 0.0
    if (
        candidate.strategy.startswith("overtake")
        and not hard_unsafe_reason
        and behaviour in {"lane_keeping", "braking"}
        and opponent_distance < 18.0
        and min_dist >= safe_distance + 1.2
        and progress_reward >= 1.0
    ):
        extra_clearance = _clip01((min_dist - safe_distance - 1.2) / 4.0)
        clean_overtake_bonus = 0.95 + 0.35 * extra_clearance

    deterministic_cost = (
        8.0 * collision_risk
        + 4.0 * boundary_risk
        + 1.3 * racing_line_cost
        + 1.2 * smoothness_cost
        + 1.4 * behaviour_uncertainty
        + behaviour_penalty
        + 1.1 * predicted_overlap
        - 1.6 * progress_reward
        - clean_overtake_bonus
    )
    if outside_boundary:
        deterministic_cost += 6.0
    if not smooth_ok:
        deterministic_cost += 4.0

    is_safe = len(hard_unsafe_reason) == 0
    safety_margin = float(min_dist - safe_distance)
    explanation = (
        f"{candidate.strategy}: min clearance {min_dist:.2f} m at {closest_time:.2f} s; "
        f"opponent behaviour {behaviour} ({behaviour_conf:.2f}); "
        f"boundary clearance {min_boundary_clearance:.2f} m"
    )
    if hard_unsafe_reason:
        explanation += "; unsafe because " + ", ".join(hard_unsafe_reason)
    elif clean_overtake_bonus > 0.0:
        explanation += f"; clean overtake bonus {clean_overtake_bonus:.2f}"

    return CandidateScore(
        candidate_strategy=candidate.strategy,
        target_track_id=target_tid,
        min_distance_to_opponent=float(min_dist),
        time_to_closest_approach=float(closest_time),
        collision_risk_score=float(collision_risk),
        boundary_risk_score=float(boundary_risk),
        behaviour_uncertainty_score=float(behaviour_uncertainty),
        racing_line_cost=float(racing_line_cost),
        progress_reward=float(progress_reward),
        smoothness_cost=float(smoothness_cost),
        deterministic_cost=float(deterministic_cost),
        learned_risk_score=0.0,
        final_cost=float(deterministic_cost),
        is_safe=bool(is_safe),
        opponent_distance=float(opponent_distance),
        opponent_behaviour=behaviour,
        opponent_behaviour_confidence=float(behaviour_conf),
        lateral_offset=float(lateral_offset),
        predicted_path_overlap=float(predicted_overlap),
        safety_margin=safety_margin,
        explanation=explanation,
    )


FEATURE_COLUMNS = [
    "ego_speed",
    "ego_acceleration",
    "opponent_distance",
    "min_predicted_distance",
    "time_to_closest_approach",
    "collision_risk_score",
    "boundary_risk_score",
    "racing_line_cost",
    "behaviour_uncertainty_score",
    "opponent_behaviour_code",
    "candidate_strategy_code",
    "progress_reward",
    "smoothness_cost",
    "lateral_offset",
    "predicted_path_overlap",
]


def build_learning_dataset(risk_assessment: pd.DataFrame) -> pd.DataFrame:
    """Build pseudo-labelled training data from deterministic candidate scores."""
    if risk_assessment is None or risk_assessment.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_ns",
                "candidate_strategy",
                "target_track_id",
                *FEATURE_COLUMNS,
                "pseudo_label_safe",
                "pseudo_label_selected",
                "pseudo_label_risk_level",
                "pseudo_label_risk_score",
            ]
        )
    df = risk_assessment.copy()
    df["opponent_behaviour_code"] = df.get("opponent_behaviour", "unknown").map(BEHAVIOUR_CODE).fillna(4).astype(float)
    strategy_map = {name: i for i, name in enumerate(STRATEGIES)}
    df["candidate_strategy_code"] = df["candidate_strategy"].map(strategy_map).fillna(len(strategy_map)).astype(float)
    df["min_predicted_distance"] = df["min_distance_to_opponent"]
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["pseudo_label_safe"] = df["is_safe"].astype(bool).astype(int)
    df["pseudo_label_selected"] = 0
    for _, grp in df.groupby("timestamp_ns", sort=False):
        safe = grp[grp["is_safe"].astype(bool)]
        source = safe if not safe.empty else grp
        if not source.empty:
            idx = source["deterministic_cost"].astype(float).idxmin()
            df.loc[idx, "pseudo_label_selected"] = 1
    risk_score = (
        0.42 * df["collision_risk_score"].astype(float)
        + 0.18 * df["boundary_risk_score"].astype(float)
        + 0.16 * df["behaviour_uncertainty_score"].astype(float)
        + 0.10 * df["racing_line_cost"].astype(float).clip(0.0, 1.0)
        + 0.08 * df["smoothness_cost"].astype(float).clip(0.0, 1.0)
        + 0.06 * df["predicted_path_overlap"].astype(float).clip(0.0, 1.0)
    )
    risk_score += (~df["is_safe"].astype(bool)).astype(float) * 0.20
    df["pseudo_label_risk_score"] = risk_score.clip(0.0, 1.0)
    df["pseudo_label_risk_level"] = pd.cut(
        df["pseudo_label_risk_score"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["low", "medium", "high"],
    ).astype(str)
    cols = [
        "timestamp_ns",
        "candidate_strategy",
        "target_track_id",
        *FEATURE_COLUMNS,
        "pseudo_label_safe",
        "pseudo_label_selected",
        "pseudo_label_risk_level",
        "pseudo_label_risk_score",
    ]
    return df[cols]


def train_learning_risk_model(
    training_data: pd.DataFrame,
    train_learning: bool = True,
) -> LearningModel:
    """Train a lightweight risk regressor from pseudo-labels when feasible."""
    if not train_learning:
        return LearningModel(
            model=None,
            feature_columns=FEATURE_COLUMNS,
            model_type="deterministic_fallback",
            model_used=False,
            train_score=None,
            validation_score=None,
            fallback_reason="training disabled by caller",
            top_features=[],
        )
    if training_data is None or len(training_data) < 20:
        return LearningModel(
            model=None,
            feature_columns=FEATURE_COLUMNS,
            model_type="deterministic_fallback",
            model_used=False,
            train_score=None,
            validation_score=None,
            fallback_reason=f"insufficient training samples ({0 if training_data is None else len(training_data)})",
            top_features=[],
        )
    if training_data["pseudo_label_risk_score"].nunique() < 2:
        return LearningModel(
            model=None,
            feature_columns=FEATURE_COLUMNS,
            model_type="deterministic_fallback",
            model_used=False,
            train_score=None,
            validation_score=None,
            fallback_reason="pseudo-label risk score has no variation",
            top_features=[],
        )
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # noqa: BLE001
        return LearningModel(
            model=None,
            feature_columns=FEATURE_COLUMNS,
            model_type="deterministic_fallback",
            model_used=False,
            train_score=None,
            validation_score=None,
            fallback_reason=f"sklearn unavailable: {exc}",
            top_features=[],
        )

    x = training_data[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = training_data["pseudo_label_risk_score"].astype(float)
    if len(training_data) >= 40:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=7)
    else:
        x_train, x_val, y_train, y_val = x, x, y, y
    model = RandomForestRegressor(
        n_estimators=90,
        max_depth=6,
        min_samples_leaf=3,
        random_state=7,
    )
    model.fit(x_train, y_train)
    train_score = float(model.score(x_train, y_train))
    validation_score = float(model.score(x_val, y_val)) if len(x_val) > 1 else None
    importances = getattr(model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS)))
    top_features = sorted(
        [(name, float(value)) for name, value in zip(FEATURE_COLUMNS, importances)],
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    return LearningModel(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        model_type="RandomForestRegressor",
        model_used=True,
        train_score=train_score,
        validation_score=validation_score,
        fallback_reason="",
        top_features=top_features,
    )


def apply_learning_risk_model(
    risk_assessment: pd.DataFrame,
    model_info: LearningModel,
    config: PlannerConfig,
) -> pd.DataFrame:
    """Add learned risk and final cost while preserving hard safety flags."""
    if risk_assessment is None or risk_assessment.empty:
        return pd.DataFrame()
    out = risk_assessment.copy()
    if "opponent_behaviour_code" not in out.columns:
        out["opponent_behaviour_code"] = out.get("opponent_behaviour", "unknown").map(BEHAVIOUR_CODE).fillna(4).astype(float)
    strategy_map = {name: i for i, name in enumerate(STRATEGIES)}
    out["candidate_strategy_code"] = out["candidate_strategy"].map(strategy_map).fillna(len(strategy_map)).astype(float)
    out["min_predicted_distance"] = out["min_distance_to_opponent"]
    if model_info.model_used and model_info.model is not None:
        features = out[model_info.feature_columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["learned_risk_score"] = np.clip(model_info.model.predict(features), 0.0, 1.0)
        out["model_used"] = True
    else:
        out["learned_risk_score"] = 0.0
        out["model_used"] = False
    out["final_cost"] = out["deterministic_cost"].astype(float) + config.learning_alpha * out["learned_risk_score"].astype(float)
    return out


def select_strategy(risk_assessment: pd.DataFrame, config: PlannerConfig) -> pd.DataFrame:
    """Select one candidate per timestamp using hard safety and final cost."""
    if risk_assessment.empty:
        return pd.DataFrame()
    selected_rows: list[pd.Series] = []
    for _, grp in risk_assessment.groupby("timestamp_ns", sort=True):
        safe = grp[grp["is_safe"].astype(bool)]
        if not safe.empty:
            chosen = safe.loc[safe["final_cost"].astype(float).idxmin()].copy()
        else:
            emergency = grp[grp["candidate_strategy"] == "emergency_brake"]
            if not emergency.empty:
                chosen = emergency.iloc[0].copy()
            else:
                chosen = grp.loc[grp["min_distance_to_opponent"].astype(float).idxmax()].copy()
            chosen["explanation"] = str(chosen.get("explanation", "")) + "; no candidate satisfied all hard constraints"
        selected_rows.append(chosen)
    raw = pd.DataFrame(selected_rows)
    return smooth_decisions(risk_assessment, raw, config)


def smooth_decisions(
    risk_assessment: pd.DataFrame,
    selected: pd.DataFrame,
    config: PlannerConfig,
) -> pd.DataFrame:
    """Apply simple hysteresis to prevent frame-by-frame strategy flicker."""
    if selected.empty:
        return selected
    rows: list[pd.Series] = []
    prev_strategy: str | None = None
    for _, row in selected.sort_values("timestamp_ns").iterrows():
        current = row.copy()
        ts = int(current["timestamp_ns"])
        if prev_strategy is not None and prev_strategy != current["candidate_strategy"]:
            grp = risk_assessment[risk_assessment["timestamp_ns"] == ts]
            prev_rows = grp[(grp["candidate_strategy"] == prev_strategy) & (grp["is_safe"].astype(bool))]
            if not prev_rows.empty:
                prev_row = prev_rows.iloc[0].copy()
                best_cost = float(current["final_cost"])
                prev_cost = float(prev_row["final_cost"])
                allowed = best_cost + abs(best_cost) * config.hysteresis_cost_margin + 0.25
                if prev_cost <= allowed:
                    prev_row["explanation"] = str(prev_row.get("explanation", "")) + "; kept by hysteresis"
                    current = prev_row
        prev_strategy = str(current["candidate_strategy"])
        rows.append(current)
    return pd.DataFrame(rows).reset_index(drop=True)


def _score_to_row(
    score: CandidateScore,
    ego: EgoState,
    timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "timestamp_ns": int(timestamp_ns),
        "time_s": ego.time_s,
        "candidate_strategy": score.candidate_strategy,
        "target_track_id": "" if score.target_track_id is None else int(score.target_track_id),
        "ego_speed": ego.speed,
        "ego_acceleration": ego.acceleration,
        "opponent_distance": score.opponent_distance,
        "min_distance_to_opponent": score.min_distance_to_opponent,
        "time_to_closest_approach": score.time_to_closest_approach,
        "collision_risk_score": score.collision_risk_score,
        "boundary_risk_score": score.boundary_risk_score,
        "behaviour_uncertainty_score": score.behaviour_uncertainty_score,
        "racing_line_cost": score.racing_line_cost,
        "progress_reward": score.progress_reward,
        "smoothness_cost": score.smoothness_cost,
        "deterministic_cost": score.deterministic_cost,
        "learned_risk_score": score.learned_risk_score,
        "final_cost": score.final_cost,
        "is_safe": score.is_safe,
        "opponent_behaviour": score.opponent_behaviour,
        "opponent_behaviour_confidence": score.opponent_behaviour_confidence,
        "lateral_offset": score.lateral_offset,
        "predicted_path_overlap": score.predicted_path_overlap,
        "safety_margin": score.safety_margin,
        "explanation": score.explanation,
    }


def _trajectory_rows(
    timestamp_ns: int,
    candidate: CandidateTrajectory,
    selected_strategy: str,
    is_selected: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plan_id = f"{int(timestamp_ns)}_{candidate.strategy}"
    for _, pt in candidate.points.iterrows():
        rows.append(
            {
                "timestamp_ns": int(timestamp_ns),
                "plan_id": plan_id,
                "selected_strategy": candidate.strategy,
                "horizon_time": float(pt["horizon_time"]),
                "step_index": int(pt["step_index"]),
                "planned_x": float(pt["planned_x"]),
                "planned_y": float(pt["planned_y"]),
                "planned_speed": float(pt["planned_speed"]),
                "planned_heading": float(pt["planned_heading"]),
                "is_selected": bool(is_selected and candidate.strategy == selected_strategy),
            }
        )
    return rows


def _build_decisions(
    selected: pd.DataFrame,
    ego_states: dict[int, EgoState],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        ts = int(row["timestamp_ns"])
        ego = ego_states[ts]
        strategy = str(row["candidate_strategy"])
        side = "left" if strategy.endswith("left") else "right" if strategy.endswith("right") else ""
        confidence = float(
            np.clip(
                0.35
                + 0.35 * bool(row["is_safe"])
                + 0.20 * (1.0 - float(row["collision_risk_score"]))
                + 0.10 * (1.0 - float(row["behaviour_uncertainty_score"])),
                0.0,
                1.0,
            )
        )
        rows.append(
            {
                "timestamp_ns": ts,
                "time_s": ego.time_s,
                "ego_x": ego.x,
                "ego_y": ego.y,
                "ego_speed": ego.speed,
                "ego_heading": ego.heading,
                "selected_strategy": strategy,
                "selected_side": side,
                "target_track_id": row["target_track_id"],
                "confidence": confidence,
                "safety_margin": float(row["safety_margin"]),
                "min_predicted_distance": float(row["min_distance_to_opponent"]),
                "collision_risk_score": float(row["collision_risk_score"]),
                "boundary_risk_score": float(row["boundary_risk_score"]),
                "behaviour_uncertainty_score": float(row["behaviour_uncertainty_score"]),
                "deterministic_cost": float(row["deterministic_cost"]),
                "learned_risk_score": float(row["learned_risk_score"]),
                "final_cost": float(row["final_cost"]),
                "explanation": str(row["explanation"]),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    decisions: pd.DataFrame,
    risk_assessment: pd.DataFrame,
    model_info: LearningModel,
    notes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = int(len(decisions))
    unsafe_frames = int((decisions["safety_margin"] < 0.0).sum()) if not decisions.empty else 0
    rows.extend(
        [
            {"metric": "number_of_planning_frames", "strategy": "", "value": n, "percentage": "", "notes": ""},
            {
                "metric": "mean_safety_margin",
                "strategy": "",
                "value": float(decisions["safety_margin"].mean()) if n else np.nan,
                "percentage": "",
                "notes": "m",
            },
            {
                "metric": "minimum_safety_margin",
                "strategy": "",
                "value": float(decisions["safety_margin"].min()) if n else np.nan,
                "percentage": "",
                "notes": "m",
            },
            {"metric": "number_of_unsafe_frames", "strategy": "", "value": unsafe_frames, "percentage": "", "notes": ""},
            {
                "metric": "number_of_overtaking_attempts",
                "strategy": "",
                "value": int(decisions["selected_strategy"].str.startswith("overtake").sum()) if n else 0,
                "percentage": "",
                "notes": "",
            },
            {
                "metric": "number_of_aborted_overtakes",
                "strategy": "",
                "value": int((decisions["selected_strategy"] == "abort_overtake").sum()) if n else 0,
                "percentage": "",
                "notes": "",
            },
            {
                "metric": "number_of_emergency_brakes",
                "strategy": "",
                "value": int((decisions["selected_strategy"] == "emergency_brake").sum()) if n else 0,
                "percentage": "",
                "notes": "",
            },
            {
                "metric": "learning_based_model_used",
                "strategy": "",
                "value": bool(model_info.model_used),
                "percentage": "",
                "notes": model_info.model_type,
            },
            {
                "metric": "notes_about_missing_inputs",
                "strategy": "",
                "value": "",
                "percentage": "",
                "notes": " | ".join(notes),
            },
        ]
    )
    if n:
        counts = decisions["selected_strategy"].value_counts()
        for strategy, count in counts.items():
            rows.append(
                {
                    "metric": "selected_strategy_count",
                    "strategy": strategy,
                    "value": int(count),
                    "percentage": 100.0 * float(count) / max(n, 1),
                    "notes": "",
                }
            )
    if not risk_assessment.empty:
        rows.append(
            {
                "metric": "unsafe_candidate_fraction",
                "strategy": "",
                "value": float((~risk_assessment["is_safe"].astype(bool)).mean()),
                "percentage": "",
                "notes": "candidate-level",
            }
        )
    return pd.DataFrame(rows)


def _build_model_summary(model_info: LearningModel, n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_type": model_info.model_type,
                "number_of_training_samples": int(n_samples),
                "number_of_features": len(model_info.feature_columns),
                "pseudo_label_type": "normalized deterministic risk score",
                "train_score": "" if model_info.train_score is None else model_info.train_score,
                "validation_score": "" if model_info.validation_score is None else model_info.validation_score,
                "fallback_reason": model_info.fallback_reason,
                "top_features": "; ".join(f"{name}:{value:.3f}" for name, value in model_info.top_features),
            }
        ]
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        else:
            text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else str(v).replace("|", "\\|"))
    columns = list(text_df.columns)
    rows = text_df.astype(str).values.tolist()
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(columns)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    return "\n".join(
        [
            fmt([str(c) for c in columns]),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(fmt(row) for row in rows),
        ]
    )


def write_report(
    output_dir: Path,
    notes: list[str],
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    config: PlannerConfig,
) -> None:
    lines = [
        "# Task 7 - Overtake Strategy / Planning",
        "",
        "## Purpose",
        "",
        "Task 7 plans short-horizon ego manoeuvres that avoid predicted opponents, decide when an overtake is allowed, and explain the chosen racing strategy.",
        "",
        "## Inputs Used",
        "",
        "- Task 3 `tracks.csv`: opponent state, velocity, confidence, and track IDs.",
        "- Task 5 `predictions.csv`: future opponent trajectories when source timestamps match.",
        "- Task 6 `behaviour_predictions.csv`: behaviour labels and confidence for behaviour-aware margins.",
        "- Task 4 `track_boundaries.geojson`: loaded for map context; local planning uses an ego-frame corridor inferred from tracks because Task 3 outputs are ego-frame.",
        "- `data/ego.mat` when available: ego speed estimate only; planning coordinates remain ego-frame.",
        "",
        "## Planner Architecture",
        "",
        "The innovation is a hybrid architecture: a deterministic safety-constrained receding-horizon planner generates candidate manoeuvres, while a lightweight learned risk model ranks them using features from tracking, mapping, trajectory prediction, and behaviour prediction. This gives data-driven adaptation without sacrificing safety or explainability.",
        "",
        "This is not an end-to-end black-box planner. The learned model never overrides hard constraints; unsafe candidates are rejected before selection unless every candidate is unsafe, in which case the planner falls back to emergency braking or maximum-clearance evasive behaviour.",
        "",
        "## Candidate Trajectories",
        "",
        "At each timestamp the planner samples `keep_racing_line`, `follow_opponent`, `brake`, `evasive_left`, `evasive_right`, `overtake_left`, `overtake_right`, `abort_overtake`, and `emergency_brake`. Candidates use constant acceleration plus smooth lateral offset profiles over the receding horizon.",
        "",
        "## Risk And Cost Function",
        "",
        "- Collision risk from minimum distance to predicted opponent paths.",
        "- Boundary risk from distance to the inferred racing corridor.",
        "- Racing-line cost from lateral deviation.",
        "- Progress reward for forward motion.",
        "- Clean-overtake bonus when a side pass is already safe, improves clearance, and preserves progress.",
        "- Smoothness cost from acceleration, lateral acceleration, and yaw-rate proxies.",
        "- Behaviour-aware penalties for braking, overtaking attempts, recovery, or unknown behaviour.",
        "",
        "## Behaviour-Aware Logic",
        "",
        "- Braking opponents increase front-collision cost and favor braking/following unless a side path is clear.",
        "- Overtaking opponents block the side they are predicted to move toward.",
        "- Trajectory recovery and unknown labels increase uncertainty margins and block overtake attempts.",
        "- Lane keeping uses normal margins.",
        "",
        "## Learning-Based Innovation Module",
        "",
        "The training table is built from every candidate trajectory. Pseudo-labels mark safe/unsafe, deterministic selected/not selected, low/medium/high risk, and a normalized risk score. A RandomForestRegressor is trained when enough data and scikit-learn are available. If training is unavailable, the deterministic planner is used alone and the fallback reason is written to `learning_model_summary.csv`.",
        "",
        "## Outputs",
        "",
        "- `planning_decisions.csv`",
        "- `planned_trajectories.csv`",
        "- `risk_assessment.csv`",
        "- `planning_summary.csv`",
        "- `strategy_timeline.png`",
        "- `planning_bev.png`",
        "- `overtake_scenarios.png`",
        "- `learned_risk_training_data.csv`",
        "- `learned_risk_scores.csv`",
        "- `learning_model_summary.csv`",
        "- `learning_feature_importance.png` when feature importance is available.",
        "",
        "## Limitations",
        "",
        "- Task 3 and Task 5 are ego-frame planning inputs, while Task 4 boundaries are world-frame. The planner therefore uses a conservative local corridor and draws boundaries as context rather than hard geometry.",
        "- Pseudo-label learning is useful for strategy ranking, but it is not a substitute for labelled planning data.",
        "- Candidate generation is deliberately lightweight and does not solve a full nonlinear optimal-control problem.",
        "",
        "## How To Run",
        "",
        "```bash",
        "python src/task7_overtake_planning.py --tracks outputs/task3/tracks.csv --predictions outputs/task5/predictions.csv --behaviour outputs/task6/behaviour_predictions.csv --map-dir outputs/task4 --output-dir outputs/task7",
        "```",
        "",
        "The default command also works:",
        "",
        "```bash",
        "python src/task7_overtake_planning.py",
        "```",
        "",
        "## Configuration Snapshot",
        "",
        f"- Horizon: {config.horizon_s:.2f} s",
        f"- Step: {config.dt_s:.2f} s",
        f"- Minimum safe distance: {config.min_safe_distance_m:.2f} m",
        f"- Track half-width: {config.track_half_width_m:.2f} m before inference",
        "",
        "## Run Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Planning Summary", "", _dataframe_to_markdown(summary.head(25)), ""])
    lines.extend(["## Learning Model Summary", "", _dataframe_to_markdown(model_summary), ""])
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_corridor(ax: plt.Axes, corridor: TrackCorridor) -> None:
    hw = corridor.half_width_m
    ax.axhline(hw, color="#9E9E9E", lw=1.0, ls="--", label="corridor edge")
    ax.axhline(-hw, color="#9E9E9E", lw=1.0, ls="--")
    ax.axhline(0.0, color="#212121", lw=0.8, alpha=0.4, label="racing line")


def render_strategy_timeline(decisions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if decisions.empty:
        ax.text(0.5, 0.5, "No planning decisions", ha="center", va="center")
        ax.axis("off")
    else:
        order = STRATEGIES
        ymap = {name: i for i, name in enumerate(order)}
        colors = [STRATEGY_COLORS.get(s, "#6D6D6D") for s in decisions["selected_strategy"]]
        ax.scatter(decisions["time_s"], decisions["selected_strategy"].map(ymap), c=colors, s=28)
        ax.step(decisions["time_s"], decisions["selected_strategy"].map(ymap), where="post", color="#37474F", lw=1.0, alpha=0.55)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("selected strategy")
        ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Task 7 - Selected Strategy Timeline")
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_planning_bev(
    decisions: pd.DataFrame,
    trajectories: pd.DataFrame,
    tracks_df: pd.DataFrame,
    corridor: TrackCorridor,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    _plot_corridor(ax, corridor)
    if tracks_df is not None and not tracks_df.empty:
        for tid, grp in tracks_df.groupby("track_id"):
            ax.plot(grp["x"], grp["y"], lw=0.8, alpha=0.45, color="#78909C")
            last = grp.iloc[-1]
            ax.scatter(last["x"], last["y"], s=14, color="#455A64")
            ax.text(last["x"], last["y"], str(int(tid)), fontsize=6, color="#263238")
    selected = trajectories[trajectories["is_selected"].astype(bool)] if not trajectories.empty else pd.DataFrame()
    if not selected.empty:
        sample_ts = set(decisions["timestamp_ns"].iloc[:: max(1, len(decisions) // 40)].astype(int))
        selected = selected[selected["timestamp_ns"].astype(int).isin(sample_ts)]
        for plan_id, grp in selected.groupby("plan_id"):
            strategy = str(grp["selected_strategy"].iloc[0])
            ax.plot(
                grp["planned_x"],
                grp["planned_y"],
                color=STRATEGY_COLORS.get(strategy, "#6D6D6D"),
                lw=1.8,
                alpha=0.75,
                label=strategy,
            )
    ax.scatter([0.0], [0.0], marker="^", s=90, color="#111111", label="ego")
    ax.set_xlabel("x ego-frame forward (m)")
    ax.set_ylabel("y ego-frame left (m)")
    ax.set_title("Task 7 - Planning BEV")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "planning_bev.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_overtake_scenarios(
    decisions: pd.DataFrame,
    trajectories: pd.DataFrame,
    tracks_df: pd.DataFrame,
    predictions_by_ts: dict[int, dict[int, pd.DataFrame]],
    corridor: TrackCorridor,
    output_dir: Path,
) -> None:
    if decisions.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No scenarios available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_dir / "overtake_scenarios.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return
    preferred = [
        "follow_opponent",
        "brake",
        "evasive_left",
        "evasive_right",
        "overtake_left",
        "overtake_right",
        "abort_overtake",
        "emergency_brake",
        "keep_racing_line",
    ]
    chosen_indices: list[int] = []
    for strategy in preferred:
        matches = decisions.index[decisions["selected_strategy"] == strategy].tolist()
        if matches:
            chosen_indices.append(matches[0])
        if len(chosen_indices) >= 6:
            break
    if not chosen_indices:
        chosen_indices = decisions.index[: min(6, len(decisions))].tolist()
    n = len(chosen_indices)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 4.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), chosen_indices):
        ax.axis("on")
        row = decisions.loc[idx]
        ts = int(row["timestamp_ns"])
        strategy = str(row["selected_strategy"])
        _plot_corridor(ax, corridor)
        current_tracks = tracks_df[tracks_df["timestamp_ns"] == ts]
        if not current_tracks.empty:
            ax.scatter(current_tracks["x"], current_tracks["y"], c="#455A64", s=25, label="opponent")
            for _, tr in current_tracks.iterrows():
                ax.text(tr["x"], tr["y"], str(int(tr["track_id"])), fontsize=7)
        for tid, pred in predictions_by_ts.get(ts, {}).items():
            ax.plot(pred["pred_x"], pred["pred_y"], color="#D84315", lw=1.0, alpha=0.55)
        selected = trajectories[
            (trajectories["timestamp_ns"].astype(int) == ts)
            & (trajectories["selected_strategy"] == strategy)
            & (trajectories["is_selected"].astype(bool))
        ]
        if not selected.empty:
            ax.plot(
                selected["planned_x"],
                selected["planned_y"],
                color=STRATEGY_COLORS.get(strategy, "#1565C0"),
                lw=2.4,
                label=strategy,
            )
        ax.scatter([0.0], [0.0], marker="^", color="#111111", s=70)
        ax.set_title(f"t={float(row['time_s']):.2f}s - {strategy}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Task 7 - Representative Planning Scenarios")
    fig.tight_layout()
    fig.savefig(output_dir / "overtake_scenarios.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def render_learning_feature_importance(model_info: LearningModel, output_dir: Path) -> None:
    if not model_info.top_features:
        return
    names = [name for name, _ in model_info.top_features][::-1]
    values = [value for _, value in model_info.top_features][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color="#1565C0")
    ax.set_xlabel("importance")
    ax.set_title("Task 7 - Learned Risk Feature Importance")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(
    output_dir: Path,
    decisions: pd.DataFrame,
    trajectories: pd.DataFrame,
    tracks_df: pd.DataFrame,
    predictions_by_ts: dict[int, dict[int, pd.DataFrame]],
    corridor: TrackCorridor,
    model_info: LearningModel,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_strategy_timeline(decisions, output_dir)
    render_planning_bev(decisions, trajectories, tracks_df, corridor, output_dir)
    render_overtake_scenarios(decisions, trajectories, tracks_df, predictions_by_ts, corridor, output_dir)
    render_learning_feature_importance(model_info, output_dir)


def write_outputs(
    output_dir: Path,
    decisions: pd.DataFrame,
    trajectories: pd.DataFrame,
    risk_assessment: pd.DataFrame,
    summary: pd.DataFrame,
    training_data: pd.DataFrame,
    learned_scores: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_dir / "planning_decisions.csv", index=False)
    trajectories.to_csv(output_dir / "planned_trajectories.csv", index=False)
    risk_assessment.to_csv(output_dir / "risk_assessment.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    training_data.to_csv(output_dir / "learned_risk_training_data.csv", index=False)
    learned_scores.to_csv(output_dir / "learned_risk_scores.csv", index=False)
    model_summary.to_csv(output_dir / "learning_model_summary.csv", index=False)


def _learned_scores_table(risk_assessment: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if risk_assessment.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_ns",
                "candidate_strategy",
                "deterministic_cost",
                "learned_risk_score",
                "final_cost",
                "model_used",
                "is_safe",
                "was_selected",
            ]
        )
    out = risk_assessment[
        [
            "timestamp_ns",
            "candidate_strategy",
            "deterministic_cost",
            "learned_risk_score",
            "final_cost",
            "model_used",
            "is_safe",
        ]
    ].copy()
    selected_pairs = {
        (int(row["timestamp_ns"]), str(row["candidate_strategy"]))
        for _, row in selected.iterrows()
    }
    out["was_selected"] = [
        (int(row["timestamp_ns"]), str(row["candidate_strategy"])) in selected_pairs
        for _, row in out.iterrows()
    ]
    return out


def run_overtake_planning(
    tracks_path: Path | str = Path("outputs/task3/tracks.csv"),
    predictions_path: Path | str | None = Path("outputs/task5/predictions.csv"),
    behaviour_path: Path | str | None = Path("outputs/task6/behaviour_predictions.csv"),
    map_dir: Path | str | None = Path("outputs/task4"),
    output_dir: Path | str = Path("outputs/task7"),
    horizon_s: float = 2.0,
    dt_s: float = 0.25,
    max_frames: int | None = None,
    train_learning: bool = True,
) -> Task7Result:
    """Run Task 7 end-to-end and write all required challenge artifacts."""
    output_dir = Path(output_dir)
    config = PlannerConfig(horizon_s=horizon_s, dt_s=dt_s, max_frames=max_frames)
    tracks_df, predictions_df, behaviour_df, notes = load_inputs(
        tracks_path=tracks_path,
        predictions_path=predictions_path,
        behaviour_path=behaviour_path,
    )
    corridor = load_or_infer_track_corridor(tracks_df, map_dir, config, notes)
    config.track_half_width_m = corridor.half_width_m

    if tracks_df.empty:
        timestamps = [0]
    else:
        timestamps = sorted(int(ts) for ts in tracks_df["timestamp_ns"].dropna().unique())
    if max_frames is not None and max_frames > 0:
        timestamps = timestamps[:max_frames]
        notes.append(f"planning: limited to first {len(timestamps)} frames by --max-frames")
    if not timestamps:
        timestamps = [0]

    ego_states = _estimate_ego_states(timestamps, tracks_df, config, notes)
    risk_rows: list[dict[str, Any]] = []
    candidate_by_key: dict[tuple[int, str], CandidateTrajectory] = {}
    predictions_by_ts: dict[int, dict[int, pd.DataFrame]] = {}

    for ts in timestamps:
        ego = ego_states[int(ts)]
        predictions = predict_opponents_if_needed(tracks_df, predictions_df, int(ts), config)
        predictions_by_ts[int(ts)] = predictions
        closest_tid, _ = _closest_opponent(predictions, ego)
        behaviour = _behaviour_at_time(behaviour_df, int(ts), list(predictions.keys()))
        candidates = generate_candidate_trajectories(ego, config, target_track_id=closest_tid)
        for candidate in candidates:
            score = score_candidate_trajectory(
                candidate,
                ego,
                predictions,
                behaviour_by_track=behaviour,
                config=config,
                corridor=corridor,
            )
            risk_rows.append(_score_to_row(score, ego, int(ts)))
            candidate_by_key[(int(ts), candidate.strategy)] = candidate

    risk_assessment = pd.DataFrame(risk_rows)
    training_data = build_learning_dataset(risk_assessment)
    model_info = train_learning_risk_model(training_data, train_learning=train_learning)
    risk_assessment = apply_learning_risk_model(risk_assessment, model_info, config)
    selected = select_strategy(risk_assessment, config)
    decisions = _build_decisions(selected, ego_states)

    selected_pairs = {
        (int(row["timestamp_ns"]), str(row["candidate_strategy"]))
        for _, row in selected.iterrows()
    }
    traj_rows: list[dict[str, Any]] = []
    for key, candidate in candidate_by_key.items():
        ts, strategy = key
        selected_strategy = strategy
        traj_rows.extend(
            _trajectory_rows(
                ts,
                candidate,
                selected_strategy=selected_strategy,
                is_selected=key in selected_pairs,
            )
        )
    planned_trajectories = pd.DataFrame(traj_rows)
    learned_scores = _learned_scores_table(risk_assessment, selected)
    summary = _build_summary(decisions, risk_assessment, model_info, notes)
    model_summary = _build_model_summary(model_info, len(training_data))

    write_outputs(
        output_dir,
        decisions,
        planned_trajectories,
        risk_assessment,
        summary,
        training_data,
        learned_scores,
        model_summary,
    )
    create_visualizations(
        output_dir,
        decisions,
        planned_trajectories,
        tracks_df,
        predictions_by_ts,
        corridor,
        model_info,
    )
    write_report(output_dir, notes, summary, model_summary, config)

    return Task7Result(
        decisions=decisions,
        planned_trajectories=planned_trajectories,
        risk_assessment=risk_assessment,
        summary=summary,
        learning_training_data=training_data,
        learned_scores=learned_scores,
        learning_model_summary=model_summary,
        notes=notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 7: Overtake Strategy / Planning.")
    parser.add_argument("--tracks", default="outputs/task3/tracks.csv")
    parser.add_argument("--predictions", default="outputs/task5/predictions.csv")
    parser.add_argument("--behaviour", default="outputs/task6/behaviour_predictions.csv")
    parser.add_argument("--map-dir", default="outputs/task4")
    parser.add_argument("--output-dir", default="outputs/task7")
    parser.add_argument("--horizon-s", type=float, default=2.0)
    parser.add_argument("--dt-s", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-learning", action="store_true", help="Disable pseudo-label risk model training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Task 7: Overtake Strategy / Planning")
    print(f"  tracks      : {args.tracks}")
    print(f"  predictions : {args.predictions}")
    print(f"  behaviour   : {args.behaviour}")
    print(f"  map_dir     : {args.map_dir}")
    print(f"  output_dir  : {args.output_dir}")
    result = run_overtake_planning(
        tracks_path=args.tracks,
        predictions_path=args.predictions,
        behaviour_path=args.behaviour,
        map_dir=args.map_dir,
        output_dir=args.output_dir,
        horizon_s=args.horizon_s,
        dt_s=args.dt_s,
        max_frames=args.max_frames,
        train_learning=not args.no_learning,
    )
    print("\nPlanning summary:")
    visible = result.summary[result.summary["metric"].isin(["number_of_planning_frames", "mean_safety_margin", "minimum_safety_margin", "number_of_unsafe_frames", "learning_based_model_used"])]
    print(visible.to_string(index=False))
    print(f"\nTask 7 complete. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
