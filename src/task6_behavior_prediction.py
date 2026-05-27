"""Task 6: rule-based opponent behaviour prediction.

The challenge data does not include labelled behaviour annotations, so this
module builds explainable temporal features from Task 3 tracks and applies a
robust rule classifier. Task 5 predictions and Task 4 boundaries are used as
supporting evidence when available, but the pipeline still runs with tracks
alone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BEHAVIOURS = [
    "lane_keeping",
    "braking",
    "overtaking_attempt",
    "trajectory_recovery",
    "unknown",
]

BEHAVIOUR_COLORS = {
    "lane_keeping": "#2E7D32",
    "braking": "#C62828",
    "overtaking_attempt": "#1565C0",
    "trajectory_recovery": "#F9A825",
    "unknown": "#6D6D6D",
}


@dataclass
class Task6Result:
    """In-memory result returned by :func:`run_behavior_prediction`."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    segments: pd.DataFrame
    notes: list[str]


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _wrap_angle(angle: np.ndarray | pd.Series | float) -> np.ndarray | float:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _ramp(value: pd.Series | np.ndarray | float, start: float, full: float) -> pd.Series:
    values = pd.Series(value) if not isinstance(value, pd.Series) else value
    if math.isclose(full, start):
        return (values >= full).astype(float)
    return ((values - start) / (full - start)).clip(0.0, 1.0)


def _group_derivative(df: pd.DataFrame, value_col: str) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, grp in df.groupby("track_id", sort=False):
        values = grp[value_col].astype(float)
        dt = grp["time_s"].diff()
        deriv = values.diff() / dt.replace(0.0, np.nan)
        deriv = deriv.replace([np.inf, -np.inf], np.nan)
        deriv = deriv.bfill().ffill().fillna(0.0)
        pieces.append(deriv)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _unwrap_heading_by_track(df: pd.DataFrame, heading_col: str) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, grp in df.groupby("track_id", sort=False):
        unwrapped = np.unwrap(grp[heading_col].astype(float).to_numpy())
        pieces.append(pd.Series(unwrapped, index=grp.index))
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _infer_track_axis(grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start = grp[["x", "y"]].iloc[0].to_numpy(dtype=float)
    end = grp[["x", "y"]].iloc[-1].to_numpy(dtype=float)
    axis = end - start
    norm = float(np.linalg.norm(axis))
    if norm < 0.5:
        vel = grp[["vx", "vy"]].median().to_numpy(dtype=float)
        norm = float(np.linalg.norm(vel))
        axis = vel if norm >= 0.1 else np.array([1.0, 0.0], dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    return start, axis


def _prediction_evidence(
    features: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    prediction_errors_df: pd.DataFrame | None,
    notes: dict[str, Any],
) -> pd.DataFrame:
    out = features.copy()
    out["future_lateral_shift"] = 0.0
    out["prediction_disagreement"] = 0.0
    out["prediction_error_fde_m"] = np.nan
    out["prediction_error_lateral_m"] = np.nan
    out["prediction_available"] = False

    if predictions_df is None or predictions_df.empty:
        notes["predictions_used"] = False
        return out

    required = {"track_id", "source_ts_ns", "pred_x", "pred_y"}
    if not required.issubset(predictions_df.columns):
        notes["predictions_used"] = False
        notes["prediction_note"] = "predictions.csv missing source_ts_ns/pred_x/pred_y columns"
        return out

    pred = predictions_df.copy()
    pred["track_id"] = pd.to_numeric(pred["track_id"], errors="coerce").astype("Int64")
    pred["source_ts_ns"] = pd.to_numeric(pred["source_ts_ns"], errors="coerce").astype("Int64")
    pred = pred.dropna(subset=["track_id", "source_ts_ns", "pred_x", "pred_y"])
    if pred.empty:
        notes["predictions_used"] = False
        return out

    current = out[["track_id", "timestamp_ns", "x", "y", "heading"]].rename(
        columns={"timestamp_ns": "source_ts_ns"}
    )
    merged = pred.merge(current, on=["track_id", "source_ts_ns"], how="inner")
    if merged.empty:
        notes["predictions_used"] = False
        notes["prediction_note"] = "prediction source timestamps did not match track timestamps"
        return out

    notes["prediction_note"] = (
        "used Task 5 model disagreement; pred_x/pred_y are future ego-frame states, "
        "so raw current-vs-future lateral subtraction is not used"
    )

    if {"horizon_s", "model"}.issubset(merged.columns):
        spread_rows: list[dict[str, float | int]] = []
        for keys, grp in merged.groupby(["track_id", "source_ts_ns", "horizon_s"]):
            if grp["model"].nunique() < 2:
                continue
            dx = float(grp["pred_x"].max() - grp["pred_x"].min())
            dy = float(grp["pred_y"].max() - grp["pred_y"].min())
            spread_rows.append(
                {
                    "track_id": int(keys[0]),
                    "source_ts_ns": int(keys[1]),
                    "prediction_disagreement": math.hypot(dx, dy),
                }
            )
        if spread_rows:
            disagreement = (
                pd.DataFrame(spread_rows)
                .groupby(["track_id", "source_ts_ns"])["prediction_disagreement"]
                .max()
                .reset_index()
            )
        else:
            disagreement = pd.DataFrame(
                columns=["track_id", "source_ts_ns", "prediction_disagreement"]
            )
    else:
        disagreement = pd.DataFrame(
            columns=["track_id", "source_ts_ns", "prediction_disagreement"]
        )

    if not disagreement.empty:
        out = out.merge(
            disagreement,
            left_on=["track_id", "timestamp_ns"],
            right_on=["track_id", "source_ts_ns"],
            how="left",
            suffixes=("", "_pred"),
        )
        if "prediction_disagreement_pred" in out.columns:
            out["prediction_disagreement"] = out["prediction_disagreement_pred"].fillna(0.0)
            out.drop(columns=["prediction_disagreement_pred"], inplace=True)
        out.drop(columns=[c for c in ["source_ts_ns"] if c in out.columns], inplace=True)

    out["prediction_available"] = out["prediction_disagreement"].abs() > 0.0
    notes["predictions_used"] = True

    if prediction_errors_df is not None and not prediction_errors_df.empty:
        error_required = {"track_id", "source_ts_ns"}
        if error_required.issubset(prediction_errors_df.columns):
            err = prediction_errors_df.copy()
            value_cols = []
            if "fde_m" in err.columns:
                err["prediction_error_fde_m"] = pd.to_numeric(err["fde_m"], errors="coerce")
                value_cols.append("prediction_error_fde_m")
            if "lateral_m" in err.columns:
                err["prediction_error_lateral_m"] = pd.to_numeric(
                    err["lateral_m"], errors="coerce"
                ).abs()
                value_cols.append("prediction_error_lateral_m")
            if value_cols:
                agg = (
                    err.groupby(["track_id", "source_ts_ns"])[value_cols]
                    .mean()
                    .reset_index()
                )
                out = out.merge(
                    agg,
                    left_on=["track_id", "timestamp_ns"],
                    right_on=["track_id", "source_ts_ns"],
                    how="left",
                    suffixes=("", "_err"),
                )
                for col in value_cols:
                    err_col = f"{col}_err"
                    if err_col in out.columns:
                        out[col] = out[err_col]
                        out.drop(columns=[err_col], inplace=True)
                out.drop(columns=[c for c in ["source_ts_ns"] if c in out.columns], inplace=True)
                notes["prediction_errors_used"] = True
    return out


def prepare_track_features(
    tracks_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    prediction_errors_df: pd.DataFrame | None = None,
    window_s: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize track columns and compute sliding-window behaviour features."""
    if tracks_df.empty:
        raise ValueError("tracks.csv is empty")

    notes: dict[str, Any] = {}
    df = tracks_df.copy()

    track_col = _first_existing(df.columns, ["track_id", "id", "object_id"])
    if track_col is None:
        raise ValueError("tracks input must contain a track_id column")
    df["track_id"] = pd.to_numeric(df[track_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)

    timestamp_col = _first_existing(df.columns, ["timestamp_ns", "timestamp", "time", "frame"])
    if timestamp_col is None:
        df["timestamp_ns"] = np.arange(len(df), dtype=np.int64)
        df["time_s"] = df.groupby("track_id").cumcount().astype(float)
        notes["timestamp_note"] = "no timestamp column; used row order as time"
    elif timestamp_col == "timestamp_ns":
        df["timestamp_ns"] = pd.to_numeric(df[timestamp_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=["timestamp_ns"]).copy()
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
        t0 = int(df["timestamp_ns"].min())
        df["time_s"] = (df["timestamp_ns"] - t0) / 1e9
    else:
        raw_time = pd.to_numeric(df[timestamp_col], errors="coerce")
        df = df.loc[raw_time.notna()].copy()
        raw_time = raw_time.loc[df.index]
        if raw_time.max() > 1e12:
            df["timestamp_ns"] = raw_time.astype(np.int64)
            df["time_s"] = (df["timestamp_ns"] - int(df["timestamp_ns"].min())) / 1e9
        else:
            df["time_s"] = raw_time.astype(float) - float(raw_time.min())
            df["timestamp_ns"] = (df["time_s"] * 1e9).round().astype(np.int64)
        notes["timestamp_note"] = f"used {timestamp_col} as time source"

    x_col = _first_existing(df.columns, ["center_x", "x", "position_x", "pos_x"])
    y_col = _first_existing(df.columns, ["center_y", "y", "position_y", "pos_y"])
    if x_col is None or y_col is None:
        raise ValueError("tracks input must contain x/y or center_x/center_y columns")
    df["x"] = _to_numeric(df[x_col])
    df["y"] = _to_numeric(df[y_col])

    df.sort_values(["track_id", "timestamp_ns"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    vx_col = _first_existing(df.columns, ["vx", "velocity_x", "v_x"])
    vy_col = _first_existing(df.columns, ["vy", "velocity_y", "v_y"])
    if vx_col is not None and vy_col is not None:
        df["vx"] = _to_numeric(df[vx_col])
        df["vy"] = _to_numeric(df[vy_col])
        notes["used_velocity_columns"] = True
    else:
        df["vx"] = _group_derivative(df, "x")
        df["vy"] = _group_derivative(df, "y")
        notes["used_velocity_columns"] = False

    speed_col = _first_existing(df.columns, ["speed", "velocity", "v"])
    if speed_col is not None:
        df["speed"] = _to_numeric(df[speed_col])
        notes["speed_note"] = f"used {speed_col}"
    else:
        df["speed"] = np.hypot(df["vx"], df["vy"])
        notes["speed_note"] = "inferred from vx/vy"

    heading_col = _first_existing(df.columns, ["heading", "yaw", "theta"])
    if heading_col is not None:
        df["heading"] = _to_numeric(df[heading_col])
        notes["heading_note"] = f"used {heading_col}"
    else:
        df["heading"] = np.arctan2(df["vy"], df["vx"])
        notes["heading_note"] = "inferred from velocity or position derivative"
    df["heading_unwrapped"] = _unwrap_heading_by_track(df, "heading")
    df["heading"] = _wrap_angle(df["heading_unwrapped"])

    accel_col = _first_existing(df.columns, ["acceleration", "accel", "a", "a_long"])
    if accel_col is not None:
        df["acceleration"] = _to_numeric(df[accel_col])
        notes["acceleration_note"] = f"used {accel_col}"
    else:
        df["acceleration"] = _group_derivative(df, "speed")
        notes["acceleration_note"] = "inferred from speed derivative"

    yaw_rate_col = _first_existing(df.columns, ["yaw_rate", "omega", "turn_rate"])
    if yaw_rate_col is not None:
        df["yaw_rate"] = _to_numeric(df[yaw_rate_col])
        notes["yaw_rate_note"] = f"used {yaw_rate_col}"
    else:
        df["yaw_rate"] = _group_derivative(df, "heading_unwrapped")
        notes["yaw_rate_note"] = "inferred from heading derivative"

    conf_col = _first_existing(df.columns, ["confidence", "score", "track_confidence"])
    if conf_col is not None:
        df["input_confidence"] = _to_numeric(df[conf_col], default=0.7).clip(0.0, 1.0)
        notes["confidence_note"] = f"used {conf_col}"
    else:
        df["input_confidence"] = 0.75
        notes["confidence_note"] = "missing input confidence; assumed 0.75"

    status_col = _first_existing(df.columns, ["status", "track_status"])
    if status_col is not None:
        df["status"] = df[status_col].astype(str)
        notes["status_note"] = f"used {status_col}"
    else:
        df["status"] = "UNSPECIFIED"
        notes["status_note"] = "missing status; no lifecycle penalty applied"

    # Track-relative coordinates are a coarse lane/corridor proxy when explicit
    # lane centerlines are unavailable.
    track_s = np.zeros(len(df), dtype=float)
    track_d = np.zeros(len(df), dtype=float)
    for _, grp in df.groupby("track_id", sort=False):
        origin, axis = _infer_track_axis(grp)
        left = np.array([-axis[1], axis[0]])
        pts = grp[["x", "y"]].to_numpy(dtype=float) - origin
        track_s[grp.index] = pts @ axis
        track_d[grp.index] = pts @ left
    df["track_s"] = track_s
    df["track_d"] = track_d

    feature_rows: list[dict[str, Any]] = []
    for _, grp in df.groupby("track_id", sort=False):
        grp = grp.reset_index(drop=False).rename(columns={"index": "_orig_index"})
        times = grp["time_s"].to_numpy(dtype=float)
        x = grp["x"].to_numpy(dtype=float)
        y = grp["y"].to_numpy(dtype=float)
        speed = grp["speed"].to_numpy(dtype=float)
        heading = grp["heading_unwrapped"].to_numpy(dtype=float)
        yaw_rate = grp["yaw_rate"].to_numpy(dtype=float)
        accel = grp["acceleration"].to_numpy(dtype=float)
        track_d_values = grp["track_d"].to_numpy(dtype=float)
        input_conf = grp["input_confidence"].to_numpy(dtype=float)

        for i in range(len(grp)):
            start = int(np.searchsorted(times, times[i] - window_s, side="left"))
            idx = slice(start, i + 1)
            n = i - start + 1
            duration = max(float(times[i] - times[start]), 0.0)
            if duration > 1e-6:
                ref_heading = math.atan2(y[i] - y[start], x[i] - x[start])
                if math.hypot(x[i] - x[start], y[i] - y[start]) < 0.2:
                    ref_heading = float(heading[i])
            else:
                ref_heading = float(heading[i])
            forward = np.array([math.cos(ref_heading), math.sin(ref_heading)])
            left = np.array([-math.sin(ref_heading), math.cos(ref_heading)])
            delta = np.array([x[i] - x[start], y[i] - y[start]])

            lateral_disp = float(delta @ left)
            long_progress = float(delta @ forward)
            lat_velocity = lateral_disp / duration if duration > 1e-6 else 0.0
            ego_lateral_disp = float(y[i] - y[start])
            ego_long_progress = float(x[i] - x[start])
            track_offset_delta = float(track_d_values[i] - track_d_values[start])
            speed_change = float(speed[i] - speed[start])
            window_accel = speed_change / duration if duration > 1e-6 else float(accel[i])
            heading_delta = float(_wrap_angle(heading[i] - heading[start]))

            if n >= 4:
                mid = start + max((n // 2) - 1, 0)
                previous_yaw_peak = float(np.nanmax(np.abs(yaw_rate[start : mid + 1])))
                recent_yaw_peak = float(np.nanmax(np.abs(yaw_rate[mid + 1 : i + 1])))
                previous_heading_span = float(np.ptp(heading[start : mid + 1]))
                recent_heading_span = float(np.ptp(heading[mid + 1 : i + 1]))
                previous_offset_peak = float(np.nanmax(np.abs(track_d_values[start : mid + 1])))
                recent_offset_mean = float(np.nanmean(np.abs(track_d_values[mid + 1 : i + 1])))
            else:
                previous_yaw_peak = 0.0
                recent_yaw_peak = float(np.nanmax(np.abs(yaw_rate[idx]))) if n else 0.0
                previous_heading_span = 0.0
                recent_heading_span = float(np.ptp(heading[idx])) if n else 0.0
                previous_offset_peak = 0.0
                recent_offset_mean = float(abs(track_d_values[i]))

            feature_rows.append(
                {
                    "_orig_index": int(grp["_orig_index"].iloc[i]),
                    "window_count": n,
                    "window_duration_s": duration,
                    "window_lateral_displacement": lateral_disp,
                    "window_longitudinal_progress": long_progress,
                    "window_lateral_velocity": lat_velocity,
                    "ego_lateral_displacement": ego_lateral_disp,
                    "ego_longitudinal_progress": ego_long_progress,
                    "track_offset_delta": track_offset_delta,
                    "window_speed_change": speed_change,
                    "window_acceleration": window_accel,
                    "window_speed_std": float(np.nanstd(speed[idx])),
                    "window_yaw_abs_mean": float(np.nanmean(np.abs(yaw_rate[idx]))),
                    "window_yaw_peak": float(np.nanmax(np.abs(yaw_rate[idx]))),
                    "window_heading_delta": heading_delta,
                    "window_heading_span": float(np.ptp(heading[idx])),
                    "window_curvature": float(
                        np.nanmean(np.abs(yaw_rate[idx])) / max(np.nanmean(speed[idx]), 0.3)
                    ),
                    "previous_yaw_peak": previous_yaw_peak,
                    "recent_yaw_peak": recent_yaw_peak,
                    "previous_heading_span": previous_heading_span,
                    "recent_heading_span": recent_heading_span,
                    "previous_track_offset_peak": previous_offset_peak,
                    "recent_track_offset_mean": recent_offset_mean,
                    "track_offset_reduction": previous_offset_peak - abs(track_d_values[i]),
                    "window_input_confidence": float(np.nanmean(input_conf[idx])),
                    "data_quality": min(1.0, n / 4.0) * min(1.0, duration / max(window_s, 1e-6)),
                }
            )

    feature_df = pd.DataFrame(feature_rows).set_index("_orig_index")
    df = df.join(feature_df, how="left")
    df["data_quality"] = df["data_quality"].fillna(0.0)
    df["window_input_confidence"] = df["window_input_confidence"].fillna(df["input_confidence"])

    return _prediction_evidence(df, predictions_df, prediction_errors_df, notes), notes


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    stable_heading = (
        0.5 * (1.0 - _ramp(out["window_heading_delta"].abs(), 0.08, 0.55))
        + 0.5 * (1.0 - _ramp(out["window_yaw_abs_mean"], 0.05, 0.35))
    ).clip(0.0, 1.0)
    low_lateral = (1.0 - _ramp(out["window_lateral_displacement"].abs(), 0.35, 2.0)).clip(
        0.0, 1.0
    )
    low_accel = (1.0 - _ramp(out["window_acceleration"].abs(), 0.35, 1.8)).clip(0.0, 1.0)
    speed_consistency = (1.0 - _ramp(out["window_speed_std"], 0.25, 1.8)).clip(0.0, 1.0)
    prediction_stable = (1.0 - _ramp(out["future_lateral_shift"].abs(), 0.8, 3.0)).clip(
        0.0, 1.0
    )

    decel_strength = _ramp(-out["window_acceleration"], 0.6, 3.0)
    speed_drop_strength = _ramp(-out["window_speed_change"], 0.4, 2.8)

    ego_lateral_signal = out["ego_lateral_displacement"].abs().clip(upper=3.0) * 0.45
    lateral_abs = pd.concat(
        [
            out["window_lateral_displacement"].abs(),
            out["track_offset_delta"].abs(),
            out["future_lateral_shift"].abs(),
            ego_lateral_signal,
        ],
        axis=1,
    ).max(axis=1)
    lateral_velocity_abs = pd.concat(
        [
            out["window_lateral_velocity"].abs(),
            lateral_abs / out["window_duration_s"].replace(0.0, np.nan),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    longitudinal_progress = pd.concat(
        [
            out["window_longitudinal_progress"],
            out["ego_longitudinal_progress"].abs(),
        ],
        axis=1,
    ).max(axis=1)

    lateral_strength = _ramp(lateral_abs, 0.55, 2.4)
    lateral_velocity_strength = _ramp(lateral_velocity_abs, 0.30, 1.4)
    yaw_change_strength = np.maximum(
        _ramp(out["window_heading_delta"].abs(), 0.08, 0.45),
        _ramp(out["window_yaw_peak"], 0.18, 0.75),
    )
    forward_strength = _ramp(longitudinal_progress, 1.0, 4.5)
    speed_maintained = (1.0 - _ramp(-out["window_speed_change"], 0.3, 2.3)).clip(0.0, 1.0)
    prediction_lateral = _ramp(out["future_lateral_shift"].abs(), 1.0, 4.0)
    prediction_disagreement = _ramp(out["prediction_disagreement"], 0.5, 3.0)

    past_instability = np.maximum.reduce(
        [
            _ramp(out["previous_yaw_peak"], 0.20, 0.80).to_numpy(),
            _ramp(out["previous_heading_span"], 0.20, 0.90).to_numpy(),
            _ramp(out["previous_track_offset_peak"], 0.8, 3.0).to_numpy(),
        ]
    )
    yaw_stabilizing = _ramp(out["previous_yaw_peak"] - out["recent_yaw_peak"], 0.05, 0.45)
    heading_stabilizing = _ramp(
        out["previous_heading_span"] - out["recent_heading_span"], 0.05, 0.45
    )
    lateral_return = _ramp(out["track_offset_reduction"], 0.25, 1.6)
    current_stable = (1.0 - _ramp(out["recent_yaw_peak"], 0.08, 0.40)).clip(0.0, 1.0)

    out["braking_score"] = (
        0.42 * decel_strength
        + 0.32 * speed_drop_strength
        + 0.16 * stable_heading
        + 0.10 * low_lateral
    ).clip(0.0, 1.0)
    out["overtaking_score"] = (
        0.28 * lateral_strength
        + 0.18 * lateral_velocity_strength
        + 0.16 * yaw_change_strength
        + 0.15 * forward_strength
        + 0.11 * speed_maintained
        + 0.08 * prediction_lateral
        + 0.04 * prediction_disagreement
    ).clip(0.0, 1.0)
    out["recovery_score"] = (
        0.30 * pd.Series(past_instability, index=out.index)
        + 0.25 * yaw_stabilizing
        + 0.15 * heading_stabilizing
        + 0.20 * lateral_return
        + 0.10 * current_stable
    ).clip(0.0, 1.0)
    out["lane_keeping_score"] = (
        0.25 * (1.0 - _ramp(out["window_yaw_abs_mean"], 0.04, 0.25))
        + 0.22 * (1.0 - _ramp(out["window_heading_delta"].abs(), 0.05, 0.35))
        + 0.18 * low_lateral
        + 0.18 * low_accel
        + 0.12 * speed_consistency
        + 0.05 * prediction_stable
    ).clip(0.0, 1.0)
    out["lateral_motion_score"] = np.maximum.reduce(
        [
            lateral_strength.to_numpy(),
            lateral_velocity_strength.to_numpy(),
            prediction_lateral.to_numpy(),
        ]
    )
    out["score_lateral_displacement"] = lateral_abs
    out["score_longitudinal_progress"] = longitudinal_progress
    return out


def _status_weight(status: Any) -> float:
    text = str(status).upper()
    if "CONFIRMED" in text:
        return 1.0
    if "COAST" in text:
        return 0.78
    if "TENT" in text:
        return 0.62
    if "UNSPECIFIED" in text:
        return 0.92
    return 0.75


def _choose_raw_label(row: pd.Series) -> tuple[str, float]:
    scores = {
        "lane_keeping": float(row["lane_keeping_score"]),
        "braking": float(row["braking_score"]),
        "overtaking_attempt": float(row["overtaking_score"]),
        "trajectory_recovery": float(row["recovery_score"]),
    }
    max_label = max(scores, key=scores.get)
    max_score = scores[max_label]

    # Prefer explicit manoeuvre evidence over lane keeping when the scores are
    # close. This prevents a stable heading from hiding clear braking/lateral cues.
    if scores["braking"] >= 0.48 and scores["braking"] >= scores["lane_keeping"] - 0.08:
        return "braking", scores["braking"]
    if (
        scores["trajectory_recovery"] >= 0.52
        and scores["trajectory_recovery"] >= scores["overtaking_attempt"] - 0.06
    ):
        return "trajectory_recovery", scores["trajectory_recovery"]
    if (
        scores["overtaking_attempt"] >= 0.50
        and scores["overtaking_attempt"] >= scores["lane_keeping"] - 0.08
    ):
        return "overtaking_attempt", scores["overtaking_attempt"]
    if (
        row.get("lateral_motion_score", 0.0) >= 0.40
        and scores["overtaking_attempt"] >= 0.46
        and scores["overtaking_attempt"] >= scores["lane_keeping"] - 0.18
        and row.get("window_speed_change", 0.0) > -1.5
    ):
        return "overtaking_attempt", scores["overtaking_attempt"]
    return max_label, max_score


def _smooth_track_behaviours(grp: pd.DataFrame) -> list[str]:
    labels = grp["raw_behaviour"].tolist()
    confidences = grp["raw_confidence"].to_numpy(dtype=float)
    smoothed: list[str] = []
    pending_label: str | None = None
    pending_count = 0

    for i, label in enumerate(labels):
        if label == "unknown":
            smoothed.append("unknown")
            pending_label = None
            pending_count = 0
            continue

        if not smoothed or smoothed[-1] == "unknown":
            smoothed.append(label)
            pending_label = None
            pending_count = 0
            continue

        previous = smoothed[-1]
        if label == previous:
            smoothed.append(label)
            pending_label = None
            pending_count = 0
            continue

        if pending_label == label:
            pending_count += 1
        else:
            pending_label = label
            pending_count = 1

        prev_score_col = f"{previous}_score"
        label_score_col = f"{label}_score"
        prev_score = float(grp.iloc[i].get(prev_score_col, 0.0))
        label_score = float(grp.iloc[i].get(label_score_col, 0.0))

        if confidences[i] >= 0.74 or pending_count >= 2 or label_score > prev_score + 0.18:
            smoothed.append(label)
            pending_label = None
            pending_count = 0
        else:
            smoothed.append(previous)

    return smoothed


def _make_explanation(row: pd.Series) -> str:
    label = row["predicted_behaviour"]
    if label == "unknown":
        if row["window_count"] < row.get("min_points", 4):
            return f"insufficient data: {int(row['window_count'])} points in window"
        if row["window_input_confidence"] < 0.35:
            return f"low input confidence: {row['window_input_confidence']:.2f}"
        return f"weak rule evidence: top score {row['selected_score']:.2f}"
    if label == "braking":
        return (
            f"braking: acceleration {row['window_acceleration']:.2f} m/s^2, "
            f"speed change {row['window_speed_change']:.2f} m/s"
        )
    if label == "overtaking_attempt":
        return (
            f"overtaking attempt: lateral shift "
            f"{row.get('score_lateral_displacement', row['window_lateral_displacement']):.2f} m, "
            f"forward progress {row.get('score_longitudinal_progress', row['window_longitudinal_progress']):.2f} m"
        )
    if label == "trajectory_recovery":
        return (
            f"trajectory recovery: yaw peak {row['previous_yaw_peak']:.2f}->"
            f"{row['recent_yaw_peak']:.2f} rad/s, "
            f"offset reduction {row['track_offset_reduction']:.2f} m"
        )
    return (
        f"lane keeping: stable heading, yaw mean "
        f"{row['window_yaw_abs_mean']:.2f} rad/s, "
        f"lateral shift {row['window_lateral_displacement']:.2f} m"
    )


def classify_behaviours(
    features_df: pd.DataFrame,
    min_points: int = 4,
    unknown_threshold: float = 0.38,
) -> pd.DataFrame:
    """Apply rule scores, confidence weighting, and temporal smoothing."""
    scored = _score_rows(features_df)
    raw_labels: list[str] = []
    raw_scores: list[float] = []

    for _, row in scored.iterrows():
        label, score = _choose_raw_label(row)
        too_little_data = row["window_count"] < min_points or row["window_duration_s"] < 0.25
        low_input_conf = row["window_input_confidence"] < 0.30
        if too_little_data or low_input_conf or score < unknown_threshold:
            label = "unknown"
        raw_labels.append(label)
        raw_scores.append(score)

    scored["raw_behaviour"] = raw_labels
    scored["selected_score"] = raw_scores
    quality = (0.55 + 0.45 * scored["data_quality"].clip(0.0, 1.0))
    input_quality = (0.55 + 0.45 * scored["window_input_confidence"].clip(0.0, 1.0))
    status_quality = scored["status"].apply(_status_weight)
    scored["raw_confidence"] = (
        scored["selected_score"] * quality * input_quality * status_quality
    ).clip(0.0, 1.0)
    scored.loc[scored["raw_behaviour"] == "unknown", "raw_confidence"] = np.minimum(
        scored.loc[scored["raw_behaviour"] == "unknown", "raw_confidence"], 0.34
    )

    pieces: list[pd.DataFrame] = []
    for _, grp in scored.groupby("track_id", sort=False):
        grp = grp.sort_values("timestamp_ns").copy()
        grp["predicted_behaviour"] = _smooth_track_behaviours(grp)
        pieces.append(grp)
    out = pd.concat(pieces).sort_values(["track_id", "timestamp_ns"]).reset_index(drop=True)

    final_scores = []
    final_conf = []
    for _, row in out.iterrows():
        label = row["predicted_behaviour"]
        if label == "unknown":
            score = min(float(row["selected_score"]), 0.34)
        else:
            score = float(row.get(f"{label}_score", row["selected_score"]))
        final_scores.append(score)
        conf = score * (0.55 + 0.45 * row["data_quality"]) * (
            0.55 + 0.45 * row["window_input_confidence"]
        ) * _status_weight(row["status"])
        if label == "unknown":
            conf = min(conf, 0.34)
        final_conf.append(float(np.clip(conf, 0.0, 1.0)))
    out["selected_score"] = final_scores
    out["confidence"] = final_conf
    out["min_points"] = min_points
    out["explanation"] = out.apply(_make_explanation, axis=1)

    output_columns = [
        "timestamp_ns",
        "time_s",
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
        "status",
        "window_count",
        "window_duration_s",
        "window_lateral_displacement",
        "window_longitudinal_progress",
        "ego_lateral_displacement",
        "ego_longitudinal_progress",
        "track_offset_delta",
        "score_lateral_displacement",
        "score_longitudinal_progress",
        "future_lateral_shift",
        "prediction_disagreement",
        "prediction_error_fde_m",
        "prediction_error_lateral_m",
        "data_quality",
    ]
    return out[[c for c in output_columns if c in out.columns]]


def build_behaviour_segments(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive rows with the same behaviour per track."""
    rows: list[dict[str, Any]] = []
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "behaviour",
                "start_time",
                "end_time",
                "duration",
                "mean_confidence",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
            ]
        )

    for track_id, grp in predictions_df.sort_values(["track_id", "timestamp_ns"]).groupby(
        "track_id", sort=False
    ):
        grp = grp.reset_index(drop=True)
        if len(grp) > 1:
            dt = grp["time_s"].diff().dropna()
            gap_limit = max(0.75, float(dt.median()) * 2.5) if not dt.empty else 0.75
        else:
            gap_limit = 0.75
        start_idx = 0
        for i in range(1, len(grp) + 1):
            changed = i == len(grp)
            if i < len(grp):
                changed = (
                    grp.loc[i, "predicted_behaviour"]
                    != grp.loc[i - 1, "predicted_behaviour"]
                ) or (grp.loc[i, "time_s"] - grp.loc[i - 1, "time_s"] > gap_limit)
            if not changed:
                continue
            segment = grp.iloc[start_idx:i]
            first = segment.iloc[0]
            last = segment.iloc[-1]
            rows.append(
                {
                    "track_id": int(track_id),
                    "behaviour": first["predicted_behaviour"],
                    "start_time": float(first["time_s"]),
                    "end_time": float(last["time_s"]),
                    "duration": float(last["time_s"] - first["time_s"]),
                    "mean_confidence": float(segment["confidence"].mean()),
                    "start_x": float(first["x"]),
                    "start_y": float(first["y"]),
                    "end_x": float(last["x"]),
                    "end_y": float(last["y"]),
                    "start_timestamp_ns": int(first["timestamp_ns"]),
                    "end_timestamp_ns": int(last["timestamp_ns"]),
                    "n_frames": int(len(segment)),
                }
            )
            start_idx = i

    return pd.DataFrame(rows)


def build_behaviour_summary(predictions_df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    total_frames = int(len(predictions_df))
    total_tracks = int(predictions_df["track_id"].nunique()) if total_frames else 0
    counts = predictions_df["predicted_behaviour"].value_counts() if total_frames else pd.Series()
    rows: list[dict[str, Any]] = []
    low_unknown = int(
        (
            (predictions_df["predicted_behaviour"] == "unknown")
            | (predictions_df["confidence"] < 0.40)
        ).sum()
    ) if total_frames else 0
    notes_text = " | ".join(notes) if notes else "all optional inputs loaded"

    for behaviour in BEHAVIOURS:
        count = int(counts.get(behaviour, 0))
        sub = predictions_df[predictions_df["predicted_behaviour"] == behaviour]
        rows.append(
            {
                "number_of_tracks_analysed": total_tracks,
                "number_of_frames_analysed": total_frames,
                "behaviour": behaviour,
                "count": count,
                "percentage": (100.0 * count / total_frames) if total_frames else 0.0,
                "average_confidence": float(sub["confidence"].mean()) if not sub.empty else np.nan,
                "low_confidence_unknown_count": low_unknown,
                "notes": notes_text,
            }
        )
    return pd.DataFrame(rows)


def _load_optional_csv(path: Path | None, label: str, notes: list[str]) -> pd.DataFrame | None:
    if path is None:
        notes.append(f"{label} not requested")
        return None
    if not path.exists():
        notes.append(f"{label} missing at {path}; degraded mode")
        print(f"WARNING: {label} missing at {path}; continuing without it", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(path)
        notes.append(f"loaded {label}: {len(df)} rows")
        return df
    except Exception as exc:  # pragma: no cover - defensive CLI path
        notes.append(f"{label} failed to load: {exc}")
        print(f"WARNING: failed to load {label} from {path}: {exc}", file=sys.stderr)
        return None


def load_boundaries(map_dir: Path | None, notes: list[str]) -> list[np.ndarray]:
    if map_dir is None:
        notes.append("map directory not requested")
        return []
    path = map_dir / "track_boundaries.geojson"
    if not path.exists():
        notes.append(f"track boundaries missing at {path}; map context skipped")
        print(f"WARNING: track boundaries missing at {path}; continuing", file=sys.stderr)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive CLI path
        notes.append(f"track boundaries failed to load: {exc}")
        print(f"WARNING: failed to load boundaries: {exc}", file=sys.stderr)
        return []

    boundaries: list[np.ndarray] = []
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString" and len(coords) >= 2:
            boundaries.append(np.asarray(coords, dtype=float))
        elif gtype == "MultiLineString":
            for line in coords:
                if len(line) >= 2:
                    boundaries.append(np.asarray(line, dtype=float))
    notes.append(f"loaded Task 4 boundaries: {len(boundaries)} polylines")
    return boundaries


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return float("inf")
    p = point.astype(float)
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(closest - p, axis=1)
    return float(np.min(dist))


def attach_boundary_distance(
    features_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    boundaries: list[np.ndarray],
) -> pd.DataFrame:
    out = features_df.copy()
    out["future_boundary_distance_m"] = np.nan
    if predictions_df is None or predictions_df.empty or not boundaries:
        return out
    required = {"track_id", "source_ts_ns", "world_x", "world_y"}
    if not required.issubset(predictions_df.columns):
        return out

    pred = predictions_df.dropna(subset=["track_id", "source_ts_ns", "world_x", "world_y"]).copy()
    if pred.empty:
        return out

    distance_rows: list[dict[str, float | int]] = []
    for _, row in pred.iterrows():
        point = np.array([float(row["world_x"]), float(row["world_y"])])
        distance = min(_point_to_polyline_distance(point, b) for b in boundaries)
        distance_rows.append(
            {
                "track_id": int(row["track_id"]),
                "timestamp_ns": int(row["source_ts_ns"]),
                "future_boundary_distance_m": distance,
            }
        )
    if not distance_rows:
        return out
    dist_df = (
        pd.DataFrame(distance_rows)
        .groupby(["track_id", "timestamp_ns"])["future_boundary_distance_m"]
        .min()
        .reset_index()
    )
    out = out.drop(columns=["future_boundary_distance_m"], errors="ignore").merge(
        dist_df, on=["track_id", "timestamp_ns"], how="left"
    )
    return out


def render_behaviour_timeline(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictions_df.empty:
        return
    tracks = sorted(predictions_df["track_id"].unique())
    y_map = {tid: i for i, tid in enumerate(tracks)}
    height = min(18.0, max(6.0, 0.18 * len(tracks) + 3.0))
    fig, ax = plt.subplots(figsize=(13, height))
    for behaviour in BEHAVIOURS:
        sub = predictions_df[predictions_df["predicted_behaviour"] == behaviour]
        if sub.empty:
            continue
        alpha = np.clip(sub["confidence"].to_numpy(dtype=float), 0.25, 0.95)
        ax.scatter(
            sub["time_s"],
            sub["track_id"].map(y_map),
            s=18,
            c=BEHAVIOUR_COLORS[behaviour],
            alpha=float(np.nanmean(alpha)),
            marker="s",
            linewidths=0,
            label=behaviour,
        )
    ax.set_xlabel("Time since first track timestamp (s)")
    ax.set_ylabel("track_id")
    step = max(1, len(tracks) // 30)
    tick_tracks = tracks[::step]
    ax.set_yticks([y_map[t] for t in tick_tracks])
    ax.set_yticklabels([str(t) for t in tick_tracks])
    ax.set_title("Task 6 - Behaviour Timeline")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "behaviour_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _merge_world_prediction_labels(
    predictions_df: pd.DataFrame | None,
    behaviour_df: pd.DataFrame,
) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame()
    required = {"track_id", "source_ts_ns", "world_x", "world_y"}
    if not required.issubset(predictions_df.columns):
        return pd.DataFrame()
    pred = predictions_df.copy()
    if "horizon_s" in pred.columns:
        min_horizon = pred["horizon_s"].dropna().min()
        pred = pred[pred["horizon_s"] == min_horizon]
    if "model" in pred.columns and "ctrv" in set(pred["model"].astype(str)):
        pred = pred[pred["model"].astype(str) == "ctrv"]
    labels = behaviour_df[["track_id", "timestamp_ns", "predicted_behaviour", "confidence"]]
    return pred.merge(
        labels,
        left_on=["track_id", "source_ts_ns"],
        right_on=["track_id", "timestamp_ns"],
        how="inner",
    )


def render_behaviour_bev(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    boundaries: list[np.ndarray] | None = None,
    task5_predictions: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = boundaries or []
    world = _merge_world_prediction_labels(task5_predictions, predictions_df)
    use_world_panel = bool(boundaries and not world.empty)

    if use_world_panel:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax_ego, ax_world = axes
    else:
        fig, ax_ego = plt.subplots(figsize=(9, 8))
        ax_world = None

    for _, grp in predictions_df.groupby("track_id", sort=False):
        ax_ego.plot(grp["x"], grp["y"], color="#B0B0B0", lw=0.8, alpha=0.35)
    for behaviour in BEHAVIOURS:
        sub = predictions_df[predictions_df["predicted_behaviour"] == behaviour]
        if sub.empty:
            continue
        ax_ego.scatter(
            sub["x"],
            sub["y"],
            s=16,
            c=BEHAVIOUR_COLORS[behaviour],
            alpha=0.72,
            label=behaviour,
            linewidths=0,
        )
    ax_ego.plot(0, 0, "k^", ms=9, label="ego")
    ax_ego.set_aspect("equal", adjustable="box")
    ax_ego.set_xlabel("Ego-frame X (m)")
    ax_ego.set_ylabel("Ego-frame Y (m)")
    ax_ego.set_title("Task 6 - Opponent Tracks by Behaviour")
    ax_ego.grid(True, alpha=0.2)

    if ax_world is not None:
        for i, boundary in enumerate(boundaries):
            ax_world.plot(
                boundary[:, 0],
                boundary[:, 1],
                color="#424242",
                lw=1.0,
                alpha=0.65,
                label="Task 4 boundary" if i == 0 else "_nolegend_",
            )
        for behaviour in BEHAVIOURS:
            sub = world[world["predicted_behaviour"] == behaviour]
            if sub.empty:
                continue
            ax_world.scatter(
                sub["world_x"],
                sub["world_y"],
                s=14,
                c=BEHAVIOUR_COLORS[behaviour],
                alpha=0.65,
                linewidths=0,
                label=behaviour,
            )
        ax_world.set_aspect("equal", adjustable="box")
        ax_world.set_xlabel("World X (m)")
        ax_world.set_ylabel("World Y (m)")
        ax_world.set_title("Task 5 World Predictions over Task 4 Boundaries")
        ax_world.grid(True, alpha=0.2)
        ax_world.legend(loc="best", fontsize=8)

    handles = [Patch(color=BEHAVIOUR_COLORS[b], label=b) for b in BEHAVIOURS]
    ax_ego.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "behaviour_bev.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_interesting_track_plots(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("track_*_behaviour_signals.png"):
        old_plot.unlink()
    if predictions_df.empty:
        return
    non_lane = predictions_df[
        ~predictions_df["predicted_behaviour"].isin(["lane_keeping", "unknown"])
    ]
    if non_lane.empty:
        return
    interesting_ids = (
        non_lane.groupby("track_id").size().sort_values(ascending=False).head(3).index.tolist()
    )
    for tid in interesting_ids:
        grp = predictions_df[predictions_df["track_id"] == tid].sort_values("time_s")
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(grp["time_s"], grp["speed"], color="#263238", lw=1.5)
        axes[0].set_ylabel("speed (m/s)")
        axes[1].plot(grp["time_s"], grp["acceleration"], color="#C62828", lw=1.5)
        axes[1].set_ylabel("accel (m/s^2)")
        axes[2].plot(grp["time_s"], grp["yaw_rate"], color="#1565C0", lw=1.5)
        axes[2].set_ylabel("yaw rate")
        axes[2].set_xlabel("time (s)")
        ymin, ymax = axes[0].get_ylim()
        for _, seg in build_behaviour_segments(grp).iterrows():
            color = BEHAVIOUR_COLORS.get(seg["behaviour"], "#888888")
            for ax in axes:
                ax.axvspan(seg["start_time"], seg["end_time"], color=color, alpha=0.12)
                ax.grid(True, alpha=0.2)
        axes[0].set_ylim(ymin, ymax)
        fig.suptitle(f"Task 6 - Behaviour Signals (track_id={tid})")
        fig.tight_layout()
        fig.savefig(output_dir / f"track_{int(tid)}_behaviour_signals.png", dpi=130)
        plt.close(fig)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer."""
    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        else:
            text_df[col] = text_df[col].map(
                lambda v: "" if pd.isna(v) else str(v).replace("|", "\\|")
            )
    columns = list(text_df.columns)
    rows = text_df.astype(str).values.tolist()
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(columns)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    header = fmt([str(c) for c in columns])
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt(row) for row in rows]
    return "\n".join([header, sep, *body])


def write_report(
    output_dir: Path,
    notes: list[str],
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    window_s: float,
    min_points: int,
) -> None:
    lines = [
        "# Task 6 - Behaviour Prediction",
        "",
        "## Inputs Used",
        "",
        "- Task 3 `tracks.csv`: position, velocity, heading/yaw, tracker status, confidence.",
        "- Task 5 `predictions.csv`: CV/CTRV model disagreement and world-frame future points when available.",
        "- Task 5 `prediction_errors.csv`: retrospective prediction error context when available.",
        "- Task 4 `track_boundaries.geojson`: world-frame boundary overlay and future boundary distance when compatible with Task 5 world predictions.",
        "",
        "## Features",
        "",
        f"- Sliding temporal window: {window_s:.2f} s.",
        f"- Minimum evidence points before non-unknown classification: {min_points}.",
        "- Speed, acceleration, heading, and yaw rate are inferred from positions when explicit columns are missing.",
        "- Lateral displacement, longitudinal progress, curvature proxy, yaw peaks, speed change, and track-relative offset are computed per window.",
        "",
        "## Rule Logic",
        "",
        "- `lane_keeping`: stable heading, low yaw rate, low lateral displacement, and low acceleration.",
        "- `braking`: sustained negative acceleration and speed drop with mostly stable heading.",
        "- `overtaking_attempt`: lateral shift plus forward progress, maintained speed, yaw change, or Task 5 prediction disagreement.",
        "- `trajectory_recovery`: previous yaw/offset instability followed by lower recent yaw and decreasing track-relative offset.",
        "- `unknown`: too few points, low confidence, or weak rule evidence.",
        "",
        "Temporal hysteresis is applied per track to reduce frame-by-frame flicker.",
        "",
        "## Outputs",
        "",
        "- `behaviour_predictions.csv`: per-frame scores, label, confidence, and explanation.",
        "- `track_behaviour_segments.csv`: consecutive same-behaviour intervals.",
        "- `behaviour_summary.csv`: counts, percentages, confidence, and missing-input notes.",
        "- `behaviour_timeline.png`: behaviour class over time for each track.",
        "- `behaviour_bev.png`: BEV track plot, with Task 4 boundary context when available.",
        "- `track_*_behaviour_signals.png`: optional signal plots for interesting manoeuvre tracks.",
        "",
        "## Limitations",
        "",
        "- This is an explainable rule-based classifier because no labelled behaviour data is available.",
        "- Task 3 tracks are ego-frame while Task 4 boundaries are world-frame; direct boundary distances use Task 5 world predictions rather than raw track positions.",
        "- Short-lived or coasting tracks are intentionally assigned lower confidence.",
        "- Thresholds are conservative to favor stable labels over noisy manoeuvre flicker.",
        "",
        "## Run Command",
        "",
        "```bash",
        "uv run python src/task6_behavior_prediction.py --tracks outputs/task3/tracks.csv --predictions outputs/task5/predictions.csv --prediction-errors outputs/task5/prediction_errors.csv --map-dir outputs/task4 --output-dir outputs/task6",
        "```",
        "",
        "## Run Notes",
        "",
    ]
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Summary Snapshot",
            "",
            _dataframe_to_markdown(summary_df) if not summary_df.empty else "No frames analysed.",
            "",
        ]
    )
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def run_behavior_prediction(
    tracks_path: Path | str,
    predictions_path: Path | str | None = Path("outputs/task5/predictions.csv"),
    prediction_errors_path: Path | str | None = Path("outputs/task5/prediction_errors.csv"),
    map_dir: Path | str | None = Path("outputs/task4"),
    output_dir: Path | str = Path("outputs/task6"),
    window_s: float = 1.5,
    min_points: int = 4,
) -> Task6Result:
    """Run Task 6 end-to-end and write all challenge artifacts."""
    tracks_path = Path(tracks_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks file not found: {tracks_path}")

    notes: list[str] = []
    tracks_df = pd.read_csv(tracks_path)
    notes.append(f"loaded tracks: {len(tracks_df)} rows from {tracks_path}")

    pred_df = _load_optional_csv(
        Path(predictions_path) if predictions_path is not None else None,
        "Task 5 predictions",
        notes,
    )
    pred_err_df = _load_optional_csv(
        Path(prediction_errors_path) if prediction_errors_path is not None else None,
        "Task 5 prediction errors",
        notes,
    )
    boundaries = load_boundaries(Path(map_dir) if map_dir is not None else None, notes)

    features, feature_notes = prepare_track_features(
        tracks_df,
        predictions_df=pred_df,
        prediction_errors_df=pred_err_df,
        window_s=window_s,
    )
    for key, value in feature_notes.items():
        notes.append(f"{key}: {value}")

    if boundaries:
        features = attach_boundary_distance(features, pred_df, boundaries)
        if features["future_boundary_distance_m"].notna().any():
            notes.append("used Task 4 boundaries through Task 5 world prediction distances")
        else:
            notes.append("Task 4 boundaries loaded for plotting; no compatible world prediction distances")

    behaviour_df = classify_behaviours(features, min_points=min_points)
    if "future_boundary_distance_m" in features.columns:
        behaviour_df = behaviour_df.merge(
            features[["track_id", "timestamp_ns", "future_boundary_distance_m"]],
            on=["track_id", "timestamp_ns"],
            how="left",
        )
    segments_df = build_behaviour_segments(behaviour_df)
    summary_df = build_behaviour_summary(behaviour_df, notes)

    behaviour_df.to_csv(output_dir / "behaviour_predictions.csv", index=False)
    segments_df.to_csv(output_dir / "track_behaviour_segments.csv", index=False)
    summary_df.to_csv(output_dir / "behaviour_summary.csv", index=False)

    render_behaviour_timeline(behaviour_df, output_dir)
    render_behaviour_bev(behaviour_df, output_dir, boundaries=boundaries, task5_predictions=pred_df)
    render_interesting_track_plots(behaviour_df, output_dir)
    write_report(output_dir, notes, behaviour_df, summary_df, window_s, min_points)

    return Task6Result(
        predictions=behaviour_df,
        summary=summary_df,
        segments=segments_df,
        notes=notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6: Behaviour Prediction.")
    parser.add_argument("--tracks", default="outputs/task3/tracks.csv")
    parser.add_argument("--predictions", default="outputs/task5/predictions.csv")
    parser.add_argument("--prediction-errors", default="outputs/task5/prediction_errors.csv")
    parser.add_argument("--map-dir", default="outputs/task4")
    parser.add_argument("--output-dir", default="outputs/task6")
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--min-points", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Task 6: Behaviour Prediction")
    print(f"  tracks            : {args.tracks}")
    print(f"  predictions       : {args.predictions}")
    print(f"  prediction_errors : {args.prediction_errors}")
    print(f"  map_dir           : {args.map_dir}")
    print(f"  output_dir        : {args.output_dir}")
    result = run_behavior_prediction(
        tracks_path=args.tracks,
        predictions_path=args.predictions,
        prediction_errors_path=args.prediction_errors,
        map_dir=args.map_dir,
        output_dir=args.output_dir,
        window_s=args.window_s,
        min_points=args.min_points,
    )
    print("\nBehaviour summary:")
    print(result.summary[["behaviour", "count", "percentage", "average_confidence"]].to_string(index=False))
    print(f"\nTask 6 complete. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
