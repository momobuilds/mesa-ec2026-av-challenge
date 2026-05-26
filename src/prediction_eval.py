"""Self-evaluation harness for Task 5 trajectory prediction.

For each confirmed track with >= min_track_len frames in tracks.csv:
  - Walk source timestamps t_source.
  - For each horizon h in horizons_s: find the actual track row closest to
    t_source + h*1e9 within +-100 ms tolerance.
  - Run the predictor and compute FDE, lateral/longitudinal errors, NLL.
  - Record prediction jitter between consecutive source timestamps.

Writes:
  predictions.csv       -- every (source_ts, track_id, horizon, model) prediction
  prediction_errors.csv -- per-prediction FDE and decomposed errors
  prediction_summary.csv -- aggregated mean/median/p95 FDE, NLL, jitter per (model, horizon)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.ego_trajectory import EgoTrajectory
from src.task2_fusion import _X, _Y, _YAW
from src.tf_utils import EgoStateLog
from src.trajectory_predictor import Prediction, TrajectoryPredictor

_MATCH_WINDOW_NS = int(100e6)  # +-100 ms


def run_evaluation(
    tracks_csv: Path,
    ego_traj: EgoTrajectory,
    ego_log: EgoStateLog,
    output_dir: Path,
    horizons_s: list[float] = None,
    models: list[str] = None,
    min_track_len: int = 5,
    dt_substep_s: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full evaluation. Returns (predictions_df, errors_df, summary_df)."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    if models is None:
        models = ["cv", "ctrv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(tracks_csv)

    # Only confirmed tracks with enough frames
    confirmed = df_all[df_all["status"] == "CONFIRMED"].copy()
    track_counts = confirmed.groupby("track_id")["timestamp_ns"].count()
    valid_ids = track_counts[track_counts >= min_track_len].index
    confirmed = confirmed[confirmed["track_id"].isin(valid_ids)].copy()
    confirmed.sort_values(["track_id", "timestamp_ns"], inplace=True)

    # Build per-track history for CTRV omega estimation
    history: dict[int, pd.DataFrame] = {
        tid: grp.reset_index(drop=True)
        for tid, grp in confirmed.groupby("track_id")
    }

    pred_rows: list[dict] = []
    err_rows: list[dict] = []

    for model_name in models:
        predictor = TrajectoryPredictor(
            ego_traj=ego_traj,
            ego_log=ego_log,
            model=model_name,
            dt_substep_s=dt_substep_s,
        )
        predictor.set_track_history(history)

        for track_id, grp in confirmed.groupby("track_id"):
            grp = grp.reset_index(drop=True)
            timestamps = grp["timestamp_ns"].values

            for src_idx, src_ts in enumerate(timestamps):
                row = grp.iloc[src_idx]
                source_state = np.array([
                    row["center_x"], row["center_y"],
                    row["vx"], row["vy"],
                    row["yaw"], row["length"], row["width"],
                ], dtype=np.float64)
                source_P = np.eye(7) * 0.25
                source_P[0, 0] = row["cov_xx"]
                source_P[1, 1] = row["cov_yy"]

                preds = predictor.predict(
                    track_id=int(track_id),
                    source_state=source_state,
                    source_P=source_P,
                    source_ts_ns=int(src_ts),
                    horizons_s=horizons_s,
                )

                for pred in preds:
                    pred_rows.append({
                        "model": model_name,
                        "track_id": int(track_id),
                        "source_ts_ns": pred.source_ts_ns,
                        "horizon_s": pred.horizon_s,
                        "target_ts_ns": pred.target_ts_ns,
                        "pred_x": pred.state[_X],
                        "pred_y": pred.state[_Y],
                        "cov_xx": pred.covariance[_X, _X],
                        "cov_yy": pred.covariance[_Y, _Y],
                        "world_x": pred.world_xy[0],
                        "world_y": pred.world_xy[1],
                    })

                    # Ground truth lookup
                    target_ns = pred.target_ts_ns
                    candidates = grp[
                        (grp["timestamp_ns"] >= target_ns - _MATCH_WINDOW_NS) &
                        (grp["timestamp_ns"] <= target_ns + _MATCH_WINDOW_NS)
                    ]
                    missed = len(candidates) == 0
                    fde = np.nan
                    lateral = np.nan
                    longitudinal = np.nan
                    nll = np.nan

                    if not missed:
                        closest = candidates.iloc[
                            (candidates["timestamp_ns"] - target_ns).abs().argmin()
                        ]
                        actual_ego = np.array([[closest["center_x"], closest["center_y"]]])
                        actual_ts = int(closest["timestamp_ns"])
                        actual_world = ego_traj.ego_to_world(actual_ego, actual_ts)[0]

                        dx = actual_world[0] - pred.world_xy[0]
                        dy = actual_world[1] - pred.world_xy[1]
                        fde = math.hypot(dx, dy)

                        # Decompose into lateral/longitudinal wrt predicted heading
                        pred_heading = pred.state[_YAW]
                        lon_dir = np.array([math.cos(pred_heading), math.sin(pred_heading)])
                        lat_dir = np.array([-math.sin(pred_heading), math.cos(pred_heading)])
                        err_vec = np.array([dx, dy])
                        longitudinal = float(np.dot(err_vec, lon_dir))
                        lateral = float(np.dot(err_vec, lat_dir))

                        # NLL under predicted 2D Gaussian
                        Sigma = pred.covariance[:2, :2]
                        try:
                            Sigma_reg = Sigma + np.eye(2) * 1e-6
                            Sigma_inv = np.linalg.inv(Sigma_reg)
                            sign, logdet = np.linalg.slogdet(Sigma_reg)
                            if sign <= 0:
                                nll = np.nan
                            else:
                                maha2 = float(err_vec @ Sigma_inv @ err_vec)
                                nll = 0.5 * (maha2 + logdet + 2 * math.log(2 * math.pi))
                        except np.linalg.LinAlgError:
                            nll = np.nan

                    err_rows.append({
                        "model": model_name,
                        "track_id": int(track_id),
                        "source_ts_ns": pred.source_ts_ns,
                        "horizon_s": pred.horizon_s,
                        "fde_m": fde,
                        "lateral_m": lateral,
                        "longitudinal_m": longitudinal,
                        "nll": nll,
                        "missed": missed,
                    })

    pred_df = pd.DataFrame(pred_rows)
    err_df  = pd.DataFrame(err_rows)

    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    err_df.to_csv(output_dir / "prediction_errors.csv", index=False)

    # Aggregate summary
    summary_rows = []
    for model_name in models:
        for h in horizons_s:
            sub = err_df[(err_df["model"] == model_name) & (err_df["horizon_s"] == h)]
            valid = sub[~sub["missed"] & sub["fde_m"].notna()]
            if len(valid) == 0:
                print(f"  WARNING: no valid samples for model={model_name} horizon={h}s")
            jitter = _compute_jitter(pred_df, model_name, h)
            summary_rows.append({
                "model": model_name,
                "horizon_s": h,
                "n_samples": len(valid),
                "mean_fde": valid["fde_m"].mean(),
                "median_fde": valid["fde_m"].median(),
                "p95_fde": valid["fde_m"].quantile(0.95),
                "mean_lateral": valid["lateral_m"].abs().mean(),
                "mean_longitudinal": valid["longitudinal_m"].abs().mean(),
                "mean_nll": valid["nll"].mean(),
                "miss_rate": sub["missed"].mean(),
                "mean_jitter": jitter,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "prediction_summary.csv", index=False)

    return pred_df, err_df, summary_df


def _compute_jitter(
    pred_df: pd.DataFrame,
    model: str,
    horizon_s: float,
) -> float:
    """Mean jitter: L2 distance between consecutive source-timestamp world_xy predictions.

    For source timestamps t1 < t2, both predict their own target time at t+h.
    Jitter is how much the predicted position shifts between consecutive frames.
    """
    sub = pred_df[
        (pred_df["model"] == model) & (pred_df["horizon_s"] == horizon_s)
    ].copy().sort_values(["track_id", "source_ts_ns"])

    jitters = []
    for _, grp in sub.groupby("track_id"):
        rows = grp.reset_index(drop=True)
        for i in range(len(rows) - 1):
            r1 = rows.iloc[i]
            r2 = rows.iloc[i + 1]
            dx = r2["world_x"] - r1["world_x"]
            dy = r2["world_y"] - r1["world_y"]
            jitters.append(math.hypot(dx, dy))

    return float(np.mean(jitters)) if jitters else float("nan")
