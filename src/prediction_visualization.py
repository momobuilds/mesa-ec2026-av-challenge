"""Visualisations for Task 5 trajectory prediction.

Three plot types:
1. prediction_bev_NNN.png — BEV at selected frames with fading polylines and
   uncertainty ellipses at each horizon (CV=blue, CTRV=orange).
2. prediction_error_hist.png — FDE histograms per horizon, CV vs CTRV.
3. manoeuvre_case_study.png — zoom on a single turning track.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from src.ego_trajectory import EgoTrajectory
from src.task2_fusion import _X, _Y, _YAW
from src.trajectory_predictor import TrajectoryPredictor

_CV_COLOR     = "#2979ff"   # blue family for CV
_CTRV_COLOR   = "#ff6d00"   # orange family for CTRV
_ACTUAL_COLOR = "#212121"   # near-black for ground truth


def _uncertainty_ellipse(ax, center_xy, cov2x2, n_sigma=2, **kwargs):
    """Draw a 2D uncertainty ellipse from a 2x2 covariance sub-matrix."""
    vals, vecs = np.linalg.eigh(cov2x2 + np.eye(2) * 1e-6)
    vals = np.maximum(vals, 0)
    width  = 2 * n_sigma * math.sqrt(vals[1])
    height = 2 * n_sigma * math.sqrt(vals[0])
    angle  = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ellipse = Ellipse(
        xy=center_xy, width=width, height=height, angle=angle,
        fill=False, linewidth=1.0, **kwargs,
    )
    ax.add_patch(ellipse)


def render_bev_frames(
    frames_ts: list[int],
    tracks_df: pd.DataFrame,
    pred_df,
    predictor_cv: TrajectoryPredictor,
    predictor_ctrv: TrajectoryPredictor,
    ego_traj: EgoTrajectory,
    output_dir: Path,
    horizons_s: list[float] = None,
    view_range: float = 60.0,
) -> None:
    """Render BEV plots for selected timestamps."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"]

    for frame_idx, ts in enumerate(frames_ts):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-view_range, view_range)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f5")
        ax.set_xlabel("Ego-frame X (m)")
        ax.set_ylabel("Ego-frame Y (m)")
        ax.set_title(f"Task 5 — Prediction BEV  (frame {frame_idx})")

        # Ego marker
        ax.plot(0, 0, "k^", markersize=10, label="Ego")

        # Find tracks alive at this timestamp (within 200 ms)
        alive = confirmed[
            (confirmed["timestamp_ns"] >= ts - int(200e6)) &
            (confirmed["timestamp_ns"] <= ts + int(200e6))
        ]
        if alive.empty:
            cv_patch   = mpatches.Patch(color=_CV_COLOR,   label="CV")
            ctrv_patch = mpatches.Patch(color=_CTRV_COLOR, label="CTRV")
            ax.legend(handles=[cv_patch, ctrv_patch], loc="upper right")
            out_path = output_dir / f"prediction_bev_{frame_idx:03d}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_path.name} (no tracks)")
            continue

        idx = alive.groupby("track_id")["timestamp_ns"].apply(
            lambda s: (s - ts).abs().idxmin()
        )
        closest_rows = alive.loc[idx.values].reset_index(drop=True)

        for _, row in closest_rows.iterrows():
            cx, cy = row["center_x"], row["center_y"]
            ax.plot(cx, cy, "ks", markersize=6)

            src_state = np.array([
                row["center_x"], row["center_y"],
                row["vx"], row["vy"], row["yaw"],
                row["length"], row["width"],
            ])
            src_P = np.eye(7) * 0.25
            src_P[0, 0] = row["cov_xx"]
            src_P[1, 1] = row["cov_yy"]
            src_ts = int(row["timestamp_ns"])
            tid = int(row["track_id"])

            for predictor, color in [
                (predictor_cv, _CV_COLOR),
                (predictor_ctrv, _CTRV_COLOR),
            ]:
                preds = predictor.predict(
                    track_id=tid,
                    source_state=src_state,
                    source_P=src_P,
                    source_ts_ns=src_ts,
                    horizons_s=horizons_s,
                )
                if not preds:
                    continue
                # Fading polyline: source → each horizon pred position (ego frame)
                xs = [cx] + [p.state[_X] for p in preds]
                ys = [cy] + [p.state[_Y] for p in preds]
                n = len(xs)
                for i in range(n - 1):
                    alpha = 0.8 - 0.5 * (i / max(n - 2, 1))
                    ax.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=alpha, lw=1.5)

                # Uncertainty ellipses at each horizon
                for p in preds:
                    _uncertainty_ellipse(
                        ax,
                        center_xy=(p.state[_X], p.state[_Y]),
                        cov2x2=p.covariance[:2, :2],
                        color=color, alpha=0.5, linestyle="--",
                    )
                    ax.plot(p.state[_X], p.state[_Y], "o",
                            color=color, markersize=4, alpha=0.8)

        cv_patch   = mpatches.Patch(color=_CV_COLOR,   label="CV")
        ctrv_patch = mpatches.Patch(color=_CTRV_COLOR, label="CTRV")
        ax.legend(handles=[cv_patch, ctrv_patch], loc="upper right")

        out_path = output_dir / f"prediction_bev_{frame_idx:03d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def render_error_histograms(
    err_df: pd.DataFrame,
    output_dir: Path,
    horizons_s: list[float] = None,
) -> None:
    """FDE histograms per horizon, CV vs CTRV side-by-side."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    n_h = len(horizons_s)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 4), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons_s):
        for color, model_name in [(_CV_COLOR, "cv"), (_CTRV_COLOR, "ctrv")]:
            sub = err_df[
                (err_df["model"] == model_name) &
                (err_df["horizon_s"] == h) &
                ~err_df["missed"] &
                err_df["fde_m"].notna()
            ]["fde_m"]
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=30, color=color, alpha=0.55, label=model_name.upper(), density=True)
            ax.axvline(sub.median(), color=color, lw=1.5, linestyle="--")
        ax.set_title(f"FDE  h={h} s")
        ax.set_xlabel("FDE (m)")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Task 5 — Prediction Error Histograms", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "prediction_error_hist.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def render_manoeuvre_case_study(
    tracks_df: pd.DataFrame,
    err_df: pd.DataFrame,
    predictor_cv: TrajectoryPredictor,
    predictor_ctrv: TrajectoryPredictor,
    ego_traj: EgoTrajectory,
    output_dir: Path,
    horizons_s: list[float] = None,
) -> None:
    """Zoom on a single turning track: actual vs CV vs CTRV."""
    if horizons_s is None:
        horizons_s = [0.5, 1.0, 2.0]
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"].copy()

    # Pick track with highest yaw variance (most interesting turn)
    yaw_var = confirmed.groupby("track_id")["yaw"].var()
    if yaw_var.empty:
        print("  No tracks for case study — skipping.")
        return
    best_tid = int(yaw_var.idxmax())

    grp = confirmed[confirmed["track_id"] == best_tid].sort_values("timestamp_ns").reset_index(drop=True)
    mid_idx = len(grp) // 2
    row = grp.iloc[mid_idx]
    src_ts = int(row["timestamp_ns"])
    src_state = np.array([
        row["center_x"], row["center_y"],
        row["vx"], row["vy"], row["yaw"],
        row["length"], row["width"],
    ])
    src_P = np.eye(7) * 0.25
    src_P[0, 0] = row["cov_xx"]
    src_P[1, 1] = row["cov_yy"]

    preds_cv   = predictor_cv.predict(best_tid, src_state, src_P, src_ts, horizons_s)
    preds_ctrv = predictor_ctrv.predict(best_tid, src_state, src_P, src_ts, horizons_s)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    # Actual trajectory
    ax.plot(grp["center_x"], grp["center_y"], color=_ACTUAL_COLOR, lw=1.5, label="Actual")
    ax.plot(row["center_x"], row["center_y"], "ko", markersize=8, label="Source")

    # CV prediction
    xs_cv = [row["center_x"]] + [p.state[_X] for p in preds_cv]
    ys_cv = [row["center_y"]] + [p.state[_Y] for p in preds_cv]
    ax.plot(xs_cv, ys_cv, color=_CV_COLOR, lw=2, marker="o", markersize=5, label="CV")
    for p in preds_cv:
        _uncertainty_ellipse(ax, (p.state[_X], p.state[_Y]),
                             p.covariance[:2, :2], color=_CV_COLOR, linestyle="--")

    # CTRV prediction
    xs_cr = [row["center_x"]] + [p.state[_X] for p in preds_ctrv]
    ys_cr = [row["center_y"]] + [p.state[_Y] for p in preds_ctrv]
    ax.plot(xs_cr, ys_cr, color=_CTRV_COLOR, lw=2, marker="o", markersize=5, label="CTRV")
    for p in preds_ctrv:
        _uncertainty_ellipse(ax, (p.state[_X], p.state[_Y]),
                             p.covariance[:2, :2], color=_CTRV_COLOR, linestyle="--")

    ax.legend()
    ax.set_title(f"Task 5 — Manoeuvre Case Study  (track_id={best_tid})")
    ax.set_xlabel("Ego-frame X (m)")
    ax.set_ylabel("Ego-frame Y (m)")

    out_path = output_dir / "manoeuvre_case_study.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
