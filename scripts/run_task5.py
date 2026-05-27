"""CLI entry point for EC2026 Task 5 (Trajectory Prediction)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import parse_bag_summary
from src.ego_trajectory import EgoTrajectory
from src.prediction_eval import run_evaluation
from src.prediction_visualization import (
    render_bev_frames,
    render_error_histograms,
    render_manoeuvre_case_study,
)
from src.tf_utils import EgoStateLog
from src.trajectory_predictor import TrajectoryPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 5: Trajectory Prediction.")
    p.add_argument("--tracks", default="outputs/task3/tracks.csv")
    p.add_argument("--bag-dir", default="data")
    p.add_argument("--output-dir", default="outputs/task5")
    p.add_argument("--horizons", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--models", nargs="+", choices=["cv", "ctrv"], default=["cv", "ctrv"])
    p.add_argument("--bev-frames", type=int, default=6)
    p.add_argument("--min-track-len", type=int, default=5)
    p.add_argument("--substep-ms", type=float, default=50.0)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip ground-truth evaluation (no prediction_errors.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = Path(args.tracks)
    dt_substep = args.substep_ms / 1000.0

    print("Task 5: Trajectory Prediction")
    print(f"  tracks    : {tracks_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  horizons  : {args.horizons} s")
    print(f"  models    : {args.models}")
    print(f"  substep   : {args.substep_ms} ms")

    # ── Load ego data ─────────────────────────────────────────────────────────
    bag_summary = parse_bag_summary(args.bag_dir)
    bag_start_ns = int(bag_summary.starting_time_ns or 0)
    mat_candidates = list(Path(args.bag_dir).glob("**/*.mat"))
    if not mat_candidates:
        print("ERROR: no ego.mat found under --bag-dir", file=sys.stderr)
        sys.exit(1)
    mat_path = mat_candidates[0]
    print(f"  ego.mat   : {mat_path}")

    # ego_log is used exclusively for substep ego-motion compensation (Doppler
    # correction: provides ekf_vx/vy at each prediction substep).
    ego_log = EgoStateLog(mat_path, bag_start_ns)
    print(f"  EgoStateLog: {ego_log}")

    # ego_traj is used for ego-frame → world-frame coordinate conversion.
    # Prefer the TF-based approach (no drift) over velocity integration.
    try:
        ego_traj = EgoTrajectory.from_bag_tf(args.bag_dir)
        print(f"  EgoTrajectory: from /tf ({ego_traj._timestamps_ns.shape[0]} samples)")
    except Exception as exc:
        print(f"  /tf unavailable ({exc}), falling back to ego.mat integration")
        ego_traj = EgoTrajectory.from_ego_log(ego_log)

    # ── Load tracks ───────────────────────────────────────────────────────────
    tracks_df = pd.read_csv(tracks_path)
    print(f"  tracks.csv: {len(tracks_df)} rows, "
          f"{tracks_df['track_id'].nunique()} unique track IDs")

    # ── Build history for CTRV omega estimation ───────────────────────────────
    confirmed = tracks_df[tracks_df["status"] == "CONFIRMED"].copy()
    confirmed.sort_values(["track_id", "timestamp_ns"], inplace=True)
    history = {
        tid: grp.reset_index(drop=True)
        for tid, grp in confirmed.groupby("track_id")
    }

    # ── Evaluation ───────────────────────────────────────────────────────────
    pred_df = None
    err_df  = None
    summary_df = None

    if not args.no_eval:
        print("\nRunning evaluation harness …")
        pred_df, err_df, summary_df = run_evaluation(
            tracks_csv=tracks_path,
            ego_traj=ego_traj,
            ego_log=ego_log,
            output_dir=output_dir,
            horizons_s=args.horizons,
            models=args.models,
            min_track_len=args.min_track_len,
            dt_substep_s=dt_substep,
        )
        print("\nPrediction Summary:")
        print(summary_df.to_string(index=False))

        # Sanity check: CV FDE at 0.5 s
        cv_05 = summary_df[
            (summary_df["model"] == "cv") & (summary_df["horizon_s"] == 0.5)
        ]
        if not cv_05.empty:
            mean_fde = float(cv_05["mean_fde"].iloc[0])
            flag = "OK" if mean_fde <= 2.0 else "WARNING — exceeds 2.0 m target!"
            print(f"\n[Sanity check] CV mean FDE @ 0.5 s = {mean_fde:.3f} m  [{flag}]")
    else:
        print("  --no-eval: skipping ground-truth evaluation")

    # ── Build predictors for visualisation ───────────────────────────────────
    print("\nRendering visualisations …")
    predictor_cv = TrajectoryPredictor(
        ego_traj=ego_traj, ego_log=ego_log, model="cv", dt_substep_s=dt_substep,
    )
    predictor_cv.set_track_history(history)

    predictor_ctrv = TrajectoryPredictor(
        ego_traj=ego_traj, ego_log=ego_log, model="ctrv", dt_substep_s=dt_substep,
    )
    predictor_ctrv.set_track_history(history)

    # Select BEV frame timestamps evenly across the run
    all_ts = sorted(confirmed["timestamp_ns"].unique())
    if len(all_ts) < args.bev_frames:
        bev_ts = all_ts
    else:
        indices = np.linspace(0, len(all_ts) - 1, args.bev_frames, dtype=int)
        bev_ts = [all_ts[i] for i in indices]

    render_bev_frames(
        frames_ts=bev_ts,
        tracks_df=tracks_df,
        pred_df=pred_df,
        predictor_cv=predictor_cv,
        predictor_ctrv=predictor_ctrv,
        ego_traj=ego_traj,
        output_dir=output_dir,
        horizons_s=args.horizons,
    )

    if err_df is not None and not err_df.empty:
        render_error_histograms(err_df, output_dir, horizons_s=args.horizons)
        render_manoeuvre_case_study(
            tracks_df=tracks_df,
            err_df=err_df,
            predictor_cv=predictor_cv,
            predictor_ctrv=predictor_ctrv,
            ego_traj=ego_traj,
            output_dir=output_dir,
            horizons_s=args.horizons,
        )

    print(f"\nTask 5 complete. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
