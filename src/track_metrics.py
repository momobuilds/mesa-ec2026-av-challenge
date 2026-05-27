"""Proxy tracking metrics for Task 3 (no ground truth required).

Metrics reported
----------------
- Number of confirmed tracks
- Track lifetimes (n_frames_observed, n_frames_coasted)
- Total path length and mean / max speed per track
- ID-switch proxy: a new tentative track that appears within 2 m of a
  recently deleted confirmed track within 1 s is flagged as a probable
  ID switch (i.e., the same physical target lost and re-initiated).
- MOTA-lite: not computable without ground truth, but we report
  configuration choices and why they were made.

All functions operate on the Track objects returned by the tracker, or
on the flat rows list (dicts matching TRACKS_CSV_COLUMNS).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.track_state import Track, TrackStatus, TRACKS_CSV_COLUMNS


SUMMARY_COLUMNS = [
    "track_id", "start_ts", "end_ts",
    "n_frames_observed", "n_frames_coasted",
    "total_path_m", "mean_speed_mps", "max_speed_mps",
    "last_status",
]


def build_tracks_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert the flat row list to a DataFrame with correct column order."""
    if not rows:
        return pd.DataFrame(columns=TRACKS_CSV_COLUMNS)
    return pd.DataFrame(rows, columns=TRACKS_CSV_COLUMNS)


def build_summary_df(all_tracks: list[Track]) -> pd.DataFrame:
    """One row per track: lifetime, path length, speed statistics."""
    records = []
    for trk in all_tracks:
        if trk.hits == 0:
            continue
        hist = trk.history
        if len(hist) < 2:
            n_observed = trk.hits
            n_coasted = trk.time_since_update
            path_m = 0.0
            mean_spd = 0.0
            max_spd = 0.0
            start_ts = hist[0][0] if hist else 0
            end_ts = hist[-1][0] if hist else 0
        else:
            start_ts = hist[0][0]
            end_ts = hist[-1][0]
            # n_observed = total EKF update hits.
            # n_coasted  = history entries appended during COASTING (missed frames
            #              while CONFIRMED/COASTING) — these are appended by
            #              MultiObjectTracker.step() on mark_missed, distinct from
            #              the creation entry (hist[0]) and update entries.
            # Formula: create_track appends hist[0]; update_track appends one per
            # hit; mark_missed + step appends one per coasted frame (conf/coast only).
            # So len(hist) = 1 (creation) + hits + coasted_frames.
            n_observed = trk.hits
            n_coasted = max(0, len(hist) - 1 - trk.hits)  # subtract creation entry

            # Path length from history positions
            path_m = 0.0
            speeds: list[float] = []
            for i in range(1, len(hist)):
                dt_ns = hist[i][0] - hist[i - 1][0]
                dt_s = dt_ns * 1e-9
                dx = hist[i][1] - hist[i - 1][1]
                dy = hist[i][2] - hist[i - 1][2]
                dist = math.hypot(dx, dy)
                path_m += dist
                if dt_s > 0:
                    speeds.append(dist / dt_s)

            mean_spd = float(np.mean(speeds)) if speeds else 0.0
            max_spd = float(np.max(speeds)) if speeds else 0.0

        records.append({
            "track_id": trk.track_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "n_frames_observed": n_observed,
            "n_frames_coasted": n_coasted,
            "total_path_m": round(path_m, 2),
            "mean_speed_mps": round(mean_spd, 3),
            "max_speed_mps": round(max_spd, 3),
            "last_status": trk.status.name,
        })

    if not records:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS)


def detect_id_switches(
    all_tracks: list[Track],
    *,
    position_threshold_m: float = 2.0,
    time_threshold_ns: int = int(1e9),
) -> list[dict[str, Any]]:
    """Detect probable ID switches (proxy, no ground truth).

    A new track is flagged as a probable ID switch if it is initialised
    within *position_threshold_m* of a recently-deleted confirmed track
    and within *time_threshold_ns* of its deletion.

    Returns list of dicts: {new_track_id, deleted_track_id, distance_m, time_gap_ns}
    """
    deleted = [
        t for t in all_tracks
        if t.status == TrackStatus.DELETED and t.hits >= 3 and t.history
    ]
    new_tracks = [
        t for t in all_tracks
        if t.history and t.track_id not in {d.track_id for d in deleted}
    ]

    switches = []
    for new_trk in new_tracks:
        if not new_trk.history:
            continue
        new_start_ts = new_trk.history[0][0]
        new_x, new_y = new_trk.history[0][1], new_trk.history[0][2]

        for del_trk in deleted:
            del_end_ts = del_trk.history[-1][0]
            if not (0 <= new_start_ts - del_end_ts <= time_threshold_ns):
                continue
            del_x, del_y = del_trk.history[-1][1], del_trk.history[-1][2]
            dist = math.hypot(new_x - del_x, new_y - del_y)
            if dist <= position_threshold_m:
                switches.append({
                    "new_track_id": new_trk.track_id,
                    "deleted_track_id": del_trk.track_id,
                    "distance_m": round(dist, 3),
                    "time_gap_ns": new_start_ts - del_end_ts,
                })
    return switches


def print_metrics_summary(
    rows: list[dict[str, Any]],
    all_tracks: list[Track],
) -> None:
    """Print a human-readable tracking quality summary."""
    df = build_tracks_df(rows)
    summary = build_summary_df(all_tracks)
    id_switches = detect_id_switches(all_tracks)

    confirmed = [t for t in all_tracks if t.hits >= 3]
    coasting_events = sum(t.time_since_update > 0 for t in confirmed)

    print("\n── Tracking metrics ──────────────────────────────────────────────")
    print(f"  Total unique track IDs       : {len(all_tracks)}")
    print(f"  Confirmed tracks (≥3 hits)   : {len(confirmed)}")
    print(f"  Total track-rows emitted     : {len(rows)}")
    if not summary.empty:
        long = summary[summary["n_frames_observed"] > 0]
        print(f"  Mean track lifetime (obs)    : {long['n_frames_observed'].mean():.1f} frames")
        print(f"  Max track lifetime (obs)     : {long['n_frames_observed'].max()} frames")
        n_long = (summary["n_frames_observed"] > 30).sum()
        print(f"  Tracks with >30 obs frames   : {n_long}")
        print(f"  Mean path length             : {long['total_path_m'].mean():.1f} m")
    print(f"  Probable ID switches         : {len(id_switches)}")
    if id_switches:
        for sw in id_switches[:5]:
            print(f"    track {sw['new_track_id']} likely same as {sw['deleted_track_id']}"
                  f" (dist={sw['distance_m']:.1f}m, gap={sw['time_gap_ns']/1e6:.0f}ms)")
    print("─────────────────────────────────────────────────────────────────")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _track_color(track_id: int) -> str:
    """Stable color from track ID (deterministic, visually distinct)."""
    palette = [
        "#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
        "#00BCD4", "#FF5722", "#607D8B", "#795548", "#009688",
        "#F44336", "#3F51B5", "#CDDC39", "#FFC107", "#8BC34A",
    ]
    return palette[track_id % len(palette)]


def plot_tracking_bev(
    frame_rows: list[dict[str, Any]],
    all_tracks: list[Track],
    all_sensor_bev: dict[str, Any],
    output_path: str,
    *,
    forward_range: tuple[float, float] = (-5.0, 95.0),
    lateral_range: tuple[float, float] = (-20.0, 20.0),
    trail_frames: int = 20,
    timestamp_ns: int = 0,
    frame_idx: int = 0,
) -> None:
    """BEV plot: raw points + oriented track boxes + trails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    import math as _math

    fig, ax = plt.subplots(figsize=(10, 11))

    # Raw sensor points (faded)
    sensor_colors = {
        "lidar_front": "#2196F3", "lidar_left": "#4CAF50", "lidar_right": "#FF5722",
        "radar_front": "#9C27B0", "radar_back": "#607D8B",
        "radar_left": "#FFEB3B", "radar_right": "#F44336",
    }
    for topic, fr in all_sensor_bev.items():
        if fr is None or not hasattr(fr, "bev_points") or len(fr.bev_points) == 0:
            continue
        key = topic.split("/")[2] if "/" in topic else topic
        c = sensor_colors.get(key, "#999")
        pts = fr.bev_points
        ax.scatter(pts[:, 1], pts[:, 0], s=0.3, c=c, alpha=0.20, linewidths=0)

    # Track trails and boxes
    active_track_ids = {r["track_id"] for r in frame_rows}
    for trk in all_tracks:
        if trk.track_id not in active_track_ids:
            continue
        color = _track_color(trk.track_id)

        # Trail: last trail_frames history points
        hist = trk.history[-trail_frames:]
        if len(hist) >= 2:
            xs = [h[1] for h in hist]
            ys = [h[2] for h in hist]
            ax.plot(ys, xs, color=color, lw=1.0, alpha=0.5)

    for row in frame_rows:
        trk_id = row["track_id"]
        color = _track_color(trk_id)
        cx, cy = float(row["center_x"]), float(row["center_y"])
        yaw = float(row.get("yaw", 0.0))
        l, w = float(row.get("length", 4.0)), float(row.get("width", 2.0))
        is_coasting = row["status"] == "COASTING"

        # Oriented box
        hl, hw = l / 2, w / 2
        c_ang, s_ang = _math.cos(yaw), _math.sin(yaw)
        corners_local = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]])
        R2 = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
        corners = (R2 @ corners_local.T).T + np.array([cx, cy])
        lw_box = 1.5 if is_coasting else 2.5
        ls_box = "--" if is_coasting else "-"
        poly = Polygon(corners[:, [1, 0]], closed=True,
                       fill=False, edgecolor=color, linewidth=lw_box, linestyle=ls_box)
        ax.add_patch(poly)

        # Velocity arrow
        vx, vy = float(row.get("vx", 0)), float(row.get("vy", 0))
        spd = _math.hypot(vx, vy)
        if spd > 0.5:
            ax.annotate("", xy=(cy + vy * 2, cx + vx * 2), xytext=(cy, cx),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        # Label
        label = f"T{trk_id}"
        if is_coasting:
            label += f"(coast:{row['time_since_update']})"
        ax.text(cy, cx + 1.5, label, fontsize=7, ha="center", color=color,
                bbox={"facecolor": "white", "alpha": 0.6, "pad": 1, "edgecolor": "none"})

    ax.set_xlabel("lateral y_ego (m)")
    ax.set_ylabel("forward x_ego (m)")
    ax.set_xlim(lateral_range)
    ax.set_ylim(forward_range)
    ax.set_aspect("equal")
    ax.grid(True, color="#ddd", lw=0.6)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")

    # Legend
    legend_els = [
        Line2D([0], [0], color="k", lw=2, label="confirmed"),
        Line2D([0], [0], color="k", lw=1.5, ls="--", label="coasting"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    n_conf = sum(1 for r in frame_rows if r["status"] == "CONFIRMED")
    n_coast = sum(1 for r in frame_rows if r["status"] == "COASTING")
    ax.set_title(
        f"Task 3 tracking | frame {frame_idx} | t={timestamp_ns} ns | "
        f"confirmed={n_conf} coasting={n_coast}"
    )
    fig.tight_layout()
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tracking_timeline(
    all_tracks: list[Track],
    master_ts_list: list[int],
    output_path: str,
) -> None:
    """Gantt-style chart: track lifetime per track_id across frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    confirmed = [t for t in all_tracks if t.hits >= 3]
    if not confirmed:
        return

    # Map timestamp → frame index
    ts_to_frame = {ts: i for i, ts in enumerate(master_ts_list)}
    n_frames = len(master_ts_list)

    fig, ax = plt.subplots(figsize=(14, max(4, len(confirmed) * 0.5 + 2)))

    for row_idx, trk in enumerate(sorted(confirmed, key=lambda t: t.track_id)):
        color = _track_color(trk.track_id)
        hist = trk.history
        if not hist:
            continue

        # Group history into observed vs coasted segments
        frames_observed = set()
        for h in hist:
            fi = ts_to_frame.get(h[0], None)
            if fi is not None:
                frames_observed.add(fi)

        if not frames_observed:
            continue

        f_start = min(frames_observed)
        f_end = max(frames_observed)

        # Full lifetime bar (hatched = coasting)
        ax.barh(row_idx, f_end - f_start + 1, left=f_start,
                height=0.6, color=color, alpha=0.25,
                hatch="//", edgecolor=color, linewidth=0.5,
                label="_nolegend_")

        # Observed frames: solid blocks
        # Group consecutive observed frames into runs for efficiency
        obs_sorted = sorted(frames_observed)
        run_start = obs_sorted[0]
        prev = obs_sorted[0]
        for fi in obs_sorted[1:] + [obs_sorted[-1] + 2]:
            if fi > prev + 1:
                ax.barh(row_idx, prev - run_start + 1, left=run_start,
                        height=0.6, color=color, alpha=0.85,
                        edgecolor=color, linewidth=0.8)
                run_start = fi
            prev = fi

        ax.text(f_start - 0.3, row_idx, f"T{trk.track_id}",
                va="center", ha="right", fontsize=7, color=color)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Track")
    ax.set_title("Task 3 tracking timeline (solid=observed, hatched=coasting)")
    ax.set_xlim(0, n_frames)
    ax.set_ylim(-0.5, len(confirmed) - 0.5)
    ax.yaxis.set_visible(False)
    ax.grid(True, axis="x", color="#ddd", lw=0.5)

    legend_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.85, label="observed"),
        mpatches.Patch(facecolor="grey", alpha=0.25, hatch="//", label="coasting"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
