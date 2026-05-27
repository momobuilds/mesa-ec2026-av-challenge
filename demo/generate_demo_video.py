#!/usr/bin/env python3
"""Generate a Tasks 1-7 demo MP4 from existing challenge outputs.

The script is intentionally a visualization layer. It reads the PNG/CSV
artifacts already produced by Tasks 1-7, optionally samples raw camera frames
from the ROS bag, and composites them into a recorded video with simple labels
and overlays.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


BG = (18, 22, 29)
PANEL_BG = (26, 31, 40)
PANEL_BORDER = (72, 83, 99)
TEXT = (238, 242, 247)
MUTED = (170, 181, 196)
CYAN = (255, 210, 64)
GREEN = (116, 214, 146)
ORANGE = (95, 164, 255)
MAGENTA = (214, 128, 255)
RED = (92, 104, 255)
GRID = (55, 65, 81)

COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DemoArtifacts:
    raw_images: list[Path]
    detection_images: list[Path]
    fusion_images: list[Path]
    tracking_images: list[Path]
    map_images: list[Path]
    prediction_images: list[Path]
    planning_images: list[Path]
    bag_raw_frames: list[np.ndarray]
    detections: pd.DataFrame
    fusion: pd.DataFrame
    tracks: pd.DataFrame
    predictions: pd.DataFrame
    behaviour: pd.DataFrame
    planning_decisions: pd.DataFrame
    planned_trajectories: pd.DataFrame
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a recorded MP4 demo for challenge Tasks 1 to 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing outputs/task1 ... outputs/task7.",
    )
    parser.add_argument(
        "--bag-dir",
        "--bag",
        type=Path,
        default=Path("data"),
        dest="bag_dir",
        help="Optional ROS 2 bag directory used only for --raw-source bag.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/demo_task1_to_7.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the recorded video.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=160,
        help="Number of video frames to render. Use 0 to infer from inputs.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Output video width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Output video height in pixels.",
    )
    parser.add_argument(
        "--raw-source",
        choices=("auto", "outputs", "bag", "none"),
        default="auto",
        help=(
            "Raw sensor panel source. 'outputs' uses Task 1 raw BEV PNGs; "
            "'bag' samples compressed camera frames from the ROS bag."
        ),
    )
    parser.add_argument(
        "--camera-topic",
        default="/sensor/camera/camera_fl/image/compressed",
        help="CompressedImage topic to sample when --raw-source bag is used.",
    )
    parser.add_argument(
        "--bag-frame-limit",
        type=int,
        default=160,
        help="Maximum raw camera frames to decode from the bag.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec for the MP4 writer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.width = make_even(max(640, int(args.width)))
    args.height = make_even(max(480, int(args.height)))

    artifacts = load_artifacts(args)
    timeline = build_timeline(artifacts)
    total_frames = infer_frame_count(args.frames, artifacts, timeline)
    if total_frames <= 0:
        print("No renderable frames found. Check --outputs-dir or --raw-source.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*args.codec[:4]),
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        print(
            f"Could not open video writer for {args.output}. "
            "Try --codec avc1 or install ffmpeg-enabled OpenCV.",
            file=sys.stderr,
        )
        return 3

    image_cache: dict[Path, np.ndarray | None] = {}
    start_ts = int(timeline[0]) if len(timeline) else None

    for frame_index in range(total_frames):
        ts = select_timestamp(timeline, frame_index, total_frames)
        frame = render_frame(
            args=args,
            artifacts=artifacts,
            frame_index=frame_index,
            total_frames=total_frames,
            timestamp_ns=ts,
            start_ts=start_ts,
            image_cache=image_cache,
        )
        writer.write(frame)

    writer.release()

    print(f"Wrote {args.output} ({total_frames} frames at {args.fps:g} fps)")
    for warning in artifacts.warnings:
        print(f"warning: {warning}")
    return 0


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def load_artifacts(args: argparse.Namespace) -> DemoArtifacts:
    outputs = args.outputs_dir
    warnings: list[str] = []

    raw_images: list[Path] = []
    bag_raw_frames: list[np.ndarray] = []
    if args.raw_source in ("auto", "outputs"):
        raw_images = sorted_images(outputs / "task1", ["representative_frame_*_raw.png"])
        if not raw_images:
            raw_images = sorted_images(outputs / "task1", ["frame_*.png"])
        if args.raw_source == "outputs" and not raw_images:
            warnings.append("Task 1 raw PNGs were not found; raw sensor panel will be blank.")

    if args.raw_source == "bag":
        bag_raw_frames = load_bag_camera_frames(
            args.bag_dir,
            args.camera_topic,
            limit=max(1, args.bag_frame_limit),
            warnings=warnings,
        )
    elif args.raw_source == "auto" and not raw_images:
        bag_raw_frames = load_bag_camera_frames(
            args.bag_dir,
            args.camera_topic,
            limit=max(1, args.bag_frame_limit),
            warnings=warnings,
        )

    detection_images = sorted_images(outputs / "task1", ["frame_*.png", "representative_frame_*_final.png"])
    fusion_images = sorted_images(outputs / "task2", ["fusion_bev_*.png"])
    tracking_images = sorted_images(outputs / "task3", ["tracking_bev_*.png"])
    map_images = sorted_images(
        outputs / "task4",
        ["ego_view_*.png", "world_map_with_trajectory.png", "world_map_static.png", "freespace_map.png"],
    )
    prediction_images = sorted_images(outputs / "task5", ["prediction_bev_*.png"])
    planning_images = sorted_images(
        outputs / "task7",
        ["planning_bev.png", "overtake_scenarios.png", "strategy_timeline.png"],
    )

    for label, items in (
        ("Task 1 detection images", detection_images),
        ("Task 2 fusion images", fusion_images),
        ("Task 3 tracking images", tracking_images),
        ("Task 4 map images", map_images),
        ("Task 5 prediction images", prediction_images),
        ("Task 7 planning images", planning_images),
    ):
        if not items:
            warnings.append(f"{label} were not found.")

    detections = load_csv(outputs / "task1" / "detections.csv", warnings)
    fusion = load_csv(outputs / "task2" / "fused_detections.csv", warnings)
    tracks = load_csv(outputs / "task3" / "tracks.csv", warnings)
    predictions = load_csv(outputs / "task5" / "predictions.csv", warnings)
    behaviour = load_csv(outputs / "task6" / "behaviour_predictions.csv", warnings)
    planning_decisions = load_csv(outputs / "task7" / "planning_decisions.csv", warnings)
    planned_trajectories = load_csv(outputs / "task7" / "planned_trajectories.csv", warnings)

    return DemoArtifacts(
        raw_images=raw_images,
        detection_images=detection_images,
        fusion_images=fusion_images,
        tracking_images=tracking_images,
        map_images=map_images,
        prediction_images=prediction_images,
        planning_images=planning_images,
        bag_raw_frames=bag_raw_frames,
        detections=detections,
        fusion=fusion,
        tracks=tracks,
        predictions=predictions,
        behaviour=behaviour,
        planning_decisions=planning_decisions,
        planned_trajectories=planned_trajectories,
        warnings=warnings,
    )


def sorted_images(directory: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    return sorted({path for path in paths if path.is_file()}, key=numeric_sort_key)


def numeric_sort_key(path: Path) -> tuple[object, ...]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", path.stem))
    return (path.name.rstrip("0123456789"), numbers, path.name)


def load_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing CSV: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read {path}: {exc}")
        return pd.DataFrame()

    for column in (
        "timestamp_ns",
        "timestamp",
        "source_ts_ns",
        "track_id",
        "target_track_id",
        "frame_idx",
        "frame_index",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_bag_camera_frames(
    bag_dir: Path,
    topic: str,
    *,
    limit: int,
    warnings: list[str],
) -> list[np.ndarray]:
    if not bag_dir.exists():
        warnings.append(f"Bag directory not found: {bag_dir}")
        return []
    db_paths = sorted(bag_dir.glob("*.db3"))
    if not db_paths:
        warnings.append(f"No .db3 files found in bag directory: {bag_dir}")
        return []

    try:
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS2_JAZZY)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"rosbags unavailable; cannot decode bag camera frames: {exc}")
        return []

    samples: list[tuple[int, bytes]] = []
    for db_path in db_paths:
        if len(samples) >= limit:
            break
        try:
            with sqlite3.connect(str(db_path)) as connection:
                topic_row = connection.execute(
                    "select id, type from topics where name = ?",
                    (topic,),
                ).fetchone()
                if topic_row is None:
                    continue
                topic_id, msg_type = int(topic_row[0]), str(topic_row[1])
                rows = connection.execute(
                    "select timestamp, data from messages "
                    "where topic_id = ? order by timestamp limit ?",
                    (topic_id, limit - len(samples)),
                )
                for timestamp_ns, raw_data in rows:
                    if msg_type != COMPRESSED_IMAGE_TYPE:
                        continue
                    samples.append((int(timestamp_ns), raw_data))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not sample camera frames from {db_path}: {exc}")

    frames: list[np.ndarray] = []
    for _, raw_data in sorted(samples, key=lambda item: item[0])[:limit]:
        try:
            msg = typestore.deserialize_cdr(raw_data, COMPRESSED_IMAGE_TYPE)
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                frames.append(image)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not decode one camera frame: {exc}")
            break

    if not frames:
        warnings.append(f"No raw camera frames decoded from {topic}.")
    return frames


def build_timeline(artifacts: DemoArtifacts) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for df, column in (
        (artifacts.tracks, "timestamp_ns"),
        (artifacts.behaviour, "timestamp_ns"),
        (artifacts.planning_decisions, "timestamp_ns"),
        (artifacts.fusion, "timestamp_ns"),
        (artifacts.detections, "timestamp_ns"),
        (artifacts.predictions, "source_ts_ns"),
    ):
        if not df.empty and column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna().astype(np.int64).unique()
            if len(values):
                candidates.append(np.sort(values))

    if not candidates:
        return np.array([], dtype=np.int64)

    # Prefer Task 3/6 cadence, then fill from other tasks only if needed.
    timeline = np.unique(np.concatenate(candidates[:2] if len(candidates) >= 2 else candidates))
    if len(timeline) < 2:
        timeline = np.unique(np.concatenate(candidates))
    return np.sort(timeline).astype(np.int64)


def infer_frame_count(requested: int, artifacts: DemoArtifacts, timeline: np.ndarray) -> int:
    if requested > 0:
        return requested
    image_counts = [
        len(artifacts.raw_images),
        len(artifacts.detection_images),
        len(artifacts.fusion_images),
        len(artifacts.tracking_images),
        len(artifacts.map_images),
        len(artifacts.prediction_images),
        len(artifacts.planning_images),
        len(artifacts.bag_raw_frames),
    ]
    return max([len(timeline), *image_counts, 1])


def select_timestamp(timeline: np.ndarray, frame_index: int, total_frames: int) -> int | None:
    if len(timeline) == 0:
        return None
    idx = proportional_index(len(timeline), frame_index, total_frames)
    return int(timeline[idx])


def proportional_index(length: int, frame_index: int, total_frames: int) -> int:
    if length <= 1 or total_frames <= 1:
        return 0
    return int(round(frame_index * (length - 1) / max(total_frames - 1, 1)))


def select_path(paths: list[Path], frame_index: int, total_frames: int) -> Path | None:
    if not paths:
        return None
    return paths[proportional_index(len(paths), frame_index, total_frames)]


def select_frame(frames: list[np.ndarray], frame_index: int, total_frames: int) -> np.ndarray | None:
    if not frames:
        return None
    return frames[proportional_index(len(frames), frame_index, total_frames)]


def render_frame(
    *,
    args: argparse.Namespace,
    artifacts: DemoArtifacts,
    frame_index: int,
    total_frames: int,
    timestamp_ns: int | None,
    start_ts: int | None,
    image_cache: dict[Path, np.ndarray | None],
) -> np.ndarray:
    canvas = np.full((args.height, args.width, 3), BG, dtype=np.uint8)
    draw_header(canvas, frame_index, total_frames, timestamp_ns, start_ts, artifacts.warnings)

    rects = layout(args.width, args.height)

    raw_image = select_frame(artifacts.bag_raw_frames, frame_index, total_frames)
    if raw_image is None:
        raw_path = select_path(artifacts.raw_images, frame_index, total_frames)
        raw_image = read_image(raw_path, image_cache) if raw_path else None
    render_image_panel(
        canvas,
        rects[0],
        "Task 1 | Raw Sensor Input",
        "LiDAR/Radar BEV output or optional bag camera frame",
        raw_image,
        lines=sensor_lines(raw_image, artifacts.raw_images, artifacts.bag_raw_frames),
        missing="No raw sensor frames available.",
    )

    det_path = select_path(artifacts.detection_images, frame_index, total_frames)
    det_image = read_image(det_path, image_cache) if det_path else None
    render_image_panel(
        canvas,
        rects[1],
        "Task 1 | Opponent Detection",
        "Clusters and bounding boxes",
        det_image,
        lines=detection_lines(artifacts.detections, timestamp_ns, frame_index, total_frames),
        missing="No Task 1 detection images available.",
    )

    fusion_path = select_path(artifacts.fusion_images, frame_index, total_frames)
    fusion_image = read_image(fusion_path, image_cache) if fusion_path else None
    render_image_panel(
        canvas,
        rects[2],
        "Task 2 | Sensor Fusion",
        "Position, velocity, heading",
        fusion_image,
        lines=fusion_lines(artifacts.fusion, timestamp_ns),
        missing="No Task 2 fusion BEV images available.",
    )

    tracking_path = select_path(artifacts.tracking_images, frame_index, total_frames)
    tracking_image = read_image(tracking_path, image_cache) if tracking_path else None
    render_image_panel(
        canvas,
        rects[3],
        "Task 3 | Continuous Tracking",
        "Stable IDs and track status",
        tracking_image,
        lines=tracking_lines(artifacts.tracks, timestamp_ns),
        missing="No Task 3 tracking BEV images available.",
    )

    map_path = select_path(artifacts.map_images, frame_index, total_frames)
    map_image = read_image(map_path, image_cache) if map_path else None
    render_image_panel(
        canvas,
        rects[4],
        "Task 4 | Bird-Eye Environment Map",
        "Track boundaries, free space, ego path",
        map_image,
        lines=map_lines(artifacts.map_images),
        missing="No Task 4 environment map images available.",
    )

    prediction_path = select_path(artifacts.prediction_images, frame_index, total_frames)
    prediction_image = read_image(prediction_path, image_cache) if prediction_path else None
    render_prediction_strategy_panel(
        canvas,
        rects[5],
        artifacts=artifacts,
        timestamp_ns=timestamp_ns,
        fallback_image=prediction_image,
    )

    draw_footer(canvas, args.output)
    return canvas


def layout(width: int, height: int) -> list[Rect]:
    margin = 24
    gap = 16
    header_h = 92
    footer_h = 42
    available_w = width - 2 * margin - 2 * gap
    available_h = height - header_h - footer_h - 2 * margin - gap
    panel_w = available_w // 3
    panel_h = available_h // 2
    y0 = header_h + margin
    x0 = margin
    return [
        Rect(x0 + col * (panel_w + gap), y0 + row * (panel_h + gap), panel_w, panel_h)
        for row in range(2)
        for col in range(3)
    ]


def draw_header(
    canvas: np.ndarray,
    frame_index: int,
    total_frames: int,
    timestamp_ns: int | None,
    start_ts: int | None,
    warnings: list[str],
) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 92), (14, 18, 24), -1)
    title_scale = 0.9 if w >= 1500 else 0.68
    draw_text(canvas, "Autonomous Racing Perception and Planning Demo", (24, 35), title_scale, TEXT, 2)

    ts_text = format_timestamp(timestamp_ns, start_ts)
    draw_text(
        canvas,
        f"Tasks 1-7 | frame {frame_index + 1}/{total_frames} | {ts_text}",
        (24, 65),
        0.53 if w >= 1500 else 0.43,
        MUTED,
        1,
    )

    chips = ["T1 raw", "T1 detect", "T2 fusion", "T3 track", "T4 map", "T5 predict", "T6 behaviour", "T7 plan"]
    if w >= 1500:
        x = w - 865
        y = 24
        for chip in chips:
            chip_w = max(80, int(12 * len(chip)))
            cv2.rectangle(canvas, (x, y), (x + chip_w, y + 28), (36, 48, 64), -1)
            cv2.rectangle(canvas, (x, y), (x + chip_w, y + 28), (78, 93, 116), 1)
            draw_text(canvas, chip, (x + 8, y + 20), 0.43, TEXT, 1)
            x += chip_w + 8
    else:
        draw_text_fit(canvas, "T1-T7 demo", (w - 245, 35), 0.45, TEXT, 1, 220)

    progress_w = w - 48
    progress_y = 82
    cv2.rectangle(canvas, (24, progress_y), (24 + progress_w, progress_y + 4), (49, 57, 70), -1)
    pct = 0.0 if total_frames <= 1 else frame_index / (total_frames - 1)
    cv2.rectangle(
        canvas,
        (24, progress_y),
        (24 + int(progress_w * pct), progress_y + 4),
        CYAN,
        -1,
    )

    if warnings:
        draw_text(canvas, "degraded gracefully where artifacts are missing", (w - 360, 65), 0.43, ORANGE, 1)


def draw_footer(canvas: np.ndarray, output_path: Path) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 42), (w, h), (14, 18, 24), -1)
    draw_text_fit(
        canvas,
        f"Output: {output_path} | Generated from existing task outputs; core algorithms are not rerun.",
        (24, h - 16),
        0.47,
        MUTED,
        1,
        w - 48,
    )


def read_image(path: Path | None, cache: dict[Path, np.ndarray | None]) -> np.ndarray | None:
    if path is None:
        return None
    if path in cache:
        return cache[path]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        cache[path] = None
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = composite_alpha(image)
    cache[path] = image
    if len(cache) > 80:
        # Keep memory bounded during long renders.
        for old_key in list(cache.keys())[:20]:
            cache.pop(old_key, None)
    return image


def composite_alpha(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)


def render_image_panel(
    canvas: np.ndarray,
    rect: Rect,
    title: str,
    subtitle: str,
    image: np.ndarray | None,
    *,
    lines: list[str],
    missing: str,
) -> None:
    draw_panel_shell(canvas, rect)
    inner = Rect(rect.x + 8, rect.y + 8, rect.w - 16, rect.h - 16)
    if image is None:
        draw_missing(canvas, inner, missing)
    else:
        draw_fitted_image(canvas, inner, image)

    overlay_h = 78 + 22 * min(len(lines), 4)
    alpha_rect(canvas, rect.x, rect.y, rect.w, overlay_h, (8, 12, 18), 0.76)
    title_scale = 0.55 if rect.w >= 560 else 0.42
    body_scale = 0.42 if rect.w >= 560 else 0.33
    draw_text_fit(canvas, title, (rect.x + 14, rect.y + 26), title_scale, TEXT, 1, rect.w - 28)
    draw_text_fit(canvas, subtitle, (rect.x + 14, rect.y + 50), body_scale, MUTED, 1, rect.w - 28)
    y = rect.y + 73
    for line in lines[:4]:
        draw_text_fit(canvas, line, (rect.x + 14, y), body_scale, TEXT, 1, rect.w - 28)
        y += 22


def draw_panel_shell(canvas: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), PANEL_BG, -1)
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), PANEL_BORDER, 1)


def draw_missing(canvas: np.ndarray, rect: Rect, message: str) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (31, 36, 48), -1)
    draw_text(canvas, message, (rect.x + 20, rect.y + rect.h // 2), 0.55, MUTED, 1)


def draw_fitted_image(canvas: np.ndarray, rect: Rect, image: np.ndarray) -> None:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        draw_missing(canvas, rect, "Invalid image.")
        return
    scale = min(rect.w / src_w, rect.h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = rect.x + (rect.w - new_w) // 2
    y0 = rect.y + (rect.h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized


def sensor_lines(
    image: np.ndarray | None,
    raw_images: list[Path],
    bag_raw_frames: list[np.ndarray],
) -> list[str]:
    if image is None:
        return ["source: missing"]
    source = "ROS bag camera" if bag_raw_frames else "Task 1 raw BEV PNG"
    h, w = image.shape[:2]
    count = len(bag_raw_frames) if bag_raw_frames else len(raw_images)
    return [f"source: {source}", f"frames available: {count}", f"native size: {w} x {h}"]


def detection_lines(df: pd.DataFrame, timestamp_ns: int | None, frame_index: int, total_frames: int) -> list[str]:
    if df.empty:
        return ["detections.csv: missing"]
    rows = nearest_rows(df, "timestamp_ns", timestamp_ns)
    if rows.empty and "frame_idx" in df.columns:
        idx = proportional_index(int(df["frame_idx"].max()) + 1, frame_index, total_frames)
        rows = df[df["frame_idx"] == idx]
    count = len(rows)
    if count == 0:
        return ["detections: none at current frame"]
    mean_conf = safe_mean(rows.get("confidence"))
    points = safe_sum(rows.get("point_count"))
    return [f"clusters: {count}", f"mean confidence: {mean_conf:.2f}", f"cluster points: {points:.0f}"]


def fusion_lines(df: pd.DataFrame, timestamp_ns: int | None) -> list[str]:
    rows = nearest_rows(df, "timestamp_ns", timestamp_ns)
    if rows.empty:
        return ["fused_detections.csv: missing or no nearby rows"]
    rows = rows.sort_values("confidence", ascending=False) if "confidence" in rows.columns else rows
    lines = [f"fused objects: {len(rows)}"]
    for _, row in rows.head(3).iterrows():
        track = format_track_id(row.get("track_id", "?"))
        x = safe_float(row.get("x", row.get("center_x", np.nan)))
        y = safe_float(row.get("y", row.get("center_y", np.nan)))
        speed = safe_float(row.get("speed", np.nan))
        yaw = safe_float(row.get("yaw", np.nan))
        lines.append(f"ID {track}: pos=({x:.1f},{y:.1f}) v={speed:.1f} m/s yaw={yaw:.2f}")
    return lines


def tracking_lines(df: pd.DataFrame, timestamp_ns: int | None) -> list[str]:
    rows = nearest_rows(df, "timestamp_ns", timestamp_ns)
    if rows.empty:
        return ["tracks.csv: missing or no nearby rows"]
    lines = [f"active IDs: {len(rows)}"]
    if "status" in rows.columns:
        status_counts = rows["status"].astype(str).value_counts()
        lines.append(", ".join(f"{key}:{value}" for key, value in status_counts.head(3).items()))
    ids = ", ".join(format_track_id(v) for v in rows["track_id"].head(8)) if "track_id" in rows.columns else "unknown"
    lines.append(f"IDs: {ids}")
    return lines


def map_lines(map_images: list[Path]) -> list[str]:
    return [f"map views available: {len(map_images)}", "dynamic objects masked when Task 3 tracks exist"]


def render_prediction_strategy_panel(
    canvas: np.ndarray,
    rect: Rect,
    *,
    artifacts: DemoArtifacts,
    timestamp_ns: int | None,
    fallback_image: np.ndarray | None,
) -> None:
    draw_panel_shell(canvas, rect)
    top_h = 126 if rect.h >= 360 else 104
    plot = Rect(rect.x + 12, rect.y + top_h, rect.w - 24, rect.h - top_h - 12)

    tracks = nearest_rows(artifacts.tracks, "timestamp_ns", timestamp_ns)
    behaviour = nearest_rows(artifacts.behaviour, "timestamp_ns", timestamp_ns)
    planning = nearest_rows(artifacts.planning_decisions, "timestamp_ns", timestamp_ns)
    planned_path = nearest_rows(artifacts.planned_trajectories, "timestamp_ns", timestamp_ns)
    if tracks.empty and fallback_image is not None:
        draw_fitted_image(canvas, Rect(rect.x + 8, rect.y + 8, rect.w - 16, rect.h - 16), fallback_image)
        alpha_rect(canvas, rect.x, rect.y, rect.w, 112, (8, 12, 18), 0.78)
        draw_text(canvas, "Tasks 5-7 | Predictions, Behaviour, Planning", (rect.x + 14, rect.y + 27), 0.52, TEXT, 1)
        draw_text(canvas, "Fallback: rendered Task 5 prediction BEV image", (rect.x + 14, rect.y + 53), 0.42, MUTED, 1)
        draw_text(canvas, "Ego plan overlay unavailable because tracks.csv has no nearby rows", (rect.x + 14, rect.y + 78), 0.42, TEXT, 1)
        return

    cv2.rectangle(canvas, (plot.x, plot.y), (plot.x + plot.w, plot.y + plot.h), (15, 23, 36), -1)
    draw_bev_grid(canvas, plot)

    behaviour_by_id = behaviour_lookup(behaviour)
    pred_rows = select_prediction_rows(artifacts.predictions, timestamp_ns)
    draw_predictions(canvas, plot, tracks, pred_rows)
    draw_tracks(canvas, plot, tracks, behaviour_by_id)
    strategy = draw_planned_trajectory(canvas, plot, planning, planned_path)

    alpha_rect(canvas, rect.x, rect.y, rect.w, top_h, (8, 12, 18), 0.78)
    title_scale = 0.50 if rect.w >= 560 else 0.38
    body_scale = 0.40 if rect.w >= 560 else 0.31
    draw_text_fit(
        canvas,
        "Tasks 5-7 | Prediction, Behaviour, Ego Overtake Planning",
        (rect.x + 14, rect.y + 26),
        title_scale,
        TEXT,
        1,
        rect.w - 28,
    )
    draw_text_fit(
        canvas,
        "Opponent forecasts, behaviour labels, and selected Task 7 planned path",
        (rect.x + 14, rect.y + 50),
        body_scale,
        MUTED,
        1,
        rect.w - 28,
    )

    summary = prediction_summary_lines(tracks, pred_rows, behaviour_by_id, strategy, planning)
    y = rect.y + 72
    for line in summary[:4]:
        draw_text_fit(canvas, line, (rect.x + 18, y), body_scale, TEXT, 1, rect.w - 36)
        y += 20


def draw_bev_grid(canvas: np.ndarray, rect: Rect) -> None:
    for lateral in range(-20, 25, 5):
        x0, y0 = bev_to_pixel(0, lateral, rect)
        x1, y1 = bev_to_pixel(65, lateral, rect)
        cv2.line(canvas, (x0, y0), (x1, y1), GRID, 1)
        draw_text(canvas, f"{lateral:+d}", (x0 + 2, rect.y + rect.h - 5), 0.32, MUTED, 1)
    for forward in range(0, 70, 10):
        x0, y0 = bev_to_pixel(forward, -25, rect)
        x1, y1 = bev_to_pixel(forward, 25, rect)
        cv2.line(canvas, (x0, y0), (x1, y1), GRID, 1)
        draw_text(canvas, f"{forward}m", (rect.x + 4, y0 - 3), 0.32, MUTED, 1)
    draw_text(canvas, "forward x", (rect.x + 8, rect.y + 18), 0.34, MUTED, 1)
    draw_text(canvas, "lateral y", (rect.x + rect.w - 84, rect.y + rect.h - 10), 0.34, MUTED, 1)


def bev_to_pixel(x_forward: float, y_lateral: float, rect: Rect) -> tuple[int, int]:
    x_min, x_max = -5.0, 70.0
    y_min, y_max = -25.0, 25.0
    px = rect.x + int((y_lateral - y_min) / (y_max - y_min) * rect.w)
    py = rect.y + rect.h - int((x_forward - x_min) / (x_max - x_min) * rect.h)
    return px, py


def draw_tracks(
    canvas: np.ndarray,
    rect: Rect,
    tracks: pd.DataFrame,
    behaviour_by_id: dict[int, tuple[str, float]],
) -> None:
    if tracks.empty:
        draw_text(canvas, "No current tracks", (rect.x + 20, rect.y + 42), 0.55, MUTED, 1)
        return

    tracks = tracks.copy()
    if "center_x" not in tracks.columns or "center_y" not in tracks.columns:
        return
    tracks = tracks.sort_values("center_x").head(10)
    for _, row in tracks.iterrows():
        track_id = int(safe_float(row.get("track_id", -1), -1))
        x = safe_float(row.get("center_x", np.nan))
        y = safe_float(row.get("center_y", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        color = color_for_id(track_id)
        length = max(2.5, safe_float(row.get("length", 4.5), 4.5))
        width = max(1.0, safe_float(row.get("width", 2.0), 2.0))
        yaw = safe_float(row.get("yaw", 0.0), 0.0)
        corners = oriented_box_pixels(x, y, length, width, yaw, rect)
        cv2.polylines(canvas, [corners], True, color, 2, cv2.LINE_AA)
        px, py = bev_to_pixel(x, y, rect)
        cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)

        vx = safe_float(row.get("vx", 0.0), 0.0)
        vy = safe_float(row.get("vy", 0.0), 0.0)
        end_px, end_py = bev_to_pixel(x + 0.35 * vx, y + 0.35 * vy, rect)
        cv2.arrowedLine(canvas, (px, py), (end_px, end_py), GREEN, 1, cv2.LINE_AA, tipLength=0.25)

        behaviour, confidence = behaviour_by_id.get(track_id, ("unknown", 0.0))
        label = f"ID {track_id} {behaviour} {confidence:.2f}"
        draw_label(canvas, label, (px + 7, py - 7), color)


def oriented_box_pixels(
    cx: float,
    cy: float,
    length: float,
    width: float,
    yaw: float,
    rect: Rect,
) -> np.ndarray:
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=float,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    points = local @ rot.T
    points[:, 0] += cx
    points[:, 1] += cy
    return np.array([bev_to_pixel(float(x), float(y), rect) for x, y in points], dtype=np.int32)


def draw_predictions(
    canvas: np.ndarray,
    rect: Rect,
    tracks: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    if predictions.empty:
        return

    active_ids = set()
    if not tracks.empty and "track_id" in tracks.columns:
        active_ids = {int(safe_float(v, -1)) for v in tracks["track_id"].tolist()}

    for track_id, group in predictions.groupby("track_id"):
        tid = int(safe_float(track_id, -1))
        if active_ids and tid not in active_ids:
            continue
        group = group.sort_values("horizon_s") if "horizon_s" in group.columns else group
        points: list[tuple[int, int]] = []
        current = tracks[tracks["track_id"] == tid] if not tracks.empty and "track_id" in tracks.columns else pd.DataFrame()
        if not current.empty:
            row = current.iloc[0]
            points.append(bev_to_pixel(safe_float(row.get("center_x", 0.0)), safe_float(row.get("center_y", 0.0)), rect))
        for _, row in group.iterrows():
            px, py = bev_to_pixel(safe_float(row.get("pred_x", np.nan)), safe_float(row.get("pred_y", np.nan)), rect)
            points.append((px, py))
        if len(points) >= 2:
            color = color_for_id(tid)
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            for idx, point in enumerate(points[1:], start=1):
                cv2.circle(canvas, point, max(2, 5 - idx), color, -1, cv2.LINE_AA)


def draw_planned_trajectory(
    canvas: np.ndarray,
    rect: Rect,
    planning: pd.DataFrame,
    planned_path: pd.DataFrame,
) -> str:
    if not planned_path.empty and {"planned_x", "planned_y"}.issubset(planned_path.columns):
        path = planned_path.copy()
        if "is_selected" in path.columns:
            selected = path["is_selected"].astype(str).str.lower().isin(("true", "1", "yes"))
            if selected.any():
                path = path[selected]
        if "step_index" in path.columns:
            path = path.sort_values("step_index")
        pts = np.array(
            [
                bev_to_pixel(safe_float(row.get("planned_x", 0.0)), safe_float(row.get("planned_y", 0.0)), rect)
                for _, row in path.iterrows()
            ],
            dtype=np.int32,
        )
    else:
        xs = np.linspace(0.0, 65.0, 40)
        ys = np.zeros_like(xs)
        pts = np.array([bev_to_pixel(float(x), float(y), rect) for x, y in zip(xs, ys)], dtype=np.int32)

    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, CYAN, 3, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(pts[-3]), tuple(pts[-1]), CYAN, 3, cv2.LINE_AA, tipLength=0.4)

    ego_box = oriented_box_pixels(0.0, 0.0, 4.6, 2.0, 0.0, rect)
    cv2.polylines(canvas, [ego_box], True, TEXT, 2, cv2.LINE_AA)
    ego_px = bev_to_pixel(0.0, 0.0, rect)
    draw_label(canvas, "EGO", (ego_px[0] + 8, ego_px[1] - 8), TEXT)
    if planning.empty:
        return "Task 7 decision: unavailable"
    row = planning.iloc[0]
    strategy = str(row.get("selected_strategy", "unknown"))
    target = format_track_id(row.get("target_track_id", "?"))
    margin = safe_float(row.get("safety_margin", np.nan))
    confidence = safe_float(row.get("confidence", np.nan))
    return f"Task 7: {strategy} | target ID {target} | margin {margin:.2f} m | conf {confidence:.2f}"


def behaviour_lookup(behaviour: pd.DataFrame) -> dict[int, tuple[str, float]]:
    lookup: dict[int, tuple[str, float]] = {}
    if behaviour.empty or "track_id" not in behaviour.columns:
        return lookup
    for _, row in behaviour.iterrows():
        track_id = int(safe_float(row.get("track_id", -1), -1))
        label = str(row.get("predicted_behaviour", "unknown"))
        confidence = safe_float(row.get("confidence", 0.0), 0.0)
        lookup[track_id] = (label, confidence)
    return lookup


def select_prediction_rows(predictions: pd.DataFrame, timestamp_ns: int | None) -> pd.DataFrame:
    if predictions.empty or "source_ts_ns" not in predictions.columns or timestamp_ns is None:
        return pd.DataFrame()
    values = predictions["source_ts_ns"].dropna().astype(np.int64).unique()
    if len(values) == 0:
        return pd.DataFrame()
    source_ts = int(values[np.argmin(np.abs(values.astype(np.int64) - int(timestamp_ns)))])
    rows = predictions[predictions["source_ts_ns"] == source_ts].copy()
    if rows.empty:
        return rows
    if "model" in rows.columns:
        preferred_model = "ctrv" if (rows["model"].astype(str).str.lower() == "ctrv").any() else str(rows["model"].iloc[0])
        rows = rows[rows["model"].astype(str).str.lower() == preferred_model.lower()]
    return rows


def prediction_summary_lines(
    tracks: pd.DataFrame,
    predictions: pd.DataFrame,
    behaviour_by_id: dict[int, tuple[str, float]],
    strategy: str,
    planning: pd.DataFrame,
) -> list[str]:
    labels = {}
    for label, _ in behaviour_by_id.values():
        labels[label] = labels.get(label, 0) + 1
    label_text = ", ".join(f"{key}:{value}" for key, value in sorted(labels.items())[:3]) or "none"
    safety = "Safety: unavailable"
    if not planning.empty:
        row = planning.iloc[0]
        risk = safe_float(row.get("collision_risk_score", np.nan))
        min_dist = safe_float(row.get("min_predicted_distance", np.nan))
        safety = f"Safety: risk={risk:.2f}, min distance={min_dist:.2f} m"
    return [
        f"tracked objects: {0 if tracks.empty else len(tracks)} | prediction rows: {0 if predictions.empty else len(predictions)}",
        f"behaviour labels: {label_text}",
        safety,
        strategy,
    ]


def nearest_rows(df: pd.DataFrame, timestamp_column: str, timestamp_ns: int | None) -> pd.DataFrame:
    if df.empty or timestamp_column not in df.columns:
        return pd.DataFrame()
    clean = df.dropna(subset=[timestamp_column])
    if clean.empty:
        return pd.DataFrame()
    if timestamp_ns is None:
        target = int(clean[timestamp_column].iloc[0])
    else:
        target = int(timestamp_ns)
    values = clean[timestamp_column].astype(np.int64).unique()
    nearest = int(values[np.argmin(np.abs(values.astype(np.int64) - target))])
    return clean[clean[timestamp_column] == nearest].copy()


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def safe_mean(values: object) -> float:
    if values is None:
        return float("nan")
    arr = pd.to_numeric(values, errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def safe_sum(values: object) -> float:
    if values is None:
        return 0.0
    arr = pd.to_numeric(values, errors="coerce").dropna()
    return float(arr.sum()) if len(arr) else 0.0


def format_track_id(value: object) -> str:
    number = safe_float(value)
    if np.isfinite(number):
        return str(int(number))
    return str(value)


def format_timestamp(timestamp_ns: int | None, start_ts: int | None) -> str:
    if timestamp_ns is None:
        return "timestamp unavailable"
    dt = 0.0 if start_ts is None else (int(timestamp_ns) - int(start_ts)) / 1_000_000_000.0
    wall = datetime.fromtimestamp(int(timestamp_ns) / 1_000_000_000.0, tz=timezone.utc)
    return f"t={dt:6.2f}s | {wall.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC"


def color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (82, 196, 255),
        (116, 214, 146),
        (95, 164, 255),
        (214, 128, 255),
        (255, 210, 64),
        (129, 140, 248),
        (248, 113, 113),
        (45, 212, 191),
    ]
    return palette[abs(track_id) % len(palette)]


def draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    alpha_rect(canvas, x - 3, y - th - 5, tw + 8, th + 9, (8, 12, 18), 0.70)
    draw_text(canvas, text, (x, y), 0.36, color, 1)


def alpha_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas.shape[1], x + w)
    y1 = min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    overlay = canvas[y0:y1, x0:x1].copy()
    overlay[:, :] = color
    cv2.addWeighted(overlay, alpha, canvas[y0:y1, x0:x1], 1.0 - alpha, 0.0, canvas[y0:y1, x0:x1])


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_fit(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    max_width: int,
) -> None:
    fitted = str(text)
    if max_width > 12:
        while len(fitted) > 4:
            (tw, _), _ = cv2.getTextSize(fitted, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            if tw <= max_width:
                break
            fitted = fitted[:-4].rstrip() + "..."
    draw_text(canvas, fitted, origin, scale, color, thickness)


if __name__ == "__main__":
    raise SystemExit(main())
