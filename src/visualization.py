"""BEV visualization helpers for Task 1 detections."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_bev_frame(
    points: np.ndarray,
    detections: list[Mapping[str, float | int | str]],
    output_path: str | Path,
    *,
    title: str,
    forward_range: tuple[float, float],
    lateral_range: tuple[float, float],
) -> None:
    """Save a top-down plot with raw points and detection boxes.

    ``points`` is expected to use columns ``[forward_m, lateral_m]``.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    if len(points):
        ax.scatter(
            points[:, 1],
            points[:, 0],
            s=1,
            c="#7f7f7f",
            alpha=0.35,
            linewidths=0,
            label="raw points",
        )

    colors = plt.cm.tab10.colors
    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        min_x = float(detection["min_x"])
        max_x = float(detection["max_x"])
        min_y = float(detection["min_y"])
        max_y = float(detection["max_y"])
        detection_id = detection["detection_id"]

        rect = Rectangle(
            (min_y, min_x),
            max_y - min_y,
            max_x - min_x,
            fill=False,
            linewidth=2,
            edgecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            float(detection["center_y"]),
            float(detection["center_x"]),
            str(detection_id),
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        )

    ax.set_title(title)
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_xlim(lateral_range)
    ax.set_ylim(forward_range)
    ax.grid(True, color="#d8d8d8", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
