from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_task1
from src.task1_detection import DetectionConfig, FrameResult, PipelineResult
from src.bag_inspector import BagSummary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "data" / "eva.urdf"


class Task1UrdfGeometryTest(unittest.TestCase):
    def test_lidar_front_transform_is_resolved_from_urdf(self) -> None:
        from src.task1_detection import load_urdf_geometry

        geometry = load_urdf_geometry(URDF_PATH)
        transform = geometry.transform_sensor_to_base("lidar_front", "center_of_gravity")

        np.testing.assert_allclose(transform[:3, 3], [0.662, 0.0, 0.610], atol=1e-3)
        np.testing.assert_allclose(
            transform[:3, :3] @ np.array([0.0, 0.0, 1.0]),
            [0.99995, 0.00841, 0.00480],
            atol=1e-4,
        )

    def test_ego_filter_removes_points_inside_configured_safety_box(self) -> None:
        from src.task1_detection import filter_base_points_for_detection

        config = DetectionConfig(
            ego_length=4.0,
            ego_width=2.0,
            ego_height=1.5,
            safety_margin=0.1,
            forward_range=(-10.0, 20.0),
            lateral_range=(-10.0, 10.0),
            ground_band_half_height=0.0,
        )
        base_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.05, 0.0, 0.0],
                [2.11, 0.0, 0.0],
                [8.0, 1.0, 0.5],
            ],
            dtype=np.float64,
        )

        filtered = filter_base_points_for_detection(base_xyz, config)

        self.assertEqual(filtered.stats["ego_self_points_rejected"], 2)
        np.testing.assert_allclose(filtered.after_ego_points[:, :2], [[2.11, 0.0], [8.0, 1.0]])
        np.testing.assert_allclose(filtered.final_bev_points, [[2.11, 0.0], [8.0, 1.0]])


class Task1CliUrdfTest(unittest.TestCase):
    def test_default_max_frames_is_all_and_debug_limit_is_separate(self) -> None:
        with patch.object(sys, "argv", ["run_task1.py"]):
            args = run_task1.parse_args()

        self.assertIsNone(args.max_frames)
        self.assertIsNone(args.debug_max_frames)

        with patch.object(sys, "argv", ["run_task1.py", "--max-frames", "all"]):
            args = run_task1.parse_args()

        self.assertIsNone(args.max_frames)

        with patch.object(sys, "argv", ["run_task1.py", "--debug-max-frames", "7"]):
            args = run_task1.parse_args()

        self.assertEqual(args.debug_max_frames, 7)

    def test_pipeline_accepts_urdf_arguments(self) -> None:
        import inspect
        from src.task1_detection import run_task1_pipeline

        signature = inspect.signature(run_task1_pipeline)

        self.assertIn("urdf_path", signature.parameters)
        self.assertIn("allow_identity_transform", signature.parameters)
        self.assertIn("base_frame", signature.parameters)


class Task1EvaluationMetricsTest(unittest.TestCase):
    def test_metrics_state_no_labelled_accuracy_and_filter_reductions(self) -> None:
        from src.task1_detection import build_evaluation_metrics

        detections = pd.DataFrame(
            [
                {
                    "width": 2.0,
                    "length": 4.5,
                    "point_count": 30,
                    "confidence": 0.8,
                }
            ]
        )
        result = PipelineResult(
            summary=BagSummary(
                bag_dir=PROJECT_ROOT / "data",
                storage_identifier="sqlite3",
                ros_distro="jazzy",
                duration_s=1.0,
                starting_time_ns=0,
                message_count=1,
                topics=[],
                storage_files=[],
            ),
            selected_topic=None,
            axis_mapping=None,
            frames=[
                FrameResult(
                    frame_index=0,
                    timestamp_ns=1,
                    topic="/sensor/lidar_front/points",
                    source_frame="lidar_front",
                    bev_points=np.zeros((0, 2), dtype=np.float32),
                    detections=[{"detection_id": "0000_00"}],
                    sensor_frame="lidar_front",
                    base_frame="center_of_gravity",
                    filter_stats={
                        "raw_cluster_count": 10,
                        "after_ground_cluster_count": 7,
                        "after_ego_cluster_count": 6,
                        "ground_filter_clusters_reduced": 3,
                        "ego_self_filter_clusters_reduced": 1,
                        "ground_filter_points_rejected": 12,
                        "ego_self_points_rejected": 2,
                    },
                    cluster_stats={
                        "point_count_filter_rejected_clusters": 4,
                        "vehicle_size_filter_rejected_clusters": 1,
                        "all_cluster_point_counts": [5, 30],
                        "retained_cluster_point_counts": [30],
                    },
                )
            ],
            detections=detections,
        )

        metrics = build_evaluation_metrics(result, DetectionConfig(max_frames=None))

        self.assertFalse(metrics["labelled_metrics_available"])
        self.assertEqual(metrics["frames"]["max_frames_argument"], "all")
        self.assertEqual(metrics["filters"]["raw_clusters_before_filters"], 10)
        self.assertEqual(metrics["filters"]["ground_filter_clusters_reduced"], 3)
        self.assertEqual(metrics["filters"]["ego_self_clusters_reduced"], 1)
        self.assertEqual(metrics["filters"]["final_detections_after_all_filters"], 1)


if __name__ == "__main__":
    unittest.main()
