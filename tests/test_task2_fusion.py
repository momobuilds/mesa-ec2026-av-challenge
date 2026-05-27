from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_task2 import _format_projection_label, _select_camera_projection_frames
from src.task2_fusion import _annotate_camera, _limit_velocity_pair


class Task2FusionTest(unittest.TestCase):
    def test_camera_projection_does_not_claim_camera_sensor_support(self) -> None:
        row = {
            "timestamp_ns": 100,
            "center_x": 30.0,
            "center_y": 0.0,
            "length": 4.0,
            "width": 2.0,
            "yaw": 0.0,
            "confidence": 0.70,
            "sensors": "lidar_front",
            "n_camera": 0,
        }
        camera_index = {
            "topic": "/sensor/camera/camera_fl/image/compressed",
            "timestamps_ns": np.array([100], dtype=np.int64),
        }

        _annotate_camera(row, camera_index, window_ns=50)

        self.assertTrue(row["camera_validated"])
        self.assertEqual(row["n_camera"], 0)
        self.assertEqual(row["sensors"], "lidar_front")
        self.assertEqual(row["confidence"], 0.70)

    def test_velocity_limit_defaults_to_visualization_safe_speed(self) -> None:
        vx, vy = _limit_velocity_pair(100.0, 0.0)

        self.assertLessEqual((vx**2 + vy**2) ** 0.5, 35.0)

    def test_camera_projection_frame_selection_prefers_visible_observed_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "timestamp_ns": 1,
                    "is_predicted": False,
                    "camera_validated": True,
                    "camera_u": 900,
                    "camera_v": 260,
                    "center_x": 45,
                    "confidence": 0.8,
                },
                {
                    "timestamp_ns": 1,
                    "is_predicted": True,
                    "camera_validated": True,
                    "camera_u": 910,
                    "camera_v": 260,
                    "center_x": 50,
                    "confidence": 0.9,
                },
                {
                    "timestamp_ns": 2,
                    "is_predicted": False,
                    "camera_validated": True,
                    "camera_u": 2000,
                    "camera_v": 260,
                    "center_x": 35,
                    "confidence": 0.9,
                },
                {
                    "timestamp_ns": 3,
                    "is_predicted": False,
                    "camera_validated": True,
                    "camera_u": 950,
                    "camera_v": 265,
                    "center_x": 30,
                    "confidence": 0.7,
                },
            ]
        )

        selected = _select_camera_projection_frames(df, max_images=2)

        self.assertEqual(selected, [3, 1])

    def test_projection_label_includes_identity_distance_and_timestamp(self) -> None:
        label = _format_projection_label(
            {
                "track_id": 7,
                "center_x": 42.3,
                "timestamp_ns": 123456,
                "sensors": "lidar_front+radar_front",
            }
        )

        self.assertIn("T7", label)
        self.assertIn("42.3m", label)
        self.assertIn("t=123456", label)
        self.assertIn("lidar_front+radar_front", label)


if __name__ == "__main__":
    unittest.main()
