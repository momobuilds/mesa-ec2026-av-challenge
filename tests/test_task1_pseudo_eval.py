from __future__ import annotations

import unittest

import pandas as pd

from src.task1_pseudo_eval import evaluate_task1_against_tracks, evaluate_task1_temporal_consistency


class Task1PseudoEvaluationTest(unittest.TestCase):
    def test_matches_detections_to_nearest_track_timestamp(self) -> None:
        detections = pd.DataFrame(
            [
                {
                    "timestamp_ns": 1_000,
                    "detection_id": "0000_00",
                    "center_x_base": 10.0,
                    "center_y_base": 0.0,
                    "length": 4.0,
                    "width": 2.0,
                },
                {
                    "timestamp_ns": 1_000,
                    "detection_id": "0000_01",
                    "center_x_base": 30.0,
                    "center_y_base": 0.0,
                    "length": 4.0,
                    "width": 2.0,
                },
            ]
        )
        tracks = pd.DataFrame(
            [
                {
                    "timestamp_ns": 1_020,
                    "track_id": 7,
                    "status": "CONFIRMED",
                    "center_x": 10.5,
                    "center_y": 0.2,
                    "length": 4.0,
                    "width": 2.0,
                }
            ]
        )

        result = evaluate_task1_against_tracks(
            detections,
            tracks,
            timestamp_tolerance_ns=50,
            distance_gate_m=2.0,
        )

        self.assertEqual(result.summary["matched_pairs"], 1)
        self.assertEqual(result.summary["detections_evaluated"], 2)
        self.assertEqual(result.summary["pseudo_tracks_evaluated"], 1)
        self.assertAlmostEqual(result.summary["pseudo_precision"], 0.5)
        self.assertAlmostEqual(result.summary["pseudo_recall"], 1.0)
        self.assertGreater(result.summary["mean_axis_aligned_iou"], 0.0)
        self.assertEqual(len(result.matches), 1)

    def test_temporal_consistency_promotes_repeated_detections(self) -> None:
        detections = pd.DataFrame(
            [
                {"timestamp_ns": 1_000, "detection_id": "0", "center_x_base": 10.0, "center_y_base": 0.0, "length": 4.0, "width": 2.0},
                {"timestamp_ns": 2_000, "detection_id": "1", "center_x_base": 10.5, "center_y_base": 0.1, "length": 4.0, "width": 2.0},
                {"timestamp_ns": 3_000, "detection_id": "2", "center_x_base": 11.0, "center_y_base": 0.2, "length": 4.0, "width": 2.0},
                {"timestamp_ns": 3_000, "detection_id": "noise", "center_x_base": 80.0, "center_y_base": 10.0, "length": 4.0, "width": 2.0},
            ]
        )

        result = evaluate_task1_temporal_consistency(
            detections,
            distance_gate_m=2.0,
            min_confirmed_hits=3,
            max_gap_frames=1,
        )

        self.assertEqual(result.summary["confirmed_tracks"], 1)
        self.assertEqual(result.summary["detections_on_confirmed_tracks"], 3)
        self.assertAlmostEqual(result.summary["temporal_pseudo_precision"], 0.75)


if __name__ == "__main__":
    unittest.main()
