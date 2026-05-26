from __future__ import annotations

import inspect
import sys
import unittest
from unittest.mock import patch

from scripts import run_task1
from src.task1_detection import run_task1_pipeline


class Task1CliTest(unittest.TestCase):
    def test_defaults_process_and_plot_200_frames(self) -> None:
        with patch.object(sys, "argv", ["run_task1.py"]):
            args = run_task1.parse_args()

        self.assertEqual(args.max_frames, 200)
        self.assertEqual(args.max_plots, 200)

    def test_max_plots_argument_is_forwarded_to_pipeline_api(self) -> None:
        signature = inspect.signature(run_task1_pipeline)

        self.assertIn("max_plots", signature.parameters)


if __name__ == "__main__":
    unittest.main()
