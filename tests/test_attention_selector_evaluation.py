from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AttentionSelectorEvaluationTest(unittest.TestCase):
    def test_offline_evaluation_outputs_expected_split_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            trials_path = output_directory / "trials.csv"
            summary_path = output_directory / "summary.csv"
            metadata_path = output_directory / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    "src/evaluate_attention_config_selectors.py",
                    "--trials-output",
                    str(trials_path),
                    "--summary-output",
                    str(summary_path),
                    "--metadata-output",
                    str(metadata_path),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with trials_path.open(newline="") as handle:
                trials = list(csv.DictReader(handle))
            with summary_path.open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text())

            self.assertEqual(len(trials), 2448)
            self.assertEqual(len(summary), 12)
            self.assertEqual({row["split"] for row in trials}, {"held_out"})
            self.assertEqual(metadata["development_workload_count"], 10)
            self.assertEqual(metadata["held_out_workload_count"], 6)
            self.assertEqual(metadata["config_count"], 48)
            self.assertEqual(len(metadata["input_sweeps"]), 1)
            self.assertEqual(metadata["best_fixed_development_config"], "bm32_bn64_w4_s2")

            tree_top_two = next(
                row
                for row in summary
                if row["method"] == "decision_tree_ranker" and row["budget"] == "2"
            )
            self.assertEqual(tree_top_two["selection_success_rate"], "1.0")
            self.assertAlmostEqual(
                float(tree_top_two["geomean_oracle_relative_performance"]),
                0.9260293463256642,
            )
            self.assertEqual(tree_top_two["target_90pct_oracle_within_25pct_configs"], "True")

    def test_repeated_sweeps_use_geometric_mean_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            trials_path = output_directory / "trials.csv"
            summary_path = output_directory / "summary.csv"
            metadata_path = output_directory / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    "src/evaluate_attention_config_selectors.py",
                    "--sweep",
                    "results/expanded_attention_sweep.csv",
                    "results/expanded_attention_sweep_seed1.csv",
                    "results/expanded_attention_sweep_seed2.csv",
                    "--trials-output",
                    str(trials_path),
                    "--summary-output",
                    str(summary_path),
                    "--metadata-output",
                    str(metadata_path),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with summary_path.open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text())

            self.assertEqual(len(summary), 12)
            self.assertEqual(len(metadata["input_sweeps"]), 3)
            self.assertEqual(
                metadata["latency_aggregation"],
                "geometric mean of per-run median latency",
            )
            self.assertIn(metadata["best_fixed_development_config"], {
                "bm32_bn32_w4_s2",
                "bm32_bn64_w4_s2",
            })


if __name__ == "__main__":
    unittest.main()
