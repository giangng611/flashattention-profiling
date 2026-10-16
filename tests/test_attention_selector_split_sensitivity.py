from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AttentionSelectorSplitSensitivityTest(unittest.TestCase):
    def test_balanced_split_evaluation_is_complete_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            trials_path = output_directory / "trials.csv"
            summary_path = output_directory / "summary.csv"
            metadata_path = output_directory / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    "src/evaluate_attention_selector_split_sensitivity.py",
                    "--split-count",
                    "5",
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

            self.assertEqual(len(trials), 5 * 6 * 8)
            self.assertEqual(len(summary), 8)
            self.assertEqual(metadata["split_count"], 5)
            self.assertEqual(metadata["development_count"], 10)
            self.assertEqual(metadata["held_out_count"], 6)
            self.assertEqual(metadata["splits"][0]["split_id"], "original")
            self.assertEqual(len({row["split_id"] for row in trials}), 5)
            self.assertTrue(all(int(row["held_out_evaluations"]) == 30 for row in summary))


if __name__ == "__main__":
    unittest.main()
