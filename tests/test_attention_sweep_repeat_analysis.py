from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AttentionSweepRepeatAnalysisTest(unittest.TestCase):
    def test_repeat_analysis_outputs_complete_workload_and_config_grids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            config_path = output_directory / "config.csv"
            workload_path = output_directory / "workload.csv"
            metadata_path = output_directory / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    "src/analyze_attention_sweep_repetitions.py",
                    "--config-output",
                    str(config_path),
                    "--workload-output",
                    str(workload_path),
                    "--metadata-output",
                    str(metadata_path),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with config_path.open(newline="") as handle:
                config_rows = list(csv.DictReader(handle))
            with workload_path.open(newline="") as handle:
                workload_rows = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text())

            self.assertEqual(len(config_rows), 16 * 48)
            self.assertEqual(len(workload_rows), 16)
            self.assertEqual(metadata["run_count"], 3)
            self.assertEqual(metadata["workload_count"], 16)
            self.assertEqual(metadata["config_count_per_workload"], 48)
            self.assertGreaterEqual(
                metadata["common_exact_oracle_workloads"],
                metadata["all_run_top1_agreement_workloads"],
            )
            self.assertEqual({row["split"] for row in workload_rows}, {"development", "held_out"})
            self.assertTrue(
                all(int(row["valid_config_count_all_runs"]) >= 40 for row in workload_rows)
            )


if __name__ == "__main__":
    unittest.main()
