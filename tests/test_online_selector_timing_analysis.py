from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze_online_selector_timing import (  # noqa: E402
    build_comparison_rows,
    load_summary,
    validate_compatible_metadata,
)


class OnlineSelectorTimingAnalysisTest(unittest.TestCase):
    def test_preserved_outputs_are_compatible_and_complete(self) -> None:
        top_k = load_summary(REPO_ROOT / "results/online_selector_top4_summary.csv")
        exhaustive = load_summary(
            REPO_ROOT / "results/online_selector_exhaustive_summary.csv"
        )
        top_k_metadata = json.loads(
            (REPO_ROOT / "results/online_selector_top4_metadata.json").read_text()
        )
        exhaustive_metadata = json.loads(
            (REPO_ROOT / "results/online_selector_exhaustive_metadata.json").read_text()
        )
        validate_compatible_metadata(top_k_metadata, exhaustive_metadata)
        rows = build_comparison_rows(top_k, exhaustive)

        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["top_k_budget"] == 4 for row in rows))
        self.assertTrue(all(row["exhaustive_budget"] == 48 for row in rows))
        self.assertEqual(sum(row["top_k_runtime_errors"] for row in rows), 0)
        self.assertEqual(sum(row["exhaustive_runtime_errors"] for row in rows), 32)
        self.assertEqual(
            sum(
                row["top_k_correctness_failures"]
                + row["exhaustive_correctness_failures"]
                for row in rows
            ),
            0,
        )
        self.assertTrue(
            math.isclose(
                sum(row["top_k_tuning_wall_time_ms"] for row in rows),
                1986.945029348135,
            )
        )
        self.assertTrue(
            math.isclose(
                sum(row["exhaustive_tuning_wall_time_ms"] for row in rows),
                12600.03530420363,
            )
        )

    def test_cli_writes_six_row_comparison_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.csv"
            metadata = Path(directory) / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "src/analyze_online_selector_timing.py"),
                    "--top-k-summary",
                    str(REPO_ROOT / "results/online_selector_top4_summary.csv"),
                    "--top-k-metadata",
                    str(REPO_ROOT / "results/online_selector_top4_metadata.json"),
                    "--exhaustive-summary",
                    str(REPO_ROOT / "results/online_selector_exhaustive_summary.csv"),
                    "--exhaustive-metadata",
                    str(REPO_ROOT / "results/online_selector_exhaustive_metadata.json"),
                    "--output",
                    str(output),
                    "--metadata-output",
                    str(metadata),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            result_metadata = json.loads(metadata.read_text())
            self.assertEqual(len(rows), 6)
            self.assertAlmostEqual(
                result_metadata["direct_geomean_oracle_relative_performance"],
                0.9696556610286062,
            )


if __name__ == "__main__":
    unittest.main()
