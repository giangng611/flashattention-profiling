from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_attention_selector_online import build_candidate_plan  # noqa: E402


SWEEPS = [
    REPO_ROOT / "results/expanded_attention_sweep.csv",
    REPO_ROOT / "results/expanded_attention_sweep_seed1.csv",
    REPO_ROOT / "results/expanded_attention_sweep_seed2.csv",
]
CONFIGS = REPO_ROOT / "configs/attention_config_space.csv"


class OnlineAttentionSelectorPlanTest(unittest.TestCase):
    def test_tree_and_exhaustive_plans_cover_original_held_out_workloads(self) -> None:
        tree_plan, _, _, config_ids, _ = build_candidate_plan(SWEEPS, CONFIGS, "tree", 4)
        exhaustive_plan, _, _, _, _ = build_candidate_plan(
            SWEEPS, CONFIGS, "exhaustive", 4
        )

        self.assertEqual(len(tree_plan), 6)
        self.assertEqual(set(tree_plan), set(exhaustive_plan))
        self.assertTrue(all(len(candidates) == 4 for candidates in tree_plan.values()))
        self.assertTrue(
            all(candidates == config_ids for candidates in exhaustive_plan.values())
        )


if __name__ == "__main__":
    unittest.main()
