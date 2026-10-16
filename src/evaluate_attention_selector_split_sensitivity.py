"""Evaluate selector sensitivity to the development/held-out workload split.

The evaluator is CPU-only. It reuses aggregate latency from repeated exhaustive
sweeps, includes the repository's original 10/6 split, and adds deterministic
balanced random 10/6 splits. Raw measurements are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from evaluate_attention_config_selectors import (
    DEFAULT_CONFIG_ID,
    Workload,
    aggregate_sweeps,
    build_tree,
    evaluate_candidate_set,
    load_config_ids,
    load_sweep_wall_times,
    nearest_development_workload,
    percentile,
    rank_configs,
)


DEFAULT_SWEEPS = [
    Path("results/expanded_attention_sweep.csv"),
    Path("results/expanded_attention_sweep_seed1.csv"),
    Path("results/expanded_attention_sweep_seed2.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, nargs="+", default=DEFAULT_SWEEPS)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=Path("results/expanded_attention_sweep_summary.csv"),
    )
    parser.add_argument(
        "--configs", type=Path, default=Path("configs/attention_config_space.csv")
    )
    parser.add_argument("--tutorial-config", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--split-count", type=int, default=100)
    parser.add_argument("--held-out-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-max-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf", type=int, default=2)
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path("results/attention_selector_split_sensitivity_trials.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/attention_selector_split_sensitivity_summary.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/attention_selector_split_sensitivity_metadata.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_coverage(workloads: list[Workload]) -> bool:
    """Require both sides of major categorical features and two head counts."""
    return (
        len({workload.dtype for workload in workloads}) == 2
        and len({workload.head_dim for workload in workloads}) == 2
        and len({workload.causal for workload in workloads}) == 2
        and len({workload.query_heads for workload in workloads}) >= 2
    )


def make_splits(
    workloads: dict[str, Workload],
    split_count: int,
    held_out_count: int,
    seed: int,
) -> list[dict[str, object]]:
    names = sorted(workloads)
    original_held_out = tuple(
        sorted(name for name, workload in workloads.items() if workload.split == "held_out")
    )
    if len(original_held_out) != held_out_count:
        raise ValueError("the original split does not match --held-out-count")

    def valid(held_out_names: tuple[str, ...]) -> bool:
        held_out_set = set(held_out_names)
        held_out = [workloads[name] for name in names if name in held_out_set]
        development = [workloads[name] for name in names if name not in held_out_set]
        return feature_coverage(held_out) and feature_coverage(development)

    if not valid(original_held_out):
        raise ValueError("the original split does not satisfy feature-coverage constraints")

    splits: list[dict[str, object]] = [
        {
            "split_id": "original",
            "split_seed": None,
            "held_out_names": original_held_out,
        }
    ]
    seen = {original_held_out}
    candidate_seed = seed
    attempts = 0
    while len(splits) < split_count:
        rng = random.Random(candidate_seed)
        held_out_names = tuple(sorted(rng.sample(names, held_out_count)))
        attempts += 1
        if held_out_names not in seen and valid(held_out_names):
            splits.append(
                {
                    "split_id": f"balanced_{len(splits):03d}",
                    "split_seed": candidate_seed,
                    "held_out_names": held_out_names,
                }
            )
            seen.add(held_out_names)
        candidate_seed += 1
        if attempts > 100_000:
            raise ValueError("could not generate enough unique balanced splits")
    return splits


def split_geomean(rows: list[dict[str, object]]) -> float:
    performances = [float(row["oracle_relative_performance"]) for row in rows]
    if any(value <= 0.0 for value in performances):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in performances))


def summarize_trials(
    rows: list[dict[str, object]], config_count: int
) -> list[dict[str, object]]:
    per_split: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["budget"]), str(row["split_id"]))
        per_split.setdefault(key, []).append(row)

    split_metrics: dict[tuple[str, int], list[dict[str, float]]] = {}
    for (method, budget, _), group in per_split.items():
        successful_regrets = [
            float(row["regret_pct"]) for row in group if row["regret_pct"] is not None
        ]
        split_metrics.setdefault((method, budget), []).append(
            {
                "geomean_performance": split_geomean(group),
                "success_rate": statistics.fmean(bool(row["selection_success"]) for row in group),
                "mean_regret": (
                    statistics.fmean(successful_regrets) if successful_regrets else math.inf
                ),
                "worst_regret": max(successful_regrets) if successful_regrets else math.inf,
            }
        )

    summaries: list[dict[str, object]] = []
    for (method, budget), metrics in sorted(
        split_metrics.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        performances = [item["geomean_performance"] for item in metrics]
        success_rates = [item["success_rate"] for item in metrics]
        mean_regrets = [item["mean_regret"] for item in metrics]
        worst_regrets = [item["worst_regret"] for item in metrics]
        config_fraction = budget / config_count
        summaries.append(
            {
                "method": method,
                "budget": budget,
                "split_count": len(metrics),
                "held_out_evaluations": sum(
                    1
                    for row in rows
                    if row["method"] == method and int(row["budget"]) == budget
                ),
                "original_split_geomean_oracle_performance": metrics[0][
                    "geomean_performance"
                ],
                "mean_split_geomean_oracle_performance": statistics.fmean(performances),
                "median_split_geomean_oracle_performance": statistics.median(performances),
                "p10_split_geomean_oracle_performance": percentile(performances, 0.10),
                "minimum_split_geomean_oracle_performance": min(performances),
                "maximum_split_geomean_oracle_performance": max(performances),
                "split_standard_deviation": (
                    statistics.stdev(performances) if len(performances) > 1 else 0.0
                ),
                "fraction_splits_at_least_90pct_oracle": statistics.fmean(
                    performance >= 0.90 for performance in performances
                ),
                "fraction_splits_at_least_95pct_oracle": statistics.fmean(
                    performance >= 0.95 for performance in performances
                ),
                "mean_split_selection_success_rate": statistics.fmean(success_rates),
                "minimum_split_selection_success_rate": min(success_rates),
                "mean_successful_regret_pct": statistics.fmean(mean_regrets),
                "p90_split_worst_regret_pct": percentile(worst_regrets, 0.90),
                "maximum_workload_regret_pct": max(worst_regrets),
                "config_fraction": config_fraction,
                "fraction_splits_meeting_90pct_oracle_within_25pct_configs": statistics.fmean(
                    performance >= 0.90 and config_fraction <= 0.25
                    for performance in performances
                ),
                "fraction_splits_meeting_95pct_oracle_within_25pct_configs": statistics.fmean(
                    performance >= 0.95 and config_fraction <= 0.25
                    for performance in performances
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    budgets = sorted(set(args.budgets))
    if args.split_count <= 0 or args.held_out_count <= 0:
        raise ValueError("split and held-out counts must be positive")
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must be positive")

    config_ids = load_config_ids(args.configs)
    if budgets[-1] > len(config_ids):
        raise ValueError("budget cannot exceed the configuration count")
    workloads, measurements, marked_oracles = aggregate_sweeps(args.sweep, config_ids)
    wall_times = load_sweep_wall_times(args.sweep_summary)
    splits = make_splits(workloads, args.split_count, args.held_out_count, args.seed)

    rows: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    all_names = set(workloads)
    for split in splits:
        held_out_names = set(split["held_out_names"])
        development_names = sorted(all_names - held_out_names)
        held_out = [replace(workloads[name], split="held_out") for name in sorted(held_out_names)]
        development = [
            replace(workloads[name], split="development") for name in development_names
        ]
        best_fixed = rank_configs(development_names, config_ids, measurements)[0]
        tree = build_tree(
            development_names,
            {workload.name: workload for workload in development + held_out},
            config_ids,
            measurements,
            args.tree_max_depth,
            args.tree_min_leaf,
        )
        nearest = {
            workload.name: nearest_development_workload(workload, development).name
            for workload in held_out
        }
        split_records.append(
            {
                "split_id": split["split_id"],
                "split_seed": split["split_seed"],
                "development_workloads": development_names,
                "held_out_workloads": sorted(held_out_names),
                "best_fixed_config": best_fixed,
            }
        )

        def add(
            method: str, budget: int, workload: Workload, candidate_ids: list[str]
        ) -> None:
            result = evaluate_candidate_set(
                method=method,
                budget=budget,
                seed=None,
                workload=workload,
                candidate_ids=candidate_ids,
                config_ids=config_ids,
                measurements=measurements,
                marked_oracle=marked_oracles[workload.name],
                tutorial_config=args.tutorial_config,
                sweep_wall_time_ms=wall_times[workload.name],
            )
            result = {
                "split_id": split["split_id"],
                "split_seed": split["split_seed"],
                "development_workload_count": len(development),
                "held_out_workload_count": len(held_out),
                **result,
            }
            rows.append(result)

        for workload in held_out:
            add("tutorial_derived_default", 1, workload, [args.tutorial_config])
            add("best_fixed_development", 1, workload, [best_fixed])
            add(
                "nearest_workload_reuse",
                1,
                workload,
                [marked_oracles[nearest[workload.name]]],
            )
            ranking = tree.rank(workload)
            for budget in budgets:
                add("decision_tree_ranker", budget, workload, ranking[:budget])
            add("exhaustive_local_oracle", len(config_ids), workload, config_ids)

    summaries = summarize_trials(rows, len(config_ids))
    write_csv(args.trials_output, rows)
    write_csv(args.summary_output, summaries)
    metadata = {
        "python_version": sys.version,
        "input_sweeps": [
            {"path": str(path), "sha256": sha256(path)} for path in args.sweep
        ],
        "input_sweep_summary": {
            "path": str(args.sweep_summary),
            "sha256": sha256(args.sweep_summary),
        },
        "input_configs": {"path": str(args.configs), "sha256": sha256(args.configs)},
        "latency_aggregation": "geometric mean of three per-run median latencies",
        "split_count": len(splits),
        "split_protocol": (
            "repository original split plus deterministic unique random 10/6 splits; both sides "
            "must contain FP16 and BF16, head dimensions 64 and 128, causal and non-causal "
            "workloads, and at least two query-head counts"
        ),
        "original_split_included": True,
        "random_split_seed_start": args.seed,
        "development_count": len(workloads) - args.held_out_count,
        "held_out_count": args.held_out_count,
        "budgets": budgets,
        "tree_max_depth": args.tree_max_depth,
        "tree_min_leaf": args.tree_min_leaf,
        "random_search_note": (
            "Random search is omitted because it does not train on the development partition; "
            "the earlier 100-seed held-out comparison remains its direct baseline."
        ),
        "splits": split_records,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Splits evaluated: {len(splits)}")
    print(f"Held-out workloads per split: {args.held_out_count}")
    print(f"Trial rows: {len(rows)}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Wrote {args.trials_output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
