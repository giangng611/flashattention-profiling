"""Evaluate lightweight configuration-selection baselines offline.

The input is the exhaustive expanded attention sweep. Selection policies are
fit only on development workloads and evaluated only on held-out workloads.
No CUDA device is required because candidate latency and tuning measurements
come from the preserved sweep records.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG_ID = "bm64_bn64_w4_s3"
TREE_FEATURES = (
    "log2_query_length",
    "query_heads",
    "head_dim",
    "dtype_bfloat16",
    "causal",
)


@dataclass(frozen=True)
class Workload:
    name: str
    split: str
    query_length: int
    query_heads: int
    head_dim: int
    dtype: str
    causal: bool

    def feature(self, name: str) -> float:
        values = {
            "log2_query_length": math.log2(self.query_length),
            "query_heads": float(self.query_heads),
            "head_dim": float(self.head_dim),
            "dtype_bfloat16": float(self.dtype == "bfloat16"),
            "causal": float(self.causal),
        }
        return values[name]


@dataclass(frozen=True)
class Measurement:
    status: str
    correctness_passed: bool
    latency_ms: float | None
    compile_or_cache_prepare_ms: float | None
    benchmark_wall_time_ms: float | None
    is_marked_oracle: bool

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.correctness_passed and self.latency_ms is not None


@dataclass
class TreeNode:
    workload_names: list[str]
    ranking: list[str]
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def rank(self, workload: Workload) -> list[str]:
        if self.feature is None or self.threshold is None:
            return self.ranking
        child = self.left if workload.feature(self.feature) <= self.threshold else self.right
        if child is None:
            return self.ranking
        return child.rank(workload)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "development_workloads": self.workload_names,
            "ranking": self.ranking,
        }
        if self.feature is not None:
            result.update(
                {
                    "feature": self.feature,
                    "threshold": self.threshold,
                    "left": self.left.to_dict() if self.left else None,
                    "right": self.right.to_dict() if self.right else None,
                }
            )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        type=Path,
        nargs="+",
        default=[Path("results/expanded_attention_sweep.csv")],
    )
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=Path("results/expanded_attention_sweep_summary.csv"),
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs/attention_config_space.csv"),
    )
    parser.add_argument("--tutorial-config", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-max-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf", type=int, default=2)
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path("results/attention_selector_trials.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/attention_selector_summary.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/attention_selector_metadata.json"),
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def load_config_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    config_ids = [row["config_id"] for row in rows]
    if not config_ids or len(config_ids) != len(set(config_ids)):
        raise ValueError("configuration file must contain unique config_id values")
    return config_ids


def load_sweep(
    path: Path,
) -> tuple[dict[str, Workload], dict[tuple[str, str], Measurement], dict[str, str]]:
    workloads: dict[str, Workload] = {}
    measurements: dict[tuple[str, str], Measurement] = {}
    marked_oracles: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "triton_fused_attention" or row["split"] == "reference_only":
                continue
            workload = Workload(
                name=row["workload"],
                split=row["split"],
                query_length=int(row["query_length"]),
                query_heads=int(row["query_heads"]),
                head_dim=int(row["head_dim"]),
                dtype=row["dtype"],
                causal=parse_bool(row["causal"]),
            )
            existing = workloads.get(workload.name)
            if existing is not None and existing != workload:
                raise ValueError(f"inconsistent workload metadata for {workload.name}")
            workloads[workload.name] = workload
            config_id = row["config_id"]
            key = (workload.name, config_id)
            if key in measurements:
                raise ValueError(f"duplicate measurement for {workload.name}/{config_id}")
            measurement = Measurement(
                status=row["status"],
                correctness_passed=parse_bool(row["correctness_passed"]),
                latency_ms=optional_float(row["median_latency_ms"]),
                compile_or_cache_prepare_ms=optional_float(row["compile_or_cache_prepare_ms"]),
                benchmark_wall_time_ms=optional_float(row["benchmark_wall_time_ms"]),
                is_marked_oracle=parse_bool(row["is_oracle"]),
            )
            measurements[key] = measurement
            if measurement.is_marked_oracle:
                if workload.name in marked_oracles:
                    raise ValueError(f"multiple marked oracles for {workload.name}")
                marked_oracles[workload.name] = config_id
    return workloads, measurements, marked_oracles


def aggregate_sweeps(
    paths: list[Path],
    config_ids: list[str],
) -> tuple[dict[str, Workload], dict[tuple[str, str], Measurement], dict[str, str]]:
    loaded = [load_sweep(path) for path in paths]
    workloads = loaded[0][0]
    expected_keys = set(loaded[0][1])
    for path, (candidate_workloads, measurements, _) in zip(paths[1:], loaded[1:]):
        if candidate_workloads != workloads:
            raise ValueError(f"workload definitions differ in {path}")
        if set(measurements) != expected_keys:
            raise ValueError(f"workload/configuration grid differs in {path}")

    if len(paths) == 1:
        return loaded[0]

    aggregated: dict[tuple[str, str], Measurement] = {}
    for key in sorted(expected_keys):
        repeated = [measurements[key] for _, measurements, _ in loaded]
        validity = [measurement.valid for measurement in repeated]
        if len(set(validity)) != 1:
            raise ValueError(f"validity differs across repeated sweeps for {key}")
        if validity[0]:
            latencies = [
                measurement.latency_ms
                for measurement in repeated
                if measurement.latency_ms is not None
            ]
            latency = math.exp(statistics.fmean(math.log(value) for value in latencies))
            status = "ok"
            correctness_passed = True
        else:
            latency = None
            status = repeated[0].status
            correctness_passed = False
        aggregated[key] = Measurement(
            status=status,
            correctness_passed=correctness_passed,
            latency_ms=latency,
            compile_or_cache_prepare_ms=repeated[0].compile_or_cache_prepare_ms,
            benchmark_wall_time_ms=repeated[0].benchmark_wall_time_ms,
            is_marked_oracle=False,
        )

    marked_oracles: dict[str, str] = {}
    for workload_name in workloads:
        valid_configs = [
            config_id
            for config_id in config_ids
            if aggregated[(workload_name, config_id)].valid
        ]
        if not valid_configs:
            raise ValueError(f"no valid repeated measurements for {workload_name}")
        marked_oracles[workload_name] = min(
            valid_configs,
            key=lambda config_id: (
                aggregated[(workload_name, config_id)].latency_ms,
                config_id,
            ),
        )
    aggregated = {
        key: Measurement(
            status=measurement.status,
            correctness_passed=measurement.correctness_passed,
            latency_ms=measurement.latency_ms,
            compile_or_cache_prepare_ms=measurement.compile_or_cache_prepare_ms,
            benchmark_wall_time_ms=measurement.benchmark_wall_time_ms,
            is_marked_oracle=key[1] == marked_oracles[key[0]],
        )
        for key, measurement in aggregated.items()
    }
    return workloads, aggregated, marked_oracles


def load_sweep_wall_times(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        return {
            row["workload"]: float(row["sweep_wall_time_ms"])
            for row in csv.DictReader(handle)
            if row["status"] == "ok"
        }


def valid_latency(
    measurements: dict[tuple[str, str], Measurement], workload: str, config_id: str
) -> float | None:
    measurement = measurements.get((workload, config_id))
    return measurement.latency_ms if measurement is not None and measurement.valid else None


def oracle_latency(
    measurements: dict[tuple[str, str], Measurement], workload: str, config_ids: Iterable[str]
) -> float:
    values = [
        latency
        for config_id in config_ids
        if (latency := valid_latency(measurements, workload, config_id)) is not None
    ]
    if not values:
        raise ValueError(f"no valid configurations for {workload}")
    return min(values)


def oracle_ties(
    measurements: dict[tuple[str, str], Measurement], workload: str, config_ids: Iterable[str]
) -> set[str]:
    best = oracle_latency(measurements, workload, config_ids)
    return {
        config_id
        for config_id in config_ids
        if (latency := valid_latency(measurements, workload, config_id)) is not None
        and math.isclose(latency, best, rel_tol=1e-12, abs_tol=1e-12)
    }


def rank_configs(
    workload_names: list[str],
    config_ids: list[str],
    measurements: dict[tuple[str, str], Measurement],
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for config_id in config_ids:
        log_ratios: list[float] = []
        for workload in workload_names:
            latency = valid_latency(measurements, workload, config_id)
            if latency is None:
                break
            best = oracle_latency(measurements, workload, config_ids)
            log_ratios.append(math.log(latency / best))
        else:
            scored.append((statistics.fmean(log_ratios), config_id))
    return [config_id for _, config_id in sorted(scored)]


def node_loss(
    workload_names: list[str],
    config_ids: list[str],
    measurements: dict[tuple[str, str], Measurement],
) -> float:
    ranking = rank_configs(workload_names, config_ids, measurements)
    if not ranking:
        return math.inf
    chosen = ranking[0]
    return sum(
        math.log(
            valid_latency(measurements, workload, chosen)
            / oracle_latency(measurements, workload, config_ids)
        )
        for workload in workload_names
    )


def build_tree(
    workload_names: list[str],
    workloads: dict[str, Workload],
    config_ids: list[str],
    measurements: dict[tuple[str, str], Measurement],
    max_depth: int,
    min_leaf: int,
    depth: int = 0,
) -> TreeNode:
    ranking = rank_configs(workload_names, config_ids, measurements)
    if not ranking:
        raise ValueError("tree node has no configuration valid across its development workloads")
    node = TreeNode(workload_names=sorted(workload_names), ranking=ranking)
    if depth >= max_depth or len(workload_names) < 2 * min_leaf:
        return node

    parent_loss = node_loss(workload_names, config_ids, measurements)
    best_split: tuple[float, str, float, list[str], list[str]] | None = None
    for feature in TREE_FEATURES:
        values = sorted({workloads[name].feature(feature) for name in workload_names})
        thresholds = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            left_names = [
                name for name in workload_names if workloads[name].feature(feature) <= threshold
            ]
            right_names = [name for name in workload_names if name not in left_names]
            if len(left_names) < min_leaf or len(right_names) < min_leaf:
                continue
            loss = node_loss(left_names, config_ids, measurements) + node_loss(
                right_names, config_ids, measurements
            )
            candidate = (loss, feature, threshold, left_names, right_names)
            if best_split is None or candidate[:3] < best_split[:3]:
                best_split = candidate

    if best_split is None or best_split[0] >= parent_loss - 1e-12:
        return node
    _, node.feature, node.threshold, left_names, right_names = best_split
    node.left = build_tree(
        left_names,
        workloads,
        config_ids,
        measurements,
        max_depth,
        min_leaf,
        depth + 1,
    )
    node.right = build_tree(
        right_names,
        workloads,
        config_ids,
        measurements,
        max_depth,
        min_leaf,
        depth + 1,
    )
    return node


def nearest_development_workload(
    target: Workload, development: list[Workload]
) -> Workload:
    ranges: dict[str, tuple[float, float]] = {}
    for feature in TREE_FEATURES:
        values = [workload.feature(feature) for workload in development]
        ranges[feature] = (min(values), max(values))

    def distance(candidate: Workload) -> tuple[float, str]:
        squared = 0.0
        for feature, (low, high) in ranges.items():
            scale = high - low
            difference = target.feature(feature) - candidate.feature(feature)
            squared += (difference / scale if scale else difference) ** 2
        return squared, candidate.name

    return min(development, key=distance)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_candidate_set(
    *,
    method: str,
    budget: int,
    seed: int | None,
    workload: Workload,
    candidate_ids: list[str],
    config_ids: list[str],
    measurements: dict[tuple[str, str], Measurement],
    marked_oracle: str,
    tutorial_config: str,
    sweep_wall_time_ms: float,
) -> dict[str, object]:
    candidate_ids = list(dict.fromkeys(candidate_ids))
    if not candidate_ids:
        raise ValueError(f"{method} returned no candidates for {workload.name}")
    valid_candidates = [
        config_id
        for config_id in candidate_ids
        if valid_latency(measurements, workload.name, config_id) is not None
    ]
    selected_config = min(
        valid_candidates,
        key=lambda config_id: (valid_latency(measurements, workload.name, config_id), config_id),
        default=None,
    )
    selected_latency = (
        valid_latency(measurements, workload.name, selected_config) if selected_config else None
    )
    best_latency = oracle_latency(measurements, workload.name, config_ids)
    tied_oracles = oracle_ties(measurements, workload.name, config_ids)
    within_one_percent = {
        config_id
        for config_id in config_ids
        if (latency := valid_latency(measurements, workload.name, config_id)) is not None
        and latency <= best_latency * 1.01
    }
    observed_compile = sum(
        measurement.compile_or_cache_prepare_ms or 0.0
        for config_id in candidate_ids
        if (measurement := measurements[(workload.name, config_id)]).valid
    )
    observed_benchmark = sum(
        measurement.benchmark_wall_time_ms or 0.0
        for config_id in candidate_ids
        if (measurement := measurements[(workload.name, config_id)]).valid
    )
    estimated_total_tuning = sweep_wall_time_ms * len(candidate_ids) / len(config_ids)
    tutorial_latency = valid_latency(measurements, workload.name, tutorial_config)
    latency_saved = (
        tutorial_latency - selected_latency
        if tutorial_latency is not None and selected_latency is not None
        else None
    )
    incremental_tuning = max(0.0, estimated_total_tuning - sweep_wall_time_ms / len(config_ids))
    break_even = (
        incremental_tuning / latency_saved
        if latency_saved is not None and latency_saved > 0.0
        else None
    )
    return {
        "method": method,
        "budget": budget,
        "seed": seed,
        "workload": workload.name,
        "split": workload.split,
        "candidate_config_ids": ";".join(candidate_ids),
        "selected_config_id": selected_config,
        "selection_success": selected_config is not None,
        "selected_latency_ms": selected_latency,
        "marked_oracle_config_id": marked_oracle,
        "oracle_tie_config_ids": ";".join(sorted(tied_oracles)),
        "oracle_latency_ms": best_latency,
        "oracle_relative_performance": (
            best_latency / selected_latency if selected_latency is not None else 0.0
        ),
        "regret_pct": (
            (selected_latency / best_latency - 1.0) * 100.0
            if selected_latency is not None
            else None
        ),
        "oracle_covered": bool(tied_oracles.intersection(candidate_ids)),
        "within_1pct_oracle_covered": bool(within_one_percent.intersection(candidate_ids)),
        "configs_compiled_or_measured": len(candidate_ids),
        "valid_configs_measured": len(valid_candidates),
        "failed_configs_measured": len(candidate_ids) - len(valid_candidates),
        "config_fraction": len(candidate_ids) / len(config_ids),
        "observed_compile_or_cache_prepare_ms": observed_compile,
        "observed_benchmark_wall_time_ms": observed_benchmark,
        "estimated_total_tuning_time_ms": estimated_total_tuning,
        "estimated_tuning_time_saved_pct": (1.0 - len(candidate_ids) / len(config_ids))
        * 100.0,
        "tutorial_config_latency_ms": tutorial_latency,
        "latency_saved_vs_tutorial_ms": latency_saved,
        "estimated_break_even_vs_tutorial_calls": break_even,
    }


def summarize_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["budget"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (method, budget), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        performances = [float(row["oracle_relative_performance"]) for row in group]
        successful_regrets = [
            float(row["regret_pct"])
            for row in group
            if row["regret_pct"] is not None
        ]
        positive_performances = [value for value in performances if value > 0.0]
        success_rate = statistics.fmean(bool(row["selection_success"]) for row in group)
        geomean_performance = (
            math.exp(statistics.fmean(math.log(value) for value in positive_performances))
            if len(positive_performances) == len(performances)
            else 0.0
        )
        break_even_values = [
            float(row["estimated_break_even_vs_tutorial_calls"])
            for row in group
            if row["estimated_break_even_vs_tutorial_calls"] is not None
        ]
        seed_mean_performances: list[float] = []
        seeds = sorted({int(row["seed"]) for row in group if row["seed"] is not None})
        for seed in seeds:
            seed_rows = [row for row in group if row["seed"] == seed]
            seed_mean_performances.append(
                statistics.fmean(float(row["oracle_relative_performance"]) for row in seed_rows)
            )
        config_fraction = statistics.fmean(float(row["config_fraction"]) for row in group)
        summaries.append(
            {
                "method": method,
                "budget": budget,
                "random_seed_count": len(seeds),
                "held_out_evaluations": len(group),
                "selection_success_rate": success_rate,
                "mean_oracle_relative_performance": statistics.fmean(performances),
                "geomean_oracle_relative_performance": geomean_performance,
                "p10_oracle_relative_performance": percentile(performances, 0.10),
                "median_oracle_relative_performance": statistics.median(performances),
                "stddev_seed_mean_oracle_relative_performance": (
                    statistics.stdev(seed_mean_performances)
                    if len(seed_mean_performances) > 1
                    else None
                ),
                "min_seed_mean_oracle_relative_performance": (
                    min(seed_mean_performances) if seed_mean_performances else None
                ),
                "max_seed_mean_oracle_relative_performance": (
                    max(seed_mean_performances) if seed_mean_performances else None
                ),
                "mean_regret_pct_successful": statistics.fmean(successful_regrets),
                "median_regret_pct_successful": statistics.median(successful_regrets),
                "worst_regret_pct_successful": max(successful_regrets),
                "oracle_coverage_rate": statistics.fmean(bool(row["oracle_covered"]) for row in group),
                "within_1pct_oracle_coverage_rate": statistics.fmean(
                    bool(row["within_1pct_oracle_covered"]) for row in group
                ),
                "mean_configs_compiled_or_measured": statistics.fmean(
                    int(row["configs_compiled_or_measured"]) for row in group
                ),
                "mean_failed_configs_measured": statistics.fmean(
                    int(row["failed_configs_measured"]) for row in group
                ),
                "mean_config_fraction": config_fraction,
                "mean_observed_compile_or_cache_prepare_ms": statistics.fmean(
                    float(row["observed_compile_or_cache_prepare_ms"]) for row in group
                ),
                "mean_observed_benchmark_wall_time_ms": statistics.fmean(
                    float(row["observed_benchmark_wall_time_ms"]) for row in group
                ),
                "mean_estimated_total_tuning_time_ms": statistics.fmean(
                    float(row["estimated_total_tuning_time_ms"]) for row in group
                ),
                "mean_estimated_tuning_time_saved_pct": statistics.fmean(
                    float(row["estimated_tuning_time_saved_pct"]) for row in group
                ),
                "mean_estimated_break_even_vs_tutorial_calls": (
                    statistics.fmean(break_even_values) if break_even_values else None
                ),
                "target_90pct_oracle_within_25pct_configs": (
                    success_rate == 1.0
                    and geomean_performance >= 0.90
                    and config_fraction <= 0.25
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.random_seeds <= 0:
        raise ValueError("--random-seeds must be positive")
    if args.tree_max_depth < 0 or args.tree_min_leaf <= 0:
        raise ValueError("tree depth must be nonnegative and min leaf must be positive")
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must be positive")

    config_ids = load_config_ids(args.configs)
    if budgets[-1] > len(config_ids):
        raise ValueError("budget cannot exceed the configuration count")
    if args.tutorial_config not in config_ids:
        raise ValueError("tutorial/default configuration is outside the candidate space")

    workloads, measurements, marked_oracles = aggregate_sweeps(args.sweep, config_ids)
    wall_times = load_sweep_wall_times(args.sweep_summary)
    development = sorted(
        (workload for workload in workloads.values() if workload.split == "development"),
        key=lambda workload: workload.name,
    )
    held_out = sorted(
        (workload for workload in workloads.values() if workload.split == "held_out"),
        key=lambda workload: workload.name,
    )
    if not development or not held_out:
        raise ValueError("both development and held-out workloads are required")
    expected_keys = {
        (workload.name, config_id)
        for workload in development + held_out
        for config_id in config_ids
    }
    missing_keys = expected_keys.difference(measurements)
    if missing_keys:
        raise ValueError(f"sweep is missing {len(missing_keys)} workload/config measurements")
    if set(marked_oracles) != {workload.name for workload in development + held_out}:
        raise ValueError("every supported workload must have exactly one marked oracle")

    development_names = [workload.name for workload in development]
    best_fixed_ranking = rank_configs(development_names, config_ids, measurements)
    best_fixed = best_fixed_ranking[0]
    tree = build_tree(
        development_names,
        workloads,
        config_ids,
        measurements,
        args.tree_max_depth,
        args.tree_min_leaf,
    )
    nearest: dict[str, str] = {}
    for workload in held_out:
        neighbor = nearest_development_workload(workload, development)
        nearest[workload.name] = neighbor.name

    rows: list[dict[str, object]] = []

    def add(method: str, budget: int, seed: int | None, workload: Workload, candidates: list[str]) -> None:
        rows.append(
            evaluate_candidate_set(
                method=method,
                budget=budget,
                seed=seed,
                workload=workload,
                candidate_ids=candidates,
                config_ids=config_ids,
                measurements=measurements,
                marked_oracle=marked_oracles[workload.name],
                tutorial_config=args.tutorial_config,
                sweep_wall_time_ms=wall_times[workload.name],
            )
        )

    for workload in held_out:
        add("tutorial_derived_default", 1, None, workload, [args.tutorial_config])
        add("best_fixed_development", 1, None, workload, [best_fixed])
        add(
            "nearest_workload_reuse",
            1,
            None,
            workload,
            [marked_oracles[nearest[workload.name]]],
        )
        tree_ranking = tree.rank(workload)
        for budget in budgets:
            add("decision_tree_ranker", budget, None, workload, tree_ranking[:budget])
        add("exhaustive_local_oracle", len(config_ids), None, workload, config_ids)

    for seed_offset in range(args.random_seeds):
        seed = args.seed + seed_offset
        for workload_index, workload in enumerate(held_out):
            rng = random.Random(seed * 1_000_003 + workload_index)
            order = config_ids.copy()
            rng.shuffle(order)
            for budget in budgets:
                add("random_search", budget, seed, workload, order[:budget])

    summary = summarize_trials(rows)
    write_csv(args.trials_output, rows)
    write_csv(args.summary_output, summary)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python_version": sys.version,
        "input_sweeps": [
            {"path": str(path), "sha256": sha256(path)} for path in args.sweep
        ],
        "latency_aggregation": (
            "single-run median latency"
            if len(args.sweep) == 1
            else "geometric mean of per-run median latency"
        ),
        "timing_measurement_source": str(args.sweep[0]),
        "input_sweep_summary": str(args.sweep_summary),
        "input_sweep_summary_sha256": sha256(args.sweep_summary),
        "input_configs": str(args.configs),
        "input_configs_sha256": sha256(args.configs),
        "development_workload_count": len(development),
        "held_out_workload_count": len(held_out),
        "config_count": len(config_ids),
        "tutorial_derived_default_config": args.tutorial_config,
        "tutorial_default_note": (
            "This is the fixed default used by this project's tutorial-derived benchmark, "
            "not a claim that the official Triton tutorial defines one universal default."
        ),
        "best_fixed_development_config": best_fixed,
        "best_fixed_selection_metric": (
            "minimum geometric mean latency ratio to the local oracle across all development workloads"
        ),
        "nearest_workload_features": list(TREE_FEATURES),
        "nearest_development_workloads": nearest,
        "random_seed_start": args.seed,
        "random_seed_count": args.random_seeds,
        "budgets": budgets,
        "decision_tree_max_depth": args.tree_max_depth,
        "decision_tree_min_leaf": args.tree_min_leaf,
        "decision_tree_objective": (
            "greedy reduction in summed log latency ratio; each leaf ranks configurations "
            "by geometric-mean oracle-relative latency on development workloads"
        ),
        "decision_tree": tree.to_dict(),
        "oracle_coverage_note": (
            "Coverage counts any configuration exactly tied for the minimum measured median latency."
        ),
        "tuning_time_note": (
            "Observed compile/cache and benchmark sums exclude unavailable time for failed configurations. "
            "Estimated total tuning time scales each workload's measured exhaustive sweep wall time by "
            "the fraction of configurations selected; it is an estimate, not a new wall-clock run."
        ),
        "break_even_note": (
            "Break-even compares incremental estimated tuning time beyond one fixed-config measurement "
            "with latency saved per call versus the tutorial-derived default."
        ),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Development workloads: {len(development)}")
    print(f"Held-out workloads: {len(held_out)}")
    print(f"Candidate configurations: {len(config_ids)}")
    print(f"Best fixed development config: {best_fixed}")
    print(f"Wrote {args.trials_output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
