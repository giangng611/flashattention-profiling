"""Analyze configuration-ranking stability across repeated attention sweeps.

This script is intentionally CPU-only. It consumes preserved detailed sweep
CSVs, checks that the workload/configuration grids agree, and reports both
per-configuration aggregate latency and per-workload ranking stability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SWEEPS = [
    Path("results/expanded_attention_sweep.csv"),
    Path("results/expanded_attention_sweep_seed1.csv"),
    Path("results/expanded_attention_sweep_seed2.csv"),
]
DEFAULT_SUMMARIES = [
    Path("results/expanded_attention_sweep_summary.csv"),
    Path("results/expanded_attention_sweep_seed1_summary.csv"),
    Path("results/expanded_attention_sweep_seed2_summary.csv"),
]


@dataclass(frozen=True)
class Workload:
    name: str
    split: str
    query_length: int
    query_heads: int
    head_dim: int
    dtype: str
    causal: bool


@dataclass(frozen=True)
class Measurement:
    status: str
    correctness_passed: bool
    latency_ms: float | None

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.correctness_passed and self.latency_ms is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=Path, nargs="+", default=DEFAULT_SWEEPS)
    parser.add_argument("--summaries", type=Path, nargs="+", default=DEFAULT_SUMMARIES)
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("results/attention_sweep_repeat_config_stability.csv"),
    )
    parser.add_argument(
        "--workload-output",
        type=Path,
        default=Path("results/attention_sweep_repeat_workload_stability.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/attention_sweep_repeat_metadata.json"),
    )
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sweep(
    path: Path,
) -> tuple[dict[str, Workload], dict[tuple[str, str], Measurement]]:
    workloads: dict[str, Workload] = {}
    measurements: dict[tuple[str, str], Measurement] = {}
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
            if workload.name in workloads and workloads[workload.name] != workload:
                raise ValueError(f"inconsistent metadata for {workload.name} in {path}")
            workloads[workload.name] = workload
            key = (workload.name, row["config_id"])
            if key in measurements:
                raise ValueError(f"duplicate measurement for {key} in {path}")
            measurements[key] = Measurement(
                status=row["status"],
                correctness_passed=parse_bool(row["correctness_passed"]),
                latency_ms=optional_float(row["median_latency_ms"]),
            )
    return workloads, measurements


def load_total_wall_time(path: Path) -> float:
    with path.open(newline="") as handle:
        return sum(
            float(row["sweep_wall_time_ms"])
            for row in csv.DictReader(handle)
            if row["status"] == "ok"
        )


def ordinal_ranks(latencies: dict[str, float]) -> dict[str, int]:
    return {
        config_id: rank
        for rank, (config_id, _) in enumerate(
            sorted(latencies.items(), key=lambda item: (item[1], item[0])), start=1
        )
    }


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal lists with at least two values")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.sweeps) < 2:
        raise ValueError("at least two repeated sweeps are required")
    if len(args.summaries) != len(args.sweeps):
        raise ValueError("--summaries must provide one file per sweep")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    loaded = [load_sweep(path) for path in args.sweeps]
    workloads = loaded[0][0]
    measurement_runs = [item[1] for item in loaded]
    expected_keys = set(measurement_runs[0])
    for path, (candidate_workloads, measurements) in zip(args.sweeps[1:], loaded[1:]):
        if candidate_workloads != workloads:
            raise ValueError(f"workload definitions differ in {path}")
        if set(measurements) != expected_keys:
            raise ValueError(f"workload/configuration grid differs in {path}")

    config_ids_by_workload: dict[str, list[str]] = {}
    for workload_name, config_id in expected_keys:
        config_ids_by_workload.setdefault(workload_name, []).append(config_id)
    config_ids_by_workload = {
        workload: sorted(config_ids) for workload, config_ids in config_ids_by_workload.items()
    }

    config_rows: list[dict[str, object]] = []
    workload_rows: list[dict[str, object]] = []
    for workload_name in sorted(workloads):
        workload = workloads[workload_name]
        config_ids = config_ids_by_workload[workload_name]
        valid_by_run: list[dict[str, float]] = []
        ranks_by_run: list[dict[str, int]] = []
        for measurements in measurement_runs:
            valid = {
                config_id: measurement.latency_ms
                for config_id in config_ids
                if (measurement := measurements[(workload_name, config_id)]).valid
                and measurement.latency_ms is not None
            }
            valid_by_run.append(valid)
            ranks_by_run.append(ordinal_ranks(valid))

        common_valid = set.intersection(*(set(run) for run in valid_by_run))
        if not common_valid:
            raise ValueError(f"no configuration is valid in every run for {workload_name}")
        per_run_oracles = [min(run.values()) for run in valid_by_run]
        per_run_oracle_sets = [
            {
                config_id
                for config_id, latency in run.items()
                if math.isclose(latency, oracle, rel_tol=1e-12, abs_tol=1e-12)
            }
            for run, oracle in zip(valid_by_run, per_run_oracles)
        ]
        per_run_top1 = [min(run, key=lambda config: (run[config], config)) for run in valid_by_run]
        common_exact_oracles = set.intersection(*per_run_oracle_sets)
        aggregate_latency = {
            config_id: geometric_mean([run[config_id] for run in valid_by_run])
            for config_id in common_valid
        }
        aggregate_ranks = ordinal_ranks(aggregate_latency)
        aggregate_oracle = min(
            aggregate_latency,
            key=lambda config: (aggregate_latency[config], config),
        )
        aggregate_best_latency = aggregate_latency[aggregate_oracle]

        for config_id in config_ids:
            measurements = [run[(workload_name, config_id)] for run in measurement_runs]
            latencies = [measurement.latency_ms for measurement in measurements]
            common = config_id in common_valid
            row: dict[str, object] = {
                "workload": workload_name,
                "split": workload.split,
                "query_length": workload.query_length,
                "query_heads": workload.query_heads,
                "head_dim": workload.head_dim,
                "dtype": workload.dtype,
                "causal": workload.causal,
                "config_id": config_id,
                "valid_in_all_runs": common,
                "valid_run_count": sum(measurement.valid for measurement in measurements),
                "aggregate_geomean_latency_ms": aggregate_latency.get(config_id),
                "aggregate_rank": aggregate_ranks.get(config_id),
                "aggregate_regret_pct": (
                    (aggregate_latency[config_id] / aggregate_best_latency - 1.0) * 100.0
                    if common
                    else None
                ),
                "max_run_regret_pct": (
                    max(
                        (valid_by_run[index][config_id] / per_run_oracles[index] - 1.0) * 100.0
                        for index in range(len(valid_by_run))
                    )
                    if common
                    else None
                ),
            }
            for index, measurement in enumerate(measurements):
                row[f"run{index}_status"] = measurement.status
                row[f"run{index}_correctness_passed"] = measurement.correctness_passed
                row[f"run{index}_median_latency_ms"] = latencies[index]
                row[f"run{index}_rank"] = ranks_by_run[index].get(config_id)
            config_rows.append(row)

        correlations: list[float] = []
        top_k_jaccards: list[float] = []
        for left_index in range(len(valid_by_run)):
            for right_index in range(left_index + 1, len(valid_by_run)):
                pair_configs = sorted(
                    set(valid_by_run[left_index]).intersection(valid_by_run[right_index])
                )
                correlations.append(
                    pearson(
                        [ranks_by_run[left_index][config] for config in pair_configs],
                        [ranks_by_run[right_index][config] for config in pair_configs],
                    )
                )
                left_top = {
                    config
                    for config, _ in sorted(
                        valid_by_run[left_index].items(), key=lambda item: (item[1], item[0])
                    )[: args.top_k]
                }
                right_top = {
                    config
                    for config, _ in sorted(
                        valid_by_run[right_index].items(), key=lambda item: (item[1], item[0])
                    )[: args.top_k]
                }
                top_k_jaccards.append(len(left_top & right_top) / len(left_top | right_top))

        aggregate_oracle_run_regrets = [
            (run[aggregate_oracle] / oracle - 1.0) * 100.0
            for run, oracle in zip(valid_by_run, per_run_oracles)
        ]
        row = {
            "workload": workload_name,
            "split": workload.split,
            "query_length": workload.query_length,
            "query_heads": workload.query_heads,
            "head_dim": workload.head_dim,
            "dtype": workload.dtype,
            "causal": workload.causal,
            "valid_config_count_all_runs": len(common_valid),
            "invalid_config_count_any_run": len(config_ids) - len(common_valid),
            "top1_config_ids_by_run": ";".join(per_run_top1),
            "distinct_top1_config_count": len(set(per_run_top1)),
            "all_run_top1_agree": len(set(per_run_top1)) == 1,
            "exact_oracle_config_sets_by_run": ";".join(
                "|".join(sorted(configs)) for configs in per_run_oracle_sets
            ),
            "common_exact_oracle_config_ids": "|".join(sorted(common_exact_oracles)),
            "common_exact_oracle_exists": bool(common_exact_oracles),
            "aggregate_oracle_config_id": aggregate_oracle,
            "aggregate_oracle_geomean_latency_ms": aggregate_best_latency,
            "aggregate_oracle_mean_run_regret_pct": statistics.fmean(
                aggregate_oracle_run_regrets
            ),
            "aggregate_oracle_max_run_regret_pct": max(aggregate_oracle_run_regrets),
            "aggregate_oracle_within_1pct_every_run": max(aggregate_oracle_run_regrets) <= 1.0,
            "aggregate_oracle_within_3pct_every_run": max(aggregate_oracle_run_regrets) <= 3.0,
            "aggregate_oracle_within_5pct_every_run": max(aggregate_oracle_run_regrets) <= 5.0,
            "aggregate_configs_within_1pct": sum(
                latency <= aggregate_best_latency * 1.01 for latency in aggregate_latency.values()
            ),
            "aggregate_configs_within_3pct": sum(
                latency <= aggregate_best_latency * 1.03 for latency in aggregate_latency.values()
            ),
            "aggregate_configs_within_5pct": sum(
                latency <= aggregate_best_latency * 1.05 for latency in aggregate_latency.values()
            ),
            "mean_pairwise_spearman_rank_correlation": statistics.fmean(correlations),
            "minimum_pairwise_spearman_rank_correlation": min(correlations),
            f"mean_pairwise_top{args.top_k}_jaccard": statistics.fmean(top_k_jaccards),
            f"minimum_pairwise_top{args.top_k}_jaccard": min(top_k_jaccards),
        }
        for index, (top1, oracle_latency) in enumerate(zip(per_run_top1, per_run_oracles)):
            row[f"run{index}_top1_config_id"] = top1
            row[f"run{index}_oracle_latency_ms"] = oracle_latency
        workload_rows.append(row)

    write_csv(args.config_output, config_rows)
    write_csv(args.workload_output, workload_rows)
    total_wall_times = [load_total_wall_time(path) for path in args.summaries]
    metadata = {
        "python_version": sys.version,
        "aggregation_method": "geometric mean of per-run median latency",
        "rank_correlation_method": "Spearman correlation using deterministic ordinal ranks",
        "top_k_overlap_metric": f"pairwise Jaccard similarity of top-{args.top_k} configuration sets",
        "input_sweeps": [
            {"path": str(path), "sha256": sha256(path)} for path in args.sweeps
        ],
        "input_summaries": [
            {"path": str(path), "sha256": sha256(path)} for path in args.summaries
        ],
        "run_count": len(args.sweeps),
        "workload_count": len(workloads),
        "config_count_per_workload": len(next(iter(config_ids_by_workload.values()))),
        "all_run_top1_agreement_workloads": sum(
            bool(row["all_run_top1_agree"]) for row in workload_rows
        ),
        "common_exact_oracle_workloads": sum(
            bool(row["common_exact_oracle_exists"]) for row in workload_rows
        ),
        "aggregate_oracle_within_1pct_every_run_workloads": sum(
            bool(row["aggregate_oracle_within_1pct_every_run"]) for row in workload_rows
        ),
        "aggregate_oracle_within_3pct_every_run_workloads": sum(
            bool(row["aggregate_oracle_within_3pct_every_run"]) for row in workload_rows
        ),
        "aggregate_oracle_within_5pct_every_run_workloads": sum(
            bool(row["aggregate_oracle_within_5pct_every_run"]) for row in workload_rows
        ),
        "mean_workload_pairwise_spearman_rank_correlation": statistics.fmean(
            float(row["mean_pairwise_spearman_rank_correlation"]) for row in workload_rows
        ),
        f"mean_workload_pairwise_top{args.top_k}_jaccard": statistics.fmean(
            float(row[f"mean_pairwise_top{args.top_k}_jaccard"]) for row in workload_rows
        ),
        "total_sweep_wall_time_ms_by_run": total_wall_times,
        "warm_cache_interpretation": (
            "Later runs may reuse Triton disk-cache artifacts. Their wall times support repeated "
            "latency/ranking analysis but must not be interpreted as cold compile cost."
        ),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Runs analyzed: {len(args.sweeps)}")
    print(f"Supported workloads: {len(workloads)}")
    print(f"Configurations per workload: {metadata['config_count_per_workload']}")
    print(
        "Exact top-1 agreement: "
        f"{metadata['all_run_top1_agreement_workloads']}/{len(workloads)} workloads"
    )
    print(
        "Aggregate oracle within 3% in every run: "
        f"{metadata['aggregate_oracle_within_3pct_every_run_workloads']}/{len(workloads)} workloads"
    )
    print(f"Wrote {args.config_output}")
    print(f"Wrote {args.workload_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
