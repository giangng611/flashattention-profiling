"""Measure direct warm-cache tuning cost for one selector policy on CUDA.

Run the tree top-k policy and exhaustive control in separate processes so each
measurement starts with a fresh Python/Triton in-memory state. Existing Triton
disk-cache artifacts are intentionally retained and disclosed in metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from evaluate_attention_config_selectors import (
    aggregate_sweeps,
    build_tree,
    load_config_ids,
    oracle_latency,
    valid_latency,
)


DEFAULT_SWEEPS = [
    Path("results/expanded_attention_sweep.csv"),
    Path("results/expanded_attention_sweep_seed1.csv"),
    Path("results/expanded_attention_sweep_seed2.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tree", "exhaustive"], required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--aggregate-sweep", type=Path, nargs="+", default=DEFAULT_SWEEPS)
    parser.add_argument(
        "--workloads",
        type=Path,
        default=Path("workloads/expanded_dense_prefill_workloads.csv"),
    )
    parser.add_argument(
        "--configs", type=Path, default=Path("configs/attention_config_space.csv")
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--output", type=Path, default=Path("results/online_selector_measurements.csv")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/online_selector_summary.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/online_selector_metadata.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config_tuples(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        row["config_id"]: (
            int(row["block_m"]),
            int(row["block_n"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("configuration IDs must be unique")
    return result


def build_candidate_plan(
    aggregate_sweeps: list[Path],
    configs: Path,
    mode: str,
    budget: int,
) -> tuple[
    dict[str, list[str]],
    dict[str, object],
    dict[tuple[str, str], object],
    list[str],
    object,
]:
    config_ids = load_config_ids(configs)
    if budget <= 0 or budget > len(config_ids):
        raise ValueError("budget must be between 1 and the configuration count")
    workloads, measurements, _ = aggregate_sweeps_fn(aggregate_sweeps, config_ids)
    development_names = sorted(
        workload.name for workload in workloads.values() if workload.split == "development"
    )
    held_out_names = sorted(
        workload.name for workload in workloads.values() if workload.split == "held_out"
    )
    if not development_names or not held_out_names:
        raise ValueError("original development and held-out workloads are required")
    tree = build_tree(
        development_names,
        workloads,
        config_ids,
        measurements,
        max_depth=2,
        min_leaf=2,
    )
    if mode == "exhaustive":
        plan = {name: config_ids.copy() for name in held_out_names}
    else:
        plan = {name: tree.rank(workloads[name])[:budget] for name in held_out_names}
    return plan, workloads, measurements, config_ids, tree


# Alias keeps the public helper argument name readable without shadowing the import.
aggregate_sweeps_fn = aggregate_sweeps


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.trials <= 0:
        raise ValueError("warmup must be nonnegative and trials must be positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("correctness tolerances must be nonnegative")

    plan, evaluation_workloads, aggregate_measurements, config_ids, tree = build_candidate_plan(
        args.aggregate_sweep,
        args.configs,
        args.mode,
        args.budget,
    )
    config_tuples = load_config_tuples(args.configs)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the online selector benchmark")

    from benchmark_triton_fused_attention import load_workloads, metadata
    from sweep_triton_fused_attention_configs import (
        benchmark_torch_once,
        benchmark_triton_configs,
        kernel_args,
        write_csv,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark_workloads = {
        workload.name: workload
        for workload in load_workloads(args.workloads)
        if workload.name in plan
    }
    if set(benchmark_workloads) != set(plan):
        missing = sorted(set(plan) - set(benchmark_workloads))
        raise ValueError(f"workload file is missing held-out workloads: {missing}")

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for workload_name in sorted(plan):
        workload = benchmark_workloads[workload_name]
        candidate_ids = plan[workload_name]
        candidates = [config_tuples[config_id] for config_id in candidate_ids]
        print(f"\nworkload={workload_name} mode={args.mode} budget={len(candidate_ids)}")

        torch_row, q, k, v = benchmark_torch_once(workload, args, candidates[0])
        torch_row["selection_mode"] = "torch_reference"
        torch_row["selection_budget"] = 0
        torch_row["candidate_rank"] = None
        detailed_rows.append(torch_row)

        inference_start = time.perf_counter()
        if args.mode == "tree":
            selected_candidate_ids = tree.rank(evaluation_workloads[workload_name])[: args.budget]
        else:
            selected_candidate_ids = config_ids
        selector_inference_ms = (time.perf_counter() - inference_start) * 1000.0

        tuning_start = time.perf_counter()
        triton_rows = benchmark_triton_configs(
            workload,
            q,
            k,
            v,
            args,
            float(torch_row["median_latency_ms"]),
            [config_tuples[config_id] for config_id in selected_candidate_ids],
        )
        tuning_wall_time_ms = (time.perf_counter() - tuning_start) * 1000.0
        rank_by_config = {
            config_id: rank for rank, config_id in enumerate(selected_candidate_ids, start=1)
        }
        for row in triton_rows:
            row["selection_mode"] = args.mode
            row["selection_budget"] = len(selected_candidate_ids)
            row["candidate_rank"] = rank_by_config[str(row["config_id"])]
            detailed_rows.append(row)

        valid_rows = [
            row
            for row in triton_rows
            if row["status"] == "ok" and row["correctness_passed"] is True
        ]
        selected = min(
            valid_rows,
            key=lambda row: (float(row["median_latency_ms"]), str(row["config_id"])),
            default=None,
        )
        selected_config_id = str(selected["config_id"]) if selected else None
        aggregate_selected_latency = (
            valid_latency(aggregate_measurements, workload_name, selected_config_id)
            if selected_config_id is not None
            else None
        )
        aggregate_oracle = oracle_latency(
            aggregate_measurements, workload_name, config_ids
        )
        summary_rows.append(
            {
                "workload": workload_name,
                "selection_mode": args.mode,
                "selection_budget": len(selected_candidate_ids),
                "candidate_config_ids": ";".join(selected_candidate_ids),
                "selector_inference_ms": selector_inference_ms,
                "tuning_wall_time_ms": tuning_wall_time_ms,
                "configs_attempted": len(selected_candidate_ids),
                "valid_configs_measured": len(valid_rows),
                "runtime_error_configs": sum(row["status"] == "error" for row in triton_rows),
                "correctness_failed_configs": sum(
                    row["status"] == "ok" and row["correctness_passed"] is False
                    for row in triton_rows
                ),
                "selected_config_id": selected_config_id,
                "selected_median_latency_ms": (
                    selected["median_latency_ms"] if selected else None
                ),
                "aggregate_selected_config_latency_ms": aggregate_selected_latency,
                "aggregate_oracle_latency_ms": aggregate_oracle,
                "aggregate_oracle_relative_performance": (
                    aggregate_oracle / aggregate_selected_latency
                    if aggregate_selected_latency is not None
                    else 0.0
                ),
                "compile_or_cache_prepare_total_ms": sum(
                    float(row["compile_or_cache_prepare_ms"])
                    for row in valid_rows
                    if row["compile_or_cache_prepare_ms"] is not None
                ),
                "benchmark_wall_time_total_ms": sum(
                    float(row["benchmark_wall_time_ms"])
                    for row in valid_rows
                    if row["benchmark_wall_time_ms"] is not None
                ),
                "torch_sdpa_median_latency_ms": torch_row["median_latency_ms"],
            }
        )

    write_csv(args.output, detailed_rows)
    write_csv(args.summary_output, summary_rows)
    meta_args = kernel_args(args, *config_tuples[plan[sorted(plan)[0]][0]])
    result_metadata = metadata(meta_args, list(benchmark_workloads.values()))
    result_metadata.update(
        {
            "selection_mode": args.mode,
            "selection_budget": len(next(iter(plan.values()))),
            "candidate_plan": plan,
            "aggregate_sweeps": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.aggregate_sweep
            ],
            "config_file": str(args.configs),
            "config_file_sha256": sha256(args.configs),
            "workload_file": str(args.workloads),
            "workload_file_sha256": sha256(args.workloads),
            "timing_scope": (
                "tuning_wall_time_ms directly times compilation/cache preparation, correctness, "
                "and steady-state benchmarking for the selected candidate set; PyTorch reference "
                "timing and input creation are excluded"
            ),
            "cache_note": (
                "This benchmark retains the existing Triton disk cache. Run tree and exhaustive "
                "modes in separate Python processes; results measure warm-cache online selection."
            ),
        }
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(result_metadata, indent=2) + "\n")

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
