"""Sweep fixed Triton fused-attention configurations.

This script is the next small step after the first fused-attention benchmark.
It measures whether different workload shapes prefer different tile and launch
choices. The fastest correct Triton result per workload is treated as the oracle
configuration for that small search space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)

from benchmark_triton_fused_attention import (
    add_metrics,
    base_row,
    compile_triton_attention,
    load_workloads,
    metadata,
    skipped_rows,
    time_cuda,
    torch_sdpa,
    triton_attention,
    workload_supported,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, default=Path("workloads/small_attention_workloads.csv"))
    parser.add_argument("--configs", type=Path, default=None, help="Optional explicit configuration CSV.")
    parser.add_argument("--block-ms", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--block-ns", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--num-warps-values", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--num-stages-values", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, default=Path("results/triton_fused_attention_config_sweep.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/triton_fused_attention_config_sweep_summary.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/triton_fused_attention_config_sweep_metadata.json"),
    )
    return parser.parse_args()


def kernel_args(
    args: argparse.Namespace,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        warmup=args.warmup,
        trials=args.trials,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        skip_correctness=False,
        output=args.output,
        metadata_output=args.metadata_output,
    )


def load_config_candidates(args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    if args.configs is None:
        candidates = [
            (block_m, block_n, num_warps, num_stages)
            for block_m in args.block_ms
            for block_n in args.block_ns
            for num_warps in args.num_warps_values
            for num_stages in args.num_stages_values
        ]
    else:
        with args.configs.open() as handle:
            rows = list(csv.DictReader(handle))
        candidates = [
            (int(row["block_m"]), int(row["block_n"]), int(row["num_warps"]), int(row["num_stages"]))
            for row in rows
        ]
    if not candidates:
        raise ValueError(f"No configurations found in {args.configs}")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"Duplicate configurations found in {args.configs}")
    for block_m, block_n, num_warps, num_stages in candidates:
        if block_m <= 0 or block_m & (block_m - 1):
            raise ValueError(f"block_m must be a positive power of two, got {block_m}")
        if block_n <= 0 or block_n & (block_n - 1):
            raise ValueError(f"block_n must be a positive power of two, got {block_n}")
        if num_warps not in {1, 2, 4, 8}:
            raise ValueError(f"num_warps must be one of 1, 2, 4, 8, got {num_warps}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
    return candidates


def add_config_metrics(row: dict[str, object], torch_median_ms: float | None) -> dict[str, object]:
    if row["status"] != "ok" or torch_median_ms is None:
        row["speedup_over_torch_sdpa"] = None
        row["latency_gap_to_torch_pct"] = None
        return row

    median_ms = float(row["median_latency_ms"])
    row["speedup_over_torch_sdpa"] = torch_median_ms / median_ms
    row["latency_gap_to_torch_pct"] = (median_ms / torch_median_ms - 1.0) * 100.0
    return row


def benchmark_torch_once(
    workload,
    args: argparse.Namespace,
    first_config: tuple[int, int, int, int],
) -> tuple[dict[str, object] | None, object, object, object]:
    supported, reason = workload_supported(workload)
    if not supported:
        return None, None, None, None

    config_args = kernel_args(
        args,
        *first_config,
    )
    q, k, v = benchmark_inputs(workload)
    row = base_row(workload, "torch_sdpa", config_args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    benchmark_start = time.perf_counter()
    row.update(time_cuda(lambda: torch_sdpa(q, k, v, workload), args.warmup, args.trials))
    row["benchmark_wall_time_ms"] = (time.perf_counter() - benchmark_start) * 1000.0
    row["memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    row.update(
        {
            "correctness_max_abs_diff": 0.0,
            "correctness_mean_abs_diff": 0.0,
            "correctness_max_rel_diff": 0.0,
            "correctness_atol": args.atol,
            "correctness_rtol": args.rtol,
            "correctness_passed": True,
            "compile_or_cache_prepare_ms": None,
        }
    )
    row = add_metrics(row, workload)
    row["config_id"] = "torch_sdpa"
    row["is_oracle"] = False
    row["oracle_gap_pct"] = None
    row["speedup_over_torch_sdpa"] = 1.0
    row["latency_gap_to_torch_pct"] = 0.0
    return row, q, k, v


def benchmark_inputs(workload):
    from benchmark_triton_fused_attention import make_inputs

    return make_inputs(workload)


def benchmark_triton_configs(
    workload,
    q,
    k,
    v,
    args: argparse.Namespace,
    torch_median_ms: float,
    candidates: list[tuple[int, int, int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block_m, block_n, num_warps, num_stages in candidates:
        config_args = kernel_args(args, block_m, block_n, num_warps, num_stages)
        config_id = f"bm{block_m}_bn{block_n}_w{num_warps}_s{num_stages}"
        print(
            "  triton config "
            f"block_m={block_m} block_n={block_n} "
            f"num_warps={num_warps} num_stages={num_stages}"
        )
        try:
            workload_rows = benchmark_workload_with_existing_inputs(workload, q, k, v, config_args)
        except Exception as exc:
            row = base_row(workload, "triton_fused_attention", config_args)
            row.update(
                {
                    "status": "error",
                    "error": type(exc).__name__ + ": " + str(exc),
                    "mean_latency_ms": None,
                    "median_latency_ms": None,
                    "min_latency_ms": None,
                    "max_latency_ms": None,
                    "std_latency_ms": None,
                    "memory_allocated_bytes": None,
                    "correctness_max_abs_diff": None,
                    "correctness_mean_abs_diff": None,
                    "correctness_max_rel_diff": None,
                    "correctness_atol": args.atol,
                    "correctness_rtol": args.rtol,
                    "correctness_passed": False,
                    "compile_or_cache_prepare_ms": None,
                    "benchmark_wall_time_ms": None,
                }
            )
            workload_rows = [add_metrics(row, workload)]

        for row in workload_rows:
            row["config_id"] = config_id
            row["is_oracle"] = False
            row["oracle_gap_pct"] = None
            rows.append(add_config_metrics(row, torch_median_ms))
    return rows


def benchmark_workload_with_existing_inputs(workload, q, k, v, args) -> list[dict[str, object]]:
    from benchmark_triton_fused_attention import correctness

    reference = torch_sdpa(q, k, v, workload)
    torch.cuda.synchronize()
    compile_or_cache_prepare_ms = compile_triton_attention(q, k, v, workload, args)
    triton_out = triton_attention(q, k, v, workload, args)
    torch.cuda.synchronize()
    correctness_info = correctness(reference, triton_out, args.atol, args.rtol)

    row = base_row(workload, "triton_fused_attention", args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    benchmark_start = time.perf_counter()
    row.update(time_cuda(lambda: triton_attention(q, k, v, workload, args), args.warmup, args.trials))
    row["benchmark_wall_time_ms"] = (time.perf_counter() - benchmark_start) * 1000.0
    row["compile_or_cache_prepare_ms"] = compile_or_cache_prepare_ms
    row["memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    row.update(correctness_info)
    return [add_metrics(row, workload)]


def mark_oracle(rows: list[dict[str, object]]) -> None:
    by_workload: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["method"] == "triton_fused_attention" and row["status"] == "ok" and row["correctness_passed"] is True:
            by_workload.setdefault(str(row["workload"]), []).append(row)

    for workload_rows in by_workload.values():
        best = min(workload_rows, key=lambda row: float(row["median_latency_ms"]))
        best_latency = float(best["median_latency_ms"])
        best["is_oracle"] = True
        for row in workload_rows:
            row["oracle_gap_pct"] = (float(row["median_latency_ms"]) / best_latency - 1.0) * 100.0


def summarize(
    rows: list[dict[str, object]],
    workload_sweep_wall_time_ms: dict[str, float],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    workloads = sorted({str(row["workload"]) for row in rows})
    for workload in workloads:
        workload_rows = [row for row in rows if row["workload"] == workload]
        context = workload_rows[0]
        torch_rows = [row for row in workload_rows if row["method"] == "torch_sdpa" and row["status"] == "ok"]
        triton_ok = [
            row
            for row in workload_rows
            if row["method"] == "triton_fused_attention" and row["status"] == "ok" and row["correctness_passed"] is True
        ]
        triton_rows = [row for row in workload_rows if row["method"] == "triton_fused_attention"]
        runtime_errors = [row for row in triton_rows if row["status"] == "error"]
        correctness_failures = [
            row
            for row in triton_rows
            if row["status"] == "ok" and row["correctness_passed"] is False
        ]
        skipped = [row for row in workload_rows if row["status"] == "skipped"]
        if not triton_ok:
            reason = skipped[0]["error"] if skipped else "no valid Triton configuration"
            summary.append(
                {
                    "workload": workload,
                    "status": "skipped" if skipped else "error",
                    "reason": reason,
                    "split": context["split"],
                    "source": context["source"],
                    "dtype": context["dtype"],
                    "query_length": context["query_length"],
                    "query_heads": context["query_heads"],
                    "head_dim": context["head_dim"],
                    "causal": context["causal"],
                    "torch_median_latency_ms": None,
                    "best_config_id": None,
                    "best_block_m": None,
                    "best_block_n": None,
                    "best_num_warps": None,
                    "best_num_stages": None,
                    "best_triton_median_latency_ms": None,
                    "best_triton_median_tflops": None,
                    "best_speedup_over_torch_sdpa": None,
                    "candidate_triton_configs": 0 if skipped else len(triton_rows),
                    "tested_triton_configs": 0,
                    "runtime_error_configs": len(runtime_errors),
                    "correctness_failed_configs": len(correctness_failures),
                    "compile_or_cache_prepare_total_ms": None,
                    "benchmark_wall_time_total_ms": None,
                    "sweep_wall_time_ms": workload_sweep_wall_time_ms.get(workload),
                }
            )
            continue

        best = min(triton_ok, key=lambda row: float(row["median_latency_ms"]))
        torch_median = float(torch_rows[0]["median_latency_ms"]) if torch_rows else None
        summary.append(
            {
                "workload": workload,
                "status": "ok",
                "reason": "",
                "split": context["split"],
                "source": context["source"],
                "dtype": context["dtype"],
                "query_length": context["query_length"],
                "query_heads": context["query_heads"],
                "head_dim": context["head_dim"],
                "causal": context["causal"],
                "torch_median_latency_ms": torch_median,
                "best_config_id": best["config_id"],
                "best_block_m": best["block_m"],
                "best_block_n": best["block_n"],
                "best_num_warps": best["num_warps"],
                "best_num_stages": best["num_stages"],
                "best_triton_median_latency_ms": best["median_latency_ms"],
                "best_triton_median_tflops": best["median_tflops"],
                "best_speedup_over_torch_sdpa": best["speedup_over_torch_sdpa"],
                "candidate_triton_configs": len(triton_rows),
                "tested_triton_configs": len(triton_ok),
                "runtime_error_configs": len(runtime_errors),
                "correctness_failed_configs": len(correctness_failures),
                "compile_or_cache_prepare_total_ms": sum(
                    float(row["compile_or_cache_prepare_ms"])
                    for row in triton_ok
                    if row["compile_or_cache_prepare_ms"] is not None
                ),
                "benchmark_wall_time_total_ms": sum(
                    float(row["benchmark_wall_time_ms"])
                    for row in triton_ok
                    if row["benchmark_wall_time_ms"] is not None
                ),
                "sweep_wall_time_ms": workload_sweep_wall_time_ms.get(workload),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Correctness tolerances must be non-negative")
    if args.warmup < 0 or args.trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    workloads = load_workloads(args.workloads)
    candidates = load_config_candidates(args)
    first_config = candidates[0]
    rows: list[dict[str, object]] = []
    workload_sweep_wall_time_ms: dict[str, float] = {}
    for workload in workloads:
        print(f"\nworkload={workload.name}")
        supported, reason = workload_supported(workload)
        if workload.expected_status == "supported" and not supported:
            raise ValueError(
                f"Workload {workload.name} is labeled supported but validation failed: {reason}"
            )
        if workload.expected_status == "unsupported" and supported:
            raise ValueError(
                f"Workload {workload.name} is labeled unsupported but passes support validation"
            )
        if not supported:
            config_args = kernel_args(
                args,
                *first_config,
            )
            skipped = skipped_rows(workload, reason, config_args)
            for row in skipped:
                row["config_id"] = "unsupported"
                row["is_oracle"] = False
                row["oracle_gap_pct"] = None
                row["speedup_over_torch_sdpa"] = None
                row["latency_gap_to_torch_pct"] = None
                row["compile_or_cache_prepare_ms"] = None
                row["benchmark_wall_time_ms"] = None
            rows.extend(skipped)
            print(f"  skipped: {reason}")
            continue

        torch_row, q, k, v = benchmark_torch_once(workload, args, first_config)
        rows.append(torch_row)
        torch_median_ms = float(torch_row["median_latency_ms"])
        print(f"  torch_sdpa median={torch_median_ms:.4f} ms")
        sweep_start = time.perf_counter()
        rows.extend(benchmark_triton_configs(workload, q, k, v, args, torch_median_ms, candidates))
        workload_sweep_wall_time_ms[workload.name] = (time.perf_counter() - sweep_start) * 1000.0

    mark_oracle(rows)
    summary_rows = summarize(rows, workload_sweep_wall_time_ms)

    write_csv(args.output, rows)
    write_csv(args.summary_output, summary_rows)

    meta_args = kernel_args(
        args,
        *first_config,
    )
    meta = metadata(meta_args, workloads)
    meta.update(
        {
            "block_ms": args.block_ms if args.configs is None else None,
            "block_ns": args.block_ns if args.configs is None else None,
            "num_warps_values": args.num_warps_values if args.configs is None else None,
            "num_stages_values": args.num_stages_values if args.configs is None else None,
            "config_file": str(args.configs) if args.configs is not None else None,
            "config_candidates": [
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
                for block_m, block_n, num_warps, num_stages in candidates
            ],
            "config_count": len(candidates),
            "timing_note": (
                "compile_or_cache_prepare_ms times Triton JITFunction.warmup without a kernel launch. "
                "It may include cache lookup/load and is not guaranteed cold compilation time. "
                "benchmark_wall_time_ms covers steady-state warmup and timed trials."
            ),
            "note": "Triton fused-attention config sweep. The fastest correct Triton config per workload is marked as the oracle within this small search space.",
        }
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
