"""Sweep fixed Triton fused-attention tile configurations.

This script is the next small step after the first fused-attention benchmark.
It measures whether different workload shapes prefer different BLOCK_M/BLOCK_N
choices. The fastest correct Triton result per workload is treated as the
oracle configuration for that small search space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    benchmark_workload,
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
    parser.add_argument("--block-ms", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--block-ns", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
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


def kernel_args(args: argparse.Namespace, block_m: int, block_n: int) -> SimpleNamespace:
    return SimpleNamespace(
        block_m=block_m,
        block_n=block_n,
        warmup=args.warmup,
        trials=args.trials,
        seed=args.seed,
        skip_correctness=False,
        output=args.output,
        metadata_output=args.metadata_output,
    )


def add_config_metrics(row: dict[str, object], torch_median_ms: float | None) -> dict[str, object]:
    if row["status"] != "ok" or torch_median_ms is None:
        row["speedup_over_torch_sdpa"] = None
        row["latency_gap_to_torch_pct"] = None
        return row

    median_ms = float(row["median_latency_ms"])
    row["speedup_over_torch_sdpa"] = torch_median_ms / median_ms
    row["latency_gap_to_torch_pct"] = (median_ms / torch_median_ms - 1.0) * 100.0
    return row


def benchmark_torch_once(workload, args: argparse.Namespace) -> tuple[dict[str, object] | None, object, object, object]:
    supported, reason = workload_supported(workload)
    if not supported:
        return None, None, None, None

    config_args = kernel_args(args, args.block_ms[0], args.block_ns[0])
    q, k, v = benchmark_inputs(workload)
    row = base_row(workload, "torch_sdpa", config_args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row.update(time_cuda(lambda: torch_sdpa(q, k, v, workload), args.warmup, args.trials))
    row["memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    row.update(
        {
            "correctness_max_abs_diff": 0.0,
            "correctness_mean_abs_diff": 0.0,
            "correctness_passed": True,
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


def benchmark_triton_configs(workload, q, k, v, args: argparse.Namespace, torch_median_ms: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block_m in args.block_ms:
        for block_n in args.block_ns:
            config_args = kernel_args(args, block_m, block_n)
            print(f"  triton config block_m={block_m} block_n={block_n}")
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
                        "correctness_passed": False,
                    }
                )
                workload_rows = [add_metrics(row, workload)]

            for row in workload_rows:
                row["config_id"] = f"bm{block_m}_bn{block_n}"
                row["is_oracle"] = False
                row["oracle_gap_pct"] = None
                rows.append(add_config_metrics(row, torch_median_ms))
    return rows


def benchmark_workload_with_existing_inputs(workload, q, k, v, args) -> list[dict[str, object]]:
    from benchmark_triton_fused_attention import correctness

    reference = torch_sdpa(q, k, v, workload)
    torch.cuda.synchronize()
    triton_out = triton_attention(q, k, v, workload, args)
    torch.cuda.synchronize()
    correctness_info = correctness(reference, triton_out)

    row = base_row(workload, "triton_fused_attention", args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row.update(time_cuda(lambda: triton_attention(q, k, v, workload, args), args.warmup, args.trials))
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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    workloads = sorted({str(row["workload"]) for row in rows})
    for workload in workloads:
        workload_rows = [row for row in rows if row["workload"] == workload]
        torch_rows = [row for row in workload_rows if row["method"] == "torch_sdpa" and row["status"] == "ok"]
        triton_ok = [
            row
            for row in workload_rows
            if row["method"] == "triton_fused_attention" and row["status"] == "ok" and row["correctness_passed"] is True
        ]
        skipped = [row for row in workload_rows if row["status"] == "skipped"]
        if not triton_ok:
            reason = skipped[0]["error"] if skipped else "no valid Triton configuration"
            summary.append(
                {
                    "workload": workload,
                    "status": "skipped",
                    "reason": reason,
                    "torch_median_latency_ms": None,
                    "best_config_id": None,
                    "best_block_m": None,
                    "best_block_n": None,
                    "best_triton_median_latency_ms": None,
                    "best_triton_median_tflops": None,
                    "best_speedup_over_torch_sdpa": None,
                    "tested_triton_configs": 0,
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
                "torch_median_latency_ms": torch_median,
                "best_config_id": best["config_id"],
                "best_block_m": best["block_m"],
                "best_block_n": best["block_n"],
                "best_triton_median_latency_ms": best["median_latency_ms"],
                "best_triton_median_tflops": best["median_tflops"],
                "best_speedup_over_torch_sdpa": best["speedup_over_torch_sdpa"],
                "tested_triton_configs": len(triton_ok),
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    workloads = load_workloads(args.workloads)
    rows: list[dict[str, object]] = []
    for workload in workloads:
        print(f"\nworkload={workload.name}")
        supported, reason = workload_supported(workload)
        if not supported:
            config_args = kernel_args(args, args.block_ms[0], args.block_ns[0])
            skipped = skipped_rows(workload, reason, config_args)
            for row in skipped:
                row["config_id"] = "unsupported"
                row["is_oracle"] = False
                row["oracle_gap_pct"] = None
                row["speedup_over_torch_sdpa"] = None
                row["latency_gap_to_torch_pct"] = None
            rows.extend(skipped)
            print(f"  skipped: {reason}")
            continue

        torch_row, q, k, v = benchmark_torch_once(workload, args)
        rows.append(torch_row)
        torch_median_ms = float(torch_row["median_latency_ms"])
        print(f"  torch_sdpa median={torch_median_ms:.4f} ms")
        rows.extend(benchmark_triton_configs(workload, q, k, v, args, torch_median_ms))

    mark_oracle(rows)
    summary_rows = summarize(rows)

    write_csv(args.output, rows)
    write_csv(args.summary_output, summary_rows)

    meta_args = kernel_args(args, args.block_ms[0], args.block_ns[0])
    meta = metadata(meta_args, workloads)
    meta.update(
        {
            "block_ms": args.block_ms,
            "block_ns": args.block_ns,
            "config_count": len(args.block_ms) * len(args.block_ns),
            "note": "Triton fused-attention tile sweep. The fastest correct Triton config per workload is marked as the oracle within this small search space.",
        }
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
