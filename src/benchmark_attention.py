"""Benchmark standard attention against PyTorch FlashAttention."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Run `python3 src/check_environment.py` first.")
    sys.exit(1)

from attention import flash_attention, standard_attention


AttentionFn = Callable[..., torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--allow-sdpa-auto", action="store_true", help="Use PyTorch SDPA even if flash cannot be forced.")
    parser.add_argument("--output", type=Path, default=Path("results/attention_benchmark.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/environment_metadata.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def environment_metadata(args: argparse.Namespace) -> dict[str, object]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "device_index": device,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "batch_size": args.batch_size,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
        "warmup": args.warmup,
        "trials": args.trials,
        "seed": args.seed,
    }


def make_inputs(args: argparse.Namespace, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (args.batch_size, args.heads, seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def benchmark_one(
    name: str,
    fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(args.warmup):
        fn(q, k, v, causal=args.causal)

    torch.cuda.synchronize()
    times_ms: list[float] = []

    for _ in range(args.trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(q, k, v, causal=args.causal)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    peak_memory = torch.cuda.max_memory_allocated()
    return {
        "method": name,
        "mean_latency_ms": statistics.mean(times_ms),
        "median_latency_ms": statistics.median(times_ms),
        "min_latency_ms": min(times_ms),
        "max_latency_ms": max(times_ms),
        "std_latency_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "peak_memory_bytes": peak_memory,
    }


def validate_correctness(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    standard = standard_attention(q, k, v, causal=args.causal)
    flashed = flash_attention(
        q,
        k,
        v,
        causal=args.causal,
        force_flash=not args.allow_sdpa_auto,
    )
    torch.cuda.synchronize()

    diff = (standard - flashed).float().abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    passed = math.isfinite(max_abs) and max_abs < 5e-2
    return {
        "correctness_max_abs_diff": max_abs,
        "correctness_mean_abs_diff": mean_abs,
        "correctness_passed": passed,
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Run this benchmark on an NVIDIA GPU machine with CUDA-enabled PyTorch.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(environment_metadata(args), indent=2))

    dtype = dtype_from_name(args.dtype)
    rows: list[dict[str, object]] = []

    for seq_len in args.seq_lengths:
        q, k, v = make_inputs(args, seq_len, dtype)
        correctness: dict[str, float | bool] = {}
        if not args.skip_correctness:
            correctness = validate_correctness(q, k, v, args)
            if not correctness["correctness_passed"]:
                raise RuntimeError(f"Correctness check failed at sequence length {seq_len}: {correctness}")

        standard_row = benchmark_one("standard", standard_attention, q, k, v, args)
        flash_row = benchmark_one(
            "flash",
            lambda q, k, v, causal: flash_attention(
                q,
                k,
                v,
                causal=causal,
                force_flash=not args.allow_sdpa_auto,
            ),
            q,
            k,
            v,
            args,
        )

        standard_median = float(standard_row["median_latency_ms"])
        flash_median = float(flash_row["median_latency_ms"])
        speedup = standard_median / flash_median

        for row in (standard_row, flash_row):
            row.update(
                {
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "dtype": args.dtype,
                    "causal": args.causal,
                    "speedup_over_standard": speedup if row["method"] == "flash" else 1.0,
                    **correctness,
                }
            )
            rows.append(row)

        print(
            f"seq={seq_len}: standard median={standard_median:.3f} ms, "
            f"flash median={flash_median:.3f} ms, speedup={speedup:.2f}x"
        )

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
