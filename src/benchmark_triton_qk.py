"""Benchmark a small Triton QK^T kernel against PyTorch matmul.

This is not FlashAttention. It isolates one tiled component of attention so I
can study how head_dim changes the scaling behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)

try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton is not installed. Run `python -m pip install triton` in the project venv.")
    sys.exit(1)


@triton.jit
def qk_matmul_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    k = tl.load(
        k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dims", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/triton_qk_benchmark.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/triton_qk_benchmark_metadata.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def time_cuda(fn, warmup: int, trials: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize()

    times_ms: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "mean_latency_ms": statistics.mean(times_ms),
        "median_latency_ms": statistics.median(times_ms),
        "min_latency_ms": min(times_ms),
        "max_latency_ms": max(times_ms),
        "std_latency_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def qk_flops(seq_len: int, head_dim: int) -> int:
    return 2 * seq_len * seq_len * head_dim


def add_normalized_metrics(row: dict[str, object], seq_len: int, head_dim: int) -> dict[str, object]:
    flops = qk_flops(seq_len, head_dim)
    median_ms = float(row["median_latency_ms"])
    mean_ms = float(row["mean_latency_ms"])
    row.update(
        {
            "estimated_qk_flops": flops,
            "mean_tflops": flops / (mean_ms / 1000.0) / 1e12,
            "median_tflops": flops / (median_ms / 1000.0) / 1e12,
            "median_us_per_token": median_ms * 1000.0 / seq_len,
            "median_ns_per_qk_pair": median_ms * 1_000_000.0 / (seq_len * seq_len),
        }
    )
    return row


def benchmark_one(seq_len: int, head_dim: int, args: argparse.Namespace) -> list[dict[str, object]]:
    q = torch.randn((seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq_len, head_dim), device="cuda", dtype=torch.float16)
    torch_out = torch.empty((seq_len, seq_len), device="cuda", dtype=torch.float16)
    triton_out = torch.empty((seq_len, seq_len), device="cuda", dtype=torch.float32)
    block_d = next_power_of_2(head_dim)

    def run_torch() -> None:
        torch.matmul(q, k.transpose(0, 1), out=torch_out)

    def run_triton() -> None:
        grid = (triton.cdiv(seq_len, args.block_m), triton.cdiv(seq_len, args.block_n))
        qk_matmul_kernel[grid](
            q,
            k,
            triton_out,
            seq_len,
            head_dim,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            triton_out.stride(0),
            triton_out.stride(1),
            args.block_m,
            args.block_n,
            block_d,
        )

    correctness_max_abs_diff = None
    correctness_mean_abs_diff = None
    if not args.skip_correctness:
        run_torch()
        run_triton()
        synchronize()
        diff = (torch_out.float() - triton_out).abs()
        correctness_max_abs_diff = float(diff.max().item())
        correctness_mean_abs_diff = float(diff.mean().item())

    rows: list[dict[str, object]] = []
    for method, fn in [("torch_matmul", run_torch), ("triton_qk_matmul", run_triton)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row: dict[str, object] = {
            "method": method,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "dtype": "float16",
            "block_m": args.block_m if method == "triton_qk_matmul" else None,
            "block_n": args.block_n if method == "triton_qk_matmul" else None,
            "block_d": block_d if method == "triton_qk_matmul" else None,
            "correctness_max_abs_diff": correctness_max_abs_diff,
            "correctness_mean_abs_diff": correctness_mean_abs_diff,
        }
        row.update(time_cuda(fn, args.warmup, args.trials))
        row["memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        rows.append(add_normalized_metrics(row, seq_len, head_dim))
    return rows


def metadata(args: argparse.Namespace) -> dict[str, object]:
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "device_name": props.name,
        "device_index": device_index,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "seq_len": args.seq_len,
        "head_dims": args.head_dims,
        "warmup": args.warmup,
        "trials": args.trials,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "seed": args.seed,
        "note": "This is a QK^T matmul microbenchmark, not full FlashAttention.",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata(args), indent=2))

    rows: list[dict[str, object]] = []
    for head_dim in args.head_dims:
        print(f"\nhead_dim={head_dim}")
        for row in benchmark_one(args.seq_len, head_dim, args):
            rows.append(row)
            print(
                f"  {row['method']}: median={float(row['median_latency_ms']):.4f} ms, "
                f"{float(row['median_tflops']):.2f} TFLOP/s"
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
