"""Benchmark a small Triton fused-attention forward kernel.

This is a forward-only reproduction step toward full fused attention. It is
intentionally smaller than production FlashAttention: it supports contiguous
self-attention shapes with query heads equal to KV heads. Decode and GQA shapes
are recorded as unsupported for now instead of being silently mis-benchmarked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)

try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton is not installed. Run `python -m pip install triton` in the project venv.")
    sys.exit(1)


@dataclass(frozen=True)
class Workload:
    name: str
    phase: str
    batch: int
    query_length: int
    kv_length: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    causal: bool


DEFAULT_WORKLOADS = [
    Workload("toy-prefill-d64", "prefill", 1, 512, 512, 8, 8, 64, "float16", True),
    Workload("toy-prefill-d128", "prefill", 1, 512, 512, 8, 8, 128, "float16", True),
    Workload("qwen2.5-1.5b-like-prefill-512", "prefill", 1, 512, 512, 12, 12, 128, "float16", True),
    Workload("qwen2.5-1.5b-like-prefill-2048", "prefill", 1, 2048, 2048, 12, 12, 128, "float16", True),
    Workload("qwen2.5-1.5b-like-decode-2048", "decode", 1, 1, 2048, 12, 12, 128, "float16", True),
    Workload("qwen2.5-7b-like-prefill-512-gqa", "prefill", 1, 512, 512, 28, 4, 128, "float16", True),
]


@triton.jit
def fused_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    causal: tl.constexpr,
) -> None:
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    offs_m = start_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q_base = off_bh * q_len * head_dim
    kv_base = off_bh * kv_len * head_dim
    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((block_m,), 0.0, dtype=tl.float32)
    acc = tl.zeros((block_m, block_d), dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in range(0, kv_len, block_n):
        curr_n = start_n + offs_n
        k = tl.load(
            k_ptr + kv_base + curr_n[None, :] * head_dim + offs_d[:, None],
            mask=(curr_n[None, :] < kv_len) & (offs_d[:, None] < head_dim),
            other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale
        qk = tl.where(curr_n[None, :] < kv_len, qk, -float("inf"))
        if causal:
            qk = tl.where(offs_m[:, None] >= curr_n[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + kv_base + curr_n[:, None] * head_dim + offs_d[None, :],
            mask=(curr_n[:, None] < kv_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, default=None, help="Optional workload CSV.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/triton_fused_attention.csv"))
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/triton_fused_attention_metadata.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_workloads(path: Path | None) -> list[Workload]:
    if path is None:
        return list(DEFAULT_WORKLOADS)

    with path.open() as handle:
        reader = csv.DictReader(handle)
        return [
            Workload(
                name=row["name"],
                phase=row["phase"],
                batch=int(row["batch"]),
                query_length=int(row["query_length"]),
                kv_length=int(row["kv_length"]),
                query_heads=int(row["query_heads"]),
                kv_heads=int(row["kv_heads"]),
                head_dim=int(row["head_dim"]),
                dtype=row["dtype"],
                causal=row["causal"].lower() == "true",
            )
            for row in reader
        ]


def workload_supported(workload: Workload) -> tuple[bool, str]:
    if workload.query_length != workload.kv_length:
        return False, "initial Triton kernel supports self-attention only, q_len must equal kv_len"
    if workload.query_heads != workload.kv_heads:
        return False, "initial Triton kernel does not support GQA/MQA yet"
    if workload.head_dim not in {32, 64, 128, 256}:
        return False, "head_dim must be one of 32, 64, 128, 256"
    if workload.dtype not in {"float16", "bfloat16"}:
        return False, "dtype must be float16 or bfloat16"
    return True, ""


def make_inputs(workload: Workload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = dtype_from_name(workload.dtype)
    shape_q = (workload.batch, workload.query_heads, workload.query_length, workload.head_dim)
    shape_kv = (workload.batch, workload.kv_heads, workload.kv_length, workload.head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(shape_kv, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(shape_kv, device="cuda", dtype=dtype).contiguous()
    return q, k, v


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, workload: Workload, args: argparse.Namespace) -> torch.Tensor:
    output = torch.empty_like(q)
    grid = (triton.cdiv(workload.query_length, args.block_m), workload.batch * workload.query_heads)
    fused_attention_fwd_kernel[grid](
        q,
        k,
        v,
        output,
        workload.query_length,
        workload.kv_length,
        workload.head_dim,
        workload.head_dim**-0.5,
        args.block_m,
        args.block_n,
        next_power_of_2(workload.head_dim),
        workload.causal,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    return output


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, workload: Workload) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=workload.causal)


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


def attention_pairs(workload: Workload) -> int:
    if workload.causal and workload.query_length == workload.kv_length:
        return workload.query_length * (workload.query_length + 1) // 2
    return workload.query_length * workload.kv_length


def estimated_attention_flops(workload: Workload) -> int:
    return 4 * workload.batch * workload.query_heads * attention_pairs(workload) * workload.head_dim


def add_metrics(row: dict[str, object], workload: Workload) -> dict[str, object]:
    if row["status"] != "ok":
        row.update(
            {
                "attention_pairs": None,
                "estimated_attention_flops": None,
                "median_tflops": None,
                "mean_tflops": None,
                "median_us_per_query_token": None,
            }
        )
        return row

    flops = estimated_attention_flops(workload)
    mean_ms = float(row["mean_latency_ms"])
    median_ms = float(row["median_latency_ms"])
    row.update(
        {
            "attention_pairs": attention_pairs(workload),
            "estimated_attention_flops": flops,
            "mean_tflops": flops / (mean_ms / 1000.0) / 1e12,
            "median_tflops": flops / (median_ms / 1000.0) / 1e12,
            "median_us_per_query_token": median_ms * 1000.0 / (workload.batch * workload.query_length),
        }
    )
    return row


def correctness(reference: torch.Tensor, compared: torch.Tensor) -> dict[str, object]:
    diff = (reference.float() - compared.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    return {
        "correctness_max_abs_diff": max_abs,
        "correctness_mean_abs_diff": mean_abs,
        "correctness_passed": math.isfinite(max_abs) and max_abs < 8e-2,
    }


def base_row(workload: Workload, method: str, args: argparse.Namespace) -> dict[str, object]:
    return {
        "workload": workload.name,
        "phase": workload.phase,
        "method": method,
        "status": "ok",
        "error": "",
        "batch": workload.batch,
        "query_length": workload.query_length,
        "kv_length": workload.kv_length,
        "query_heads": workload.query_heads,
        "kv_heads": workload.kv_heads,
        "head_dim": workload.head_dim,
        "dtype": workload.dtype,
        "causal": workload.causal,
        "block_m": args.block_m if method == "triton_fused_attention" else None,
        "block_n": args.block_n if method == "triton_fused_attention" else None,
        "num_warps": args.num_warps if method == "triton_fused_attention" else None,
        "num_stages": args.num_stages if method == "triton_fused_attention" else None,
    }


def skipped_rows(workload: Workload, reason: str, args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ["torch_sdpa", "triton_fused_attention"]:
        row = base_row(workload, method, args)
        row.update(
            {
                "status": "skipped",
                "error": reason,
                "mean_latency_ms": None,
                "median_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "std_latency_ms": None,
                "memory_allocated_bytes": None,
                "correctness_max_abs_diff": None,
                "correctness_mean_abs_diff": None,
                "correctness_passed": None,
            }
        )
        rows.append(add_metrics(row, workload))
    return rows


def benchmark_workload(workload: Workload, args: argparse.Namespace) -> list[dict[str, object]]:
    supported, reason = workload_supported(workload)
    if not supported:
        return skipped_rows(workload, reason, args)

    q, k, v = make_inputs(workload)
    reference = torch_sdpa(q, k, v, workload)
    synchronize()

    triton_out = triton_attention(q, k, v, workload, args)
    synchronize()
    correctness_info = correctness(reference, triton_out) if not args.skip_correctness else {
        "correctness_max_abs_diff": None,
        "correctness_mean_abs_diff": None,
        "correctness_passed": None,
    }

    rows: list[dict[str, object]] = []
    methods = [
        ("torch_sdpa", lambda: torch_sdpa(q, k, v, workload), {}),
        ("triton_fused_attention", lambda: triton_attention(q, k, v, workload, args), correctness_info),
    ]
    for method_name, fn, method_correctness in methods:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = base_row(workload, method_name, args)
        row.update(time_cuda(fn, args.warmup, args.trials))
        row["memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        if method_name == "torch_sdpa":
            row.update(
                {
                    "correctness_max_abs_diff": 0.0,
                    "correctness_mean_abs_diff": 0.0,
                    "correctness_passed": True,
                }
            )
        else:
            row.update(method_correctness)
        rows.append(add_metrics(row, workload))
    return rows


def metadata(args: argparse.Namespace, workloads: list[Workload]) -> dict[str, object]:
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
        "warmup": args.warmup,
        "trials": args.trials,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "seed": args.seed,
        "workload_count": len(workloads),
        "note": "Forward-only Triton fused-attention benchmark. Initial kernel supports self-attention with q_heads == kv_heads.",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    workloads = load_workloads(args.workloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata(args, workloads), indent=2))

    rows: list[dict[str, object]] = []
    for workload in workloads:
        print(f"\nworkload={workload.name}")
        workload_rows = benchmark_workload(workload, args)
        rows.extend(workload_rows)
        for row in workload_rows:
            if row["status"] == "ok":
                print(
                    f"  {row['method']}: median={float(row['median_latency_ms']):.4f} ms, "
                    f"{float(row['median_tflops']):.2f} TFLOP/s"
                )
            else:
                print(f"  {row['method']}: {row['status']}: {row['error']}")

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
