"""Sweep attention configurations to look for backend-selection boundaries."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Run `python src/check_environment.py` first.")
    sys.exit(1)

from attention import flash_attention, sdpa_attention, standard_attention
from benchmark_attention import (
    MethodSpec,
    build_methods,
    clear_memory,
    dtype_from_name,
    memory_allocated,
    select_device,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 512, 2048, 4096])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--causal-values", nargs="+", choices=["false", "true"], default=["false", "true"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/cuda_boundary_sweep.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/cuda_boundary_sweep_metadata.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-standard", action="store_true")
    parser.add_argument("--stop-after", type=int, default=None, help="Optional limit for quick smoke tests.")
    return parser.parse_args()


def causal_from_name(value: str) -> bool:
    return value == "true"


def make_inputs(
    batch_size: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def method_list(device: torch.device, include_standard: bool) -> list[MethodSpec]:
    methods = build_methods(device)
    if include_standard:
        return methods
    return [method for method in methods if method.name != "standard"]


def benchmark_method(
    method: MethodSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    trials: int,
    device: torch.device,
) -> dict[str, object]:
    clear_memory(device)

    try:
        for _ in range(warmup):
            method.fn(q, k, v, causal=causal)
        synchronize(device)

        times_ms: list[float] = []
        for _ in range(trials):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                method.fn(q, k, v, causal=causal)
                end.record()
                synchronize(device)
                times_ms.append(start.elapsed_time(end))
            else:
                synchronize(device)
                start_time = time.perf_counter()
                method.fn(q, k, v, causal=causal)
                synchronize(device)
                times_ms.append((time.perf_counter() - start_time) * 1000)

        return {
            "method": method.name,
            "backend": method.backend,
            "status": "ok",
            "error": "",
            "mean_latency_ms": statistics.mean(times_ms),
            "median_latency_ms": statistics.median(times_ms),
            "min_latency_ms": min(times_ms),
            "max_latency_ms": max(times_ms),
            "std_latency_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
            "memory_allocated_bytes": memory_allocated(device),
        }
    except Exception as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "method": method.name,
            "backend": method.backend,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "mean_latency_ms": None,
            "median_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "std_latency_ms": None,
            "memory_allocated_bytes": None,
        }


def attention_pairs(seq_len: int, causal: bool) -> int:
    if causal:
        return seq_len * (seq_len + 1) // 2
    return seq_len * seq_len


def estimated_attention_flops(
    batch_size: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
) -> int:
    # Two attention matmuls: QK^T and P/V. Each multiply-add counts as 2 FLOPs.
    return 4 * batch_size * heads * attention_pairs(seq_len, causal) * head_dim


def add_normalized_metrics(row: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    if row["status"] != "ok":
        row.update(
            {
                "attention_pairs": None,
                "estimated_attention_flops": None,
                "mean_tflops": None,
                "median_tflops": None,
                "median_us_per_token": None,
                "median_ns_per_attention_pair": None,
            }
        )
        return row

    seq_len = int(config["seq_len"])
    head_dim = int(config["head_dim"])
    batch_size = int(config["batch_size"])
    heads = int(config["heads"])
    causal = bool(config["causal"])
    pairs = attention_pairs(seq_len, causal)
    flops = estimated_attention_flops(batch_size, heads, seq_len, head_dim, causal)
    mean_ms = float(row["mean_latency_ms"])
    median_ms = float(row["median_latency_ms"])

    row.update(
        {
            "attention_pairs": pairs,
            "estimated_attention_flops": flops,
            "mean_tflops": flops / (mean_ms / 1000.0) / 1e12,
            "median_tflops": flops / (median_ms / 1000.0) / 1e12,
            "median_us_per_token": median_ms * 1000.0 / (batch_size * seq_len),
            "median_ns_per_attention_pair": median_ms * 1_000_000.0 / (batch_size * heads * pairs),
        }
    )
    return row


def metadata(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    data: dict[str, object] = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "seq_lengths": args.seq_lengths,
        "head_dims": args.head_dims,
        "dtypes": args.dtypes,
        "causal_values": args.causal_values,
        "batch_size": args.batch_size,
        "heads": args.heads,
        "warmup": args.warmup,
        "trials": args.trials,
        "include_standard": args.include_standard,
        "normalized_metrics": {
            "attention_pairs": "seq_len^2 for non-causal attention, seq_len * (seq_len + 1) / 2 for causal attention",
            "estimated_attention_flops": "4 * batch_size * heads * attention_pairs * head_dim, counting multiply-add as 2 FLOPs for QK^T and P/V",
            "mean_tflops": "estimated_attention_flops divided by mean latency",
            "median_tflops": "estimated_attention_flops divided by median latency",
            "median_us_per_token": "median latency normalized by batch_size * seq_len",
            "median_ns_per_attention_pair": "median latency normalized by batch_size * heads * attention_pairs",
        },
        "seed": args.seed,
    }

    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        data.update(
            {
                "device_name": props.name,
                "device_index": index,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
            }
        )

    return data


def add_context(row: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    row.update(config)
    return row


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    methods = method_list(device, args.include_standard)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata(args, device), indent=2))

    rows: list[dict[str, object]] = []
    configs = itertools.product(
        args.seq_lengths,
        args.head_dims,
        args.dtypes,
        args.causal_values,
    )

    completed_configs = 0
    for seq_len, head_dim, dtype_name, causal_name in configs:
        if args.stop_after is not None and completed_configs >= args.stop_after:
            break

        causal = causal_from_name(causal_name)
        dtype = dtype_from_name(dtype_name)
        config = {
            "seq_len": seq_len,
            "head_dim": head_dim,
            "dtype": dtype_name,
            "causal": causal,
            "batch_size": args.batch_size,
            "heads": args.heads,
            "device": device.type,
        }
        print(f"\nconfig={config}")

        try:
            q, k, v = make_inputs(
                args.batch_size,
                args.heads,
                seq_len,
                head_dim,
                dtype,
                device,
            )
        except Exception as error:
            for method in methods:
                row = add_context(
                    {
                        "method": method.name,
                        "backend": method.backend,
                        "status": "failed",
                        "error": f"input allocation failed: {type(error).__name__}: {error}",
                        "mean_latency_ms": None,
                        "median_latency_ms": None,
                        "min_latency_ms": None,
                        "max_latency_ms": None,
                        "std_latency_ms": None,
                        "memory_allocated_bytes": None,
                    },
                    dict(config),
                )
                rows.append(add_normalized_metrics(row, config))
            continue

        for method in methods:
            row = benchmark_method(
                method,
                q,
                k,
                v,
                causal,
                args.warmup,
                args.trials,
                device,
            )
            row = add_context(row, dict(config))
            rows.append(add_normalized_metrics(row, config))
            if row["status"] == "ok":
                print(
                    f"  {method.name}: {float(row['median_latency_ms']):.4f} ms, "
                    f"{float(row['median_tflops']):.2f} TFLOP/s"
                )
            else:
                print(f"  {method.name}: failed: {row['error']}")

        completed_configs += 1

    fieldnames = [
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "batch_size",
        "heads",
        "device",
        "method",
        "backend",
        "status",
        "error",
        "mean_latency_ms",
        "median_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "std_latency_ms",
        "memory_allocated_bytes",
        "attention_pairs",
        "estimated_attention_flops",
        "mean_tflops",
        "median_tflops",
        "median_us_per_token",
        "median_ns_per_attention_pair",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
