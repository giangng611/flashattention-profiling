"""Benchmark standard attention against PyTorch SDPA and FlashAttention."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Run `python3 src/check_environment.py` first.")
    sys.exit(1)

from attention import flash_attention, sdpa_attention, standard_attention


AttentionFn = Callable[..., torch.Tensor]


class MethodSpec(NamedTuple):
    name: str
    backend: str
    fn: AttentionFn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/attention_benchmark.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/environment_metadata.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise RuntimeError("MPS was requested, but MPS is not available.")

    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_allocated(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def error_row(name: str, backend: str, error: Exception) -> dict[str, object]:
    return {
        "method": name,
        "backend": backend,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "mean_latency_ms": None,
        "median_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "std_latency_ms": None,
        "memory_allocated_bytes": None,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def environment_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "batch_size": args.batch_size,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
        "warmup": args.warmup,
        "trials": args.trials,
        "seed": args.seed,
    }

    if device.type == "cuda":
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update(
            {
                "device_name": properties.name,
                "device_index": device_index,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "benchmark_kind": "cuda_flashattention",
            }
        )
    elif device.type == "mps":
        metadata.update(
            {
                "device_name": "Apple MPS",
                "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
                "benchmark_kind": "mps_practice_sdpa",
            }
        )
    else:
        metadata.update(
            {
                "device_name": "CPU",
                "benchmark_kind": "cpu_correctness_or_smoke_test",
            }
        )

    return metadata


def make_inputs(
    args: argparse.Namespace,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (args.batch_size, args.heads, seq_len, args.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def benchmark_one(
    method: MethodSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | str]:
    clear_memory(device)

    for _ in range(args.warmup):
        method.fn(q, k, v, causal=args.causal)

    synchronize(device)
    times_ms: list[float] = []

    for _ in range(args.trials):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            method.fn(q, k, v, causal=args.causal)
            end.record()
            synchronize(device)
            times_ms.append(start.elapsed_time(end))
        else:
            synchronize(device)
            start_time = time.perf_counter()
            method.fn(q, k, v, causal=args.causal)
            synchronize(device)
            times_ms.append((time.perf_counter() - start_time) * 1000)

    allocated_memory = memory_allocated(device)
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
        "memory_allocated_bytes": allocated_memory,
    }


def correctness_against_standard(
    standard: torch.Tensor,
    compared: torch.Tensor,
) -> dict[str, float | bool]:
    diff = (standard - compared).float().abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    passed = math.isfinite(max_abs) and max_abs < 5e-2
    return {
        "correctness_max_abs_diff": max_abs,
        "correctness_mean_abs_diff": mean_abs,
        "correctness_passed": passed,
    }


def build_methods(device: torch.device) -> list[MethodSpec]:
    methods = [
        MethodSpec("standard", "explicit_matmul_softmax_matmul", standard_attention),
        MethodSpec("sdpa_auto", f"{device.type}_sdpa_auto", sdpa_attention),
    ]

    if device.type == "cuda":
        methods.append(
            MethodSpec(
                "flash_forced",
                "cuda_sdpa_flash_forced",
                lambda q, k, v, causal: flash_attention(
                    q,
                    k,
                    v,
                    causal=causal,
                    force_flash=True,
                ),
            )
        )

    return methods


def add_run_context(
    row: dict[str, object],
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    speedup: float | None,
    correctness: dict[str, object],
) -> dict[str, object]:
    row.update(
        {
            "seq_len": seq_len,
            "device": device.type,
            "batch_size": args.batch_size,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": str(dtype).replace("torch.", ""),
            "causal": args.causal,
            "speedup_over_standard": speedup,
            **correctness,
        }
    )
    return row


def compute_standard_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    standard = standard_attention(q, k, v, causal=args.causal)
    synchronize(device)
    return standard


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    if device.type != "cuda":
        print(
            "Non-CUDA device selected. Running practice benchmark with PyTorch SDPA, "
            "not CUDA FlashAttention. Use --device cuda on an NVIDIA machine for the main experiment."
        )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(environment_metadata(args, device), indent=2))

    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    methods = build_methods(device)
    rows: list[dict[str, object]] = []

    for seq_len in args.seq_lengths:
        q, k, v = make_inputs(args, seq_len, dtype, device)
        print(f"\nseq={seq_len}")

        try:
            standard_output = compute_standard_output(q, k, v, args, device)
        except Exception as error:
            for method in methods:
                rows.append(
                    add_run_context(
                        error_row(method.name, method.backend, error),
                        seq_len,
                        device,
                        dtype,
                        args,
                        None,
                        {
                            "correctness_max_abs_diff": None,
                            "correctness_mean_abs_diff": None,
                            "correctness_passed": False,
                        },
                    )
                )
            print(f"  all methods failed during standard correctness reference: {error}")
            continue

        seq_rows: list[dict[str, object]] = []
        standard_median: float | None = None

        for method in methods:
            correctness: dict[str, object] = {
                "correctness_max_abs_diff": 0.0 if method.name == "standard" else None,
                "correctness_mean_abs_diff": 0.0 if method.name == "standard" else None,
                "correctness_passed": True if method.name == "standard" else None,
            }

            try:
                if method.name != "standard" and not args.skip_correctness:
                    compared_output = method.fn(q, k, v, causal=args.causal)
                    synchronize(device)
                    correctness = correctness_against_standard(standard_output, compared_output)
                    if not correctness["correctness_passed"]:
                        raise RuntimeError(f"correctness check failed: {correctness}")

                row = benchmark_one(method, q, k, v, args, device)
            except Exception as error:
                row = error_row(method.name, method.backend, error)

            seq_rows.append(add_run_context(row, seq_len, device, dtype, args, None, correctness))
            if method.name == "standard" and row["status"] == "ok":
                standard_median = float(row["median_latency_ms"])

        for row in seq_rows:
            if row["status"] == "ok" and standard_median is not None:
                row["speedup_over_standard"] = standard_median / float(row["median_latency_ms"])
            rows.append(row)

            if row["status"] == "ok":
                print(
                    f"  {row['method']}: median={float(row['median_latency_ms']):.3f} ms, "
                    f"speedup={float(row['speedup_over_standard']):.2f}x"
                )
            else:
                print(f"  {row['method']}: failed: {row['error']}")

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
