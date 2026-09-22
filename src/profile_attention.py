"""Profile attention implementations with PyTorch Profiler.

This is a fallback profiler for machines where Nsight Systems or Nsight Compute
are not available. It records operator-level CPU/CUDA time and Chrome traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
except ImportError:
    print("PyTorch is not installed. Run `python src/check_environment.py` first.")
    sys.exit(1)

from benchmark_attention import (
    build_methods,
    dtype_from_name,
    make_inputs,
    select_device,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["standard", "sdpa_auto", "flash_forced"],
        default=None,
        help="Optional subset of attention methods to profile.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("profiling/pytorch"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=25)
    return parser.parse_args()


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
        "methods": args.methods,
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
            }
        )

    return metadata


def profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def write_events_csv(path: Path, events: list[object]) -> None:
    fieldnames = [
        "name",
        "self_cpu_time_total_us",
        "cpu_time_total_us",
        "self_cuda_time_total_us",
        "cuda_time_total_us",
        "cpu_memory_usage_bytes",
        "cuda_memory_usage_bytes",
        "count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "name": event.key,
                    "self_cpu_time_total_us": getattr(event, "self_cpu_time_total", 0.0),
                    "cpu_time_total_us": getattr(event, "cpu_time_total", 0.0),
                    "self_cuda_time_total_us": getattr(event, "self_cuda_time_total", 0.0),
                    "cuda_time_total_us": getattr(event, "cuda_time_total", 0.0),
                    "cpu_memory_usage_bytes": getattr(event, "cpu_memory_usage", 0),
                    "cuda_memory_usage_bytes": getattr(event, "cuda_memory_usage", 0),
                    "count": getattr(event, "count", 0),
                }
            )


def profile_one_method(
    args: argparse.Namespace,
    device: torch.device,
    seq_len: int,
    method: object,
    dtype: torch.dtype,
) -> dict[str, object]:
    q, k, v = make_inputs(args, seq_len, dtype, device)

    for _ in range(args.warmup):
        method.fn(q, k, v, causal=args.causal)
    synchronize(device)

    label = f"seq{seq_len}_{method.name}"
    trace_path = args.output_dir / f"{label}.trace.json"
    table_path = args.output_dir / f"{label}.txt"
    events_path = args.output_dir / f"{label}.events.csv"

    try:
        with profile(
            activities=profiler_activities(device),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with record_function(label):
                for _ in range(args.trials):
                    method.fn(q, k, v, causal=args.causal)
                synchronize(device)

        events = prof.key_averages()
        table = events.table(
            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
            row_limit=args.row_limit,
        )

        table_path.write_text(table)
        write_events_csv(events_path, list(events))
        prof.export_chrome_trace(str(trace_path))

        return {
            "seq_len": seq_len,
            "method": method.name,
            "backend": method.backend,
            "status": "ok",
            "error": "",
            "trace_path": str(trace_path),
            "table_path": str(table_path),
            "events_path": str(events_path),
        }
    except Exception as error:
        return {
            "seq_len": seq_len,
            "method": method.name,
            "backend": method.backend,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "trace_path": "",
            "table_path": "",
            "events_path": "",
        }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seq_len",
        "method",
        "backend",
        "status",
        "error",
        "trace_path",
        "table_path",
        "events_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(environment_metadata(args, device), indent=2))

    rows: list[dict[str, object]] = []
    methods = build_methods(device)
    if args.methods is not None:
        selected_methods = set(args.methods)
        methods = [method for method in methods if method.name in selected_methods]

    for seq_len in args.seq_lengths:
        print(f"\nseq={seq_len}")
        for method in methods:
            print(f"  profiling {method.name}")
            row = profile_one_method(args, device, seq_len, method, dtype)
            rows.append(row)
            if row["status"] == "ok":
                print(f"    wrote {row['table_path']}")
            else:
                print(f"    failed: {row['error']}")

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
