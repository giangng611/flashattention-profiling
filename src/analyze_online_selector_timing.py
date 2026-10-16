"""Compare direct warm-cache top-k and exhaustive selector measurements.

This CPU-only analysis keeps the preserved CUDA outputs immutable and writes a
small workload-level comparison plus aggregate provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k-summary",
        type=Path,
        default=Path("results/online_selector_top4_summary.csv"),
    )
    parser.add_argument(
        "--top-k-metadata",
        type=Path,
        default=Path("results/online_selector_top4_metadata.json"),
    )
    parser.add_argument(
        "--exhaustive-summary",
        type=Path,
        default=Path("results/online_selector_exhaustive_summary.csv"),
    )
    parser.add_argument(
        "--exhaustive-metadata",
        type=Path,
        default=Path("results/online_selector_exhaustive_metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/online_selector_comparison.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/online_selector_comparison_metadata.json"),
    )
    return parser.parse_args()


def normalized_text_sha256(path: Path) -> str:
    """Hash tracked text consistently across LF and CRLF checkouts."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_workload = {row["workload"]: row for row in rows}
    if len(by_workload) != len(rows):
        raise ValueError(f"duplicate workload in {path}")
    if not rows:
        raise ValueError(f"empty summary: {path}")
    return by_workload


def validate_compatible_metadata(top_k: dict[str, object], exhaustive: dict[str, object]) -> None:
    shared_fields = (
        "aggregate_sweeps",
        "workload_file_sha256",
        "config_file_sha256",
        "device_name",
        "compute_capability",
        "total_memory_bytes",
        "torch_version",
        "torch_cuda_version",
        "triton_version",
        "python_version",
        "seed",
        "warmup",
        "trials",
        "correctness_atol",
        "correctness_rtol",
        "workload_count",
        "timing_scope",
        "cache_note",
    )
    mismatches = [field for field in shared_fields if top_k.get(field) != exhaustive.get(field)]
    if mismatches:
        raise ValueError(f"incompatible benchmark metadata fields: {', '.join(mismatches)}")
    if top_k.get("selection_mode") != "tree":
        raise ValueError("top-k metadata must describe tree selection")
    if exhaustive.get("selection_mode") != "exhaustive":
        raise ValueError("control metadata must describe exhaustive selection")


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def build_comparison_rows(
    top_k: dict[str, dict[str, str]], exhaustive: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    if set(top_k) != set(exhaustive):
        raise ValueError("top-k and exhaustive summaries cover different workloads")
    rows: list[dict[str, object]] = []
    for workload in sorted(top_k):
        selected = top_k[workload]
        oracle = exhaustive[workload]
        top_k_wall = float(selected["tuning_wall_time_ms"])
        exhaustive_wall = float(oracle["tuning_wall_time_ms"])
        top_k_latency = float(selected["selected_median_latency_ms"])
        exhaustive_latency = float(oracle["selected_median_latency_ms"])
        latency_gap = top_k_latency - exhaustive_latency
        extra_exhaustive_tuning = exhaustive_wall - top_k_wall
        rows.append(
            {
                "workload": workload,
                "top_k_budget": int(selected["selection_budget"]),
                "exhaustive_budget": int(oracle["selection_budget"]),
                "top_k_selected_config_id": selected["selected_config_id"],
                "exhaustive_selected_config_id": oracle["selected_config_id"],
                "top_k_selected_latency_ms": top_k_latency,
                "exhaustive_selected_latency_ms": exhaustive_latency,
                "direct_oracle_relative_performance": exhaustive_latency / top_k_latency,
                "direct_regret_pct": (top_k_latency / exhaustive_latency - 1.0) * 100.0,
                "top_k_tuning_wall_time_ms": top_k_wall,
                "exhaustive_tuning_wall_time_ms": exhaustive_wall,
                "tuning_wall_time_saved_pct": (1.0 - top_k_wall / exhaustive_wall) * 100.0,
                "tuning_wall_time_speedup": exhaustive_wall / top_k_wall,
                "top_k_selector_inference_ms": float(selected["selector_inference_ms"]),
                "top_k_compile_or_cache_prepare_ms": float(
                    selected["compile_or_cache_prepare_total_ms"]
                ),
                "exhaustive_compile_or_cache_prepare_ms": float(
                    oracle["compile_or_cache_prepare_total_ms"]
                ),
                "top_k_benchmark_wall_time_ms": float(
                    selected["benchmark_wall_time_total_ms"]
                ),
                "exhaustive_benchmark_wall_time_ms": float(
                    oracle["benchmark_wall_time_total_ms"]
                ),
                "top_k_runtime_errors": int(selected["runtime_error_configs"]),
                "exhaustive_runtime_errors": int(oracle["runtime_error_configs"]),
                "top_k_correctness_failures": int(selected["correctness_failed_configs"]),
                "exhaustive_correctness_failures": int(
                    oracle["correctness_failed_configs"]
                ),
                "exhaustive_break_even_invocations": (
                    extra_exhaustive_tuning / latency_gap if latency_gap > 0.0 else None
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    top_k = load_summary(args.top_k_summary)
    exhaustive = load_summary(args.exhaustive_summary)
    top_k_metadata = json.loads(args.top_k_metadata.read_text())
    exhaustive_metadata = json.loads(args.exhaustive_metadata.read_text())
    validate_compatible_metadata(top_k_metadata, exhaustive_metadata)
    rows = build_comparison_rows(top_k, exhaustive)
    write_csv(args.output, rows)

    top_k_wall = sum(float(row["top_k_tuning_wall_time_ms"]) for row in rows)
    exhaustive_wall = sum(float(row["exhaustive_tuning_wall_time_ms"]) for row in rows)
    direct_performance = [
        float(row["direct_oracle_relative_performance"]) for row in rows
    ]
    metadata = {
        "input_files": [
            {"path": str(path), "normalized_text_sha256": normalized_text_sha256(path)}
            for path in (
                args.top_k_summary,
                args.top_k_metadata,
                args.exhaustive_summary,
                args.exhaustive_metadata,
            )
        ],
        "workload_count": len(rows),
        "top_k_budget": int(top_k_metadata["selection_budget"]),
        "exhaustive_budget": int(exhaustive_metadata["selection_budget"]),
        "top_k_total_tuning_wall_time_ms": top_k_wall,
        "exhaustive_total_tuning_wall_time_ms": exhaustive_wall,
        "total_tuning_wall_time_saved_pct": (1.0 - top_k_wall / exhaustive_wall) * 100.0,
        "total_tuning_wall_time_speedup": exhaustive_wall / top_k_wall,
        "direct_geomean_oracle_relative_performance": geometric_mean(direct_performance),
        "mean_direct_regret_pct": statistics.fmean(
            float(row["direct_regret_pct"]) for row in rows
        ),
        "worst_direct_regret_pct": max(float(row["direct_regret_pct"]) for row in rows),
        "top_k_total_selector_inference_ms": sum(
            float(row["top_k_selector_inference_ms"]) for row in rows
        ),
        "top_k_total_runtime_errors": sum(int(row["top_k_runtime_errors"]) for row in rows),
        "exhaustive_total_runtime_errors": sum(
            int(row["exhaustive_runtime_errors"]) for row in rows
        ),
        "total_correctness_failures": sum(
            int(row["top_k_correctness_failures"])
            + int(row["exhaustive_correctness_failures"])
            for row in rows
        ),
        "cache_note": top_k_metadata["cache_note"],
        "timing_scope": top_k_metadata["timing_scope"],
        "break_even_definition": (
            "Additional exhaustive tuning wall time divided by the direct latency "
            "advantage of the exhaustive-selected configuration."
        ),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Compared {len(rows)} workloads")
    print(f"Top-k tuning wall time: {top_k_wall:.3f} ms")
    print(f"Exhaustive tuning wall time: {exhaustive_wall:.3f} ms")
    print(
        "Tuning wall-time saving: "
        f"{metadata['total_tuning_wall_time_saved_pct']:.2f}% "
        f"({metadata['total_tuning_wall_time_speedup']:.2f}x)"
    )
    print(
        "Direct geometric-mean oracle-relative performance: "
        f"{metadata['direct_geomean_oracle_relative_performance'] * 100.0:.2f}%"
    )


if __name__ == "__main__":
    main()
