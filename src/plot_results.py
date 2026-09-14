"""Create plots from the attention benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/attention_benchmark.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def save_latency_plot(data: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in data.groupby("method"):
        group = group.sort_values("seq_len")
        ax.plot(group["seq_len"], group["median_latency_ms"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Attention latency vs sequence length")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_vs_sequence_length.png", dpi=200)
    plt.close(fig)


def save_memory_plot(data: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = data.copy()
    data["peak_memory_mb"] = data["memory_allocated_bytes"] / (1024**2)
    for method, group in data.groupby("method"):
        group = group.sort_values("seq_len")
        ax.plot(group["seq_len"], group["peak_memory_mb"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Recorded allocated memory (MiB)")
    ax.set_title("Recorded memory vs sequence length")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "peak_memory_vs_sequence_length.png", dpi=200)
    plt.close(fig)


def save_speedup_plot(data: pd.DataFrame, output_dir: Path) -> None:
    compared = data[data["method"] != "standard"].sort_values("seq_len")
    compared_label = compared["method"].iloc[0] if not compared.empty else "optimized"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(compared["seq_len"], compared["speedup_over_standard"], marker="o", color="tab:green")
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup over standard attention")
    ax.set_title(f"{compared_label.upper()} speedup vs sequence length")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_vs_sequence_length.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)

    save_latency_plot(data, args.output_dir)
    save_memory_plot(data, args.output_dir)
    save_speedup_plot(data, args.output_dir)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
