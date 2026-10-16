"""Validate expanded workload and configuration CSVs without requiring CUDA."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        type=Path,
        default=Path("workloads/expanded_dense_prefill_workloads.csv"),
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs/attention_config_space.csv"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_workloads(rows: list[dict[str, str]]) -> None:
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Workload names must be unique")

    supported = [row for row in rows if row["expected_status"] == "supported"]
    unsupported = [row for row in rows if row["expected_status"] == "unsupported"]
    if not 12 <= len(supported) <= 20:
        raise ValueError(f"Expected 12-20 supported workloads, found {len(supported)}")
    if not unsupported:
        raise ValueError("Expected at least one explicit unsupported reference workload")

    for row in supported:
        if row["phase"] != "prefill":
            raise ValueError(f"Supported workload must be prefill: {row['name']}")
        if row["query_length"] != row["kv_length"]:
            raise ValueError(f"Supported workload must be self-attention: {row['name']}")
        if row["query_heads"] != row["kv_heads"]:
            raise ValueError(f"Supported workload must be dense attention: {row['name']}")
        if row["split"] not in {"development", "held_out"}:
            raise ValueError(f"Supported workload has invalid split: {row['name']}")

    if {row["split"] for row in supported} != {"development", "held_out"}:
        raise ValueError("Supported workloads must include development and held-out splits")
    if {row["dtype"] for row in supported} != {"float16", "bfloat16"}:
        raise ValueError("Supported workloads must include FP16 and BF16")
    if {row["causal"] for row in supported} != {"true", "false"}:
        raise ValueError("Supported workloads must include causal and non-causal cases")
    if len({row["query_length"] for row in supported}) < 4:
        raise ValueError("Sequence-length coverage is too narrow")
    if len({row["head_dim"] for row in supported}) < 2:
        raise ValueError("Head-dimension coverage is too narrow")
    if len({row["query_heads"] for row in supported}) < 3:
        raise ValueError("Head-count coverage is too narrow")


def validate_configs(rows: list[dict[str, str]]) -> None:
    configs = [
        (
            int(row["block_m"]),
            int(row["block_n"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        for row in rows
    ]
    if not 32 <= len(configs) <= 64:
        raise ValueError(f"Expected 32-64 candidate configurations, found {len(configs)}")
    if len(configs) != len(set(configs)):
        raise ValueError("Configurations must be unique")
    for block_m, block_n, num_warps, num_stages in configs:
        if not is_power_of_two(block_m) or not is_power_of_two(block_n):
            raise ValueError(f"Block sizes must be powers of two: {(block_m, block_n)}")
        if num_warps not in {1, 2, 4, 8}:
            raise ValueError(f"Unsupported num_warps: {num_warps}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive: {num_stages}")
        expected_id = f"bm{block_m}_bn{block_n}_w{num_warps}_s{num_stages}"
        matching_rows = [
            row
            for row in rows
            if (
                int(row["block_m"]),
                int(row["block_n"]),
                int(row["num_warps"]),
                int(row["num_stages"]),
            )
            == (block_m, block_n, num_warps, num_stages)
        ]
        if matching_rows[0]["config_id"] != expected_id:
            raise ValueError(
                f"Configuration ID {matching_rows[0]['config_id']} does not match {expected_id}"
            )


def main() -> None:
    args = parse_args()
    workload_rows = read_csv(args.workloads)
    config_rows = read_csv(args.configs)
    validate_workloads(workload_rows)
    validate_configs(config_rows)
    supported = sum(row["expected_status"] == "supported" for row in workload_rows)
    print(
        f"Validated {supported} supported workloads, "
        f"{len(workload_rows) - supported} unsupported references, "
        f"and {len(config_rows)} candidate configurations."
    )


if __name__ == "__main__":
    main()
