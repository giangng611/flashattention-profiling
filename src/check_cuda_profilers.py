"""Check whether NVIDIA profiling tools are visible on the current machine."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


TOOL_NAMES = ["nsys", "ncu"]
COMMON_BIN_DIRS = [
    "/usr/local/cuda/bin",
    "/opt/nvidia/nsight-systems",
    "/opt/nvidia/nsight-compute",
    "/usr/local/NVIDIA-Nsight-Systems/bin",
    "/usr/local/NVIDIA-Nsight-Compute",
]


def run_version(path: str) -> str:
    commands = [[path, "--version"], [path, "-v"]]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as error:
            return f"{type(error).__name__}: {error}"

        output = (result.stdout + result.stderr).strip()
        if output:
            return output.splitlines()[0]
    return "version output unavailable"


def candidate_paths(tool_name: str) -> list[str]:
    paths: list[str] = []
    found = shutil.which(tool_name)
    if found is not None:
        paths.append(found)

    home = Path.home()
    search_roots = [home / "nsight-systems", home / "nsight-compute", home / ".local"]
    for root in search_roots:
        if root.exists():
            paths.extend(str(path) for path in root.rglob(tool_name))

    for directory in COMMON_BIN_DIRS:
        path = Path(directory) / tool_name
        if path.exists():
            paths.append(str(path))

    return sorted(set(paths))


def main() -> None:
    report: dict[str, object] = {
        "path": os.environ.get("PATH", ""),
        "tools": {},
        "hint": "If nsys or ncu exists but is not on PATH, add its bin directory to PATH before profiling.",
    }

    tools: dict[str, object] = {}
    for tool_name in TOOL_NAMES:
        paths = candidate_paths(tool_name)
        tools[tool_name] = {
            "available": bool(paths),
            "paths": paths,
            "version": run_version(paths[0]) if paths else None,
        }
    report["tools"] = tools

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
