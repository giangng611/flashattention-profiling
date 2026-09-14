"""Report local Python, PyTorch, CUDA, and GPU information."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "pytorch": {
            "installed": False,
        },
        "nvidia_smi": {
            "available": shutil.which("nvidia-smi") is not None,
        },
    }

    if info["nvidia_smi"]["available"]:
        info["nvidia_smi"]["summary"] = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )

    try:
        import torch
    except ImportError as exc:
        info["pytorch"]["import_error"] = str(exc)
        return info

    info["pytorch"].update(
        {
            "installed": True,
            "version": torch.__version__,
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gb": round(
                        properties.total_memory / (1024**3), 2
                    ),
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
        info["pytorch"]["cuda_device_count"] = torch.cuda.device_count()
        info["pytorch"]["devices"] = devices

    return info


def main() -> None:
    info = collect_environment()
    print(json.dumps(info, indent=2))

    if not info["pytorch"]["installed"]:
        print("\nPyTorch is not installed in this Python environment.")
        print("You can still develop locally, but CUDA benchmarks cannot run yet.")
    elif not info["pytorch"].get("cuda_is_available"):
        print("\nPyTorch is installed, but CUDA is not available.")
        print("Run the benchmark later on a CUDA-capable NVIDIA GPU machine.")
    else:
        print("\nCUDA is available. This machine can run the benchmark.")


if __name__ == "__main__":
    main()
