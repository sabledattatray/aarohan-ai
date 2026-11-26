"""Report the local Python/PyTorch execution environment without requiring CUDA."""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_bytes": None,
        "ram_bytes": None,
        "ram_source": "psutil if installed; otherwise not queried",
    }

    try:
        import torch
    except ImportError:
        report["torch_import_error"] = "PyTorch is not installed in this Python environment."
    else:
        report["torch_version"] = torch.__version__
        report["cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            report["gpu_name"] = properties.name
            report["gpu_memory_bytes"] = properties.total_memory

    try:
        import psutil
    except ImportError:
        pass
    else:
        report["ram_bytes"] = psutil.virtual_memory().total
    return report


def main() -> None:
    print(json.dumps(environment_report(), indent=2))


if __name__ == "__main__":
    main()

