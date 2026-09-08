"""Capture compute resources and numerical modes for reproducible pilot runs."""

from __future__ import annotations

import contextlib
import io
import os
import platform
import resource
import shutil
from pathlib import Path

import numpy as np


def collect_runtime_environment(directory: Path, *, device: str) -> dict:
    """Record available resources without selecting a fallback device."""
    memory = {}
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk_free_bytes": shutil.disk_usage(directory).free,
        "numpy_version": np.__version__,
        "numpy_build_config": blas.getvalue(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "requested_device": device,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    try:
        import torch
    except ImportError:
        result["torch"] = {"available": False, "reason": "torch is not installed"}
        return result
    available = torch.cuda.is_available()
    devices = []
    for index in range(torch.cuda.device_count() if available else 0):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": total,
                "free_memory_bytes": free,
            }
        )
    result["torch"] = {
        "available": True,
        "version": torch.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "devices": devices,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return result
