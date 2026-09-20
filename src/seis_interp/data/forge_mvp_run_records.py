"""MVP execution identity and process resource records."""

import importlib.metadata
import platform
import resource
import subprocess
from datetime import datetime, timezone

import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def execution_identity(repo):
    return {
        "git_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_worktree": bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        ),
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "gpus": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
        },
        "software_versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "pyarrow", "torch", "segyio", "PyYAML")
            },
        },
    }


def peak_memory(device):
    gpu = torch.device(device)
    return {
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(gpu))
        if gpu.type == "cuda" and torch.cuda.is_available()
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(gpu))
        if gpu.type == "cuda" and torch.cuda.is_available()
        else 0,
    }
