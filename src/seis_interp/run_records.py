"""Write model-independent execution records for run pipelines."""

from __future__ import annotations

import json
import resource
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import yaml

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.data.file_checksums import file_sha256

CONFIG_FILE_NAME = "config.resolved.yaml"
INPUTS_LOCK_FILE_NAME = "inputs.lock.json"
METRICS_FILE_NAME = "metrics.json"
RUN_FILE_NAME = "run.json"
CHECKPOINT_RELATIVE_PATH = Path("artifacts") / "best.pt"


def check_new_output_directory(directory: Path) -> None:
    """Reject an existing run output path without creating a missing one."""
    if directory.exists():
        raise FileExistsError(f"run output path already exists: {directory}")


def current_git_commit() -> str:
    """Return the repository HEAD commit recorded in run metadata."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not determine the current Git commit") from error
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    return commit


def current_git_metadata() -> dict[str, str | bool]:
    """Capture HEAD and staged, unstaged, or untracked changes at run start."""
    commit = current_git_commit()
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not determine the current Git worktree status") from error
    return {"git_commit": commit, "git_worktree_dirty": bool(completed.stdout.strip())}


def utc_timestamp() -> str:
    """Return the current UTC time with second precision and a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_hashes(
    directory: Path,
    file_names: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    """Return the SHA-256 digest of each named file in one directory."""
    return {file_name: {"sha256": file_sha256(directory / file_name)} for file_name in file_names}


def runtime_resource_metadata(device: object) -> dict[str, object]:
    """Return resource usage and, on CUDA, device and numerical-mode metadata."""
    # Local import keeps run_records importable without the optional ml dependency.
    import torch

    result: dict[str, object] = {
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cudnn_benchmark": device.type == "cuda" and torch.backends.cudnn.benchmark,
        "cudnn_deterministic": device.type == "cuda" and torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cuda_device_name": properties.name,
                "cuda_device_capability": [properties.major, properties.minor],
                "cuda_total_memory_bytes": properties.total_memory,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
    return result


def write_run_outputs(
    output_directory: Path,
    config: Mapping[str, object],
    inputs_lock: Mapping[str, object],
    metrics: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> None:
    """Write the resolved config, inputs lock, metrics, and run metadata files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / CONFIG_FILE_NAME).write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    (output_directory / INPUTS_LOCK_FILE_NAME).write_text(
        json.dumps(inputs_lock, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / METRICS_FILE_NAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / RUN_FILE_NAME).write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_run_progress(
    output_directory: Path,
    metrics: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> None:
    """Replace progress records individually, leaving the starting inputs intact.

    Metrics are committed before run metadata. Each file remains readable if a
    write is interrupted; the two replacements are not a single transaction.
    """
    records = {
        name: json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for name, payload in ((METRICS_FILE_NAME, metrics), (RUN_FILE_NAME, run_metadata))
    }
    for name, text in records.items():
        with tempfile.NamedTemporaryFile(
            dir=output_directory, prefix=f".{name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(output_directory / name)
        finally:
            temporary.unlink(missing_ok=True)
