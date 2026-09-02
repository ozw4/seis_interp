"""Write model-independent run records for training pipelines."""

from __future__ import annotations

import json
import resource
import subprocess
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
    """Return process and CUDA resource usage for one training device."""
    # Local import keeps run_records importable without the optional ml dependency.
    import torch

    result: dict[str, object] = {
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cudnn_benchmark": device.type == "cuda" and torch.backends.cudnn.benchmark,
        "cudnn_deterministic": device.type == "cuda" and torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
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
