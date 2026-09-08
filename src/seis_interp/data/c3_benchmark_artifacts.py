"""Small immutable JSON, YAML, and metadata inputs for C3 preparation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.file_checksums import file_sha256


def write_benchmark_json(path: Path, payload: object) -> None:
    """Create a strict JSON record, refusing to replace any existing file."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def read_benchmark_json(path: Path) -> dict[str, object]:
    """Read a JSON object, rejecting nonstandard numeric constants."""

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    result = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return result


def write_benchmark_yaml(path: Path, payload: object) -> None:
    """Create a resolved configuration without overwriting earlier preparation."""
    with Path(path).open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def benchmark_relative_path(path: Path, base: Path) -> str:
    """Use the manifest directory as the one explicit path resolution base."""
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def benchmark_file_record(path: Path, base: Path) -> dict[str, object]:
    """Hash the current file contents and record a portable relative reference."""
    return {"path": benchmark_relative_path(path, base), "sha256": file_sha256(path)}


def load_c3_geometry_inputs(interim_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read table/time only; geometry QC never opens the amplitude array."""
    table = pd.read_parquet(Path(interim_dir) / "traces.parquet")
    times = np.load(Path(interim_dir) / "time_s.npy", allow_pickle=False)
    return table, times
