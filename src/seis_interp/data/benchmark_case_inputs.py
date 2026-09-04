"""Collect and verify the files bound by a benchmark case."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import OUTPUT_FILE_NAMES as MASK_FILE_NAMES
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES


def collect_benchmark_input_hashes(
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return SHA-256 records for all files bound by a benchmark case."""
    input_paths = _benchmark_input_paths(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
    )
    missing = [
        f"{group}/{file_name}"
        for group, paths in input_paths.items()
        for file_name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"benchmark case inputs are missing required files: {missing}")

    return {
        group: {file_name: {"sha256": file_sha256(path)} for file_name, path in paths.items()}
        for group, paths in input_paths.items()
    }


def verify_benchmark_case_inputs(
    case: Mapping[str, object],
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
) -> None:
    """Require current input hashes to exactly match one benchmark case."""
    if not isinstance(case, Mapping):
        raise ValueError("benchmark case must be an object")
    current = collect_benchmark_input_hashes(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
    )
    if case.get("input_files") != current:
        raise ValueError("benchmark case input_files do not match the current input files")


def _benchmark_input_paths(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
) -> dict[str, dict[str, Path]]:
    directories = {
        "interim": Path(interim_dir),
        "processed": Path(processed_dir),
        "mask": Path(mask_dir),
    }
    file_names = {
        "interim": INTERIM_FILE_NAMES,
        "processed": PREPARED_FILE_NAMES,
        "mask": MASK_FILE_NAMES,
    }
    return {
        group: {file_name: directory / file_name for file_name in file_names[group]}
        for group, directory in directories.items()
    }
