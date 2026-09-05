"""Verify the benchmark case bound by a C3 volume artifact."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    load_benchmark_case,
)
from seis_interp.data.c3_volume_index_store import validate_c3_volume_metadata
from seis_interp.data.file_checksums import file_sha256


def load_bound_benchmark_case(
    volume_metadata: Mapping[str, object],
    *,
    case_dir: Path,
) -> dict[str, object]:
    """Load and require the exact benchmark case referenced by a volume."""
    validate_c3_volume_metadata(volume_metadata)
    case_path = Path(case_dir) / BENCHMARK_CASE_FILE_NAME
    case = load_benchmark_case(case_dir)
    binding = volume_metadata["benchmark_case"]
    if file_sha256(case_path) != binding["sha256"]:  # type: ignore[index]
        raise ValueError("benchmark case SHA-256 does not match the volume binding")
    if case["case_id"] != binding["case_id"]:  # type: ignore[index]
        raise ValueError("benchmark case ID does not match the volume binding")
    if case["dataset_id"] != volume_metadata["dataset_id"]:
        raise ValueError("benchmark case dataset ID does not match the volume")
    if case["partition"] != volume_metadata["partition"]:
        raise ValueError("benchmark case partition does not match the volume")
    return case
