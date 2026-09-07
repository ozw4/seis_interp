"""Collect and verify the files bound by a benchmark case."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np

from seis_interp.data.benchmark_case_store import validate_benchmark_case
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import OUTPUT_FILE_NAMES as MASK_FILE_NAMES
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
)
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.trace_store import METADATA_FILE_NAME, TRACES_FILE_NAME
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.processing.interpolation_masks import MASK_KINDS
from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

_PARTITIONS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


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
    validate_benchmark_case(case)
    current = collect_benchmark_input_hashes(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
    )
    if case["input_files"] != current:
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


def validate_benchmark_preparation(
    dataset_metadata: Mapping[str, object],
    preparation: Mapping[str, object],
    *,
    interim_hashes: Mapping[str, object],
) -> str:
    """Verify preparation metadata and its exact interim input hashes."""
    dataset_id = _metadata_text(dataset_metadata, "dataset_id", METADATA_FILE_NAME)
    prepared_dataset_id = _metadata_text(preparation, "dataset_id", PREPARATION_FILE_NAME)
    if prepared_dataset_id != dataset_id:
        raise ValueError(
            f"dataset_id mismatch: {METADATA_FILE_NAME} has {dataset_id!r}, "
            f"{PREPARATION_FILE_NAME} has {prepared_dataset_id!r}"
        )

    for key in ("trace_count", "sample_count"):
        expected = _metadata_nonnegative_integer(dataset_metadata, key, METADATA_FILE_NAME)
        actual = _metadata_nonnegative_integer(preparation, key, PREPARATION_FILE_NAME)
        if actual != expected:
            raise ValueError(
                f"{PREPARATION_FILE_NAME} {key} is {actual}, but {METADATA_FILE_NAME} has "
                f"{expected}"
            )

    if preparation.get("input_files") != interim_hashes:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} input_files do not match the current interim files"
        )
    expected_files = {
        "trace_split": TRACE_SPLIT_FILE_NAME,
        "normalization": NORMALIZATION_FILE_NAME,
    }
    if preparation.get("files") != expected_files:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} files must be {expected_files!r}, "
            f"got {preparation.get('files')!r}"
        )
    return dataset_id


def validated_benchmark_mask_summary(
    metadata: Mapping[str, object],
    *,
    mask_row_count: int,
    trace_count: int,
    mask_array_rows: np.ndarray,
    dataset_id: str,
    input_hashes: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Validate mask provenance and return the benchmark-case summary."""
    mask_dataset_id = _metadata_text(metadata, "dataset_id", "interpolation_mask.json")
    if mask_dataset_id != dataset_id:
        raise ValueError(
            "dataset_id mismatch: "
            f"interpolation_mask.json has {mask_dataset_id!r}, expected {dataset_id!r}"
        )

    partition = metadata.get("partition")
    if not isinstance(partition, str) or partition not in _PARTITIONS:
        raise ValueError(f"interpolation_mask.json partition must be one of {list(_PARTITIONS)}")
    kind = metadata.get("kind")
    if not isinstance(kind, str) or kind not in MASK_KINDS:
        raise ValueError(f"interpolation_mask.json kind must be one of {list(MASK_KINDS)}")

    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("interpolation_mask.json counts must be an object")
    candidate_trace_count = _metadata_positive_integer(
        metadata,
        "candidate_trace_count",
        "interpolation_mask.json",
    )
    if counts.get("total") != candidate_trace_count or candidate_trace_count != mask_row_count:
        raise ValueError(
            "interpolation_mask.json candidate_trace_count, counts.total, and mask row count "
            "must match"
        )
    candidate_ffid_count = _metadata_positive_integer(
        metadata,
        "candidate_ffid_count",
        "interpolation_mask.json",
    )

    expected_mask_inputs = {
        "interim": {
            file_name: input_hashes["interim"][file_name]
            for file_name in (TRACES_FILE_NAME, METADATA_FILE_NAME)
        },
        "processed": {
            file_name: input_hashes["processed"][file_name]
            for file_name in (TRACE_SPLIT_FILE_NAME, PREPARATION_FILE_NAME)
        },
    }
    if metadata.get("input_files") != expected_mask_inputs:
        raise ValueError(
            "interpolation_mask.json input_files do not match the current interim and "
            "prepared partition files"
        )

    if np.any(mask_array_rows < 0) or np.any(mask_array_rows >= trace_count):
        raise ValueError("observation_mask.parquet array_row values are outside the interim range")

    duplicate_summary = metadata.get("duplicate_physical_coordinates")
    if not isinstance(duplicate_summary, Mapping):
        raise ValueError("interpolation_mask.json duplicate_physical_coordinates must be an object")
    if "policy" not in duplicate_summary or "removed_trace_count" not in duplicate_summary:
        raise ValueError(
            "interpolation_mask.json duplicate_physical_coordinates must contain policy and "
            "removed_trace_count"
        )

    return {
        "kind": kind,
        "missing_fraction": metadata.get("missing_fraction"),
        "random_seed": metadata.get("random_seed"),
        "candidate_trace_count": candidate_trace_count,
        "candidate_ffid_count": candidate_ffid_count,
        "counts": dict(counts),
        "duplicate_physical_coordinates": {
            "policy": duplicate_summary["policy"],
            "removed_trace_count": duplicate_summary["removed_trace_count"],
        },
    }


def _metadata_text(metadata: Mapping[str, object], key: str, description: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{description} {key} must be a trimmed non-empty string")
    return value


def _metadata_nonnegative_integer(
    metadata: Mapping[str, object],
    key: str,
    description: str,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{description} {key} must be a nonnegative integer")
    return int(value)


def _metadata_positive_integer(
    metadata: Mapping[str, object],
    key: str,
    description: str,
) -> int:
    value = _metadata_nonnegative_integer(metadata, key, description)
    if value == 0:
        raise ValueError(f"{description} {key} must be a positive integer")
    return value
