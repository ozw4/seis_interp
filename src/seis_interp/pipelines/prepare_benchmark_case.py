"""Bind one prepared partition and interpolation mask into a benchmark case."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from seis_interp.data.benchmark_case_inputs import collect_benchmark_input_hashes
from seis_interp.data.benchmark_case_store import (
    EVALUATION_TARGET_AMPLITUDE_USE,
    MASK_DOMAIN,
    validated_case_id,
    write_benchmark_case,
)
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
)
from seis_interp.data.trace_store import (
    METADATA_FILE_NAME,
    TRACES_FILE_NAME,
)
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    MASK_KINDS,
    OBSERVED_ROLE,
)
from seis_interp.processing.normalization import read_normalization_parameters
from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

_PARTITIONS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def prepare_benchmark_case(
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    case_id: str,
    config_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate existing artifacts and bind them by exact hash."""
    stored_case_id = validated_case_id(case_id)
    stored_config_source = _validated_config_source(config_source)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")

    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    mask_directory = Path(mask_dir)
    input_hashes = collect_benchmark_input_hashes(
        interim_directory,
        processed_directory,
        mask_directory,
    )

    dataset = load_interim_trace_dataset(
        interim_directory,
        memory_map_amplitudes=True,
        amplitude_validation_rows=np.empty(0, dtype=np.int64),
    )
    preparation = _read_json_object(
        processed_directory / PREPARATION_FILE_NAME,
        description=PREPARATION_FILE_NAME,
    )
    read_normalization_parameters(processed_directory / NORMALIZATION_FILE_NAME)
    dataset_id = _validate_preparation(
        dataset.metadata,
        preparation,
        interim_hashes=input_hashes["interim"],
    )

    mask_table, mask_metadata = load_interpolation_mask(mask_directory)
    mask_summary = _validated_mask_summary(
        mask_metadata,
        mask_row_count=len(mask_table),
        trace_count=len(dataset.trace_table),
        mask_array_rows=mask_table["array_row"].to_numpy(dtype=np.int64),
        dataset_id=dataset_id,
        input_hashes=input_hashes,
    )

    case: dict[str, object] = {
        "case_id": stored_case_id,
        "dataset_id": dataset_id,
        "partition": mask_metadata["partition"],
        "config_source": stored_config_source,
        "role_contract": {
            "domain": MASK_DOMAIN,
            "observed_role": OBSERVED_ROLE,
            "evaluation_target_role": EVALUATION_TARGET_ROLE,
            "evaluation_target_amplitude_use": EVALUATION_TARGET_AMPLITUDE_USE,
        },
        "mask": mask_summary,
        "input_files": input_hashes,
    }
    return write_benchmark_case(
        Path(output_dir),
        case,
        overwrite=overwrite,
    )


def _validated_config_source(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("config_source must be a non-empty repository-relative path")
    if "\\" in value:
        raise ValueError("config_source must use POSIX path separators")

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("config_source must not be an absolute path")
    if not posix_path.parts:
        raise ValueError("config_source must identify a configuration file")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("config_source must not escape the repository")
    return posix_path.as_posix()


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} does not contain valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _validate_preparation(
    dataset_metadata: Mapping[str, object],
    preparation: Mapping[str, object],
    *,
    interim_hashes: Mapping[str, object],
) -> str:
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


def _validated_mask_summary(
    metadata: Mapping[str, object],
    *,
    mask_row_count: int,
    trace_count: int,
    mask_array_rows: np.ndarray,
    dataset_id: str,
    input_hashes: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
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
