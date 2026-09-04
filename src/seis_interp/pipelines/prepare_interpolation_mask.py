"""Prepare an interpolation mask for one existing dataset partition."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import write_interpolation_mask
from seis_interp.data.trace_store import METADATA_FILE_NAME, TRACES_FILE_NAME
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.pipelines.prepare_baseline import (
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
    WHOLE_FFID_SPLIT_SCOPE,
)
from seis_interp.processing.interpolation_masks import (
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
    make_random_trace_mask,
    make_random_whole_ffid_mask,
    validate_interpolation_mask,
)
from seis_interp.processing.trace_canonicalization import (
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

_MASK_PARTITIONS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
_STORED_SPLITS = (*_MASK_PARTITIONS, EXCLUDED_SPLIT)


def prepare_interpolation_mask(
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    partition: str,
    kind: str,
    missing_fraction: float,
    random_seed: int,
    config_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate and store visibility roles within one dataset partition."""
    stored_partition = _validated_partition(partition)
    stored_kind = _validated_kind(kind)
    stored_config_source = _validated_config_source(config_source)

    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    input_paths = {
        "interim": {
            TRACES_FILE_NAME: interim_directory / TRACES_FILE_NAME,
            METADATA_FILE_NAME: interim_directory / METADATA_FILE_NAME,
        },
        "processed": {
            TRACE_SPLIT_FILE_NAME: processed_directory / TRACE_SPLIT_FILE_NAME,
            PREPARATION_FILE_NAME: processed_directory / PREPARATION_FILE_NAME,
        },
    }
    _require_input_files(input_paths)

    dataset_metadata = _read_json_object(
        input_paths["interim"][METADATA_FILE_NAME],
        description=METADATA_FILE_NAME,
    )
    preparation = _read_json_object(
        input_paths["processed"][PREPARATION_FILE_NAME],
        description=PREPARATION_FILE_NAME,
    )
    trace_table = pd.read_parquet(input_paths["interim"][TRACES_FILE_NAME])
    split_table = pd.read_parquet(input_paths["processed"][TRACE_SPLIT_FILE_NAME])

    source_rows = validated_array_rows(trace_table, require_contiguous=True)
    split_rows = _validated_split_rows(split_table, expected_array_rows=source_rows)
    _validate_input_metadata(
        dataset_metadata,
        preparation,
        trace_count=len(trace_table),
        traces_path=input_paths["interim"][TRACES_FILE_NAME],
    )

    split_by_array_row = dict(zip(split_rows, split_table[SPLIT_COLUMN].to_numpy(), strict=True))
    trace_partitions = trace_table["array_row"].map(split_by_array_row)
    joined_table = trace_table.assign(**{SPLIT_COLUMN: trace_partitions})
    canonical_table, duplicate_audit = canonicalize_eligible_physical_coordinates(joined_table)
    candidate_table = canonical_table.loc[canonical_table[SPLIT_COLUMN].eq(stored_partition)].copy()
    if candidate_table.empty:
        raise ValueError(f"dataset partition {stored_partition!r} is empty")
    candidate_ffid_count = _candidate_ffid_count(candidate_table)

    if stored_kind == RANDOM_TRACE_MASK_KIND:
        mask_table = make_random_trace_mask(
            candidate_table,
            missing_fraction=missing_fraction,
            random_seed=random_seed,
        )
    else:
        _validate_whole_ffid_partition(
            trace_table,
            trace_partitions,
            candidate_table,
            preparation,
            partition=stored_partition,
        )
        mask_table = make_random_whole_ffid_mask(
            candidate_table,
            missing_fraction=missing_fraction,
            random_seed=random_seed,
        )

    candidate_rows = candidate_table["array_row"].to_numpy(dtype=np.int64)
    validate_interpolation_mask(mask_table, expected_array_rows=candidate_rows)

    dataset_id = _metadata_text(dataset_metadata, "dataset_id", METADATA_FILE_NAME)
    metadata: dict[str, object] = {
        "dataset_id": dataset_id,
        "partition": stored_partition,
        "kind": stored_kind,
        "missing_fraction": float(missing_fraction),
        "random_seed": int(random_seed),
        "config_source": stored_config_source,
        "candidate_trace_count": int(len(candidate_table)),
        "candidate_ffid_count": candidate_ffid_count,
        "duplicate_physical_coordinates": {
            "policy": duplicate_audit["policy"],
            "removed_trace_count": duplicate_audit["removed_trace_count"],
        },
        "input_files": {
            group: {file_name: {"sha256": file_sha256(path)} for file_name, path in paths.items()}
            for group, paths in input_paths.items()
        },
    }
    return write_interpolation_mask(
        Path(output_dir),
        mask_table,
        metadata,
        overwrite=overwrite,
    )


def _validated_partition(value: str) -> str:
    if not isinstance(value, str) or value not in _MASK_PARTITIONS:
        raise ValueError(f"partition must be one of {list(_MASK_PARTITIONS)}, got {value!r}")
    return value


def _validated_kind(value: str) -> str:
    if value not in (RANDOM_TRACE_MASK_KIND, RANDOM_WHOLE_FFID_MASK_KIND):
        raise ValueError(
            "kind must be one of "
            f"{[RANDOM_TRACE_MASK_KIND, RANDOM_WHOLE_FFID_MASK_KIND]}, got {value!r}"
        )
    return value


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


def _require_input_files(input_paths: Mapping[str, Mapping[str, Path]]) -> None:
    missing = [
        f"{group}/{file_name}"
        for group, paths in input_paths.items()
        for file_name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"interpolation mask inputs are missing required files: {missing}")


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} does not contain valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _validated_split_rows(
    split_table: pd.DataFrame,
    *,
    expected_array_rows: np.ndarray,
) -> np.ndarray:
    if SPLIT_COLUMN not in split_table.columns:
        raise ValueError(f"{TRACE_SPLIT_FILE_NAME} is missing required column: {SPLIT_COLUMN}")
    split_rows = validated_array_rows(split_table)
    if not np.array_equal(np.sort(split_rows), np.sort(expected_array_rows)):
        raise ValueError(
            f"{TRACE_SPLIT_FILE_NAME} array_row values do not exactly match {TRACES_FILE_NAME}"
        )

    known = split_table[SPLIT_COLUMN].isin(_STORED_SPLITS)
    if not bool(known.all()):
        unknown = sorted({repr(value) for value in split_table.loc[~known, SPLIT_COLUMN].tolist()})
        raise ValueError(f"{TRACE_SPLIT_FILE_NAME} contains unknown split values: {unknown}")
    return split_rows


def _validate_input_metadata(
    dataset_metadata: Mapping[str, object],
    preparation: Mapping[str, object],
    *,
    trace_count: int,
    traces_path: Path,
) -> None:
    dataset_id = _metadata_text(dataset_metadata, "dataset_id", METADATA_FILE_NAME)
    prepared_dataset_id = _metadata_text(preparation, "dataset_id", PREPARATION_FILE_NAME)
    if prepared_dataset_id != dataset_id:
        raise ValueError(
            f"dataset_id mismatch: {METADATA_FILE_NAME} has {dataset_id!r}, "
            f"{PREPARATION_FILE_NAME} has {prepared_dataset_id!r}"
        )

    prepared_trace_count = preparation.get("trace_count")
    if (
        isinstance(prepared_trace_count, bool)
        or not isinstance(prepared_trace_count, Integral)
        or int(prepared_trace_count) != trace_count
    ):
        raise ValueError(
            f"{PREPARATION_FILE_NAME} trace_count is {prepared_trace_count!r}, "
            f"but {TRACES_FILE_NAME} contains {trace_count} rows"
        )

    expected_hash = _recorded_trace_hash(preparation)
    if expected_hash is not None:
        actual_hash = file_sha256(traces_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"{PREPARATION_FILE_NAME} input hash for {TRACES_FILE_NAME} does not match "
                "the current file"
            )


def _metadata_text(metadata: Mapping[str, object], key: str, description: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} {key} must be a non-empty string")
    return value


def _recorded_trace_hash(preparation: Mapping[str, object]) -> str | None:
    if "input_files" not in preparation:
        return None
    input_files = preparation["input_files"]
    if not isinstance(input_files, Mapping):
        raise ValueError(f"{PREPARATION_FILE_NAME} input_files must be an object")
    if TRACES_FILE_NAME not in input_files:
        return None
    trace_record = input_files[TRACES_FILE_NAME]
    if not isinstance(trace_record, Mapping):
        raise ValueError(
            f"{PREPARATION_FILE_NAME} input_files.{TRACES_FILE_NAME} must be an object"
        )
    if "sha256" not in trace_record:
        return None
    value = trace_record["sha256"]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} input_files.{TRACES_FILE_NAME}.sha256 "
            "must be a non-empty string"
        )
    return value


def _candidate_ffid_count(candidate_table: pd.DataFrame) -> int:
    if "ffid" not in candidate_table.columns:
        raise ValueError(f"{TRACES_FILE_NAME} is missing required column: ffid")
    if candidate_table["ffid"].isna().any():
        raise ValueError(f"{TRACES_FILE_NAME} ffid contains missing values")
    return int(candidate_table["ffid"].nunique())


def _validate_whole_ffid_partition(
    trace_table: pd.DataFrame,
    trace_partitions: pd.Series,
    candidate_table: pd.DataFrame,
    preparation: Mapping[str, object],
    *,
    partition: str,
) -> None:
    if preparation.get("split_scope") != WHOLE_FFID_SPLIT_SCOPE:
        raise ValueError(
            f"{RANDOM_WHOLE_FFID_MASK_KIND} requires "
            f"{PREPARATION_FILE_NAME} split_scope={WHOLE_FFID_SPLIT_SCOPE!r}"
        )

    candidate_ffids = candidate_table["ffid"].drop_duplicates()
    memberships = pd.DataFrame(
        {
            "ffid": trace_table["ffid"].to_numpy(),
            SPLIT_COLUMN: trace_partitions.to_numpy(),
        }
    )
    relevant = memberships.loc[
        memberships["ffid"].isin(candidate_ffids) & memberships[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)
    ]
    split_counts = relevant.groupby("ffid", dropna=False)[SPLIT_COLUMN].nunique()
    crossing_ffids = split_counts.index[split_counts.gt(1)].tolist()
    if crossing_ffids:
        raise ValueError(
            f"{RANDOM_WHOLE_FFID_MASK_KIND} requires complete FFIDs in partition "
            f"{partition!r}; FFIDs also occur in another non-excluded partition: "
            f"{crossing_ffids}"
        )
