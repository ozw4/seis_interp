"""Prepare trace splits and normalization metadata for baseline evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    OUTPUT_FILE_NAMES,
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
)
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.data.trace_store import canonical_source_files
from seis_interp.processing.normalization import (
    fit_normalization_parameters,
    write_normalization_parameters,
)
from seis_interp.processing.trace_amplitude_filter import (
    TraceAmplitudeFilterConfig,
    TraceAmplitudeFilterResult,
    filter_trace_amplitudes,
)
from seis_interp.processing.trace_splits import (
    C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
    EXCLUDED_SPLIT,
    GLOBAL_SPLIT_SCOPE,
    PER_FFID_SPLIT_SCOPE,
    SPLIT_COLUMN,
    SPLIT_SCOPES,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    WHOLE_FFID_ASSIGNMENT_SPLIT_SCOPES,
    WHOLE_FFID_SPLIT_SCOPE,
    assign_c3_source_line_block_splits,
    assign_random_trace_splits,
    assign_random_trace_splits_by_ffid,
    assign_random_whole_ffid_splits,
    validated_c3_source_line_ranges,
    validated_random_seed,
)

COORDINATE_NORMALIZATION_METHOD = "train_minmax_linear_plus_azimuth_sin_cos"
AMPLITUDE_NORMALIZATION_METHOD = "train_global_rms"


def prepare_baseline_dataset(
    interim_dir: Path,
    output_dir: Path,
    *,
    holdout_fraction: float | None,
    validation_fraction_of_holdout: float | None,
    random_seed: int,
    coordinate_normalization: str = COORDINATE_NORMALIZATION_METHOD,
    amplitude_normalization: str = AMPLITUDE_NORMALIZATION_METHOD,
    split_scope: str = GLOBAL_SPLIT_SCOPE,
    source_line_ranges: Mapping[str, object] | None = None,
    trace_amplitude_filter: TraceAmplitudeFilterConfig | None = None,
    config_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create split assignments and train-only normalization parameters."""
    input_directory = Path(interim_dir)
    output_directory = Path(output_dir)
    stored_coordinate_normalization = _validated_normalization_method(
        "coordinate_normalization",
        coordinate_normalization,
        COORDINATE_NORMALIZATION_METHOD,
    )
    stored_amplitude_normalization = _validated_normalization_method(
        "amplitude_normalization",
        amplitude_normalization,
        AMPLITUDE_NORMALIZATION_METHOD,
    )
    stored_config_source = _validated_config_source(config_source)
    stored_split_scope = _validated_split_scope(split_scope)
    stored_random_seed = validated_random_seed(random_seed)
    stored_trace_filter = _validated_trace_amplitude_filter(trace_amplitude_filter)

    dataset = load_interim_trace_dataset(input_directory, memory_map_amplitudes=True)
    trace_filter_result = (
        filter_trace_amplitudes(dataset.amplitudes, stored_trace_filter)
        if stored_trace_filter is not None
        else None
    )
    stored_source_line_ranges: dict[str, tuple[int, int]] | None = None
    if stored_split_scope == C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE:
        if holdout_fraction is not None or validation_fraction_of_holdout is not None:
            raise ValueError(
                "holdout fractions are not used with split_scope "
                f"{C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE!r}"
            )
        if source_line_ranges is None:
            raise ValueError(
                "source_line_ranges is required for split_scope "
                f"{C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE!r}"
            )
        stored_source_line_ranges = validated_c3_source_line_ranges(source_line_ranges)
        full_split_table = assign_c3_source_line_block_splits(
            dataset.trace_table,
            source_line_ranges=stored_source_line_ranges,
        )
        eligible_split_table = _eligible_trace_table(full_split_table, trace_filter_result)
    else:
        eligible_trace_table = _eligible_trace_table(dataset.trace_table, trace_filter_result)
    if stored_split_scope == GLOBAL_SPLIT_SCOPE:
        eligible_split_table = assign_random_trace_splits(
            eligible_trace_table,
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction_of_holdout,
            random_seed=stored_random_seed,
        )
    elif stored_split_scope == PER_FFID_SPLIT_SCOPE:
        eligible_split_table = assign_random_trace_splits_by_ffid(
            eligible_trace_table,
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction_of_holdout,
            random_seed=stored_random_seed,
        )
    elif stored_split_scope == WHOLE_FFID_SPLIT_SCOPE:
        eligible_split_table = assign_random_whole_ffid_splits(
            eligible_trace_table,
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction_of_holdout,
            random_seed=stored_random_seed,
        )
    normalization = fit_normalization_parameters(
        eligible_split_table,
        dataset.amplitudes,
        dataset.time_s,
        amplitudes_are_finite=True,
    )

    stored_splits = _stored_split_table(dataset.trace_table, eligible_split_table)
    split_counts = {
        split: int((stored_splits[SPLIT_COLUMN] == split).sum())
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    source_provenance = _source_provenance(dataset.metadata)
    preparation: dict[str, object] = {
        "dataset_id": _metadata_text(dataset.metadata, "dataset_id"),
        **source_provenance,
        "input_files": {
            file_name: {"sha256": file_sha256(input_directory / file_name)}
            for file_name in INTERIM_FILE_NAMES
        },
        "trace_count": int(dataset.metadata["trace_count"]),
        "sample_count": int(dataset.metadata["sample_count"]),
        "config_source": stored_config_source,
        "normalization": {
            "coordinates": stored_coordinate_normalization,
            "amplitude": stored_amplitude_normalization,
        },
        "random_seed": stored_random_seed,
        "split_scope": stored_split_scope,
        "ffid_count": int(eligible_split_table["ffid"].nunique()),
        "split_counts": split_counts,
        "files": {
            "trace_split": TRACE_SPLIT_FILE_NAME,
            "normalization": NORMALIZATION_FILE_NAME,
        },
    }
    if stored_split_scope == C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE:
        assert stored_source_line_ranges is not None
        preparation["source_line_ranges"] = {
            split: list(stored_source_line_ranges[split])
            for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
        }
    else:
        assert holdout_fraction is not None
        assert validation_fraction_of_holdout is not None
        preparation["holdout_fraction"] = float(holdout_fraction)
        preparation["validation_fraction_of_holdout"] = float(validation_fraction_of_holdout)
    if stored_split_scope in WHOLE_FFID_ASSIGNMENT_SPLIT_SCOPES:
        preparation["ffid_split_counts"] = {
            split: int(
                eligible_split_table.loc[
                    eligible_split_table[SPLIT_COLUMN].eq(split), "ffid"
                ].nunique()
            )
            for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
        }
    if stored_trace_filter is not None:
        assert trace_filter_result is not None
        preparation["trace_amplitude_filter"] = stored_trace_filter.to_dict()
        preparation["trace_quality"] = _trace_quality_metadata(
            dataset.trace_table,
            trace_filter_result,
        )
    preparation_json = json.dumps(preparation, indent=2, sort_keys=True) + "\n"

    _check_output_directory(output_directory, overwrite=overwrite)
    output_directory.mkdir(parents=True, exist_ok=True)
    stored_splits.to_parquet(output_directory / TRACE_SPLIT_FILE_NAME, index=False)
    write_normalization_parameters(
        output_directory / NORMALIZATION_FILE_NAME,
        normalization,
    )
    (output_directory / PREPARATION_FILE_NAME).write_text(
        preparation_json,
        encoding="utf-8",
    )
    return preparation


def _validated_trace_amplitude_filter(
    value: TraceAmplitudeFilterConfig | None,
) -> TraceAmplitudeFilterConfig | None:
    if value is not None and not isinstance(value, TraceAmplitudeFilterConfig):
        raise TypeError("trace_amplitude_filter must be a TraceAmplitudeFilterConfig or None")
    return value


def _eligible_trace_table(
    trace_table: pd.DataFrame,
    result: TraceAmplitudeFilterResult | None,
) -> pd.DataFrame:
    """Return the rows eligible for split assignment and normalization fitting."""
    if result is None:
        return trace_table
    eligible_by_array_row = np.zeros(len(trace_table), dtype=bool)
    eligible_by_array_row[result.eligible_array_rows] = True
    source_rows = trace_table["array_row"].to_numpy(dtype=np.int64)
    eligible = trace_table.loc[eligible_by_array_row[source_rows]].copy()
    if eligible.empty:
        raise ValueError("trace amplitude filter excluded every trace")
    return eligible


def _stored_split_table(
    trace_table: pd.DataFrame,
    eligible_split_table: pd.DataFrame,
) -> pd.DataFrame:
    """Keep every source row and label amplitude-QC failures as excluded."""
    stored = trace_table[["array_row"]].copy()
    split_by_array_row = np.full(len(stored), EXCLUDED_SPLIT, dtype=object)
    eligible_rows = eligible_split_table["array_row"].to_numpy(dtype=np.int64)
    split_by_array_row[eligible_rows] = eligible_split_table[SPLIT_COLUMN].to_numpy()
    stored_rows = stored["array_row"].to_numpy(dtype=np.int64)
    stored[SPLIT_COLUMN] = split_by_array_row[stored_rows]
    return stored


def _trace_quality_metadata(
    trace_table: pd.DataFrame,
    result: TraceAmplitudeFilterResult,
) -> dict[str, object]:
    excluded_rows = np.setdiff1d(
        trace_table["array_row"].to_numpy(dtype=np.int64),
        result.eligible_array_rows,
        assume_unique=True,
    )
    source_rows = trace_table["array_row"].to_numpy(dtype=np.int64)
    ffid_by_array_row = np.empty(len(trace_table), dtype=np.int64)
    ffid_by_array_row[source_rows] = trace_table["ffid"].to_numpy(dtype=np.int64)
    excluded_ffids = sorted(int(value) for value in np.unique(ffid_by_array_row[excluded_rows]))
    eligible_ffids = {
        int(value) for value in np.unique(ffid_by_array_row[result.eligible_array_rows])
    }
    fully_excluded_ffids = sorted(set(excluded_ffids) - eligible_ffids)
    return {
        "input_trace_count": int(len(trace_table)),
        "eligible_trace_count": int(len(result.eligible_array_rows)),
        "excluded_trace_count": int(len(excluded_rows)),
        "all_zero_trace_count": int(len(result.all_zero_array_rows)),
        "excess_amplitude_trace_count": int(len(result.excess_amplitude_array_rows)),
        "excluded_array_rows": [int(row) for row in excluded_rows],
        "affected_ffids": excluded_ffids,
        "fully_excluded_ffids": fully_excluded_ffids,
    }


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"input dataset metadata {key} must be a non-empty string")
    return value


def _source_provenance(metadata: Mapping[str, object]) -> dict[str, object]:
    """Preserve legacy source keys and use the canonical list for multi-source data."""
    source_files = canonical_source_files(metadata)
    if len(source_files) == 1:
        source_file = source_files[0]
        return {
            "source_file": source_file["name"],
            "source_sha256": source_file["sha256"],
        }
    return {"source_files": [dict(source_file) for source_file in source_files]}


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


def _validated_normalization_method(name: str, value: str, expected: str) -> str:
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}, got {value!r}")
    return expected


def _validated_split_scope(value: str) -> str:
    if not isinstance(value, str) or value not in SPLIT_SCOPES:
        raise ValueError(f"split_scope must be one of {sorted(SPLIT_SCOPES)}, got {value!r}")
    return value


def _check_output_directory(directory: Path, *, overwrite: bool) -> None:
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"output path is not a directory: {directory}")
    if directory.exists() and not overwrite and any(directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {directory}; pass overwrite=True to replace "
            "the generated files"
        )

    if overwrite and directory.exists():
        invalid_targets = [
            file_name
            for file_name in OUTPUT_FILE_NAMES
            if (directory / file_name).exists() and not (directory / file_name).is_file()
        ]
        if invalid_targets:
            raise FileExistsError(
                f"generated output paths are not files in {directory}: {invalid_targets}"
            )
