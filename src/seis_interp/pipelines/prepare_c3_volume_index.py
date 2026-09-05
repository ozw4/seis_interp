"""Prepare a dense C3 volume index bound to one benchmark case."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.benchmark_case_inputs import verify_benchmark_case_inputs
from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    load_benchmark_case,
    validated_config_source,
)
from seis_interp.data.c3_volume_index_store import (
    validated_volume_id,
    write_c3_volume_index,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.trace_store import TIME_FILE_NAME, TRACES_FILE_NAME
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    build_c3_volume_index,
    selected_spatial_shape,
    validated_index_range,
)
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
)

_TRACE_COLUMNS = (
    "array_row",
    "ffid",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)


def prepare_c3_volume_index(
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    output_dir: Path,
    *,
    volume_id: str,
    time_range: tuple[int, int],
    source_line_range: tuple[int, int],
    shot_in_line_range: tuple[int, int],
    relative_receiver_x_range: tuple[int, int],
    relative_receiver_y_range: tuple[int, int],
    config_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate current artifacts and write their dense trace-to-cell mapping."""
    stored_volume_id = validated_volume_id(volume_id)
    stored_config_source = validated_config_source(config_source)
    ranges = {
        "time": validated_index_range(time_range, name="time_range"),
        "source_line": validated_index_range(source_line_range, name="source_line_range"),
        "shot_in_line": validated_index_range(shot_in_line_range, name="shot_in_line_range"),
        "relative_receiver_x": validated_index_range(
            relative_receiver_x_range, name="relative_receiver_x_range"
        ),
        "relative_receiver_y": validated_index_range(
            relative_receiver_y_range, name="relative_receiver_y_range"
        ),
    }
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")

    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    mask_directory = Path(mask_dir)
    case_directory = Path(case_dir)

    case = load_benchmark_case(case_directory)
    verify_benchmark_case_inputs(
        case,
        interim_dir=interim_directory,
        processed_dir=processed_directory,
        mask_dir=mask_directory,
    )
    case_sha256 = file_sha256(case_directory / BENCHMARK_CASE_FILE_NAME)

    trace_table = pd.read_parquet(
        interim_directory / TRACES_FILE_NAME,
        columns=list(_TRACE_COLUMNS),
    )
    validated_array_rows(trace_table, require_contiguous=True)
    time_s = np.load(interim_directory / TIME_FILE_NAME, allow_pickle=False)
    _validate_time_axis(time_s, ranges["time"])

    mask_table, mask_metadata = load_interpolation_mask(mask_directory)
    _validate_case_mask(case, mask_metadata, mask_row_count=len(mask_table))

    index_table = build_c3_volume_index(
        trace_table,
        mask_table["array_row"].to_numpy(dtype=np.int64),
        source_line_range=ranges["source_line"],
        shot_in_line_range=ranges["shot_in_line"],
        relative_receiver_x_range=ranges["relative_receiver_x"],
        relative_receiver_y_range=ranges["relative_receiver_y"],
    )
    selected_roles = index_table[["array_row"]].merge(
        mask_table,
        on="array_row",
        how="left",
        validate="one_to_one",
        sort=False,
    )[OBSERVATION_ROLE_COLUMN]
    role_counts = {
        OBSERVED_ROLE: int(selected_roles.eq(OBSERVED_ROLE).sum()),
        EVALUATION_TARGET_ROLE: int(selected_roles.eq(EVALUATION_TARGET_ROLE).sum()),
    }
    if any(count == 0 for count in role_counts.values()):
        raise ValueError(
            "selected volume must contain at least one observed and one evaluation target trace"
        )
    if sum(role_counts.values()) != len(index_table):
        raise ValueError("selected volume rows are not fully covered by mask roles")

    spatial_shape = selected_spatial_shape(
        source_line_range=ranges["source_line"],
        shot_in_line_range=ranges["shot_in_line"],
        relative_receiver_x_range=ranges["relative_receiver_x"],
        relative_receiver_y_range=ranges["relative_receiver_y"],
    )
    metadata: dict[str, object] = {
        "volume_id": stored_volume_id,
        "dataset_id": case["dataset_id"],
        "partition": case["partition"],
        "config_source": stored_config_source,
        "axis_order": list(VOLUME_AXIS_ORDER),
        "selection": {name: list(index_range) for name, index_range in ranges.items()},
        "shape": [ranges["time"][1] - ranges["time"][0], *spatial_shape],
        "trace_count": int(len(index_table)),
        "role_counts": role_counts,
        "index_contract": dict(INDEX_CONTRACT),
        "benchmark_case": {
            "case_id": case["case_id"],
            "file": BENCHMARK_CASE_FILE_NAME,
            "sha256": case_sha256,
        },
    }
    return write_c3_volume_index(
        Path(output_dir),
        index_table,
        metadata,
        overwrite=overwrite,
    )


def _validate_time_axis(time_s: np.ndarray, time_range: tuple[int, int]) -> None:
    if not isinstance(time_s, np.ndarray) or time_s.ndim != 1 or len(time_s) == 0:
        raise ValueError(f"{TIME_FILE_NAME} must be a non-empty one-dimensional array")
    if time_s.dtype.kind not in "fiu" or not np.all(np.isfinite(time_s)):
        raise ValueError(f"{TIME_FILE_NAME} must contain finite real values")
    if len(time_s) > 1 and not np.all(np.diff(time_s) > 0):
        raise ValueError(f"{TIME_FILE_NAME} must be strictly increasing")
    if time_range[1] > len(time_s):
        raise ValueError(
            f"time_range {time_range} exceeds the {len(time_s)} samples in {TIME_FILE_NAME}"
        )


def _validate_case_mask(
    case: Mapping[str, object],
    mask_metadata: Mapping[str, object],
    *,
    mask_row_count: int,
) -> None:
    if mask_metadata.get("dataset_id") != case["dataset_id"]:
        raise ValueError("mask dataset ID does not match the benchmark case")
    if mask_metadata.get("partition") != case["partition"]:
        raise ValueError("mask partition does not match the benchmark case")

    case_mask = case["mask"]
    assert isinstance(case_mask, Mapping)
    keys = (
        "kind",
        "missing_fraction",
        "random_seed",
        "candidate_trace_count",
        "candidate_ffid_count",
        "counts",
        "duplicate_physical_coordinates",
    )
    current = {key: mask_metadata.get(key) for key in keys}
    if current != dict(case_mask):
        raise ValueError("mask semantic summary does not match the benchmark case")
    if mask_row_count != case_mask["candidate_trace_count"]:
        raise ValueError("mask row count does not match the benchmark case")
