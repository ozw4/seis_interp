"""Load the shared observed-only inputs for the C3 random-80 PoC."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from seis_interp.data.c3_volume_run_inputs import (
    C3VolumeRunInputs,
    load_c3_volume_run_inputs,
)
from seis_interp.processing.c3_benchmark_contract import (
    MAIN_C3_DIMENSIONS,
    C3BenchmarkDimensions,
)
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVED_ROLE,
    RANDOM_TRACE_MASK_KIND,
)

C3_RANDOM80_POC_BENCHMARK_ID = "c3_sl25_40_random80_observed_only_poc_v2"
C3_RANDOM80_MISSING_FRACTION = 0.8


def load_c3_random80_poc_inputs(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    config: Mapping[str, object] | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> C3VolumeRunInputs:
    """Materialize only observed amplitudes and validate the shared PoC contract."""
    inputs = load_c3_volume_run_inputs(
        config={} if config is None else config,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
    )
    observed_count, target_count = _validate_poc_inputs(inputs, dimensions=dimensions)
    return replace(
        inputs,
        inputs_lock=_poc_inputs_lock(
            inputs,
            observed_count=observed_count,
            target_count=target_count,
        ),
    )


def _validate_poc_inputs(
    inputs: C3VolumeRunInputs,
    *,
    dimensions: C3BenchmarkDimensions,
) -> tuple[int, int]:
    if not isinstance(dimensions, C3BenchmarkDimensions):
        raise TypeError("dimensions must be a C3BenchmarkDimensions instance")
    expected_shape = tuple(dimensions.shape)
    _validate_dimension_contract(dimensions)
    _validate_volume_domain(inputs, dimensions=dimensions, expected_shape=expected_shape)

    volume = inputs.observed_volume
    observed = _trace_mask(
        volume.observed_trace_mask,
        name="observed_trace_mask",
        expected_shape=expected_shape[1:],
    )
    target = _trace_mask(
        volume.evaluation_target_trace_mask,
        name="evaluation_target_trace_mask",
        expected_shape=expected_shape[1:],
    )
    if np.any(observed & target):
        raise ValueError("observed and evaluation target masks must not overlap")
    if not np.all(observed | target):
        raise ValueError("observed and evaluation target masks must exactly cover the QC domain")

    _validate_outer_mask(inputs.case)
    _validate_observed_values(volume.values, observed, expected_shape=expected_shape)
    observed_count = int(np.count_nonzero(observed))
    target_count = int(np.count_nonzero(target))
    if observed_count == 0 or target_count == 0:
        raise ValueError("PoC volume must contain observed and evaluation target traces")
    _validate_role_counts(
        inputs.volume_metadata,
        observed_count=observed_count,
        target_count=target_count,
    )
    return observed_count, target_count


def _validate_dimension_contract(dimensions: C3BenchmarkDimensions) -> None:
    if len(dimensions.shape) != 5 or any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in dimensions.shape
    ):
        raise ValueError("dimensions shape must contain five positive integers")
    time_start, time_stop = dimensions.time_range
    first_sail_line, last_sail_line = dimensions.sail_line_numbers
    if dimensions.shape[0] != time_stop - time_start:
        raise ValueError("dimensions shape must agree with the time range")
    if dimensions.shape[1] != last_sail_line - first_sail_line + 1:
        raise ValueError("dimensions shape must agree with the inclusive sail-line range")


def _validate_volume_domain(
    inputs: C3VolumeRunInputs,
    *,
    dimensions: C3BenchmarkDimensions,
    expected_shape: tuple[int, ...],
) -> None:
    metadata = inputs.volume_metadata
    stored_shape = metadata.get("shape")
    if not isinstance(stored_shape, list) or tuple(stored_shape) != expected_shape:
        raise ValueError(f"benchmark volume shape must be {list(expected_shape)}")

    volume = inputs.observed_volume
    if not isinstance(volume.values, np.ndarray) or volume.values.shape != expected_shape:
        raise ValueError(f"observed volume shape must be {expected_shape}")
    spatial_shape = expected_shape[1:]
    if not isinstance(volume.array_rows, np.ndarray) or volume.array_rows.shape != spatial_shape:
        raise ValueError(f"volume array_rows shape must be {spatial_shape}")
    if len(inputs.index_table) != int(np.prod(spatial_shape, dtype=np.int64)):
        raise ValueError("volume index must contain exactly one row for every QC-domain cell")

    selection = metadata.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("benchmark volume selection must be a mapping")
    if selection.get("time") != list(dimensions.time_range):
        raise ValueError(f"benchmark volume time range must be {list(dimensions.time_range)}")
    expected_source_lines = [
        dimensions.sail_line_numbers[0],
        dimensions.sail_line_numbers[1] + 1,
    ]
    if selection.get("source_line") != expected_source_lines:
        raise ValueError(f"benchmark volume source-line range must be {expected_source_lines}")
    if (
        not isinstance(volume.time_s, np.ndarray)
        or volume.time_s.shape != (expected_shape[0],)
        or volume.time_s.dtype.kind not in "fiu"
        or volume.time_s.dtype.kind == "b"
        or not np.all(np.isfinite(volume.time_s))
    ):
        raise ValueError("observed volume time_s must be a finite numeric time vector")


def _trace_mask(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.shape != expected_shape
    ):
        raise ValueError(f"{name} must be a boolean trace mask with shape {expected_shape}")
    return value


def _validate_outer_mask(case: Mapping[str, object]) -> None:
    mask = case.get("mask")
    if not isinstance(mask, Mapping):
        raise ValueError("benchmark case mask must be a mapping")
    if mask.get("kind") != RANDOM_TRACE_MASK_KIND:
        raise ValueError(f"PoC mask kind must be {RANDOM_TRACE_MASK_KIND!r}")
    if mask.get("missing_fraction") != C3_RANDOM80_MISSING_FRACTION:
        raise ValueError(f"PoC mask missing_fraction must be {C3_RANDOM80_MISSING_FRACTION}")


def _validate_observed_values(
    values: object,
    observed: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.shape != expected_shape
        or values.dtype.kind not in "fiu"
        or values.dtype.kind == "b"
    ):
        raise ValueError(f"observed volume values must be a real numeric array of {expected_shape}")
    if not np.all(np.isfinite(values[:, observed])):
        raise ValueError("observed amplitudes must be finite")
    if np.any(values[:, ~observed] != 0):
        raise ValueError("target and QC-invalid amplitudes must not be materialized")


def _validate_role_counts(
    metadata: Mapping[str, object],
    *,
    observed_count: int,
    target_count: int,
) -> None:
    role_counts = metadata.get("role_counts")
    expected = {
        OBSERVED_ROLE: observed_count,
        EVALUATION_TARGET_ROLE: target_count,
    }
    if not isinstance(role_counts, Mapping) or dict(role_counts) != expected:
        raise ValueError(f"benchmark volume role counts must match materialized masks {expected}")


def _poc_inputs_lock(
    inputs: C3VolumeRunInputs,
    *,
    observed_count: int,
    target_count: int,
) -> dict[str, object]:
    lock = deepcopy(inputs.inputs_lock)
    mask = inputs.case["mask"]
    assert isinstance(mask, Mapping)
    input_files = inputs.case["input_files"]
    assert isinstance(input_files, Mapping)
    lock.update(
        {
            "benchmark_id": C3_RANDOM80_POC_BENCHMARK_ID,
            "case_id": inputs.case["case_id"],
            "mask": {
                "kind": mask["kind"],
                "missing_fraction": mask["missing_fraction"],
                "random_seed": mask["random_seed"],
                "unit": "complete_trace",
                "files": deepcopy(input_files["mask"]),
            },
            "volume_id": inputs.volume_metadata["volume_id"],
            "shape": list(inputs.volume_metadata["shape"]),
            "observed_trace_count": observed_count,
            "target_trace_count": target_count,
        }
    )
    return lock
