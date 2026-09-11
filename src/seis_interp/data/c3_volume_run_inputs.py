"""Load verified C3 volume inputs and their method-independent run lock."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp import config_values, run_records
from seis_interp.configuration import ConfigurationError
from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    load_benchmark_case,
)
from seis_interp.data.c3_volume_adapter import ObservedC3Volume, load_observed_c3_volume
from seis_interp.data.c3_volume_index_store import (
    OUTPUT_FILE_NAMES as VOLUME_FILE_NAMES,
)
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.file_checksums import file_sha256
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER, validated_index_range


@dataclass(frozen=True)
class C3VolumeRunInputs:
    """Observed values, verified artifacts, and input hashes for one volume run."""

    observed_volume: ObservedC3Volume
    index_table: pd.DataFrame
    case: dict[str, object]
    volume_metadata: dict[str, object]
    inputs_lock: dict[str, object]


def load_c3_volume_run_inputs(
    *,
    config: Mapping[str, object],
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
) -> C3VolumeRunInputs:
    """Verify bindings and declarations without materializing target amplitudes."""
    observed_volume = load_observed_c3_volume(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
    )
    case = load_benchmark_case(case_dir)
    index_table, volume_metadata = load_c3_volume_index(volume_dir)
    _validate_declared_inputs(config, case=case, volume_metadata=volume_metadata)
    _validate_selected_roles(observed_volume.observed_trace_mask, volume_metadata)
    if not np.array_equal(
        index_table["array_row"].to_numpy(dtype=np.int64),
        observed_volume.array_rows.reshape(-1),
    ):
        raise ValueError("volume index array_row order must match the observed volume flat order")
    inputs_lock = _inputs_lock(
        case=case,
        case_directory=case_dir,
        volume_metadata=volume_metadata,
        volume_directory=volume_dir,
    )
    return C3VolumeRunInputs(
        observed_volume=observed_volume,
        index_table=index_table,
        case=case,
        volume_metadata=volume_metadata,
        inputs_lock=inputs_lock,
    )


def _validate_declared_inputs(
    config: Mapping[str, object],
    *,
    case: Mapping[str, object],
    volume_metadata: Mapping[str, object],
) -> None:
    _validate_optional_identifier(config, "benchmark_case", "id", case["case_id"])
    _validate_optional_identifier(config, "benchmark_volume", "id", volume_metadata["volume_id"])

    benchmark_volume = config.get("benchmark_volume")
    if isinstance(benchmark_volume, Mapping) and "selection" in benchmark_volume:
        selection = validated_c3_volume_selection(benchmark_volume["selection"])
        stored_selection = volume_metadata["selection"]
        assert isinstance(stored_selection, Mapping)
        if selection != dict(stored_selection):
            raise ConfigurationError(
                "benchmark_volume.selection does not match the verified volume artifact"
            )

    mask = case["mask"]
    assert isinstance(mask, Mapping)
    interpolation_mask = config.get("interpolation_mask")
    if interpolation_mask is not None:
        if not isinstance(interpolation_mask, Mapping):
            raise ConfigurationError("interpolation_mask configuration must be a mapping")
        expected = {
            "partition": case["partition"],
            "kind": mask["kind"],
            "missing_fraction": mask["missing_fraction"],
        }
        for key, expected_value in expected.items():
            if key not in interpolation_mask:
                continue
            declared_value = interpolation_mask[key]
            if key == "missing_fraction":
                declared_value = config_values.positive_float(
                    declared_value, "interpolation_mask.missing_fraction"
                )
                if declared_value >= 1.0:
                    raise ConfigurationError(
                        "interpolation_mask.missing_fraction must be less than 1"
                    )
            if declared_value != expected_value:
                raise ConfigurationError(
                    f"interpolation_mask.{key} does not match the verified benchmark case"
                )

    project = config.get("project")
    if project is not None:
        if not isinstance(project, Mapping):
            raise ConfigurationError("project configuration must be a mapping")
        if "random_seed" in project:
            random_seed = config_values.nonnegative_integer(
                project["random_seed"], "project.random_seed"
            )
            if random_seed != mask["random_seed"]:
                raise ConfigurationError(
                    "project.random_seed does not match the verified benchmark-case mask seed"
                )


def validated_c3_volume_selection(value: object) -> dict[str, list[int]]:
    """Return a complete normalized C3 volume selection declaration."""
    if not isinstance(value, Mapping) or set(value) != set(VOLUME_AXIS_ORDER):
        raise ConfigurationError(
            f"benchmark_volume.selection must contain exactly {list(VOLUME_AXIS_ORDER)!r}"
        )
    return {
        axis: list(
            validated_index_range(
                value[axis],
                name=f"benchmark_volume.selection.{axis}",
            )
        )
        for axis in VOLUME_AXIS_ORDER
    }


def _validate_optional_identifier(
    config: Mapping[str, object],
    section_name: str,
    key: str,
    expected: object,
) -> None:
    section = config.get(section_name)
    if section is None:
        return
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{section_name} configuration must be a mapping")
    if key in section and section[key] != expected:
        raise ConfigurationError(f"{section_name}.{key} does not match the verified artifact")


def _validate_selected_roles(
    observed_trace_mask: np.ndarray,
    volume_metadata: Mapping[str, object],
) -> None:
    observed_count = int(np.count_nonzero(observed_trace_mask))
    trace_count = int(observed_trace_mask.size)
    target_count = trace_count - observed_count
    if observed_count == 0 or target_count == 0:
        raise ValueError(
            "selected volume must contain at least one observed and one evaluation target trace"
        )
    expected_counts = volume_metadata["role_counts"]
    if not isinstance(expected_counts, Mapping) or (
        expected_counts.get("observed") != observed_count
        or expected_counts.get("evaluation_target") != target_count
    ):
        raise ValueError("selected volume role counts do not match volume metadata")


def _inputs_lock(
    *,
    case: Mapping[str, object],
    case_directory: Path,
    volume_metadata: Mapping[str, object],
    volume_directory: Path,
) -> dict[str, object]:
    return {
        "benchmark_case": {
            "case_id": case["case_id"],
            "file": BENCHMARK_CASE_FILE_NAME,
            "sha256": file_sha256(case_directory / BENCHMARK_CASE_FILE_NAME),
            "input_files": deepcopy(case["input_files"]),
        },
        "benchmark_volume": {
            "volume_id": volume_metadata["volume_id"],
            "files": run_records.file_hashes(volume_directory, VOLUME_FILE_NAMES),
        },
    }
