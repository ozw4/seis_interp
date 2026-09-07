"""Bind one prepared partition and interpolation mask into a benchmark case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seis_interp.data.benchmark_case_inputs import (
    collect_benchmark_input_hashes,
    validate_benchmark_preparation,
    validated_benchmark_mask_summary,
)
from seis_interp.data.benchmark_case_store import (
    EVALUATION_TARGET_AMPLITUDE_USE,
    MASK_DOMAIN,
    validated_case_id,
    validated_config_source,
    write_benchmark_case,
)
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
)
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVED_ROLE,
)
from seis_interp.processing.normalization import read_normalization_parameters


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
    stored_config_source = validated_config_source(config_source)
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
    dataset_id = validate_benchmark_preparation(
        dataset.metadata,
        preparation,
        interim_hashes=input_hashes["interim"],
    )

    mask_table, mask_metadata = load_interpolation_mask(mask_directory)
    mask_summary = validated_benchmark_mask_summary(
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


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} does not contain valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload
