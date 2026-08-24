"""Prepare trace splits and normalization metadata for baseline evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import METADATA_FILE_NAME
from seis_interp.processing.normalization import (
    fit_normalization_parameters,
    write_normalization_parameters,
)
from seis_interp.processing.trace_splits import (
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assign_random_trace_splits,
)

TRACE_SPLIT_FILE_NAME = "trace_split.parquet"
NORMALIZATION_FILE_NAME = "normalization.json"
PREPARATION_FILE_NAME = "preparation.json"
COORDINATE_NORMALIZATION_METHOD = "train_minmax_minus_one_to_one"
AMPLITUDE_NORMALIZATION_METHOD = "train_global_rms"

OUTPUT_FILE_NAMES = (
    TRACE_SPLIT_FILE_NAME,
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
)


def prepare_baseline_dataset(
    interim_dir: Path,
    output_dir: Path,
    *,
    holdout_fraction: float,
    validation_fraction_of_holdout: float,
    random_seed: int,
    coordinate_normalization: str = COORDINATE_NORMALIZATION_METHOD,
    amplitude_normalization: str = AMPLITUDE_NORMALIZATION_METHOD,
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

    dataset = load_interim_trace_dataset(input_directory)
    split_table = assign_random_trace_splits(
        dataset.trace_table,
        holdout_fraction=holdout_fraction,
        validation_fraction_of_holdout=validation_fraction_of_holdout,
        random_seed=random_seed,
    )
    normalization = fit_normalization_parameters(
        split_table,
        dataset.amplitudes,
        dataset.time_s,
    )

    stored_splits = split_table[["array_row", SPLIT_COLUMN]].copy()
    split_counts = {
        split: int((stored_splits[SPLIT_COLUMN] == split).sum())
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    preparation: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": _metadata_text(dataset.metadata, "dataset_id"),
        "source_file": _relative_source_file(dataset.metadata),
        "source_sha256": _metadata_text(dataset.metadata, "source_sha256"),
        "input_dataset_metadata_sha256": _file_sha256(input_directory / METADATA_FILE_NAME),
        "trace_count": int(dataset.metadata["trace_count"]),
        "sample_count": int(dataset.metadata["sample_count"]),
        "config_source": stored_config_source,
        "normalization": {
            "coordinates": stored_coordinate_normalization,
            "amplitude": stored_amplitude_normalization,
        },
        "random_seed": int(random_seed),
        "holdout_fraction": float(holdout_fraction),
        "validation_fraction_of_holdout": float(validation_fraction_of_holdout),
        "split_counts": split_counts,
        "files": {
            "trace_split": TRACE_SPLIT_FILE_NAME,
            "normalization": NORMALIZATION_FILE_NAME,
        },
    }
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


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"input dataset metadata {key} must be a non-empty string")
    return value


def _relative_source_file(metadata: Mapping[str, object]) -> str:
    source_file = _metadata_text(metadata, "source_file")
    if Path(source_file).is_absolute() or PureWindowsPath(source_file).is_absolute():
        raise ValueError("input dataset metadata source_file must not be an absolute path")
    return source_file


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
