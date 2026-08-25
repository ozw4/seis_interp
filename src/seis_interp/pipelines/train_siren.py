"""Connect prepared trace data and resolved configuration to SIREN training."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.models.siren import Siren
from seis_interp.pipelines.prepare_baseline import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
)
from seis_interp.processing.normalization import (
    NormalizationParameters,
    normalize_amplitudes,
    normalize_spatial_coordinates,
    normalize_time,
    read_normalization_parameters,
)
from seis_interp.processing.trace_splits import (
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from seis_interp.training.point_sampler import RandomPointSampler, build_trace_points
from seis_interp.training.trainer import train_siren

CONFIG_FILE_NAME = "config.resolved.yaml"
INPUTS_LOCK_FILE_NAME = "inputs.lock.json"
METRICS_FILE_NAME = "metrics.json"
RUN_FILE_NAME = "run.json"
CHECKPOINT_RELATIVE_PATH = Path("artifacts") / "best.pt"
PROCESSED_INPUT_FILE_NAMES = (
    TRACE_SPLIT_FILE_NAME,
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
)


def train_siren_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Train from one prepared split and write the minimal reproducible run outputs."""
    output_directory = Path(output_dir)
    _check_new_output_directory(output_directory)
    started_at_utc = _utc_timestamp()
    git_commit = _git_commit()
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    resolved_config = deepcopy(config)
    resolved_config["training"]["device"] = device
    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    dataset = load_interim_trace_dataset(interim_directory)
    split_table, normalization, preparation = _load_processed_dataset(processed_directory)
    interim_files = _file_hashes(interim_directory, INTERIM_FILE_NAMES)
    processed_files = _file_hashes(processed_directory, PROCESSED_INPUT_FILE_NAMES)
    split_rows = _validate_split_table(split_table, len(dataset.trace_table))
    split_counts = _split_counts(split_table)
    _validate_preparation_data(
        preparation,
        dataset.metadata,
        split_counts,
        interim_files,
    )
    preparation_contract = _validated_preparation_contract(preparation, resolved_config)

    normalized_time = normalize_time(dataset.time_s, normalization)
    normalized_spatial = normalize_spatial_coordinates(dataset.trace_table, normalization)
    dataset_array_rows = validated_array_rows(dataset.trace_table, require_contiguous=True)
    spatial_by_array_row = np.empty_like(normalized_spatial)
    spatial_by_array_row[dataset_array_rows] = normalized_spatial
    normalized_amplitudes = normalize_amplitudes(dataset.amplitudes, normalization)

    train_rows = split_rows[split_table[SPLIT_COLUMN].eq(TRAIN_SPLIT).to_numpy(dtype=bool)]
    validation_rows = split_rows[
        split_table[SPLIT_COLUMN].eq(VALIDATION_SPLIT).to_numpy(dtype=bool)
    ]
    random_seed = get_required_config_value(resolved_config, "project.random_seed")
    sampler = RandomPointSampler(
        normalized_time,
        spatial_by_array_row,
        normalized_amplitudes,
        train_rows,
        random_seed=random_seed,
    )
    validation_coordinates, validation_targets = build_trace_points(
        normalized_time,
        spatial_by_array_row,
        normalized_amplitudes,
        validation_rows,
    )

    torch.manual_seed(random_seed)
    model = _build_model(resolved_config)
    _validate_training_contract(resolved_config)
    result = train_siren(
        model,
        sampler,
        validation_coordinates,
        validation_targets,
        normalization,
        device=device,
        loss=get_required_config_value(resolved_config, "training.loss"),
        learning_rate=get_required_config_value(resolved_config, "training.learning_rate"),
        batch_size=get_required_config_value(resolved_config, "training.batch_size"),
        steps_per_epoch=get_required_config_value(resolved_config, "training.steps_per_epoch"),
        max_epochs=get_required_config_value(resolved_config, "training.max_epochs"),
        early_stopping_patience=get_required_config_value(
            resolved_config, "training.early_stopping_patience"
        ),
        validation_batch_size=get_required_config_value(
            resolved_config, "training.validation_batch_size"
        ),
        validation_samples_per_trace=len(normalized_time),
        checkpoint_path=output_directory / CHECKPOINT_RELATIVE_PATH,
    )

    metrics = asdict(result)
    metrics["history"] = list(result.history)
    run_metadata = {
        "git_commit": git_commit,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_timestamp(),
        "status": "success",
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "random_seed": random_seed,
    }
    inputs_lock = _build_inputs_lock(
        interim_files=interim_files,
        processed_files=processed_files,
        preparation_contract=preparation_contract,
    )
    _write_run_outputs(
        output_directory,
        resolved_config,
        inputs_lock,
        metrics,
        run_metadata,
    )
    return metrics


def _load_processed_dataset(
    directory: Path,
) -> tuple[pd.DataFrame, NormalizationParameters, dict[str, object]]:
    missing = [name for name in PROCESSED_INPUT_FILE_NAMES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"processed dataset is missing required files in {directory}: {missing}"
        )
    split_table = pd.read_parquet(directory / TRACE_SPLIT_FILE_NAME)
    normalization = read_normalization_parameters(directory / NORMALIZATION_FILE_NAME)
    preparation = json.loads((directory / PREPARATION_FILE_NAME).read_text(encoding="utf-8"))
    if not isinstance(preparation, dict):
        raise ValueError(f"{PREPARATION_FILE_NAME} must contain a JSON object")
    return split_table, normalization, preparation


def _validate_split_table(split_table: pd.DataFrame, trace_count: int) -> np.ndarray:
    rows = validated_array_rows(split_table, require_contiguous=True)
    if len(split_table) != trace_count:
        raise ValueError(
            f"split table has {len(split_table)} rows but interim dataset has {trace_count} traces"
        )
    if SPLIT_COLUMN not in split_table.columns:
        raise ValueError(f"split table is missing required column: {SPLIT_COLUMN}")
    valid_splits = {TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT}
    invalid = sorted(set(split_table[SPLIT_COLUMN]) - valid_splits)
    if invalid:
        raise ValueError(f"split table contains invalid split values: {invalid}")
    missing = sorted(valid_splits - set(split_table[SPLIT_COLUMN]))
    if missing:
        raise ValueError(f"split table contains no rows for splits: {missing}")
    return rows


def _split_counts(split_table: pd.DataFrame) -> dict[str, int]:
    return {
        split: int(split_table[SPLIT_COLUMN].eq(split).sum())
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }


def _file_hashes(
    directory: Path,
    file_names: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    return {file_name: {"sha256": file_sha256(directory / file_name)} for file_name in file_names}


def _validate_preparation_data(
    preparation: Mapping[str, object],
    dataset_metadata: Mapping[str, object],
    split_counts: Mapping[str, int],
    interim_files: Mapping[str, object],
) -> None:
    expected_values = {
        "dataset_id": dataset_metadata.get("dataset_id"),
        "source_file": dataset_metadata.get("source_file"),
        "source_sha256": dataset_metadata.get("source_sha256"),
        "trace_count": dataset_metadata.get("trace_count"),
        "sample_count": dataset_metadata.get("sample_count"),
        "split_counts": dict(split_counts),
    }
    mismatched = [
        key for key, expected in expected_values.items() if preparation.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} does not match the interim/processed data: {mismatched}"
        )
    input_files = preparation.get("input_files")
    if not isinstance(input_files, Mapping):
        raise ValueError(f"{PREPARATION_FILE_NAME} input_files must be an object")
    if dict(input_files) != dict(interim_files):
        raise ValueError(
            f"{PREPARATION_FILE_NAME} interim file checksums do not match the current files"
        )


def _validated_preparation_contract(
    preparation: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "random_seed": get_required_config_value(config, "project.random_seed"),
        "holdout_fraction": get_required_config_value(
            config, "sampling.random_trace_holdout_fraction"
        ),
        "validation_fraction_of_holdout": get_required_config_value(
            config, "sampling.validation_fraction_of_holdout"
        ),
        "normalization": {
            "coordinates": get_required_config_value(config, "normalization.coordinates"),
            "amplitude": get_required_config_value(config, "normalization.amplitude"),
        },
    }
    mismatched = [
        key for key, expected_value in expected.items() if preparation.get(key) != expected_value
    ]
    if mismatched:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} does not match the resolved configuration: {mismatched}"
        )
    return expected


def _build_model(config: Mapping[str, object]) -> Siren:
    model_name = get_required_config_value(config, "model.name")
    if model_name != "siren":
        raise ConfigurationError(f"model.name must be 'siren', got {model_name!r}")
    input_features = get_required_config_value(config, "model.input_features")
    if input_features != len(MODEL_COORDINATE_ORDER):
        raise ConfigurationError(
            f"model.input_features must be {len(MODEL_COORDINATE_ORDER)}, got {input_features!r}"
        )
    return Siren(
        input_features=input_features,
        hidden_width=get_required_config_value(config, "model.hidden_width"),
        hidden_layers=get_required_config_value(config, "model.hidden_layers"),
        output_features=1,
        omega_0=get_required_config_value(config, "model.omega_0"),
    )


def _validate_training_contract(config: Mapping[str, object]) -> None:
    optimizer = get_required_config_value(config, "training.optimizer")
    if optimizer != "adam":
        raise ConfigurationError(f"training.optimizer must be 'adam', got {optimizer!r}")


def _build_inputs_lock(
    *,
    interim_files: Mapping[str, object],
    processed_files: Mapping[str, object],
    preparation_contract: Mapping[str, object],
) -> dict[str, object]:
    return {
        "interim_files": dict(interim_files),
        "processed_files": dict(processed_files),
        "preparation": dict(preparation_contract),
    }


def _write_run_outputs(
    output_directory: Path,
    config: Mapping[str, object],
    inputs_lock: Mapping[str, object],
    metrics: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / CONFIG_FILE_NAME).write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    (output_directory / INPUTS_LOCK_FILE_NAME).write_text(
        json.dumps(inputs_lock, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / METRICS_FILE_NAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / RUN_FILE_NAME).write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _check_new_output_directory(directory: Path) -> None:
    if directory.exists():
        raise FileExistsError(f"run output path already exists: {directory}")


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not determine the current Git commit") from error
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    return commit


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
