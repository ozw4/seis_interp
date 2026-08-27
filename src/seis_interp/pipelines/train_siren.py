"""Connect prepared trace data and resolved configuration to SIREN training."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from numbers import Integral
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
from seis_interp.data.trace_store import canonical_source_files
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.evaluation.streaming_snr import evaluate_model_global_snr_by_ffid
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
from seis_interp.training.ffid_batches import (
    FullFfidBatchSampler,
    array_rows_by_ffid_for_split,
    validate_all_ffids_have_split_rows,
)
from seis_interp.training.ffid_trainer import train_siren_by_ffid
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
RANDOM_POINTS_BATCH_MODE = "random_points"
FULL_FFID_EPOCH_BATCH_MODE = "full_ffid_epoch"
ProgressReporter = Callable[[str], None]


def train_siren_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Train from one prepared split and write the minimal reproducible run outputs."""
    output_directory = Path(output_dir)
    _check_new_output_directory(output_directory)
    started_at_utc = _utc_timestamp()
    git_commit = _git_commit()
    config = load_resolved_config(Path(config_path))
    batch_mode = _training_batch_mode(config)
    device = device_override or get_required_config_value(config, "training.device")
    resolved_config = deepcopy(config)
    resolved_config["training"]["device"] = device
    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    dataset = load_interim_trace_dataset(
        interim_directory,
        memory_map_amplitudes=batch_mode == FULL_FFID_EPOCH_BATCH_MODE,
    )
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
    preparation_contract = _validated_preparation_contract(
        preparation,
        resolved_config,
        batch_mode=batch_mode,
    )

    normalized_time = normalize_time(dataset.time_s, normalization)
    normalized_spatial = normalize_spatial_coordinates(dataset.trace_table, normalization)
    dataset_array_rows = validated_array_rows(dataset.trace_table, require_contiguous=True)
    spatial_by_array_row = np.empty_like(normalized_spatial)
    spatial_by_array_row[dataset_array_rows] = normalized_spatial
    random_seed = get_required_config_value(resolved_config, "project.random_seed")
    if batch_mode == FULL_FFID_EPOCH_BATCH_MODE:
        _seed_training(random_seed)
    else:
        # Preserve the established random-points path's observable RNG contract.
        torch.manual_seed(random_seed)
    model = _build_model(resolved_config)
    _validate_training_contract(resolved_config)
    full_training_contract: dict[str, object] | None = None
    if batch_mode == RANDOM_POINTS_BATCH_MODE:
        result = _train_random_points(
            model=model,
            normalized_time=normalized_time,
            spatial_by_array_row=spatial_by_array_row,
            amplitudes=dataset.amplitudes,
            split_table=split_table,
            split_rows=split_rows,
            normalization=normalization,
            config=resolved_config,
            random_seed=random_seed,
            device=device,
            checkpoint_path=output_directory / CHECKPOINT_RELATIVE_PATH,
        )
    else:
        result, full_training_contract = _train_full_ffid_epoch(
            model=model,
            normalized_time=normalized_time,
            spatial_by_array_row=spatial_by_array_row,
            amplitudes=dataset.amplitudes,
            trace_table=dataset.trace_table,
            split_table=split_table,
            normalization=normalization,
            config=resolved_config,
            random_seed=random_seed,
            device=device,
            checkpoint_path=output_directory / CHECKPOINT_RELATIVE_PATH,
            progress_reporter=progress_reporter,
        )

    metrics = asdict(result)
    metrics["history"] = [dict(record) for record in result.history]
    if full_training_contract is not None:
        metrics["batch_mode"] = FULL_FFID_EPOCH_BATCH_MODE
        metrics["effective_steps_per_epoch"] = full_training_contract["effective_steps_per_epoch"]
        metrics = _encode_full_ffid_infinite_snr(metrics)
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
    if full_training_contract is not None:
        run_metadata.update(full_training_contract)
    inputs_lock = _build_inputs_lock(
        interim_files=interim_files,
        processed_files=processed_files,
        preparation_contract=preparation_contract,
        source_files=(
            canonical_source_files(dataset.metadata) if full_training_contract is not None else None
        ),
        training_contract=full_training_contract,
    )
    _write_run_outputs(
        output_directory,
        resolved_config,
        inputs_lock,
        metrics,
        run_metadata,
    )
    return metrics


def _train_random_points(
    *,
    model: Siren,
    normalized_time: np.ndarray,
    spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    split_table: pd.DataFrame,
    split_rows: np.ndarray,
    normalization: NormalizationParameters,
    config: Mapping[str, object],
    random_seed: int,
    device: object,
    checkpoint_path: Path,
) -> object:
    """Run the established random-point path without changing its outputs."""
    normalized_amplitudes = normalize_amplitudes(amplitudes, normalization)
    train_rows = split_rows[split_table[SPLIT_COLUMN].eq(TRAIN_SPLIT).to_numpy(dtype=bool)]
    validation_rows = split_rows[
        split_table[SPLIT_COLUMN].eq(VALIDATION_SPLIT).to_numpy(dtype=bool)
    ]
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
    return train_siren(
        model,
        sampler,
        validation_coordinates,
        validation_targets,
        normalization,
        device=device,
        loss=get_required_config_value(config, "training.loss"),
        learning_rate=get_required_config_value(config, "training.learning_rate"),
        batch_size=get_required_config_value(config, "training.batch_size"),
        steps_per_epoch=get_required_config_value(config, "training.steps_per_epoch"),
        max_epochs=get_required_config_value(config, "training.max_epochs"),
        early_stopping_patience=get_required_config_value(
            config, "training.early_stopping_patience"
        ),
        validation_batch_size=get_required_config_value(config, "training.validation_batch_size"),
        validation_samples_per_trace=len(normalized_time),
        checkpoint_path=checkpoint_path,
    )


def _train_full_ffid_epoch(
    *,
    model: Siren,
    normalized_time: np.ndarray,
    spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    trace_table: pd.DataFrame,
    split_table: pd.DataFrame,
    normalization: NormalizationParameters,
    config: Mapping[str, object],
    random_seed: int,
    device: object,
    checkpoint_path: Path,
    progress_reporter: ProgressReporter | None,
) -> tuple[object, dict[str, object]]:
    """Build bounded-memory FFID batches and run the survey-wide trainer."""
    rows_by_split = {
        split: array_rows_by_ffid_for_split(trace_table, split_table, split=split)
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    for split, rows_by_ffid in rows_by_split.items():
        validate_all_ffids_have_split_rows(
            trace_table,
            rows_by_ffid,
            split=split,
        )

    training_rows_by_ffid = rows_by_split[TRAIN_SPLIT]
    validation_rows_by_ffid = rows_by_split[VALIDATION_SPLIT]
    validation_batch_size = _validated_positive_config_integer(
        get_required_config_value(config, "training.validation_batch_size"),
        "training.validation_batch_size",
    )
    sampler = FullFfidBatchSampler(
        normalized_time,
        spatial_by_array_row,
        amplitudes,
        training_rows_by_ffid,
        amplitude_rms=normalization.amplitude_rms,
        random_seed=random_seed,
    )

    def evaluate_validation(current_model: torch.nn.Module) -> float:
        return evaluate_model_global_snr_by_ffid(
            current_model,
            normalized_time=normalized_time,
            normalized_spatial_by_array_row=spatial_by_array_row,
            amplitudes=amplitudes,
            rows_by_ffid=validation_rows_by_ffid,
            amplitude_rms=normalization.amplitude_rms,
            prediction_batch_size=validation_batch_size,
            device=device,
        )

    result = train_siren_by_ffid(
        model,
        sampler,
        evaluate_validation,
        normalization,
        device=device,
        loss=get_required_config_value(config, "training.loss"),
        optimizer=get_required_config_value(config, "training.optimizer"),
        learning_rate=get_required_config_value(config, "training.learning_rate"),
        max_epochs=get_required_config_value(config, "training.max_epochs"),
        early_stopping_patience=get_required_config_value(
            config, "training.early_stopping_patience"
        ),
        validation_ffid_count=len(validation_rows_by_ffid),
        checkpoint_path=checkpoint_path,
        reporter=progress_reporter,
    )
    return result, _full_training_contract(
        training_rows_by_ffid,
        validation_rows_by_ffid,
        sample_count=len(normalized_time),
    )


def _full_training_contract(
    training_rows_by_ffid: Mapping[int, np.ndarray],
    validation_rows_by_ffid: Mapping[int, np.ndarray],
    *,
    sample_count: int,
) -> dict[str, object]:
    training_trace_counts = np.asarray(
        [len(rows) for rows in training_rows_by_ffid.values()], dtype=np.int64
    )
    point_counts = training_trace_counts * int(sample_count)
    ffids = sorted(training_rows_by_ffid)
    return {
        "batch_mode": FULL_FFID_EPOCH_BATCH_MODE,
        "training_ffid_count": len(training_rows_by_ffid),
        "validation_ffid_count": len(validation_rows_by_ffid),
        "effective_steps_per_epoch": len(training_rows_by_ffid),
        "ffid_range": [ffids[0], ffids[-1]],
        "training_traces_per_ffid": _integer_distribution(training_trace_counts),
        "points_per_update": _integer_distribution(point_counts),
        "amplitude_scaling": "train_global_rms",
        "validation": "all_validation_traces_streamed",
    }


def _integer_distribution(values: np.ndarray) -> dict[str, int | float]:
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "max": int(np.max(values)),
    }


def _validated_positive_config_integer(value: object, dotted_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ConfigurationError(f"{dotted_path} must be a positive integer, got {value!r}")
    return int(value)


def _encode_full_ffid_infinite_snr(metrics: dict[str, object]) -> dict[str, object]:
    """Represent a mathematically infinite full-mode S/N in strict JSON.

    The streaming metric deliberately returns positive infinity for a perfect
    prediction. JSON has no infinity number, so the full-FFID run contract uses
    the explicit string ``"inf"`` for that one mathematically valid outcome.
    Checkpoints retain the original floating-point infinity.
    """
    encoded = dict(metrics)
    encoded["best_validation_global_snr_db"] = _encode_positive_infinity(
        encoded["best_validation_global_snr_db"]
    )
    history = encoded.get("history")
    if not isinstance(history, list):
        raise RuntimeError("full_ffid_epoch metrics history must be a list")
    encoded_history: list[dict[str, object]] = []
    for raw_record in history:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("full_ffid_epoch history entries must be objects")
        record = dict(raw_record)
        record["validation_global_snr_db"] = _encode_positive_infinity(
            record["validation_global_snr_db"]
        )
        encoded_history.append(record)
    encoded["history"] = encoded_history
    return encoded


def _encode_positive_infinity(value: object) -> object:
    if isinstance(value, (float, np.floating)) and np.isposinf(value):
        return "inf"
    return value


def _seed_training(random_seed: int) -> None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _training_batch_mode(config: Mapping[str, object]) -> str:
    training = config.get("training")
    value = (
        training.get("batch_mode", RANDOM_POINTS_BATCH_MODE)
        if isinstance(training, Mapping)
        else RANDOM_POINTS_BATCH_MODE
    )
    if value not in (RANDOM_POINTS_BATCH_MODE, FULL_FFID_EPOCH_BATCH_MODE):
        raise ConfigurationError(
            f"training.batch_mode must be 'random_points' or 'full_ffid_epoch', got {value!r}"
        )
    return str(value)


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
    source_files = canonical_source_files(dataset_metadata)
    if len(source_files) == 1:
        source_contract: dict[str, object] = {
            "source_file": source_files[0]["name"],
            "source_sha256": source_files[0]["sha256"],
        }
    else:
        source_contract = {"source_files": [dict(source) for source in source_files]}
    expected_values = {
        "dataset_id": dataset_metadata.get("dataset_id"),
        **source_contract,
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
    *,
    batch_mode: str,
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
    configured_split_scope = _configured_split_scope(config)
    prepared_split_scope = preparation.get("split_scope", "global")
    if prepared_split_scope != configured_split_scope:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} does not match the resolved configuration: ['split_scope']"
        )
    if batch_mode == FULL_FFID_EPOCH_BATCH_MODE:
        if prepared_split_scope != "per_ffid":
            raise ValueError(
                "full_ffid_epoch requires preparation split_scope 'per_ffid', "
                f"got {prepared_split_scope!r}"
            )
        expected["split_scope"] = "per_ffid"
    mismatched = [
        key for key, expected_value in expected.items() if preparation.get(key) != expected_value
    ]
    if mismatched:
        raise ValueError(
            f"{PREPARATION_FILE_NAME} does not match the resolved configuration: {mismatched}"
        )
    return expected


def _configured_split_scope(config: Mapping[str, object]) -> object:
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        return "global"
    return sampling.get("split_scope", "global")


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
        hidden_omega=get_required_config_value(config, "model.hidden_omega"),
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
    source_files: tuple[dict[str, str], ...] | None = None,
    training_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    inputs_lock: dict[str, object] = {
        "interim_files": dict(interim_files),
        "processed_files": dict(processed_files),
        "preparation": dict(preparation_contract),
    }
    if source_files is not None:
        inputs_lock["source_files"] = [dict(source) for source in source_files]
    if training_contract is not None:
        inputs_lock["training"] = dict(training_contract)
    return inputs_lock


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
