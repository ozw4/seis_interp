"""Run the focused full-batch trace-correlation-loss ablation."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
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
from seis_interp.evaluation.metrics import (
    median_trace_correlation_coefficient,
    median_trace_signal_to_noise_ratio_db,
    signal_to_noise_ratio_db,
)
from seis_interp.models.siren import Siren
from seis_interp.pipelines.domain_scaling import deterministic_nested_trace_subsets
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
from seis_interp.training.correlation_loss import trace_correlation_loss
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import build_trace_points

STUDY_ID = "study_005_correlation_loss_ablation"
CONFIG_FILE_NAME = "config.resolved.yaml"
INPUTS_LOCK_FILE_NAME = "inputs.lock.json"
METRICS_FILE_NAME = "metrics.json"
RUN_FILE_NAME = "run.json"
PROCESSED_INPUT_FILE_NAMES = (
    TRACE_SPLIT_FILE_NAME,
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
)
_SELECTION_METHOD = "sorted_training_rows_single_numpy_permutation_prefix"
_EXPECTED_CONDITIONS = (
    ("mse_control", 0.0),
    ("mse_corr_0p1", 0.1),
)


def run_correlation_loss_ablation(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run both fixed full-batch conditions and write their immutable records."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    condition_paths = {
        label: output_directory / f"{run_prefix}_{label}" for label, _ in experiment["conditions"]
    }
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (*condition_paths.values(), summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    trace_count = experiment["trace_count"]
    selected_rows = deterministic_nested_trace_subsets(
        experiment_data["training_array_rows"],
        (trace_count,),
        random_seed=random_seed,
    )[trace_count]
    training_coordinates, training_targets = build_trace_points(
        experiment_data["normalized_time"],
        experiment_data["normalized_spatial_by_array_row"],
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    coordinate_tensor, target_tensor = to_model_tensors(
        training_coordinates,
        training_targets,
        device=device,
    )
    sample_count = experiment_data["sample_count"]
    expected_point_count = trace_count * sample_count
    if coordinate_tensor.shape != (expected_point_count, len(MODEL_COORDINATE_ORDER)):
        raise RuntimeError(
            f"full-batch coordinates have shape {tuple(coordinate_tensor.shape)}, "
            f"expected {(expected_point_count, len(MODEL_COORDINATE_ORDER))}"
        )
    if target_tensor.shape != (expected_point_count, 1):
        raise RuntimeError(
            f"full-batch targets have shape {tuple(target_tensor.shape)}, "
            f"expected {(expected_point_count, 1)}"
        )

    summary_runs: list[dict[str, object]] = []
    for label, correlation_weight in experiment["conditions"]:
        started_at_utc = _utc_timestamp()
        metrics = _run_condition(
            config=resolved_config,
            label=label,
            correlation_weight=correlation_weight,
            correlation_eps=experiment["correlation_eps"],
            total_updates=experiment["total_updates"],
            report_interval=experiment["report_interval"],
            coordinate_tensor=coordinate_tensor,
            target_tensor=target_tensor,
            training_targets=training_targets,
            selected_array_rows=selected_rows,
            sample_count=sample_count,
            device=device,
            random_seed=random_seed,
        )
        run_metadata = {
            "study_id": STUDY_ID,
            "condition": label,
            "correlation_weight": correlation_weight,
            "trace_count": trace_count,
            "sample_count": sample_count,
            "full_batch": True,
            "git_commit": git_commit,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_timestamp(),
            "status": "success",
            "device": device,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "random_seed": random_seed,
            "updates_completed": metrics["updates_completed"],
        }
        inputs_lock = _build_inputs_lock(
            interim_files=experiment_data["interim_files"],
            processed_files=experiment_data["processed_files"],
            preparation_contract=experiment_data["preparation_contract"],
            selected_array_rows=selected_rows,
            trace_count=trace_count,
            sample_count=sample_count,
            random_seed=random_seed,
        )
        condition_config = _resolved_condition_config(
            resolved_config,
            label=label,
            correlation_weight=correlation_weight,
        )
        output_path = condition_paths[label]
        _write_condition_outputs(
            output_path,
            condition_config,
            inputs_lock,
            metrics,
            run_metadata,
        )
        summary_runs.append(_summary_run(output_path.name, metrics))

    decision = _ablation_decision(summary_runs)
    summary: dict[str, object] = {
        "study_id": STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "decision": decision,
        "runs": summary_runs,
    }
    _write_json(summary_path, summary)
    return summary


def _run_condition(
    *,
    config: Mapping[str, object],
    label: str,
    correlation_weight: float,
    correlation_eps: float,
    total_updates: int,
    report_interval: int,
    coordinate_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    training_targets: np.ndarray,
    selected_array_rows: np.ndarray,
    sample_count: int,
    device: torch.device | str,
    random_seed: int,
) -> dict[str, object]:
    """Train one fresh model for the fixed number of full-batch updates."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    model = _build_model(config)
    model.to(device)
    learning_rate = _positive_finite_float(
        get_required_config_value(config, "training.learning_rate"),
        "training.learning_rate",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace_count = len(selected_array_rows)
    flat_target = target_tensor.reshape(-1)
    history: list[dict[str, int | float]] = []
    best_row: dict[str, int | float] | None = None

    model.train()
    for step in range(1, total_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        flat_prediction = model(coordinate_tensor).reshape(-1)
        mse_loss = torch_functional.mse_loss(flat_prediction, flat_target)
        correlation_loss = trace_correlation_loss(
            flat_prediction.reshape(trace_count, sample_count),
            flat_target.reshape(trace_count, sample_count),
            eps=correlation_eps,
        )
        total_loss = mse_loss + correlation_weight * correlation_loss
        total_loss.backward()
        optimizer.step()

        if step % report_interval != 0:
            continue
        row = _evaluate_report_point(
            model,
            coordinate_tensor=coordinate_tensor,
            target_tensor=target_tensor,
            training_targets=training_targets,
            trace_count=trace_count,
            sample_count=sample_count,
            correlation_weight=correlation_weight,
            correlation_eps=correlation_eps,
            step=step,
        )
        history.append(row)
        if (
            best_row is None
            or row["training_median_trace_snr_db"] > best_row["training_median_trace_snr_db"]
        ):
            best_row = dict(row)

    if best_row is None or not history:
        raise RuntimeError(f"condition {label!r} produced no report points")
    final_row = history[-1]
    return {
        "label": label,
        "correlation_weight": correlation_weight,
        "selected_array_rows": [int(value) for value in selected_array_rows],
        "updates_completed": total_updates,
        "best_step": best_row["step"],
        "best_training_median_trace_snr_db": best_row["training_median_trace_snr_db"],
        "best_training_global_snr_db": best_row["training_global_snr_db"],
        "best_training_median_trace_correlation": best_row["training_median_trace_correlation"],
        "best_training_prediction_target_rms_ratio": best_row[
            "training_prediction_target_rms_ratio"
        ],
        "final_training_median_trace_snr_db": final_row["training_median_trace_snr_db"],
        "final_training_global_snr_db": final_row["training_global_snr_db"],
        "final_training_median_trace_correlation": final_row["training_median_trace_correlation"],
        "final_training_prediction_target_rms_ratio": final_row[
            "training_prediction_target_rms_ratio"
        ],
        "history": history,
    }


def _evaluate_report_point(
    model: Siren,
    *,
    coordinate_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    training_targets: np.ndarray,
    trace_count: int,
    sample_count: int,
    correlation_weight: float,
    correlation_eps: float,
    step: int,
) -> dict[str, int | float]:
    """Evaluate losses and trace-balanced fit on the complete training subset."""
    model.eval()
    try:
        with torch.no_grad():
            flat_prediction = model(coordinate_tensor).reshape(-1)
            flat_target = target_tensor.reshape(-1)
            mse_loss = torch_functional.mse_loss(flat_prediction, flat_target)
            correlation_loss = trace_correlation_loss(
                flat_prediction.reshape(trace_count, sample_count),
                flat_target.reshape(trace_count, sample_count),
                eps=correlation_eps,
            )
            total_loss = mse_loss + correlation_weight * correlation_loss
            predictions = (
                flat_prediction.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
            )
    finally:
        model.train()

    trace_targets = training_targets.reshape(trace_count, sample_count)
    trace_predictions = predictions.reshape(trace_count, sample_count)
    target_float = training_targets.astype(np.float64, copy=False)
    prediction_float = predictions.astype(np.float64, copy=False)
    target_rms = float(np.sqrt(np.mean(np.square(target_float), dtype=np.float64)))
    if target_rms == 0.0:
        raise ValueError("training target RMS must be greater than zero")
    prediction_rms = float(np.sqrt(np.mean(np.square(prediction_float), dtype=np.float64)))
    row: dict[str, int | float] = {
        "step": step,
        "mse_loss": float(mse_loss.cpu().item()),
        "correlation_loss": float(correlation_loss.cpu().item()),
        "total_loss": float(total_loss.cpu().item()),
        "training_median_trace_snr_db": median_trace_signal_to_noise_ratio_db(
            trace_targets, trace_predictions
        ),
        "training_global_snr_db": signal_to_noise_ratio_db(training_targets, predictions),
        "training_median_trace_correlation": median_trace_correlation_coefficient(
            trace_targets, trace_predictions
        ),
        "training_prediction_target_rms_ratio": prediction_rms / target_rms,
    }
    if not all(math.isfinite(float(value)) for key, value in row.items() if key != "step"):
        raise ValueError(f"training report contains a non-finite value at step {step}")
    return row


def _load_experiment_data(
    interim_directory: Path,
    processed_directory: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Load, lock, validate, and normalize the existing prepared dataset."""
    dataset = load_interim_trace_dataset(interim_directory)
    split_table, normalization, preparation = _load_processed_dataset(processed_directory)
    interim_files = _file_hashes(interim_directory, INTERIM_FILE_NAMES)
    processed_files = _file_hashes(processed_directory, PROCESSED_INPUT_FILE_NAMES)
    split_rows = _validate_split_table(split_table, len(dataset.trace_table))
    split_counts = _split_counts(split_table)
    _validate_preparation_data(preparation, dataset.metadata, split_counts, interim_files)
    preparation_contract = _validated_preparation_contract(preparation, config)
    training_rows = split_rows[split_table[SPLIT_COLUMN].eq(TRAIN_SPLIT).to_numpy(dtype=bool)]

    normalized_time = normalize_time(dataset.time_s, normalization)
    normalized_spatial = normalize_spatial_coordinates(dataset.trace_table, normalization)
    dataset_array_rows = validated_array_rows(dataset.trace_table, require_contiguous=True)
    spatial_by_array_row = np.empty_like(normalized_spatial)
    spatial_by_array_row[dataset_array_rows] = normalized_spatial
    normalized_amplitudes = np.zeros_like(dataset.amplitudes)
    normalized_amplitudes[training_rows] = normalize_amplitudes(
        dataset.amplitudes[training_rows], normalization
    )
    return {
        "normalized_time": normalized_time,
        "normalized_spatial_by_array_row": spatial_by_array_row,
        "normalized_amplitudes": normalized_amplitudes,
        "training_array_rows": training_rows,
        "sample_count": int(len(normalized_time)),
        "interim_files": interim_files,
        "processed_files": processed_files,
        "preparation_contract": preparation_contract,
    }


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


def _validated_experiment_config(config: Mapping[str, object]) -> dict[str, object]:
    if get_required_config_value(config, "model.name") != "siren":
        raise ConfigurationError("model.name must be 'siren'")
    input_features = get_required_config_value(config, "model.input_features")
    if input_features != len(MODEL_COORDINATE_ORDER):
        raise ConfigurationError(f"model.input_features must be {len(MODEL_COORDINATE_ORDER)}")
    if get_required_config_value(config, "training.loss") != "l2":
        raise ConfigurationError("training.loss must be 'l2'")
    if get_required_config_value(config, "training.optimizer") != "adam":
        raise ConfigurationError("training.optimizer must be 'adam'")
    if get_required_config_value(config, "experiment.full_batch") is not True:
        raise ConfigurationError("experiment.full_batch must be true")

    conditions = _validated_conditions(get_required_config_value(config, "experiment.conditions"))
    total_updates = _positive_integer(
        get_required_config_value(config, "experiment.total_updates"),
        "experiment.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "experiment.report_interval"),
        "experiment.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError("experiment.total_updates must be divisible by report_interval")
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "correlation_eps": _positive_finite_float(
            get_required_config_value(config, "experiment.correlation_eps"),
            "experiment.correlation_eps",
        ),
        "conditions": conditions,
    }


def _validated_conditions(value: object) -> tuple[tuple[str, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("experiment.conditions must be a sequence")
    conditions: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ConfigurationError("each experiment condition must be a mapping")
        label = item.get("label")
        if not isinstance(label, str) or not label:
            raise ConfigurationError("condition label must be a non-empty string")
        weight = item.get("correlation_weight")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ConfigurationError("condition correlation_weight must be finite")
        converted_weight = float(weight)
        if not math.isfinite(converted_weight):
            raise ConfigurationError("condition correlation_weight must be finite")
        conditions.append((label, converted_weight))
    converted = tuple(conditions)
    if converted != _EXPECTED_CONDITIONS:
        raise ConfigurationError(
            "experiment.conditions must contain only mse_control=0.0 and mse_corr_0p1=0.1"
        )
    return converted


def _build_model(config: Mapping[str, object]) -> Siren:
    return Siren(
        input_features=get_required_config_value(config, "model.input_features"),
        hidden_width=get_required_config_value(config, "model.hidden_width"),
        hidden_layers=get_required_config_value(config, "model.hidden_layers"),
        output_features=1,
        omega_0=get_required_config_value(config, "model.omega_0"),
        hidden_omega=get_required_config_value(config, "model.hidden_omega"),
    )


def _ablation_decision(summary_runs: Sequence[Mapping[str, object]]) -> str:
    by_label = {str(run["label"]): run for run in summary_runs}
    if set(by_label) != {label for label, _ in _EXPECTED_CONDITIONS}:
        raise ValueError("summary runs must contain both configured conditions")
    control = by_label["mse_control"]
    correlation = by_label["mse_corr_0p1"]
    if _escaped_zero_predictor(control):
        return "full_batch_control_succeeds"
    if _escaped_zero_predictor(correlation):
        return "correlation_loss_promising"

    correlation_snr = float(correlation["best_training_median_trace_snr_db"])
    correlation_coefficient = float(correlation["best_training_median_trace_correlation"])
    correlation_rms_ratio = float(correlation["best_training_prediction_target_rms_ratio"])
    if correlation_rms_ratio > 0.1 and (correlation_snr <= 1.0 and correlation_coefficient <= 0.1):
        return "correlation_loss_inflates_amplitude_without_alignment"
    return "correlation_loss_not_effective"


def _escaped_zero_predictor(metrics: Mapping[str, object]) -> bool:
    values = (
        float(metrics["best_training_median_trace_snr_db"]),
        float(metrics["best_training_median_trace_correlation"]),
        float(metrics["best_training_prediction_target_rms_ratio"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("decision metrics must be finite")
    return values[0] > 1.0 and values[1] > 0.1 and values[2] > 0.1


def _build_inputs_lock(
    *,
    interim_files: Mapping[str, object],
    processed_files: Mapping[str, object],
    preparation_contract: Mapping[str, object],
    selected_array_rows: np.ndarray,
    trace_count: int,
    sample_count: int,
    random_seed: int,
) -> dict[str, object]:
    return {
        "interim_files": dict(interim_files),
        "processed_files": dict(processed_files),
        "preparation": dict(preparation_contract),
        "selection": {
            "source_split": TRAIN_SPLIT,
            "method": _SELECTION_METHOD,
            "random_seed": random_seed,
            "trace_count": trace_count,
            "sample_count": sample_count,
            "full_time_domain": True,
            "full_batch": True,
            "selected_array_rows": [int(value) for value in selected_array_rows],
        },
    }


def _resolved_condition_config(
    config: Mapping[str, object],
    *,
    label: str,
    correlation_weight: float,
) -> dict[str, object]:
    resolved = deepcopy(dict(config))
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment configuration must be a mapping")
    experiment["active_condition"] = {
        "label": label,
        "correlation_weight": correlation_weight,
    }
    return resolved


def _summary_run(run_id: str, metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "label",
        "correlation_weight",
        "updates_completed",
        "best_step",
        "best_training_median_trace_snr_db",
        "best_training_global_snr_db",
        "best_training_median_trace_correlation",
        "best_training_prediction_target_rms_ratio",
        "final_training_median_trace_snr_db",
        "final_training_global_snr_db",
        "final_training_median_trace_correlation",
        "final_training_prediction_target_rms_ratio",
    )
    return {"run_id": run_id, **{key: metrics[key] for key in keys}}


def _write_condition_outputs(
    output_directory: Path,
    config: Mapping[str, object],
    inputs_lock: Mapping[str, object],
    metrics: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> None:
    output_directory.mkdir(parents=True)
    (output_directory / CONFIG_FILE_NAME).write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    _write_json(output_directory / INPUTS_LOCK_FILE_NAME, inputs_lock)
    _write_json(output_directory / METRICS_FILE_NAME, metrics)
    _write_json(output_directory / RUN_FILE_NAME, run_metadata)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _preflight_output_paths(output_root: Path, targets: Sequence[Path]) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(f"output root is not a directory: {output_root}")
    existing = [path.name for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"experiment output paths already exist: {existing}")


def _file_hashes(
    directory: Path,
    file_names: Sequence[str],
) -> dict[str, dict[str, str]]:
    return {file_name: {"sha256": file_sha256(directory / file_name)} for file_name in file_names}


def _validated_random_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("project.random_seed must be an integer")
    seed = int(value)
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("project.random_seed must be within [0, 2**32 - 1]")
    return seed


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


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


def _run_id_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
