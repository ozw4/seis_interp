"""Run the full-time-domain trace-count scaling diagnostic."""

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
    median_trace_signal_to_noise_ratio_db,
    signal_to_noise_ratio_db,
    trace_signal_to_noise_ratio_db,
)
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
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import RandomPointSampler, build_trace_points
from seis_interp.training.prediction import predict_points
from seis_interp.training.trainer import build_loss

STUDY_ID = "study_004_domain_scaling"
EXPERIMENT_ID = "experiment_a"
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


def deterministic_nested_trace_subsets(
    training_array_rows: np.ndarray,
    trace_counts: Sequence[int],
    *,
    random_seed: int,
) -> dict[int, np.ndarray]:
    """Return prefix-nested subsets from one permutation of sorted training rows."""
    rows = np.asarray(training_array_rows)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("training_array_rows must be a non-empty one-dimensional array")
    if rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError("training_array_rows must have an integer dtype")
    if np.any(rows < 0):
        raise ValueError("training_array_rows must be non-negative")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("training_array_rows must not contain duplicates")

    counts = _validated_trace_counts(trace_counts)
    if counts[-1] > len(rows):
        raise ValueError(
            f"requested trace count {counts[-1]} exceeds {len(rows)} available training rows"
        )
    seed = _validated_random_seed(random_seed)
    permutation = np.random.default_rng(seed).permutation(np.sort(rows.astype(np.int64)))
    return {count: permutation[:count].copy() for count in counts}


def evaluate_training_fit(
    model: Siren,
    training_coordinates: np.ndarray,
    training_targets: np.ndarray,
    *,
    trace_count: int,
    sample_count: int,
    prediction_batch_size: int,
    device: torch.device | str,
) -> dict[str, float]:
    """Evaluate full-trace fit metrics for one selected training subset."""
    predictions = predict_points(
        model,
        training_coordinates,
        batch_size=prediction_batch_size,
        device=device,
    )
    expected_points = trace_count * sample_count
    if training_targets.shape != (expected_points,):
        raise ValueError(
            f"training_targets must have shape ({expected_points},), got {training_targets.shape}"
        )
    trace_targets = training_targets.reshape(trace_count, sample_count)
    trace_predictions = predictions.reshape(trace_count, sample_count)
    trace_snr_db = trace_signal_to_noise_ratio_db(trace_targets, trace_predictions)
    if trace_snr_db.shape != (trace_count,):
        raise RuntimeError("trace S/N calculation returned an unexpected shape")

    target_float = training_targets.astype(np.float64, copy=False)
    prediction_float = predictions.astype(np.float64, copy=False)
    target_rms = float(np.sqrt(np.mean(np.square(target_float), dtype=np.float64)))
    if target_rms == 0.0:
        raise ValueError("training target RMS must be greater than zero")
    prediction_rms = float(np.sqrt(np.mean(np.square(prediction_float), dtype=np.float64)))
    metrics = {
        "training_median_trace_snr_db": median_trace_signal_to_noise_ratio_db(
            trace_targets, trace_predictions
        ),
        "training_global_snr_db": signal_to_noise_ratio_db(training_targets, predictions),
        "training_prediction_target_rms_ratio": prediction_rms / target_rms,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("training fit metrics must be finite")
    return metrics


def run_trace_count_condition(
    *,
    config: Mapping[str, object],
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    normalized_amplitudes: np.ndarray,
    selected_array_rows: np.ndarray,
    device: torch.device | str,
) -> dict[str, object]:
    """Train one fresh SIREN for a fixed number of updates and report training fit."""
    random_seed = _validated_random_seed(get_required_config_value(config, "project.random_seed"))
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = _build_model(config)
    model.to(device)
    sampler = RandomPointSampler(
        normalized_time,
        normalized_spatial_by_array_row,
        normalized_amplitudes,
        selected_array_rows,
        random_seed=random_seed,
    )
    training_coordinates, training_targets = build_trace_points(
        normalized_time,
        normalized_spatial_by_array_row,
        normalized_amplitudes,
        selected_array_rows,
    )
    loss_function = build_loss(get_required_config_value(config, "training.loss"))
    learning_rate = _positive_finite_float(
        get_required_config_value(config, "training.learning_rate"),
        "training.learning_rate",
    )
    batch_size = _positive_integer(
        get_required_config_value(config, "training.batch_size"),
        "training.batch_size",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.steps_per_epoch"),
        "training.steps_per_epoch",
    )
    max_epochs = _positive_integer(
        get_required_config_value(config, "training.max_epochs"),
        "training.max_epochs",
    )
    prediction_batch_size = _positive_integer(
        get_required_config_value(config, "training.validation_batch_size"),
        "training.validation_batch_size",
    )
    total_updates = report_interval * max_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    interval_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    history: list[dict[str, int | float]] = []
    best_row: dict[str, int | float] | None = None

    model.train()
    for step in range(1, total_updates + 1):
        batch_coordinates, batch_targets = sampler.sample(batch_size)
        coordinate_tensor, target_tensor = to_model_tensors(
            batch_coordinates,
            batch_targets,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinate_tensor)
        loss = loss_function(prediction, target_tensor)
        loss.backward()
        optimizer.step()
        interval_loss_sum += loss.detach().to(dtype=torch.float64)

        if step % report_interval != 0:
            continue
        mean_loss = float((interval_loss_sum / report_interval).cpu().item())
        if not math.isfinite(mean_loss):
            raise ValueError(f"mean training loss is non-finite at step {step}")
        fit_metrics = evaluate_training_fit(
            model,
            training_coordinates,
            training_targets,
            trace_count=len(selected_array_rows),
            sample_count=len(normalized_time),
            prediction_batch_size=prediction_batch_size,
            device=device,
        )
        row: dict[str, int | float] = {
            "step": step,
            "mean_train_loss_since_last_report": mean_loss,
            **fit_metrics,
        }
        history.append(row)
        if (
            best_row is None
            or row["training_median_trace_snr_db"] > best_row["training_median_trace_snr_db"]
        ):
            best_row = dict(row)
        interval_loss_sum.zero_()

    if best_row is None or not history:
        raise RuntimeError("training produced no report points")
    final_row = history[-1]
    result: dict[str, object] = {
        "trace_count": int(len(selected_array_rows)),
        "selected_array_rows": [int(value) for value in selected_array_rows],
        "updates_completed": total_updates,
        "best_step": best_row["step"],
        "best_training_median_trace_snr_db": best_row["training_median_trace_snr_db"],
        "best_training_global_snr_db": best_row["training_global_snr_db"],
        "best_training_prediction_target_rms_ratio": best_row[
            "training_prediction_target_rms_ratio"
        ],
        "final_training_median_trace_snr_db": final_row["training_median_trace_snr_db"],
        "final_training_global_snr_db": final_row["training_global_snr_db"],
        "final_training_prediction_target_rms_ratio": final_row[
            "training_prediction_target_rms_ratio"
        ],
        "history": history,
    }
    result["classification"] = classify_training_fit(result)
    return result


def run_experiment_a(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run all configured nested trace-count conditions and write one summary."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device
    _validate_experiment_contract(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    trace_counts = _validated_trace_counts(
        get_required_config_value(resolved_config, "experiment_a.trace_counts")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    condition_paths = {
        count: output_directory / f"{run_prefix}_trace{count:03d}" for count in trace_counts
    }
    summary_path = output_directory / f"{run_prefix}_experiment_a_summary.json"
    _preflight_output_paths(output_directory, (*condition_paths.values(), summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    available_training_rows = len(experiment_data["training_array_rows"])
    if trace_counts[-1] != available_training_rows:
        raise ValueError(
            f"largest configured trace count {trace_counts[-1]} must equal all "
            f"{available_training_rows} available training rows"
        )
    subsets = deterministic_nested_trace_subsets(
        experiment_data["training_array_rows"],
        trace_counts,
        random_seed=random_seed,
    )
    summary_runs: list[dict[str, object]] = []
    for trace_count in trace_counts:
        started_at_utc = _utc_timestamp()
        selected_rows = subsets[trace_count]
        metrics = run_trace_count_condition(
            config=resolved_config,
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=experiment_data["normalized_amplitudes"],
            selected_array_rows=selected_rows,
            device=device,
        )
        run_metadata = {
            "study_id": STUDY_ID,
            "experiment": EXPERIMENT_ID,
            "trace_count": trace_count,
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
            trace_counts=trace_counts,
            trace_count=trace_count,
            sample_count=experiment_data["sample_count"],
            selected_array_rows=selected_rows,
            random_seed=random_seed,
        )
        output_path = condition_paths[trace_count]
        _write_condition_outputs(
            output_path,
            resolved_config,
            inputs_lock,
            metrics,
            run_metadata,
        )
        summary_runs.append(_summary_run(output_path.name, metrics))

    conclusion = _experiment_conclusion(summary_runs)
    summary: dict[str, object] = {
        "study_id": STUDY_ID,
        "experiment": EXPERIMENT_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "runs": summary_runs,
        "conclusion": conclusion,
    }
    _write_json(summary_path, summary)
    return summary


def classify_training_fit(metrics: Mapping[str, object]) -> str:
    """Classify one condition using the Experiment A diagnostic thresholds."""
    median_snr = float(metrics["best_training_median_trace_snr_db"])
    rms_ratio = float(metrics["best_training_prediction_target_rms_ratio"])
    if median_snr >= 20.0:
        return "strong_fit"
    if median_snr > 1.0 and rms_ratio > 0.1:
        return "escaped_zero_predictor"
    return "near_zero"


def _load_experiment_data(
    interim_directory: Path,
    processed_directory: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
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


def _build_model(config: Mapping[str, object]) -> Siren:
    return Siren(
        input_features=get_required_config_value(config, "model.input_features"),
        hidden_width=get_required_config_value(config, "model.hidden_width"),
        hidden_layers=get_required_config_value(config, "model.hidden_layers"),
        output_features=1,
        omega_0=get_required_config_value(config, "model.omega_0"),
    )


def _validate_experiment_contract(config: Mapping[str, object]) -> None:
    model_name = get_required_config_value(config, "model.name")
    if model_name != "siren":
        raise ConfigurationError(f"model.name must be 'siren', got {model_name!r}")
    input_features = get_required_config_value(config, "model.input_features")
    if input_features != len(MODEL_COORDINATE_ORDER):
        raise ConfigurationError(
            f"model.input_features must be {len(MODEL_COORDINATE_ORDER)}, got {input_features!r}"
        )
    loss = get_required_config_value(config, "training.loss")
    if loss != "l2":
        raise ConfigurationError(f"training.loss must be 'l2', got {loss!r}")
    optimizer = get_required_config_value(config, "training.optimizer")
    if optimizer != "adam":
        raise ConfigurationError(f"training.optimizer must be 'adam', got {optimizer!r}")


def _build_inputs_lock(
    *,
    interim_files: Mapping[str, object],
    processed_files: Mapping[str, object],
    preparation_contract: Mapping[str, object],
    trace_counts: Sequence[int],
    trace_count: int,
    sample_count: int,
    selected_array_rows: np.ndarray,
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
            "configured_trace_counts": [int(value) for value in trace_counts],
            "trace_count": trace_count,
            "sample_count": sample_count,
            "full_time_domain": True,
            "selected_array_rows": [int(value) for value in selected_array_rows],
        },
    }


def _summary_run(run_id: str, metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "trace_count",
        "best_step",
        "best_training_median_trace_snr_db",
        "best_training_global_snr_db",
        "best_training_prediction_target_rms_ratio",
        "final_training_median_trace_snr_db",
        "final_training_global_snr_db",
        "final_training_prediction_target_rms_ratio",
        "classification",
    )
    return {"run_id": run_id, **{key: metrics[key] for key in keys}}


def _experiment_conclusion(summary_runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    strong_counts = [
        int(run["trace_count"]) for run in summary_runs if run["classification"] == "strong_fit"
    ]
    escaped_counts = [
        int(run["trace_count"]) for run in summary_runs if run["classification"] != "near_zero"
    ]
    first_returned_to_near_zero: int | None = None
    escaped_smaller_subset = False
    for run in summary_runs:
        if run["classification"] != "near_zero":
            escaped_smaller_subset = True
        elif escaped_smaller_subset:
            first_returned_to_near_zero = int(run["trace_count"])
            break
    return {
        "largest_strong_fit_trace_count": max(strong_counts, default=None),
        "largest_escaped_zero_predictor_trace_count": max(escaped_counts, default=None),
        "first_larger_subset_returned_to_near_zero": first_returned_to_near_zero,
        "scaling_boundary_observed": first_returned_to_near_zero is not None,
    }


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


def _validated_trace_counts(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("experiment_a.trace_counts must be a sequence of positive integers")
    counts = tuple(_positive_integer(value, "experiment_a.trace_counts") for value in values)
    if not counts:
        raise ValueError("experiment_a.trace_counts must not be empty")
    if any(right <= left for left, right in zip(counts, counts[1:], strict=False)):
        raise ValueError("experiment_a.trace_counts must be strictly increasing")
    return counts


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
