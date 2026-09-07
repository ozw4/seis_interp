"""Fit an observed-only SIREN on one verified C3 benchmark volume."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from seis_interp import config_values, run_records
from seis_interp.configuration import (
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs, load_c3_volume_run_inputs
from seis_interp.evaluation.c3_volume_metrics import (
    evaluate_c3_volume_prediction,
    validate_c3_volume_evaluation_config,
)
from seis_interp.models.siren import Siren
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training.c3_volume_siren_data import (
    VOLUME_AMPLITUDE_SCALE_SOURCE,
    VOLUME_COORDINATE_SCOPE,
    C3VolumeSirenData,
    build_c3_volume_siren_data,
    build_c3_volume_siren_sampler,
)
from seis_interp.training.c3_volume_siren_prediction import (
    C3VolumeSirenPrediction,
    predict_c3_volume_siren,
)
from seis_interp.training.checkpoints import (
    FIXED_STEP_FINAL_CHECKPOINT_ROLE,
    VOLUME_SIREN_METHOD_VARIANT,
    save_fixed_step_siren_checkpoint,
)
from seis_interp.training.devices import resolve_device as _resolve_device
from seis_interp.training.fixed_step_siren import FixedStepSirenResult, train_siren_fixed_steps
from seis_interp.training.randomness import seed_global_model_initialization

METHOD = "siren_5d"
METHOD_VARIANT = VOLUME_SIREN_METHOD_VARIANT
PREDICTION_RELATIVE_PATH = Path("artifacts") / "prediction.npy"
CHECKPOINT_RELATIVE_PATH = Path("artifacts") / "final.pt"
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class _TrainingSettings:
    random_seed: int
    learning_rate: float
    batch_size: int
    max_steps: int
    report_interval: int
    device: str


def interpolate_siren_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Train for fixed steps, query all coordinates, then evaluate held-out truth."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    model_config = _model_constructor_config(config)
    settings = _training_settings(config)
    prediction_settings = _exact_section(config, "prediction", {"batch_size"})
    prediction_batch_size = config_values.positive_integer(
        prediction_settings["batch_size"], "prediction.batch_size"
    )
    validate_c3_volume_evaluation_config(config)
    benchmark_seed = config_values.nonnegative_integer(
        get_required_config_value(config, "project.random_seed"), "project.random_seed"
    )
    device = _resolve_device(settings.device if device_override is None else device_override)
    config["training"]["device"] = str(device)
    started_at_utc = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    timings: dict[str, float] = {}

    _report(progress_reporter, "Loading and verifying C3 inputs.")
    started = time.perf_counter()
    inputs = load_c3_volume_run_inputs(
        config=config,
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        mask_dir=Path(mask_dir),
        case_dir=Path(case_dir),
        volume_dir=Path(volume_dir),
    )
    timings["load_and_verification_seconds"] = time.perf_counter() - started
    observed = inputs.observed_volume

    _report(
        progress_reporter, "Building volume-local SIREN coordinates and observed training data."
    )
    started = time.perf_counter()
    data = build_c3_volume_siren_data(observed, inputs.index_table)
    timings["training_data_seconds"] = time.perf_counter() - started
    seed_global_model_initialization(settings.random_seed, device=device)
    sampler = build_c3_volume_siren_sampler(data, random_seed=settings.random_seed)
    model = Siren(**model_config)

    _report(progress_reporter, "Training SIREN on observed benchmark samples.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    trained = train_siren_fixed_steps(
        model,
        sampler,
        device=device,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        max_steps=settings.max_steps,
        report_interval=settings.report_interval,
        reporter=progress_reporter,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["training_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Predicting the complete benchmark volume.")
    started = time.perf_counter()
    predicted = predict_c3_volume_siren(
        model, data, observed, batch_size=prediction_batch_size, device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["prediction_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Evaluating reconstruction on evaluation-target traces.")
    started = time.perf_counter()
    metrics = evaluate_c3_volume_prediction(
        predicted.values,
        observed,
        interim_dir=Path(interim_dir),
        volume_metadata=inputs.volume_metadata,
    )
    timings["evaluation_seconds"] = time.perf_counter() - started
    metrics.update(
        {
            "method": METHOD,
            "method_variant": METHOD_VARIANT,
            "case_id": inputs.case["case_id"],
            "volume_id": inputs.volume_metadata["volume_id"],
            "training": {
                "steps_completed": trained.steps_completed,
                "final_batch_loss": trained.final_batch_loss,
                "history": [dict(record) for record in trained.history],
            },
            "observed_model_rmse_before_reinsertion": (
                predicted.observed_model_rmse_before_reinsertion
            ),
            "observed_model_max_abs_error_before_reinsertion": (
                predicted.observed_model_max_abs_error_before_reinsertion
            ),
            "uncovered_trace_count": 0,
            "uncovered_sample_count": 0,
            "warnings": [],
        }
    )

    _report(progress_reporter, "Writing final checkpoint, prediction, and immutable run records.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts").mkdir(exist_ok=False)
    save_fixed_step_siren_checkpoint(
        output / CHECKPOINT_RELATIVE_PATH,
        model,
        data.normalization,
        data.model_coordinates,
        global_step=trained.steps_completed,
        final_batch_loss=trained.final_batch_loss,
    )
    np.save(output / PREDICTION_RELATIVE_PATH, predicted.values, allow_pickle=False)
    finished_at_utc = run_records.utc_timestamp()
    metadata = _run_metadata(
        inputs=inputs,
        data=data,
        model=model,
        model_config=model_config,
        settings=settings,
        trained=trained,
        predicted=predicted,
        prediction_batch_size=prediction_batch_size,
        device=device,
        benchmark_seed=benchmark_seed,
        timings=timings,
        git_metadata=git_metadata,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
    )
    run_records.write_run_outputs(output, deepcopy(config), inputs.inputs_lock, metrics, metadata)
    return metrics


def _exact_section(config: Mapping[str, object], name: str, keys: set[str]) -> Mapping[str, object]:
    section = config.get(name)
    if not isinstance(section, Mapping) or set(section) != keys:
        raise ConfigurationError(f"{name} configuration must contain exactly {sorted(keys)}")
    return section


def _model_constructor_config(config: Mapping[str, object]) -> dict[str, object]:
    section = _exact_section(
        config,
        "model",
        {
            "name",
            "coordinate_features",
            "input_features",
            "hidden_width",
            "hidden_layers",
            "output_features",
            "omega_0",
            "hidden_omega",
            "layer_omega_schedule",
            "skip_connections",
        },
    )
    config_values.require_exact(config, "model.name", "siren")
    config_values.require_exact(config, "model.coordinate_features", "cmp_offset_azimuth")
    constructor = {
        key: config_values.positive_integer(section[key], f"model.{key}")
        for key in ("input_features", "hidden_width", "hidden_layers", "output_features")
    }
    if constructor["input_features"] != 6 or constructor["output_features"] != 1:
        raise ConfigurationError("model must have input_features=6 and output_features=1")
    constructor.update(
        {
            key: config_values.positive_float(section[key], f"model.{key}")
            for key in ("omega_0", "hidden_omega")
        }
    )
    schedule, skip = section["layer_omega_schedule"], section["skip_connections"]
    if schedule is not None and (schedule != "exponential" or constructor["hidden_layers"] < 2):
        raise ConfigurationError("model.layer_omega_schedule requires exponential and >=2 layers")
    if skip is not None and skip != "dense":
        raise ConfigurationError("model.skip_connections must be null or dense")
    constructor["layer_omega_schedule"] = schedule
    constructor["skip_connections"] = skip
    return constructor


def _training_settings(config: Mapping[str, object]) -> _TrainingSettings:
    section = _exact_section(
        config,
        "training",
        {
            "random_seed",
            "optimizer",
            "loss",
            "learning_rate",
            "batch_size",
            "max_steps",
            "report_interval",
            "device",
        },
    )
    config_values.require_exact(config, "training.optimizer", "adam")
    config_values.require_exact(config, "training.loss", "l2")
    device = section["device"]
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("training.device must be a non-empty string")
    return _TrainingSettings(
        random_seed=config_values.nonnegative_integer(
            section["random_seed"], "training.random_seed"
        ),
        learning_rate=config_values.positive_float(
            section["learning_rate"], "training.learning_rate"
        ),
        batch_size=config_values.positive_integer(section["batch_size"], "training.batch_size"),
        max_steps=config_values.positive_integer(section["max_steps"], "training.max_steps"),
        report_interval=config_values.positive_integer(
            section["report_interval"], "training.report_interval"
        ),
        device=device,
    )


def _run_metadata(
    *,
    inputs: C3VolumeRunInputs,
    data: C3VolumeSirenData,
    model: Siren,
    model_config: Mapping[str, object],
    settings: _TrainingSettings,
    trained: FixedStepSirenResult,
    predicted: C3VolumeSirenPrediction,
    prediction_batch_size: int,
    device: torch.device,
    benchmark_seed: int,
    timings: Mapping[str, float],
    git_metadata: Mapping[str, str | bool],
    started_at_utc: str,
    finished_at_utc: str,
) -> dict[str, object]:
    training = asdict(settings)
    del training["device"]
    return {
        "method": METHOD,
        "method_variant": METHOD_VARIANT,
        "case_id": inputs.case["case_id"],
        "volume_id": inputs.volume_metadata["volume_id"],
        **git_metadata,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": "success",
        "device": str(device),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "random_seed": benchmark_seed,
        "input": _input_metadata(inputs),
        "coordinates": {
            "features": data.model_coordinates.coordinate_features,
            "order": list(data.model_coordinates.coordinate_order),
            "scale_scope": VOLUME_COORDINATE_SCOPE,
            "coordinate_min": list(data.normalization.coordinate_min),
            "coordinate_max": list(data.normalization.coordinate_max),
            "time_coordinate_scale": data.model_coordinates.time_coordinate_scale,
        },
        "amplitude": {
            "scaling": "train_global_rms",
            "training_domain": "benchmark_observed_samples",
            "scale_source": VOLUME_AMPLITUDE_SCALE_SOURCE,
            "amplitude_rms": data.normalization.amplitude_rms,
        },
        "model": {
            **model_config,
            "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "optimizer": "adam",
            "loss": "l2",
            **training,
            # The shared to_model_tensors boundary converts both arrays to float32.
            "input_dtype": "float32",
            "target_dtype": "float32",
            "steps_completed": trained.steps_completed,
            "sampling": "uniform_observed_points_with_replacement",
            "stopping_rule": "fixed_optimizer_steps",
            "observed_trace_count": len(data.observed_flat_indices),
            "observed_sample_count": data.normalized_observed_amplitudes.size,
        },
        "prediction": {
            "artifact": PREDICTION_RELATIVE_PATH.as_posix(),
            "batch_size": prediction_batch_size,
            "point_count": predicted.predicted_point_count,
            "axis_order": list(VOLUME_AXIS_ORDER),
            "shape": list(predicted.values.shape),
            "dtype": predicted.values.dtype.name,
            "observed_data_consistency": "hard_reinsertion_after_model_prediction",
        },
        "checkpoint": {
            "artifact": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "role": FIXED_STEP_FINAL_CHECKPOINT_ROLE,
        },
        "resources": {
            **timings,
            **run_records.runtime_resource_metadata(device),
            "process_max_rss_scope": "whole_process",
            "torch_num_threads": torch.get_num_threads(),
        },
        "warnings": [],
    }


def _input_metadata(inputs: C3VolumeRunInputs) -> dict[str, object]:
    case, volume = inputs.case, inputs.volume_metadata
    mask, counts = case["mask"], volume["role_counts"]
    assert isinstance(mask, Mapping) and isinstance(counts, Mapping)
    target_count = int(counts["evaluation_target"])
    return {
        "dataset_id": case["dataset_id"],
        "partition": case["partition"],
        "mask": {
            "kind": mask["kind"],
            "random_seed": mask["random_seed"],
            "requested_missing_fraction": mask["missing_fraction"],
        },
        "selected_volume": {
            "selection": deepcopy(volume["selection"]),
            "shape": list(inputs.observed_volume.values.shape),
            "dtype": inputs.observed_volume.values.dtype.name,
            "observed_trace_count": int(counts["observed"]),
            "evaluation_target_trace_count": target_count,
            "actual_missing_trace_count": target_count,
            "actual_missing_fraction": target_count / int(volume["trace_count"]),
        },
    }


def _report(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)
