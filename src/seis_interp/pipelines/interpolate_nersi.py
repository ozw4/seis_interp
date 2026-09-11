"""Fit an observed-only profile-wise NeRSI on one verified C3 volume."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from seis_interp import config_values, run_records
from seis_interp.c3_poc_run_records import (
    METADATA_FILE_NAME,
    poc_compute_metadata,
    poc_coverage_metadata,
    poc_run_metadata,
    validate_poc_prediction,
)
from seis_interp.configuration import get_required_config_value, load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.models.nersi import Nersi
from seis_interp.nersi_config import NersiPocSettings, validate_nersi_poc_config
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.c3_volume_nersi_data import (
    PROFILE_AXIS_ORDER,
    PROFILE_COORDINATE_ORDER,
    C3VolumeNersiData,
    build_c3_volume_nersi_data,
)
from seis_interp.training.c3_volume_nersi_prediction import (
    C3VolumeNersiPrediction,
    predict_c3_volume_nersi,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.fixed_step_nersi import FixedStepNersiResult, train_nersi_fixed_steps
from seis_interp.training.nersi_checkpoints import (
    COORDINATE_NORMALIZATION,
    FIXED_STEP_FINAL_CHECKPOINT_ROLE,
    NERSI_METHOD_VARIANT,
    nersi_checkpoint_input_binding,
    save_fixed_step_nersi_checkpoint,
)
from seis_interp.training.randomness import seed_global_model_initialization

METHOD = "nersi"
METHOD_VARIANT = NERSI_METHOD_VARIANT
PREDICTION_RELATIVE_PATH = Path("prediction.npy")
CHECKPOINT_RELATIVE_PATH = Path("final.pt")
ProgressReporter = Callable[[str], None]


def interpolate_nersi_run(
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
    """Fit observed profiles, predict the full volume, then score target truth."""
    end_to_end_started = time.perf_counter()
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    settings = validate_nersi_poc_config(config)
    benchmark_seed = config_values.nonnegative_integer(
        get_required_config_value(config, "project.random_seed"), "project.random_seed"
    )
    requested_device = settings.training.device if device_override is None else device_override
    device = resolve_device(requested_device)
    config["training"]["device"] = str(device)
    started_at_utc = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    timings: dict[str, float] = {}

    _report(progress_reporter, "Loading and verifying C3 inputs.")
    started = time.perf_counter()
    inputs = load_c3_random80_poc_inputs(
        config=config,
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        mask_dir=Path(mask_dir),
        case_dir=Path(case_dir),
        volume_dir=Path(volume_dir),
    )
    timings["load_and_verification_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Building observed-only NeRSI profile data.")
    started = time.perf_counter()
    observed = inputs.observed_volume
    amplitude_scale = compute_observed_global_rms(
        observed.values,
        observed.observed_trace_mask,
    )
    data = build_c3_volume_nersi_data(observed, amplitude_scale=amplitude_scale)
    timings["training_data_seconds"] = time.perf_counter() - started
    model_config = settings.model_constructor_config(data.profile_shape)
    seed_global_model_initialization(settings.training.model_initialization_seed, device=device)
    model = Nersi(**model_config)

    _report(progress_reporter, "Training NeRSI on observed benchmark profile samples.")
    _synchronize_before_timing(device)
    started = time.perf_counter()
    trained = train_nersi_fixed_steps(
        model,
        data,
        device=device,
        loss_name=settings.training.loss,
        learning_rate=settings.training.learning_rate,
        profiles_per_step=settings.training.profiles_per_step,
        max_steps=settings.training.max_steps,
        report_interval=settings.training.report_interval,
        random_seed=settings.training.sampling_seed,
        reporter=progress_reporter,
    )
    _synchronize(device)
    timings["fit_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Predicting the complete benchmark volume.")
    started = time.perf_counter()
    predicted = predict_c3_volume_nersi(
        model,
        data,
        inputs.observed_volume,
        batch_size=settings.prediction_batch_size,
        device=device,
    )
    _synchronize(device)
    timings["prediction_seconds"] = time.perf_counter() - started
    target_coverage_mask = _validate_complete_prediction(
        predicted,
        inputs.observed_volume,
    )

    # This is the first call that can materialize evaluation-target amplitudes.
    _report(progress_reporter, "Evaluating reconstruction on evaluation-target traces.")
    started = time.perf_counter()
    validate_poc_prediction(predicted.values, inputs.observed_volume)
    evaluation = evaluate_c3_volume_prediction(
        predicted.values,
        inputs.observed_volume,
        interim_dir=Path(interim_dir),
        volume_metadata=inputs.volume_metadata,
        target_coverage_mask=target_coverage_mask,
    )
    timings["evaluation_seconds"] = time.perf_counter() - started
    metrics = dict(evaluation)
    metrics.update(
        {
            "method": METHOD,
            "method_variant": METHOD_VARIANT,
            "case_id": inputs.case["case_id"],
            "volume_id": inputs.volume_metadata["volume_id"],
            "training": {
                "steps_completed": trained.steps_completed,
                "optimizer_updates": trained.steps_completed,
                "final_batch_loss": trained.final_batch_loss,
                "history": [dict(record) for record in trained.history],
            },
            "timing": {
                "fit_seconds": timings["fit_seconds"],
                "prediction_seconds": timings["prediction_seconds"],
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_updates": trained.steps_completed,
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
    save_fixed_step_nersi_checkpoint(
        output / CHECKPOINT_RELATIVE_PATH,
        model,
        data,
        global_step=trained.steps_completed,
        final_batch_loss=trained.final_batch_loss,
        input_binding=nersi_checkpoint_input_binding(inputs.inputs_lock),
        model_initialization_seed=settings.training.model_initialization_seed,
        sampling_seed=settings.training.sampling_seed,
    )
    np.save(output / PREDICTION_RELATIVE_PATH, predicted.values, allow_pickle=False)
    checkpoint_sha256 = file_sha256(output / CHECKPOINT_RELATIVE_PATH)
    prediction_sha256 = file_sha256(output / PREDICTION_RELATIVE_PATH)
    timings["end_to_end_seconds"] = time.perf_counter() - end_to_end_started
    metrics["timing"]["end_to_end_seconds"] = timings["end_to_end_seconds"]
    finished_at_utc = run_records.utc_timestamp()
    metadata = _run_metadata(
        inputs=inputs,
        data=data,
        model=model,
        settings=settings,
        trained=trained,
        predicted=predicted,
        device=device,
        benchmark_seed=benchmark_seed,
        timings=timings,
        git_metadata=git_metadata,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        checkpoint_sha256=checkpoint_sha256,
        prediction_sha256=prediction_sha256,
    )
    metadata = poc_run_metadata(
        inputs.inputs_lock,
        metadata,
        normalization=metadata["normalization"],
        objective=settings.training.loss,
        operation=metadata["training"],
        coverage=poc_coverage_metadata(
            inputs.observed_volume.evaluation_target_trace_mask,
            target_coverage_mask,
            time_sample_count=len(inputs.observed_volume.time_s),
        ),
        compute=poc_compute_metadata(
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            optimizer_updates=trained.steps_completed,
            supervised_trace_presentations=trained.supervised_trace_presentations,
        ),
    )
    run_records.write_run_outputs(
        output,
        deepcopy(config),
        inputs.inputs_lock,
        evaluation,
        metadata,
        metadata_file_name=METADATA_FILE_NAME,
    )
    return metrics


def _run_metadata(
    *,
    inputs: C3VolumeRunInputs,
    data: C3VolumeNersiData,
    model: Nersi,
    settings: NersiPocSettings,
    trained: FixedStepNersiResult,
    predicted: C3VolumeNersiPrediction,
    device: torch.device,
    benchmark_seed: int,
    timings: Mapping[str, float],
    git_metadata: Mapping[str, str | bool],
    started_at_utc: str,
    finished_at_utc: str,
    checkpoint_sha256: str,
    prediction_sha256: str,
) -> dict[str, object]:
    training = asdict(settings.training)
    del training["device"]
    observed_trace_count = int(np.count_nonzero(data.observed_trace_mask))
    target_trace_count = int(np.count_nonzero(inputs.observed_volume.evaluation_target_trace_mask))
    profile_count = int(data.normalized_coordinates.shape[0])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "method": METHOD,
        "fit_domain": "O_only",
        "inner_corruption_mask": False,
        "normalization": {
            "type": "global_rms",
            "source": "O_only",
            "scale": data.amplitude_scale,
        },
        "loss": settings.training.loss,
        "checkpoint_role": "final",
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
        "model_initialization_seed": settings.training.model_initialization_seed,
        "sampling_seed": settings.training.sampling_seed,
        "input": _input_metadata(inputs),
        "profiles": {
            "coordinate_order": list(PROFILE_COORDINATE_ORDER),
            "coordinate_normalization": COORDINATE_NORMALIZATION,
            "coordinate_normalization_source": "fixed_analysis_domain",
            "coordinate_bounds": [list(bounds) for bounds in data.coordinate_bounds],
            "axis_order": list(PROFILE_AXIS_ORDER),
            "shape": list(data.profile_shape),
            "count": profile_count,
            "training_profile_count": int(len(data.training_profile_indices)),
            "stable_order": "C_order_source_line_shot_in_line_relative_receiver_x",
        },
        "amplitude": {
            "scaling": "observed_volume_global_rms",
            "training_domain": "benchmark_observed_samples",
            "scale_source": "observed_volume_trace_samples_only",
            "amplitude_scale": data.amplitude_scale,
            "target_amplitudes_used_for_scale": False,
        },
        "model": {
            **model.constructor_config(),
            "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        },
        "parameter_count": parameter_count,
        "training": {
            "optimizer": "adam",
            "loss": settings.training.loss,
            **training,
            "steps_completed": trained.steps_completed,
            "optimizer_updates": trained.steps_completed,
            "sampling": "profiles_without_replacement_per_optimizer_update",
            "stopping_rule": "fixed_optimizer_steps",
            "candidate_profile_count": int(len(data.training_profile_indices)),
            "observed_trace_count": observed_trace_count,
            "observed_sample_count": observed_trace_count * data.profile_shape[0],
            "input_dtype": "float32",
            "target_dtype": "float32",
        },
        "prediction": {
            "artifact": PREDICTION_RELATIVE_PATH.as_posix(),
            "sha256": prediction_sha256,
            "batch_size": settings.prediction_batch_size,
            "profile_count": predicted.predicted_profile_count,
            "axis_order": list(VOLUME_AXIS_ORDER),
            "shape": list(predicted.values.shape),
            "dtype": predicted.values.dtype.name,
            "amplitude_domain": "physical",
            "observed_data_consistency": "hard_reinsertion_after_model_prediction",
        },
        "coverage": {
            "target_trace_count": target_trace_count,
            "covered_target_trace_count": target_trace_count,
            "target_coverage_fraction": 1.0,
            "uncovered_trace_count": 0,
            "uncovered_sample_count": 0,
        },
        "checkpoint": {
            "artifact": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "sha256": checkpoint_sha256,
            "role": FIXED_STEP_FINAL_CHECKPOINT_ROLE,
            "scope": "one_verified_case_volume_only",
            "input_binding": nersi_checkpoint_input_binding(inputs.inputs_lock),
            "training_resume_supported": False,
        },
        "timing": {
            "fit_seconds": timings["fit_seconds"],
            "prediction_seconds": timings["prediction_seconds"],
            "end_to_end_seconds": timings["end_to_end_seconds"],
        },
        "paper_alignment": {
            "paper_specified": {
                "fourier_components_per_coordinate": 40,
                "encoder_fully_connected_layers": 2,
                "decoder_blocks": 3,
                "upsample_scales": list(model.upsample_scales),
                "nuclear_norm": False,
            },
            "repository_reimplementation_choices": {
                "configured_fourier_components_per_coordinate": model.fourier_components,
                "frequency_base": model.frequency_base,
                "activation": model.activation,
                "output_activation": model.output_activation,
                "amplitude_scaling": "observed_volume_global_rms",
            },
            "claim": "NeRSI (repository reimplementation), not an official implementation",
        },
        "resources": {
            **timings,
            **run_records.runtime_resource_metadata(device),
            "process_max_rss_scope": "whole_process",
            "torch_num_threads": torch.get_num_threads(),
        },
        "warnings": [],
    }


def _validate_complete_prediction(
    prediction: C3VolumeNersiPrediction,
    observed: ObservedC3Volume,
) -> np.ndarray:
    if not isinstance(prediction, C3VolumeNersiPrediction):
        raise TypeError("prediction must be a C3VolumeNersiPrediction")
    if not isinstance(observed, ObservedC3Volume):
        raise TypeError("observed must be an ObservedC3Volume")
    values = prediction.values
    observed_values = observed.values
    target_mask = observed.evaluation_target_trace_mask
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 5
        or values.dtype.kind not in "fiu"
        or values.dtype.kind == "b"
    ):
        raise ValueError("NeRSI prediction must be a real five-dimensional NumPy array")
    if not isinstance(observed_values, np.ndarray) or observed_values.ndim != 5:
        raise ValueError("observed values must be a five-dimensional NumPy array")
    if values.shape != observed_values.shape:
        raise ValueError("NeRSI prediction shape must match the full analysis domain")
    expected_profile_count = int(np.prod(values.shape[1:-1], dtype=np.int64))
    if prediction.predicted_profile_count != expected_profile_count:
        raise ValueError("NeRSI prediction must evaluate every analysis-domain profile")
    if target_mask.dtype != np.bool_ or target_mask.shape != values.shape[1:]:
        raise ValueError("evaluation target mask must match the prediction spatial shape")
    coverage = np.all(np.isfinite(values), axis=0)
    if not np.all(coverage[target_mask]):
        raise ValueError("NeRSI prediction must provide finite values for every target trace")
    return np.ascontiguousarray(coverage, dtype=np.bool_)


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


def _synchronize_before_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _report(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)
