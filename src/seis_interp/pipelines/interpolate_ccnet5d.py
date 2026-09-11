"""Per-volume observed-only CCNet-5D PoC fitting and interpolation."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
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
from seis_interp.ccnet5d_poc_config import CCNet5DPocSettings, validate_ccnet5d_poc_config
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import (
    C3_RANDOM80_POC_DATASET_ID,
    load_c3_random80_poc_inputs,
)
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.processing.c3_benchmark_contract import (
    MAIN_C3_DIMENSIONS,
    C3BenchmarkDimensions,
)
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_observed_training import (
    CCNet5DObservedTrainingResult,
    train_ccnet5d_observed_steps,
)
from seis_interp.training.ccnet5d_poc_checkpoints import (
    CCNET5D_POC_CHECKPOINT_ROLE,
    CCNET5D_POC_METHOD_VARIANT,
    save_ccnet5d_poc_checkpoint,
)
from seis_interp.training.ccnet5d_prediction import (
    CCNet5DVolumePrediction,
    predict_ccnet5d_volume,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.randomness import seed_global_model_initialization

METHOD = "ccnet5d"
TRAINING_DOMAIN = "O_with_inner_pseudo_mask"
PREDICTION_RELATIVE_PATH = Path("prediction.npy")
CHECKPOINT_RELATIVE_PATH = Path("final.pt")
ProgressReporter = Callable[[str], None]


def interpolate_ccnet5d_run(
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
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict[str, object]:
    """Fit on pseudo-masked observed traces, predict all targets, and evaluate."""
    end_to_end_started = time.perf_counter()
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    settings = validate_ccnet5d_poc_config(config)
    benchmark_seed = _validate_shared_config(config)
    requested_device = settings.training.device if device_override is None else device_override
    device = resolve_device(requested_device, config_key="training.device")
    config["training"]["device"] = str(device)
    started_at_utc = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    timings: dict[str, float] = {}

    _report(progress_reporter, "Loading and verifying observed-only C3 PoC inputs.")
    started = time.perf_counter()
    inputs = load_c3_random80_poc_inputs(
        config=config,
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        mask_dir=Path(mask_dir),
        case_dir=Path(case_dir),
        volume_dir=Path(volume_dir),
        dimensions=dimensions,
    )
    timings["load_and_verification_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Computing the shared global RMS from observed traces only.")
    started = time.perf_counter()
    amplitude_scale = compute_observed_global_rms(
        inputs.observed_volume.values,
        inputs.observed_volume.observed_trace_mask,
    )
    source = CCNet5DObservedPatchSource(
        inputs.observed_volume,
        amplitude_scale=amplitude_scale,
        patch_shape=settings.patches.shape,
        inner_mask_fraction=settings.patches.inner_mask_fraction,
        placement_seed=settings.patches.placement_seed,
        inner_mask_seed=settings.patches.inner_mask_seed,
    )
    timings["training_data_seconds"] = time.perf_counter() - started

    seed_global_model_initialization(settings.training.model_initialization_seed, device=device)
    model = CCNet5D(**settings.model)
    _report(progress_reporter, "Training CCNet-5D with observed-only pseudo-target traces.")
    _synchronize_before_timing(device)
    started = time.perf_counter()
    trained = train_ccnet5d_observed_steps(
        model,
        source,
        device=device,
        loss_name=settings.training.loss,
        optimizer_updates=settings.training.max_steps,
        learning_rate=settings.training.learning_rate,
        report_every_steps=settings.training.report_interval,
        reporter=progress_reporter,
    )
    _synchronize(device)
    timings["training_seconds"] = time.perf_counter() - started

    _report(progress_reporter, "Predicting every target from the complete observed set.")
    started = time.perf_counter()
    predicted = predict_ccnet5d_volume(
        model,
        inputs.observed_volume,
        amplitude_rms=amplitude_scale,
        core_shape=settings.prediction_core_shape,
        device=device,
    )
    _synchronize(device)
    timings["prediction_seconds"] = time.perf_counter() - started
    target_coverage = _target_coverage_mask(predicted, inputs)

    # This is the first boundary that may materialize evaluation-target amplitudes.
    _report(progress_reporter, "Evaluating target-only physical amplitudes.")
    started = time.perf_counter()
    validate_poc_prediction(predicted.values, inputs.observed_volume)
    evaluation = evaluate_c3_volume_prediction(
        predicted.values,
        inputs.observed_volume,
        interim_dir=Path(interim_dir),
        volume_metadata=inputs.volume_metadata,
        target_coverage_mask=target_coverage,
    )
    timings["evaluation_seconds"] = time.perf_counter() - started
    metrics = dict(evaluation)
    metrics.update(
        {
            "method": METHOD,
            "method_variant": CCNET5D_POC_METHOD_VARIANT,
            "training_domain": TRAINING_DOMAIN,
            "normalization": {
                "type": "global_rms",
                "source": "O_only",
                "scale": amplitude_scale,
            },
            "loss": settings.training.loss,
            "inner_mask_fraction": settings.patches.inner_mask_fraction,
            "checkpoint_role": CCNET5D_POC_CHECKPOINT_ROLE,
            "output_amplitude_domain": "physical",
            "case_id": inputs.case["case_id"],
            "volume_id": inputs.volume_metadata["volume_id"],
            "optimizer_updates": trained.steps_completed,
            "training": {
                "steps_completed": trained.steps_completed,
                "final_loss": trained.final_loss,
                "history": [dict(row) for row in trained.history],
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

    _report(progress_reporter, "Writing final checkpoint, prediction, and run records.")
    output.mkdir(parents=True, exist_ok=False)
    save_ccnet5d_poc_checkpoint(
        output / CHECKPOINT_RELATIVE_PATH,
        model,
        loss=settings.training.loss,
        amplitude_scale=amplitude_scale,
        patch_shape=settings.patches.shape,
        inner_mask_fraction=settings.patches.inner_mask_fraction,
        placement_seed=settings.patches.placement_seed,
        inner_mask_seed=settings.patches.inner_mask_seed,
        model_initialization_seed=settings.training.model_initialization_seed,
        optimizer_updates=trained.steps_completed,
    )
    np.save(output / PREDICTION_RELATIVE_PATH, predicted.values, allow_pickle=False)
    checkpoint_sha256 = file_sha256(output / CHECKPOINT_RELATIVE_PATH)
    prediction_sha256 = file_sha256(output / PREDICTION_RELATIVE_PATH)
    timings["end_to_end_seconds"] = time.perf_counter() - end_to_end_started
    finished_at_utc = run_records.utc_timestamp()
    metadata = _run_metadata(
        inputs=inputs,
        settings=settings,
        model=model,
        trained=trained,
        predicted=predicted,
        amplitude_scale=amplitude_scale,
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
            target_coverage,
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


def _validate_shared_config(config: Mapping[str, object]) -> int:
    project = config_values.exact_section(config, "project", {"random_seed"})
    data = config_values.exact_section(config, "data", {"dataset_id"})
    if data["dataset_id"] != C3_RANDOM80_POC_DATASET_ID:
        raise ValueError(f"data.dataset_id must be {C3_RANDOM80_POC_DATASET_ID!r}")
    config_values.exact_section(config, "benchmark_case", {"id"})
    config_values.exact_section(config, "benchmark_volume", {"id", "selection"})
    config_values.exact_section(
        config,
        "interpolation_mask",
        {"partition", "kind", "missing_fraction"},
    )
    return config_values.nonnegative_integer(project["random_seed"], "project.random_seed")


def _target_coverage_mask(
    predicted: CCNet5DVolumePrediction,
    inputs: C3VolumeRunInputs,
) -> np.ndarray:
    target = inputs.observed_volume.evaluation_target_trace_mask
    counts = predicted.coverage_counts
    if not isinstance(counts, np.ndarray) or counts.shape != target.shape:
        raise ValueError("CCNet-5D trace coverage counts must match the target mask")
    coverage = counts > 0
    if not np.all(coverage[target]):
        raise ValueError("CCNet-5D prediction must cover every evaluation target trace")
    return np.ascontiguousarray(coverage, dtype=np.bool_)


def _run_metadata(
    *,
    inputs: C3VolumeRunInputs,
    settings: CCNet5DPocSettings,
    model: CCNet5D,
    trained: CCNet5DObservedTrainingResult,
    predicted: CCNet5DVolumePrediction,
    amplitude_scale: float,
    device: torch.device,
    benchmark_seed: int,
    timings: Mapping[str, float],
    git_metadata: Mapping[str, str | bool],
    started_at_utc: str,
    finished_at_utc: str,
    checkpoint_sha256: str,
    prediction_sha256: str,
) -> dict[str, object]:
    target = inputs.observed_volume.evaluation_target_trace_mask
    target_counts = predicted.coverage_counts[target]
    return {
        "method": METHOD,
        "method_variant": CCNET5D_POC_METHOD_VARIANT,
        "training_domain": TRAINING_DOMAIN,
        **git_metadata,
        "case_id": inputs.case["case_id"],
        "volume_id": inputs.volume_metadata["volume_id"],
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": "success",
        "device": str(device),
        "random_seed": benchmark_seed,
        "model_initialization_seed": settings.training.model_initialization_seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "input": _input_metadata(inputs),
        "normalization": {
            "type": "global_rms",
            "source": "O_only",
            "scale": amplitude_scale,
            "target_amplitudes_used_for_scale": False,
        },
        "loss": settings.training.loss,
        "model": {
            **model.constructor_config(),
            "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "padding": "same_zero",
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "patches": {
            "shape": list(settings.patches.shape),
            "sampling": "random_uniform_eligible_observed_patch",
            "inner_mask_fraction": settings.patches.inner_mask_fraction,
            "placement_seed": settings.patches.placement_seed,
            "inner_mask_seed": settings.patches.inner_mask_seed,
            "pseudo_target_source": "O_only",
            "whole_trace": True,
        },
        "training": {
            "domain": TRAINING_DOMAIN,
            "optimizer": "adam",
            "optimizer_updates": trained.steps_completed,
            "learning_rate": settings.training.learning_rate,
            "final_loss": trained.final_loss,
            "history": [dict(row) for row in trained.history],
            "validation": False,
            "best_checkpoint_selection": False,
        },
        "checkpoint": {
            "artifact": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "sha256": checkpoint_sha256,
            "role": CCNET5D_POC_CHECKPOINT_ROLE,
        },
        "prediction": {
            "artifact": PREDICTION_RELATIVE_PATH.as_posix(),
            "sha256": prediction_sha256,
            "axis_order": list(VOLUME_AXIS_ORDER),
            "shape": list(predicted.values.shape),
            "dtype": predicted.values.dtype.name,
            "output_amplitude_domain": "physical",
            "core_shape": list(settings.prediction_core_shape),
            "halo_radius": model.halo_radius,
            "tile_count": predicted.tile_count,
            "maximum_input_shape": list(predicted.maximum_input_shape),
            "observed_data_consistency": "hard_reinsertion_after_model_prediction",
            "iterative_pseudo_labeling": False,
        },
        "coverage": {
            "target_trace_count": int(np.count_nonzero(target)),
            "covered_target_trace_count": int(np.count_nonzero(target_counts > 0)),
            "minimum_target_coverage_count": int(target_counts.min()),
            "maximum_target_coverage_count": int(target_counts.max()),
            "boundary_targets_included": True,
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
    volume = inputs.volume_metadata
    counts = volume["role_counts"]
    mask = inputs.case["mask"]
    assert isinstance(counts, Mapping)
    assert isinstance(mask, Mapping)
    target_count = int(counts["evaluation_target"])
    return {
        "dataset_id": inputs.case["dataset_id"],
        "partition": inputs.case["partition"],
        "mask": {
            "kind": mask["kind"],
            "random_seed": mask["random_seed"],
            "requested_missing_fraction": mask["missing_fraction"],
        },
        "selected_volume": {
            "selection": deepcopy(volume["selection"]),
            "shape": list(volume["shape"]),
            "observed_trace_count": int(counts["observed"]),
            "evaluation_target_trace_count": target_count,
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
