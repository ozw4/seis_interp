"""Frozen supervised CCNet-5D inference on a verified C3 benchmark volume."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from seis_interp import config_values, run_records
from seis_interp.configuration import ConfigurationError, load_resolved_config
from seis_interp.data.c3_volume_run_inputs import load_c3_volume_run_inputs
from seis_interp.data.ccnet5d_benchmark_inputs import validate_ccnet5d_benchmark_provenance
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import (
    evaluate_c3_volume_prediction,
    validate_c3_volume_evaluation_config,
)
from seis_interp.models.ccnet5d import ccnet5d_method_variant
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.processing.ccnet5d_tiles import validate_ccnet5d_shape
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.devices import resolve_device

METHOD = "ccnet5d"
TRAINING_REGIME = "supervised_train_partition"
PREDICTION_RELATIVE_PATH = Path("artifacts") / "prediction.npy"


def interpolate_ccnet5d_run(
    *,
    config_path: Path,
    checkpoint_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Load a saved network once, predict observed input, then evaluate target truth."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    if "model" in config or "training" in config:
        raise ConfigurationError(
            "CCNet inference model/training conditions come only from checkpoint"
        )
    prediction_settings = config_values.exact_section(
        config, "prediction", {"device", "core_shape"}
    )
    core = validate_ccnet5d_shape(prediction_settings["core_shape"], "prediction.core_shape")
    config_values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    validate_c3_volume_evaluation_config(config)
    project = config_values.exact_section(config, "project", {"random_seed"})
    seed = config_values.nonnegative_integer(project["random_seed"], "project.random_seed")
    data = config_values.exact_section(config, "data", {"dataset_id"})
    config_values.exact_section(config, "benchmark_case", {"id"})
    config_values.exact_section(config, "benchmark_volume", {"id", "selection"})
    config_values.exact_section(
        config, "interpolation_mask", {"partition", "kind", "missing_fraction"}
    )
    device = resolve_device(
        prediction_settings["device"] if device_override is None else device_override,
        config_key="prediction.device",
    )
    config["prediction"]["device"] = str(device)
    started_at = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    timings = {}
    if progress_reporter:
        progress_reporter("Loading checkpoint and verifying disjoint C3 benchmark inputs.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    checkpoint_hash = file_sha256(Path(checkpoint_path))
    loaded = load_ccnet5d_checkpoint(Path(checkpoint_path), device=device)
    warnings = []
    if loaded.training_provenance["training_run"]["git_worktree_dirty"]:
        warnings.append(
            "Training checkpoint was created from a dirty Git worktree; "
            "this inference run is nonformal."
        )
    inputs = load_c3_volume_run_inputs(
        config=config,
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        mask_dir=Path(mask_dir),
        case_dir=Path(case_dir),
        volume_dir=Path(volume_dir),
    )
    if data["dataset_id"] != inputs.case["dataset_id"]:
        raise ConfigurationError("data.dataset_id does not match the verified benchmark")
    validate_ccnet5d_benchmark_provenance(loaded.training_provenance, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["load_and_verification_seconds"] = time.perf_counter() - started
    if progress_reporter:
        progress_reporter(
            "Predicting with the frozen checkpoint and clipped receptive-field halos."
        )
    started = time.perf_counter()
    predicted = predict_ccnet5d_volume(
        loaded.model,
        inputs.observed_volume,
        amplitude_rms=loaded.amplitude_rms,
        core_shape=core,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["prediction_seconds"] = time.perf_counter() - started
    if progress_reporter:
        progress_reporter("Evaluating target-only physical amplitudes.")
    started = time.perf_counter()
    metrics = evaluate_c3_volume_prediction(
        predicted.values,
        inputs.observed_volume,
        interim_dir=Path(interim_dir),
        volume_metadata=inputs.volume_metadata,
    )
    timings["evaluation_seconds"] = time.perf_counter() - started
    identity = {
        "method": METHOD,
        "method_variant": ccnet5d_method_variant(loaded.model.output_activation),
        "training_regime": TRAINING_REGIME,
    }
    metrics.update(
        {
            **identity,
            "case_id": inputs.case["case_id"],
            "volume_id": inputs.volume_metadata["volume_id"],
            "observed_model_rmse_before_reinsertion": (
                predicted.observed_model_rmse_before_reinsertion
            ),
            "observed_model_max_abs_error_before_reinsertion": (
                predicted.observed_model_max_abs_error_before_reinsertion
            ),
            "uncovered_trace_count": 0,
            "uncovered_sample_count": 0,
            "warnings": warnings,
        }
    )
    checkpoint_record = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "training_provenance": loaded.training_provenance,
    }
    inputs_lock = {**deepcopy(inputs.inputs_lock), "checkpoint": checkpoint_record}
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts").mkdir()
    np.save(output / PREDICTION_RELATIVE_PATH, predicted.values, allow_pickle=False)
    finished_at = run_records.utc_timestamp()
    volume = inputs.volume_metadata
    counts = volume["role_counts"]
    mask = inputs.case["mask"]
    metadata = {
        **identity,
        **git_metadata,
        "case_id": inputs.case["case_id"],
        "volume_id": volume["volume_id"],
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "status": "success",
        "device": str(device),
        "random_seed": seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "input": {
            "dataset_id": inputs.case["dataset_id"],
            "partition": inputs.case["partition"],
            "mask": {
                "kind": mask["kind"],
                "random_seed": mask["random_seed"],
                "requested_missing_fraction": mask["missing_fraction"],
            },
            "selected_volume": {
                "selection": volume["selection"],
                "shape": list(predicted.values.shape),
                "dtype": predicted.values.dtype.name,
                "observed_trace_count": counts["observed"],
                "evaluation_target_trace_count": counts["evaluation_target"],
                "actual_missing_fraction": counts["evaluation_target"] / volume["trace_count"],
            },
        },
        "model": {
            **loaded.model.constructor_config(),
            "parameter_dtype": str(next(loaded.model.parameters()).dtype).removeprefix("torch."),
            "padding": "same_zero",
            "module_count": 4,
        },
        "parameter_count": sum(parameter.numel() for parameter in loaded.model.parameters()),
        "amplitude": {
            "scale_source": "checkpoint_fit_region",
            "amplitude_rms": loaded.amplitude_rms,
        },
        "checkpoint": {
            **checkpoint_record,
            "role": loaded.checkpoint_role,
            "epoch": loaded.epoch,
            "global_step": loaded.global_step,
        },
        "prediction": {
            "artifact": PREDICTION_RELATIVE_PATH.as_posix(),
            "input_dtype": "float32",
            "dtype": predicted.values.dtype.name,
            "axis_order": list(VOLUME_AXIS_ORDER),
            "shape": list(predicted.values.shape),
            "point_count": predicted.values.size,
            "core_shape": list(core),
            "halo_radius": loaded.model.halo_radius,
            "tile_count": predicted.tile_count,
            "maximum_input_shape": list(predicted.maximum_input_shape),
            "observed_data_consistency": "hard_reinsertion_after_model_prediction",
        },
        "resources": {
            **timings,
            **run_records.runtime_resource_metadata(device),
            "process_max_rss_scope": "whole_process",
            "torch_num_threads": torch.get_num_threads(),
        },
        "warnings": warnings,
    }
    run_records.write_run_outputs(output, config, inputs_lock, metrics, metadata)
    return metrics
