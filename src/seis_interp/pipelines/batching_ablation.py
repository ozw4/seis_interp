"""Run focused training-fit batching diagnostics."""

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
from seis_interp.training.point_sampler import (
    RandomPointSampler,
    RandomTraceBatchSampler,
    RandomTracePatchSampler,
    build_trace_points,
    overlapping_patch_starts,
)
from seis_interp.training.prediction import predict_points

STUDY_ID = "study_006_batching_ablation"
FULL_FFID_STUDY_ID = "study_007_full_ffid_large_batch"
FULL_FFID_TRACE_BATCH_STUDY_ID = "study_008_full_ffid_trace_batches"
FULL_FFID_TRACE_BATCH_CORRELATION_STUDY_ID = "study_009_full_ffid_trace_batch_correlation"
FULL_FFID_TEMPORAL_PATCH_STUDY_ID = "study_010_full_ffid_temporal_patches"
TRACE_POOL_CONTINUATION_STUDY_ID = "study_011_trace_pool_continuation"
OFFICIAL_SIREN_BASELINE_STUDY_ID = "study_012_official_siren_baseline"
AMPLITUDE_BALANCING_STUDY_ID = "study_013_amplitude_balancing"
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
_ALL_TRAINING_ROWS_METHOD = "all_training_rows_sorted_by_array_row"
_EXACT_LABEL = "exact_full_batch"
_RANDOM_LABEL = "random_replacement_5000"
_MSE_PLUS_TRACE_CORRELATION = "mse_plus_trace_correlation"
_STUDY_005_CORRELATION_WEIGHT = 0.1
_STUDY_005_CORRELATION_EPS = 1.0e-4
_EXPECTED_CONDITIONS = (
    (_EXACT_LABEL, "exact_full_batch", True, False),
    (_RANDOM_LABEL, "random_replacement", False, True),
)
_OFFICIAL_SIREN_CONDITIONS = (
    ("legacy_control", 300.0, 1.0),
    ("official_siren_30", 30.0, 30.0),
)
_GLOBAL_RMS_SCALING = "global_rms"
_PER_TRACE_RMS_SCALING = "per_trace_rms"
_AMPLITUDE_BALANCING_CONDITIONS = (
    ("global_rms_control", _GLOBAL_RMS_SCALING, "l2"),
    ("per_trace_rms", _PER_TRACE_RMS_SCALING, "l2"),
    ("huber_global_rms", _GLOBAL_RMS_SCALING, "huber"),
)
_AMPLITUDE_BALANCING_HUBER_DELTA = 1.0
FULL_TRACE_BATCH_ABLATION_STUDY_ID = "study_014_full_trace_batch_ablation"
_FULL_TRACE_BATCH_ABLATION_CONDITIONS = (
    ("small_batch_control", "random_replacement", False, _GLOBAL_RMS_SCALING),
    ("full_trace_batch", "random_complete_traces", False, _GLOBAL_RMS_SCALING),
    ("full_trace_batch_correlation", "random_complete_traces", True, _GLOBAL_RMS_SCALING),
    ("full_trace_batch_per_trace_rms", "random_complete_traces", False, _PER_TRACE_RMS_SCALING),
    (
        "full_trace_batch_correlation_per_trace_rms",
        "random_complete_traces",
        True,
        _PER_TRACE_RMS_SCALING,
    ),
)
STRONG_FIT_BUDGET_EXTENSION_STUDY_ID = "study_015_strong_fit_budget_extension"
_STRONG_FIT_BUDGET_EXTENSION_CONDITIONS = (
    (
        "full_trace_batch_per_trace_rms",
        "random_complete_traces",
        False,
        _PER_TRACE_RMS_SCALING,
    ),
)
_STRONG_FIT_MEDIAN_TRACE_SNR_DB = 20.0


def run_batching_ablation(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run both fixed batching conditions and write immutable records."""
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
        label: output_directory / f"{run_prefix}_{label}"
        for label, _, _, _ in experiment["conditions"]
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
    all_coordinate_tensor, all_target_tensor = to_model_tensors(
        training_coordinates,
        training_targets,
        device=device,
    )
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    _validate_full_batch_tensors(all_coordinate_tensor, all_target_tensor, point_count)
    if experiment["batch_size"] != point_count:
        raise ConfigurationError(
            f"training.batch_size must equal trace_count * sample_count ({point_count})"
        )
    point_evaluations = experiment["batch_size"] * experiment["total_updates"]

    summary_runs: list[dict[str, object]] = []
    for label, batch_mode, full_batch, replacement in experiment["conditions"]:
        started_at_utc = _utc_timestamp()
        metrics = run_training_fit_condition(
            config=resolved_config,
            label=label,
            batch_mode=batch_mode,
            full_batch=full_batch,
            replacement=replacement,
            total_updates=experiment["total_updates"],
            report_interval=experiment["report_interval"],
            batch_size=experiment["batch_size"],
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=experiment_data["normalized_amplitudes"],
            selected_array_rows=selected_rows,
            all_coordinate_tensor=all_coordinate_tensor,
            all_target_tensor=all_target_tensor,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=sample_count,
            prediction_batch_size=point_count,
            device=device,
            random_seed=random_seed,
        )
        run_metadata = _build_run_metadata(
            study_id=STUDY_ID,
            condition=label,
            batch_mode=batch_mode,
            batch_size=experiment["batch_size"],
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            point_evaluations=point_evaluations,
            full_batch=full_batch,
            replacement=replacement,
            git_commit=git_commit,
            started_at_utc=started_at_utc,
            device=device,
            random_seed=random_seed,
            updates_completed=metrics["updates_completed"],
        )
        inputs_lock = _build_inputs_lock(
            interim_files=experiment_data["interim_files"],
            processed_files=experiment_data["processed_files"],
            preparation_contract=experiment_data["preparation_contract"],
            selected_array_rows=selected_rows,
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            random_seed=random_seed,
        )
        condition_config = _resolved_condition_config(
            resolved_config,
            label=label,
            batch_mode=batch_mode,
            full_batch=full_batch,
            replacement=replacement,
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

    decision = batching_summary_decision(summary_runs)
    summary: dict[str, object] = {
        "study_id": STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "point_evaluations_per_condition": point_evaluations,
        "decision": decision,
        "runs": summary_runs,
    }
    _write_json(summary_path, summary)
    return summary


def run_full_ffid_large_batch(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run one random-replacement fit probe over every training trace."""
    return _run_full_ffid_fit_probe(
        config_path=config_path,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        output_root=output_root,
        device_override=device_override,
        study_id=FULL_FFID_STUDY_ID,
        expected_batch_mode="random_replacement",
    )


def run_full_ffid_trace_batches(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run one complete-trace mini-batch fit probe over every training trace."""
    return _run_full_ffid_fit_probe(
        config_path=config_path,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        output_root=output_root,
        device_override=device_override,
        study_id=FULL_FFID_TRACE_BATCH_STUDY_ID,
        expected_batch_mode="random_complete_traces",
    )


def run_full_ffid_trace_batch_correlation(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run one complete-trace mini-batch probe with trace correlation loss."""
    return _run_full_ffid_fit_probe(
        config_path=config_path,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        output_root=output_root,
        device_override=device_override,
        study_id=FULL_FFID_TRACE_BATCH_CORRELATION_STUDY_ID,
        expected_batch_mode="random_complete_traces",
        require_trace_correlation_loss=True,
    )


def run_full_ffid_temporal_patches(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run one shared temporal-patch mini-batch probe over every training trace."""
    return _run_full_ffid_fit_probe(
        config_path=config_path,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        output_root=output_root,
        device_override=device_override,
        study_id=FULL_FFID_TEMPORAL_PATCH_STUDY_ID,
        expected_batch_mode="random_shared_temporal_patch",
    )


def run_official_siren_baseline(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run the paired legacy and official-SIREN full-training fit probes."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_official_siren_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    condition_paths = {
        label: output_directory / f"{run_prefix}_{label}"
        for label, _, _ in experiment["conditions"]
    }
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (*condition_paths.values(), summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    selected_rows = np.sort(experiment_data["training_array_rows"]).astype(np.int64, copy=False)
    available_training_rows = len(selected_rows)
    trace_count = experiment["trace_count"]
    if trace_count != available_training_rows:
        raise ValueError(
            f"configured trace count {trace_count} must equal all "
            f"{available_training_rows} available training rows"
        )
    training_coordinates, training_targets = build_trace_points(
        experiment_data["normalized_time"],
        experiment_data["normalized_spatial_by_array_row"],
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    point_evaluations = experiment["batch_size"] * experiment["total_updates"]

    condition_summaries: list[dict[str, object]] = []
    for label, omega_0, hidden_omega in experiment["conditions"]:
        condition_config = _resolved_official_siren_condition_config(
            resolved_config,
            label=label,
            omega_0=omega_0,
            hidden_omega=hidden_omega,
        )
        started_at_utc = _utc_timestamp()
        metrics = run_training_fit_condition(
            config=condition_config,
            label=label,
            batch_mode=experiment["batch_mode"],
            full_batch=False,
            replacement=experiment["replacement"],
            total_updates=experiment["total_updates"],
            report_interval=experiment["report_interval"],
            batch_size=experiment["batch_size"],
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=experiment_data["normalized_amplitudes"],
            selected_array_rows=selected_rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=sample_count,
            prediction_batch_size=experiment["prediction_batch_size"],
            device=device,
            random_seed=random_seed,
        )
        metrics.update({"omega_0": omega_0, "hidden_omega": hidden_omega})
        run_metadata = _build_run_metadata(
            study_id=OFFICIAL_SIREN_BASELINE_STUDY_ID,
            condition=label,
            batch_mode=experiment["batch_mode"],
            batch_size=experiment["batch_size"],
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            point_evaluations=point_evaluations,
            full_batch=False,
            replacement=experiment["replacement"],
            git_commit=git_commit,
            started_at_utc=started_at_utc,
            device=device,
            random_seed=random_seed,
            updates_completed=metrics["updates_completed"],
        )
        run_metadata.update({"omega_0": omega_0, "hidden_omega": hidden_omega})
        inputs_lock = _build_inputs_lock(
            interim_files=experiment_data["interim_files"],
            processed_files=experiment_data["processed_files"],
            preparation_contract=experiment_data["preparation_contract"],
            selected_array_rows=selected_rows,
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            random_seed=random_seed,
            selection_method=_ALL_TRAINING_ROWS_METHOD,
            split_counts=experiment_data["split_counts"],
            training_contract={
                "condition": label,
                "batch_mode": experiment["batch_mode"],
                "replacement": experiment["replacement"],
                "batch_size": experiment["batch_size"],
                "total_updates": experiment["total_updates"],
                "point_evaluations": point_evaluations,
                "omega_0": omega_0,
                "hidden_omega": hidden_omega,
            },
        )
        output_path = condition_paths[label]
        _write_condition_outputs(
            output_path,
            condition_config,
            inputs_lock,
            metrics,
            run_metadata,
        )
        condition_summaries.append(
            _official_siren_condition_summary(
                output_path.name,
                omega_0=omega_0,
                hidden_omega=hidden_omega,
                metrics=metrics,
            )
        )

    classifications = {
        str(condition["label"]): str(condition["classification"])
        for condition in condition_summaries
    }
    legacy_classification = classifications["legacy_control"]
    official_classification = classifications["official_siren_30"]
    summary: dict[str, object] = {
        "study_id": OFFICIAL_SIREN_BASELINE_STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "decision": official_siren_summary_decision(
            legacy_classification=legacy_classification,
            official_classification=official_classification,
        ),
        "control_validity": legacy_classification == "near_zero",
        "point_evaluations_per_condition": point_evaluations,
        "conditions": condition_summaries,
    }
    _write_json(summary_path, summary)
    return summary


def run_amplitude_balancing(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run the paired amplitude-balancing full-training fit probes."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_amplitude_balancing_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    condition_paths = {
        label: output_directory / f"{run_prefix}_{label}"
        for label, _, _ in experiment["conditions"]
    }
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (*condition_paths.values(), summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    selected_rows = np.sort(experiment_data["training_array_rows"]).astype(np.int64, copy=False)
    available_training_rows = len(selected_rows)
    trace_count = experiment["trace_count"]
    if trace_count != available_training_rows:
        raise ValueError(
            f"configured trace count {trace_count} must equal all "
            f"{available_training_rows} available training rows"
        )
    per_trace_amplitudes, per_trace_scales = per_trace_rms_scaled_amplitudes(
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    amplitudes_by_scaling = {
        _GLOBAL_RMS_SCALING: experiment_data["normalized_amplitudes"],
        _PER_TRACE_RMS_SCALING: per_trace_amplitudes,
    }
    trace_points_by_scaling = {
        scaling: build_trace_points(
            experiment_data["normalized_time"],
            experiment_data["normalized_spatial_by_array_row"],
            amplitudes,
            selected_rows,
        )
        for scaling, amplitudes in amplitudes_by_scaling.items()
    }
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    point_evaluations = experiment["batch_size"] * experiment["total_updates"]
    per_trace_scale_stats = {
        "min": float(np.min(per_trace_scales)),
        "median": float(np.median(per_trace_scales)),
        "max": float(np.max(per_trace_scales)),
    }

    condition_summaries: list[dict[str, object]] = []
    for label, amplitude_scaling, loss_name in experiment["conditions"]:
        huber_delta = experiment["huber_delta"] if loss_name == "huber" else None
        condition_config = _resolved_amplitude_balancing_condition_config(
            resolved_config,
            label=label,
            amplitude_scaling=amplitude_scaling,
            loss_name=loss_name,
            huber_delta=huber_delta,
        )
        training_coordinates, training_targets = trace_points_by_scaling[amplitude_scaling]
        started_at_utc = _utc_timestamp()
        metrics = run_training_fit_condition(
            config=condition_config,
            label=label,
            batch_mode=experiment["batch_mode"],
            full_batch=False,
            replacement=experiment["replacement"],
            total_updates=experiment["total_updates"],
            report_interval=experiment["report_interval"],
            batch_size=experiment["batch_size"],
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=amplitudes_by_scaling[amplitude_scaling],
            selected_array_rows=selected_rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=sample_count,
            prediction_batch_size=experiment["prediction_batch_size"],
            device=device,
            random_seed=random_seed,
            loss_name=loss_name,
            huber_delta=huber_delta,
        )
        metrics.update({"amplitude_scaling": amplitude_scaling, "loss_name": loss_name})
        if huber_delta is not None:
            metrics["huber_delta"] = huber_delta
        if amplitude_scaling == _PER_TRACE_RMS_SCALING:
            metrics["per_trace_scale_stats"] = dict(per_trace_scale_stats)
        run_metadata = _build_run_metadata(
            study_id=AMPLITUDE_BALANCING_STUDY_ID,
            condition=label,
            batch_mode=experiment["batch_mode"],
            batch_size=experiment["batch_size"],
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            point_evaluations=point_evaluations,
            full_batch=False,
            replacement=experiment["replacement"],
            git_commit=git_commit,
            started_at_utc=started_at_utc,
            device=device,
            random_seed=random_seed,
            updates_completed=metrics["updates_completed"],
        )
        run_metadata.update({"amplitude_scaling": amplitude_scaling, "loss_name": loss_name})
        if huber_delta is not None:
            run_metadata["huber_delta"] = huber_delta
        training_contract = {
            "condition": label,
            "batch_mode": experiment["batch_mode"],
            "replacement": experiment["replacement"],
            "batch_size": experiment["batch_size"],
            "total_updates": experiment["total_updates"],
            "point_evaluations": point_evaluations,
            "amplitude_scaling": amplitude_scaling,
            "loss_name": loss_name,
        }
        if huber_delta is not None:
            training_contract["huber_delta"] = huber_delta
        inputs_lock = _build_inputs_lock(
            interim_files=experiment_data["interim_files"],
            processed_files=experiment_data["processed_files"],
            preparation_contract=experiment_data["preparation_contract"],
            selected_array_rows=selected_rows,
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            random_seed=random_seed,
            selection_method=_ALL_TRAINING_ROWS_METHOD,
            split_counts=experiment_data["split_counts"],
            training_contract=training_contract,
        )
        output_path = condition_paths[label]
        _write_condition_outputs(
            output_path,
            condition_config,
            inputs_lock,
            metrics,
            run_metadata,
        )
        condition_summaries.append(
            _amplitude_balancing_condition_summary(
                output_path.name,
                amplitude_scaling=amplitude_scaling,
                loss_name=loss_name,
                metrics=metrics,
            )
        )

    classifications = {
        str(condition["label"]): str(condition["classification"])
        for condition in condition_summaries
    }
    control_classification = classifications["global_rms_control"]
    summary: dict[str, object] = {
        "study_id": AMPLITUDE_BALANCING_STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "decision": amplitude_balancing_summary_decision(
            control_classification=control_classification,
            per_trace_classification=classifications["per_trace_rms"],
        ),
        "control_validity": control_classification == "near_zero",
        "point_evaluations_per_condition": point_evaluations,
        "per_trace_scale_stats": dict(per_trace_scale_stats),
        "conditions": condition_summaries,
    }
    _write_json(summary_path, summary)
    return summary


def per_trace_rms_scaled_amplitudes(
    normalized_amplitudes: np.ndarray,
    selected_array_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each selected trace to unit RMS and return the applied scales."""
    amplitudes = np.asarray(normalized_amplitudes)
    if amplitudes.ndim != 2:
        raise ValueError(
            f"normalized_amplitudes must be 2-dimensional, got shape {amplitudes.shape}"
        )
    rows = np.asarray(selected_array_rows)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("selected_array_rows must be a non-empty 1-dimensional array")
    if np.any(rows < 0) or np.any(rows >= amplitudes.shape[0]):
        raise ValueError(
            f"selected_array_rows must be within amplitude row range [0, {amplitudes.shape[0]})"
        )
    selected = amplitudes[rows].astype(np.float64, copy=False)
    trace_rms = np.sqrt(np.mean(np.square(selected), axis=1, dtype=np.float64))
    if not np.all(np.isfinite(trace_rms)) or np.any(trace_rms <= 0.0):
        raise ValueError("per-trace RMS must be positive and finite for every selected trace")
    scaled = amplitudes.copy()
    scaled[rows] = (selected / trace_rms[:, np.newaxis]).astype(scaled.dtype, copy=False)
    return scaled, trace_rms


def run_full_trace_batch_ablation(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run the Study 014 full-trace-batch ingredient ablation on the full training pool."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_full_trace_batch_ablation_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    condition_paths = {
        label: output_directory / f"{run_prefix}_{label}"
        for label, _, _, _ in experiment["conditions"]
    }
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (*condition_paths.values(), summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    selected_rows = np.sort(experiment_data["training_array_rows"]).astype(np.int64, copy=False)
    available_training_rows = len(selected_rows)
    trace_count = experiment["trace_count"]
    if trace_count != available_training_rows:
        raise ValueError(
            f"configured trace count {trace_count} must equal all "
            f"{available_training_rows} available training rows"
        )
    per_trace_amplitudes, per_trace_scales = per_trace_rms_scaled_amplitudes(
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    amplitudes_by_scaling = {
        _GLOBAL_RMS_SCALING: experiment_data["normalized_amplitudes"],
        _PER_TRACE_RMS_SCALING: per_trace_amplitudes,
    }
    trace_points_by_scaling = {
        scaling: build_trace_points(
            experiment_data["normalized_time"],
            experiment_data["normalized_spatial_by_array_row"],
            amplitudes,
            selected_rows,
        )
        for scaling, amplitudes in amplitudes_by_scaling.items()
    }
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    per_trace_scale_stats = {
        "min": float(np.min(per_trace_scales)),
        "median": float(np.median(per_trace_scales)),
        "max": float(np.max(per_trace_scales)),
    }

    condition_summaries: list[dict[str, object]] = []
    for label, batch_mode, uses_correlation, amplitude_scaling in experiment["conditions"]:
        if batch_mode == "random_replacement":
            replacement = True
            batch_size = experiment["batch_size"]
            traces_per_update = None
        else:
            replacement = False
            batch_size = point_count
            traces_per_update = trace_count
        correlation_weight = experiment["correlation_weight"] if uses_correlation else 0.0
        point_evaluations = batch_size * experiment["total_updates"]
        condition_config = _resolved_full_trace_batch_condition_config(
            resolved_config,
            label=label,
            batch_mode=batch_mode,
            correlation_weight=correlation_weight,
            amplitude_scaling=amplitude_scaling,
        )
        training_coordinates, training_targets = trace_points_by_scaling[amplitude_scaling]
        started_at_utc = _utc_timestamp()
        metrics = run_training_fit_condition(
            config=condition_config,
            label=label,
            batch_mode=batch_mode,
            full_batch=False,
            replacement=replacement,
            total_updates=experiment["total_updates"],
            report_interval=experiment["report_interval"],
            batch_size=batch_size,
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=amplitudes_by_scaling[amplitude_scaling],
            selected_array_rows=selected_rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=sample_count,
            prediction_batch_size=experiment["prediction_batch_size"],
            device=device,
            random_seed=random_seed,
            traces_per_update=traces_per_update,
            correlation_weight=correlation_weight,
            correlation_eps=experiment["correlation_eps"],
        )
        metrics["amplitude_scaling"] = amplitude_scaling
        if amplitude_scaling == _PER_TRACE_RMS_SCALING:
            metrics["per_trace_scale_stats"] = dict(per_trace_scale_stats)
        run_metadata = _build_run_metadata(
            study_id=FULL_TRACE_BATCH_ABLATION_STUDY_ID,
            condition=label,
            batch_mode=batch_mode,
            batch_size=batch_size,
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            point_evaluations=point_evaluations,
            full_batch=False,
            replacement=replacement,
            git_commit=git_commit,
            started_at_utc=started_at_utc,
            device=device,
            random_seed=random_seed,
            updates_completed=metrics["updates_completed"],
            traces_per_update=traces_per_update,
            correlation_weight=correlation_weight if uses_correlation else None,
            correlation_eps=experiment["correlation_eps"] if uses_correlation else None,
            loss_semantics=_MSE_PLUS_TRACE_CORRELATION if uses_correlation else None,
        )
        run_metadata["amplitude_scaling"] = amplitude_scaling
        training_contract: dict[str, object] = {
            "condition": label,
            "batch_mode": batch_mode,
            "replacement": replacement,
            "batch_size": batch_size,
            "total_updates": experiment["total_updates"],
            "point_evaluations": point_evaluations,
            "amplitude_scaling": amplitude_scaling,
        }
        if traces_per_update is not None:
            training_contract["traces_per_update"] = traces_per_update
        if uses_correlation:
            training_contract.update(
                {
                    "correlation_weight": correlation_weight,
                    "correlation_eps": experiment["correlation_eps"],
                    "loss_semantics": _MSE_PLUS_TRACE_CORRELATION,
                }
            )
        inputs_lock = _build_inputs_lock(
            interim_files=experiment_data["interim_files"],
            processed_files=experiment_data["processed_files"],
            preparation_contract=experiment_data["preparation_contract"],
            selected_array_rows=selected_rows,
            trace_count=trace_count,
            sample_count=sample_count,
            point_count=point_count,
            random_seed=random_seed,
            selection_method=_ALL_TRAINING_ROWS_METHOD,
            split_counts=experiment_data["split_counts"],
            training_contract=training_contract,
        )
        output_path = condition_paths[label]
        _write_condition_outputs(
            output_path,
            condition_config,
            inputs_lock,
            metrics,
            run_metadata,
        )
        condition_summaries.append(
            _full_trace_batch_condition_summary(
                output_path.name,
                batch_mode=batch_mode,
                correlation_weight=correlation_weight,
                amplitude_scaling=amplitude_scaling,
                metrics=metrics,
            )
        )

    classifications = {
        str(condition["label"]): str(condition["classification"])
        for condition in condition_summaries
    }
    control_classification = classifications["small_batch_control"]
    reproduction_classification = classifications["full_trace_batch_correlation_per_trace_rms"]
    summary: dict[str, object] = {
        "study_id": FULL_TRACE_BATCH_ABLATION_STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "decision": full_trace_batch_ablation_summary_decision(
            control_classification=control_classification,
            full_trace_batch_classification=classifications["full_trace_batch"],
            reproduction_classification=reproduction_classification,
        ),
        "control_validity": control_classification == "near_zero",
        "escape_reproduced": reproduction_classification != "near_zero",
        "correlation_weight": experiment["correlation_weight"],
        "correlation_eps": experiment["correlation_eps"],
        "per_trace_scale_stats": dict(per_trace_scale_stats),
        "conditions": condition_summaries,
    }
    _write_json(summary_path, summary)
    return summary


def run_strong_fit_budget_extension(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run the Study 015 extended-budget strong-fit check for the recommended recipe."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_strong_fit_budget_extension_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    output_directory = Path(output_root)
    ((label, batch_mode, _, amplitude_scaling),) = experiment["conditions"]
    condition_path = output_directory / f"{run_prefix}_{label}"
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (condition_path, summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    selected_rows = np.sort(experiment_data["training_array_rows"]).astype(np.int64, copy=False)
    available_training_rows = len(selected_rows)
    trace_count = experiment["trace_count"]
    if trace_count != available_training_rows:
        raise ValueError(
            f"configured trace count {trace_count} must equal all "
            f"{available_training_rows} available training rows"
        )
    per_trace_amplitudes, per_trace_scales = per_trace_rms_scaled_amplitudes(
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    training_coordinates, training_targets = build_trace_points(
        experiment_data["normalized_time"],
        experiment_data["normalized_spatial_by_array_row"],
        per_trace_amplitudes,
        selected_rows,
    )
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    if experiment["batch_size"] != point_count:
        raise ValueError(
            f"training.batch_size {experiment['batch_size']} must equal the "
            f"{point_count} points of the full complete-trace batch"
        )
    per_trace_scale_stats = {
        "min": float(np.min(per_trace_scales)),
        "median": float(np.median(per_trace_scales)),
        "max": float(np.max(per_trace_scales)),
    }
    point_evaluations = point_count * experiment["total_updates"]

    condition_config = _resolved_full_trace_batch_condition_config(
        resolved_config,
        label=label,
        batch_mode=batch_mode,
        correlation_weight=0.0,
        amplitude_scaling=amplitude_scaling,
    )
    started_at_utc = _utc_timestamp()
    metrics = run_training_fit_condition(
        config=condition_config,
        label=label,
        batch_mode=batch_mode,
        full_batch=False,
        replacement=False,
        total_updates=experiment["total_updates"],
        report_interval=experiment["report_interval"],
        batch_size=point_count,
        normalized_time=experiment_data["normalized_time"],
        normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
        normalized_amplitudes=per_trace_amplitudes,
        selected_array_rows=selected_rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=sample_count,
        prediction_batch_size=experiment["prediction_batch_size"],
        device=device,
        random_seed=random_seed,
        traces_per_update=trace_count,
    )
    metrics["amplitude_scaling"] = amplitude_scaling
    metrics["per_trace_scale_stats"] = dict(per_trace_scale_stats)
    history = metrics["history"]
    baseline_observed = best_median_trace_snr_within(
        history,
        max_step=experiment["baseline_window_updates"],
    )
    baseline_reproduced = (
        abs(baseline_observed - experiment["baseline_best_median_trace_snr_db"])
        <= experiment["baseline_tolerance_db"]
    )
    first_strong_fit_step = first_step_reaching_median_trace_snr(
        history,
        threshold_db=_STRONG_FIT_MEDIAN_TRACE_SNR_DB,
    )

    run_metadata = _build_run_metadata(
        study_id=STRONG_FIT_BUDGET_EXTENSION_STUDY_ID,
        condition=label,
        batch_mode=batch_mode,
        batch_size=point_count,
        trace_count=trace_count,
        sample_count=sample_count,
        point_count=point_count,
        point_evaluations=point_evaluations,
        full_batch=False,
        replacement=False,
        git_commit=git_commit,
        started_at_utc=started_at_utc,
        device=device,
        random_seed=random_seed,
        updates_completed=metrics["updates_completed"],
        traces_per_update=trace_count,
    )
    run_metadata["amplitude_scaling"] = amplitude_scaling
    training_contract: dict[str, object] = {
        "condition": label,
        "batch_mode": batch_mode,
        "replacement": False,
        "batch_size": point_count,
        "total_updates": experiment["total_updates"],
        "point_evaluations": point_evaluations,
        "amplitude_scaling": amplitude_scaling,
        "traces_per_update": trace_count,
    }
    inputs_lock = _build_inputs_lock(
        interim_files=experiment_data["interim_files"],
        processed_files=experiment_data["processed_files"],
        preparation_contract=experiment_data["preparation_contract"],
        selected_array_rows=selected_rows,
        trace_count=trace_count,
        sample_count=sample_count,
        point_count=point_count,
        random_seed=random_seed,
        selection_method=_ALL_TRAINING_ROWS_METHOD,
        split_counts=experiment_data["split_counts"],
        training_contract=training_contract,
    )
    _write_condition_outputs(
        condition_path,
        condition_config,
        inputs_lock,
        metrics,
        run_metadata,
    )
    condition_summary = _full_trace_batch_condition_summary(
        condition_path.name,
        batch_mode=batch_mode,
        correlation_weight=0.0,
        amplitude_scaling=amplitude_scaling,
        metrics=metrics,
    )

    summary: dict[str, object] = {
        "study_id": STRONG_FIT_BUDGET_EXTENSION_STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "decision": strong_fit_budget_extension_summary_decision(
            baseline_reproduced=baseline_reproduced,
            extension_classification=str(metrics["classification"]),
        ),
        "baseline_reproduced": baseline_reproduced,
        "baseline_window_updates": experiment["baseline_window_updates"],
        "baseline_expected_best_median_trace_snr_db": (
            experiment["baseline_best_median_trace_snr_db"]
        ),
        "baseline_observed_best_median_trace_snr_db": baseline_observed,
        "baseline_tolerance_db": experiment["baseline_tolerance_db"],
        "first_strong_fit_step": first_strong_fit_step,
        "per_trace_scale_stats": dict(per_trace_scale_stats),
        "conditions": [condition_summary],
    }
    _write_json(summary_path, summary)
    return summary


def run_trace_pool_continuation(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None = None,
) -> dict[str, object]:
    """Train one model through nested random-replacement trace pools."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_continuation_experiment_config(resolved_config)
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    _validated_random_seed(random_seed + len(experiment["trace_counts"]) - 1)
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    trace_counts = experiment["trace_counts"]
    condition_label = (
        f"continuation{trace_counts[0]}to{trace_counts[-1]}_random{experiment['batch_size']}"
    )
    output_directory = Path(output_root)
    run_path = output_directory / f"{run_prefix}_{condition_label}"
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (run_path, summary_path))

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

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    model = _build_model(resolved_config)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_positive_finite_float(
            get_required_config_value(resolved_config, "training.learning_rate"),
            "training.learning_rate",
        ),
    )

    started_at_utc = _utc_timestamp()
    sample_count = experiment_data["sample_count"]
    stages: list[dict[str, object]] = []
    cumulative_updates = 0
    anchor_reproduced = False
    for stage_index, trace_count in enumerate(trace_counts):
        selected_rows = subsets[trace_count]
        training_coordinates, training_targets = build_trace_points(
            experiment_data["normalized_time"],
            experiment_data["normalized_spatial_by_array_row"],
            experiment_data["normalized_amplitudes"],
            selected_rows,
        )
        entry_metrics = _evaluate_training_fit(
            model,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            trace_count=trace_count,
            sample_count=sample_count,
            prediction_batch_size=experiment["prediction_batch_size"],
            device=device,
            step=cumulative_updates,
        )
        sampler_seed = _validated_random_seed(random_seed + stage_index)
        stage_metrics = run_training_fit_condition(
            config=resolved_config,
            label=f"stage{stage_index + 1:02d}_trace{trace_count}",
            batch_mode=experiment["batch_mode"],
            full_batch=False,
            replacement=experiment["replacement"],
            total_updates=experiment["updates_per_stage"],
            report_interval=experiment["report_interval"],
            batch_size=experiment["batch_size"],
            normalized_time=experiment_data["normalized_time"],
            normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
            normalized_amplitudes=experiment_data["normalized_amplitudes"],
            selected_array_rows=selected_rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=sample_count,
            prediction_batch_size=experiment["prediction_batch_size"],
            device=device,
            random_seed=sampler_seed,
            model=model,
            optimizer=optimizer,
        )
        _add_continuation_stage_context(
            stage_metrics,
            stage_index=stage_index,
            sampler_seed=sampler_seed,
            cumulative_updates_before_stage=cumulative_updates,
            entry_metrics=entry_metrics,
        )
        cumulative_updates += experiment["updates_per_stage"]
        stages.append(stage_metrics)

        if stage_index == 0:
            anchor_reproduced = (
                float(stage_metrics["final_training_median_trace_snr_db"])
                >= experiment["first_stage_final_min_median_trace_snr_db"]
            )
            if not anchor_reproduced:
                break

    completed_stage = stages[-1]
    point_evaluations = experiment["batch_size"] * cumulative_updates
    planned_total_updates = experiment["updates_per_stage"] * len(trace_counts)
    planned_point_evaluations = experiment["batch_size"] * planned_total_updates
    metrics = _summary_run(condition_label, completed_stage)
    metrics.pop("run_id")
    metrics.update(
        {
            "condition": condition_label,
            "planned_stage_trace_counts": list(trace_counts),
            "completed_stage_trace_counts": [int(stage["trace_count"]) for stage in stages],
            "stages_completed": len(stages),
            "updates_per_stage": experiment["updates_per_stage"],
            "planned_total_updates": planned_total_updates,
            "updates_completed": cumulative_updates,
            "planned_point_evaluations": planned_point_evaluations,
            "point_evaluations": point_evaluations,
            "anchor_reproduced": anchor_reproduced,
            "first_stage_final_min_median_trace_snr_db": experiment[
                "first_stage_final_min_median_trace_snr_db"
            ],
            "stages": stages,
        }
    )
    final_full_ffid_classification = str(metrics["classification"]) if anchor_reproduced else None
    metrics.update(
        {
            "classification_scope": (
                "final_full_ffid_stage" if anchor_reproduced else "completed_anchor_stage"
            ),
            "final_full_ffid_classification": final_full_ffid_classification,
        }
    )
    decision = continuation_summary_decision(
        anchor_reproduced=anchor_reproduced,
        final_classification=str(metrics["classification"]),
    )

    training_contract = {
        "batch_mode": experiment["batch_mode"],
        "replacement": experiment["replacement"],
        "batch_size": experiment["batch_size"],
        "updates_per_stage": experiment["updates_per_stage"],
        "report_interval": experiment["report_interval"],
        "prediction_batch_size": experiment["prediction_batch_size"],
        "planned_stage_trace_counts": list(trace_counts),
        "completed_stage_trace_counts": [int(stage["trace_count"]) for stage in stages],
        "sampler_seed_policy": experiment["sampler_seed_policy"],
        "planned_sampler_seeds": [random_seed + index for index in range(len(trace_counts))],
        "completed_sampler_seeds": [int(stage["sampler_seed"]) for stage in stages],
        "carry_model_state": experiment["carry_model_state"],
        "carry_optimizer_state": experiment["carry_optimizer_state"],
        "reset_optimizer_between_stages": experiment["reset_optimizer_between_stages"],
        "rewind_to_best": experiment["rewind_to_best"],
        "checkpoint": experiment["checkpoint"],
        "first_stage_final_min_median_trace_snr_db": experiment[
            "first_stage_final_min_median_trace_snr_db"
        ],
        "planned_total_updates": planned_total_updates,
        "planned_point_evaluations": planned_point_evaluations,
        "updates_completed": cumulative_updates,
        "point_evaluations": point_evaluations,
    }
    completed_trace_count = int(completed_stage["trace_count"])
    completed_rows = subsets[completed_trace_count]
    inputs_lock = _build_inputs_lock(
        interim_files=experiment_data["interim_files"],
        processed_files=experiment_data["processed_files"],
        preparation_contract=experiment_data["preparation_contract"],
        selected_array_rows=completed_rows,
        trace_count=completed_trace_count,
        sample_count=sample_count,
        point_count=completed_trace_count * sample_count,
        random_seed=random_seed,
        split_counts=experiment_data["split_counts"],
        training_contract=training_contract,
    )
    selection = inputs_lock["selection"]
    if not isinstance(selection, dict):
        raise RuntimeError("continuation input selection lock must be a mapping")
    selection.update(
        {
            "planned_trace_counts": list(trace_counts),
            "planned_nested_selected_array_rows": {
                str(trace_count): [int(value) for value in subsets[trace_count]]
                for trace_count in trace_counts
            },
        }
    )

    run_metadata = _build_run_metadata(
        study_id=TRACE_POOL_CONTINUATION_STUDY_ID,
        condition=condition_label,
        batch_mode=experiment["batch_mode"],
        batch_size=experiment["batch_size"],
        trace_count=int(completed_stage["trace_count"]),
        sample_count=sample_count,
        point_count=int(completed_stage["point_count"]),
        point_evaluations=point_evaluations,
        full_batch=False,
        replacement=experiment["replacement"],
        git_commit=git_commit,
        started_at_utc=started_at_utc,
        device=device,
        random_seed=random_seed,
        updates_completed=cumulative_updates,
    )
    run_metadata.update(
        {
            "planned_stage_trace_counts": list(trace_counts),
            "completed_stage_trace_counts": [int(stage["trace_count"]) for stage in stages],
            "stages_completed": len(stages),
            "updates_per_stage": experiment["updates_per_stage"],
            "planned_total_updates": planned_total_updates,
            "planned_point_evaluations": planned_point_evaluations,
            "sampler_seed_policy": experiment["sampler_seed_policy"],
            "planned_sampler_seeds": [random_seed + index for index in range(len(trace_counts))],
            "completed_sampler_seeds": [int(stage["sampler_seed"]) for stage in stages],
            "carry_model_state": experiment["carry_model_state"],
            "carry_optimizer_state": experiment["carry_optimizer_state"],
            "reset_optimizer_between_stages": experiment["reset_optimizer_between_stages"],
            "rewind_to_best": experiment["rewind_to_best"],
            "checkpoint": experiment["checkpoint"],
            "anchor_reproduced": anchor_reproduced,
            "classification_scope": metrics["classification_scope"],
            "final_full_ffid_classification": final_full_ffid_classification,
            "first_stage_final_min_median_trace_snr_db": experiment[
                "first_stage_final_min_median_trace_snr_db"
            ],
        }
    )
    _write_condition_outputs(
        run_path,
        resolved_config,
        inputs_lock,
        metrics,
        run_metadata,
    )

    summary_run = _summary_run(run_path.name, metrics)
    summary_run.update(
        {
            "planned_stage_trace_counts": list(trace_counts),
            "completed_stage_trace_counts": [int(stage["trace_count"]) for stage in stages],
            "stages_completed": len(stages),
            "updates_per_stage": experiment["updates_per_stage"],
            "planned_total_updates": planned_total_updates,
            "planned_point_evaluations": planned_point_evaluations,
            "anchor_reproduced": anchor_reproduced,
            "classification_scope": metrics["classification_scope"],
            "final_full_ffid_classification": final_full_ffid_classification,
        }
    )
    summary: dict[str, object] = {
        "study_id": TRACE_POOL_CONTINUATION_STUDY_ID,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "planned_total_updates": planned_total_updates,
        "updates_completed": cumulative_updates,
        "planned_point_evaluations": planned_point_evaluations,
        "point_evaluations": point_evaluations,
        "anchor_reproduced": anchor_reproduced,
        "first_stage_final_min_median_trace_snr_db": experiment[
            "first_stage_final_min_median_trace_snr_db"
        ],
        "final_full_ffid_classification": final_full_ffid_classification,
        "decision": decision,
        "runs": [summary_run],
    }
    _write_json(summary_path, summary)
    return summary


def _run_full_ffid_fit_probe(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_root: Path,
    device_override: str | None,
    study_id: str,
    expected_batch_mode: str,
    require_trace_correlation_loss: bool = False,
) -> dict[str, object]:
    """Run one configured full-training fit probe and write immutable records."""
    config = load_resolved_config(Path(config_path))
    device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("training.device must be a non-empty string")
    resolved_config = deepcopy(config)
    training_config = resolved_config.get("training")
    if not isinstance(training_config, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training_config["device"] = device

    experiment = _validated_full_ffid_experiment_config(
        resolved_config,
        expected_batch_mode=expected_batch_mode,
        require_trace_correlation_loss=require_trace_correlation_loss,
    )
    random_seed = _validated_random_seed(
        get_required_config_value(resolved_config, "project.random_seed")
    )
    git_commit = _git_commit()
    run_prefix = f"{_run_id_timestamp()}_{git_commit[:7]}"
    if experiment["batch_mode"] == "random_replacement":
        condition_label = f"random{experiment['batch_size']}_trace{experiment['trace_count']}"
    elif experiment["batch_mode"] == "random_shared_temporal_patch":
        condition_label = (
            f"patch{experiment['samples_per_trace']}_"
            f"trace{experiment['traces_per_update']}_trace{experiment['trace_count']}"
        )
    elif experiment["correlation_weight"] > 0.0:
        condition_label = (
            f"tracebatch{experiment['traces_per_update']}_corr0p1_trace{experiment['trace_count']}"
        )
    else:
        condition_label = (
            f"tracebatch{experiment['traces_per_update']}_trace{experiment['trace_count']}"
        )
    output_directory = Path(output_root)
    run_path = output_directory / f"{run_prefix}_{condition_label}"
    summary_path = output_directory / f"{run_prefix}_summary.json"
    _preflight_output_paths(output_directory, (run_path, summary_path))

    experiment_data = _load_experiment_data(
        Path(interim_dir),
        Path(processed_dir),
        resolved_config,
    )
    selected_rows = np.sort(experiment_data["training_array_rows"]).astype(np.int64, copy=False)
    available_training_rows = len(selected_rows)
    trace_count = experiment["trace_count"]
    if trace_count != available_training_rows:
        raise ValueError(
            f"configured trace count {trace_count} must equal all "
            f"{available_training_rows} available training rows"
        )
    training_coordinates, training_targets = build_trace_points(
        experiment_data["normalized_time"],
        experiment_data["normalized_spatial_by_array_row"],
        experiment_data["normalized_amplitudes"],
        selected_rows,
    )
    sample_count = experiment_data["sample_count"]
    point_count = trace_count * sample_count
    traces_per_update = experiment["traces_per_update"]
    samples_per_trace = experiment["samples_per_trace"]
    patch_starts = experiment["patch_starts"]
    if experiment["batch_mode"] == "random_shared_temporal_patch":
        if samples_per_trace is None or patch_starts is None:
            raise RuntimeError("temporal-patch configuration is incomplete")
        expected_patch_starts = overlapping_patch_starts(
            sample_count,
            samples_per_trace,
            experiment["temporal_patch_overlap_fraction"],
        )
        if patch_starts != expected_patch_starts:
            raise ConfigurationError(
                "experiment.patch_starts must equal the predefined overlapping starts "
                f"{list(expected_patch_starts)}"
            )
        expected_batch_size = traces_per_update * samples_per_trace
    elif traces_per_update is not None:
        expected_batch_size = traces_per_update * sample_count
    else:
        expected_batch_size = None
    if expected_batch_size is not None and experiment["batch_size"] != expected_batch_size:
        if samples_per_trace is None:
            message = (
                "training.batch_size must equal experiment.traces_per_update * sample_count "
                f"({expected_batch_size})"
            )
        else:
            message = (
                "training.batch_size must equal experiment.traces_per_update * "
                f"experiment.samples_per_trace ({expected_batch_size})"
            )
        raise ConfigurationError(message)
    point_evaluations = experiment["batch_size"] * experiment["total_updates"]

    started_at_utc = _utc_timestamp()
    metrics = run_training_fit_condition(
        config=resolved_config,
        label=condition_label,
        batch_mode=experiment["batch_mode"],
        full_batch=False,
        replacement=experiment["replacement"],
        total_updates=experiment["total_updates"],
        report_interval=experiment["report_interval"],
        batch_size=experiment["batch_size"],
        normalized_time=experiment_data["normalized_time"],
        normalized_spatial_by_array_row=experiment_data["normalized_spatial_by_array_row"],
        normalized_amplitudes=experiment_data["normalized_amplitudes"],
        selected_array_rows=selected_rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=sample_count,
        prediction_batch_size=experiment["prediction_batch_size"],
        device=device,
        random_seed=random_seed,
        traces_per_update=traces_per_update,
        samples_per_trace=samples_per_trace,
        patch_starts=patch_starts,
        temporal_patch_overlap_fraction=experiment["temporal_patch_overlap_fraction"],
        correlation_weight=experiment["correlation_weight"],
        correlation_eps=experiment["correlation_eps"],
    )
    run_metadata = _build_run_metadata(
        study_id=study_id,
        condition=condition_label,
        batch_mode=experiment["batch_mode"],
        batch_size=experiment["batch_size"],
        trace_count=trace_count,
        sample_count=sample_count,
        point_count=point_count,
        point_evaluations=point_evaluations,
        full_batch=False,
        replacement=experiment["replacement"],
        git_commit=git_commit,
        started_at_utc=started_at_utc,
        device=device,
        random_seed=random_seed,
        updates_completed=metrics["updates_completed"],
        traces_per_update=traces_per_update,
        samples_per_trace=samples_per_trace,
        temporal_patch_overlap_fraction=experiment["temporal_patch_overlap_fraction"],
        patch_starts=patch_starts,
        shared_temporal_patch=experiment["shared_temporal_patch"],
        correlation_weight=(
            experiment["correlation_weight"] if require_trace_correlation_loss else None
        ),
        correlation_eps=(experiment["correlation_eps"] if require_trace_correlation_loss else None),
        loss_semantics=(_MSE_PLUS_TRACE_CORRELATION if require_trace_correlation_loss else None),
    )
    training_contract = {
        "batch_mode": experiment["batch_mode"],
        "replacement": experiment["replacement"],
        "batch_size": experiment["batch_size"],
        "total_updates": experiment["total_updates"],
        "point_evaluations": point_evaluations,
    }
    if traces_per_update is not None:
        training_contract["traces_per_update"] = traces_per_update
    if samples_per_trace is not None:
        training_contract.update(
            {
                "samples_per_trace": samples_per_trace,
                "temporal_patch_overlap_fraction": experiment["temporal_patch_overlap_fraction"],
                "patch_starts": list(patch_starts),
                "shared_temporal_patch": experiment["shared_temporal_patch"],
            }
        )
    if require_trace_correlation_loss:
        training_contract.update(
            {
                "correlation_weight": experiment["correlation_weight"],
                "correlation_eps": experiment["correlation_eps"],
                "loss_semantics": _MSE_PLUS_TRACE_CORRELATION,
            }
        )
    inputs_lock = _build_inputs_lock(
        interim_files=experiment_data["interim_files"],
        processed_files=experiment_data["processed_files"],
        preparation_contract=experiment_data["preparation_contract"],
        selected_array_rows=selected_rows,
        trace_count=trace_count,
        sample_count=sample_count,
        point_count=point_count,
        random_seed=random_seed,
        selection_method=_ALL_TRAINING_ROWS_METHOD,
        split_counts=experiment_data["split_counts"],
        training_contract=training_contract,
    )
    _write_condition_outputs(
        run_path,
        resolved_config,
        inputs_lock,
        metrics,
        run_metadata,
    )

    summary_run = _summary_run(run_path.name, metrics)
    decision = full_ffid_summary_decision(str(metrics["classification"]))
    summary: dict[str, object] = {
        "study_id": study_id,
        "git_commit": git_commit,
        "generated_at_utc": _utc_timestamp(),
        "point_evaluations": point_evaluations,
        "decision": decision,
        "runs": [summary_run],
    }
    if require_trace_correlation_loss:
        summary.update(
            {
                "correlation_weight": experiment["correlation_weight"],
                "correlation_eps": experiment["correlation_eps"],
                "loss_semantics": _MSE_PLUS_TRACE_CORRELATION,
            }
        )
    _write_json(summary_path, summary)
    return summary


def _print_training_progress(
    label: str,
    total_updates: int,
    row: Mapping[str, int | float],
) -> None:
    """Emit one flushed stdout line per report so long runs are observable while training."""
    print(
        f"[{label}] step {row['step']}/{total_updates} "
        f"loss={row['mean_train_loss_since_last_report']:.6f} "
        f"median_snr_db={row['training_median_trace_snr_db']:.4f} "
        f"median_corr={row['training_median_trace_correlation']:.4f} "
        f"rms_ratio={row['training_prediction_target_rms_ratio']:.4f}",
        flush=True,
    )


def run_training_fit_condition(
    *,
    config: Mapping[str, object],
    label: str,
    batch_mode: str,
    full_batch: bool,
    replacement: bool,
    total_updates: int,
    report_interval: int,
    batch_size: int,
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    normalized_amplitudes: np.ndarray,
    selected_array_rows: np.ndarray,
    all_coordinate_tensor: torch.Tensor | None,
    all_target_tensor: torch.Tensor | None,
    training_coordinates: np.ndarray,
    training_targets: np.ndarray,
    sample_count: int,
    prediction_batch_size: int,
    device: torch.device | str,
    random_seed: int,
    traces_per_update: int | None = None,
    samples_per_trace: int | None = None,
    patch_starts: Sequence[int] | None = None,
    temporal_patch_overlap_fraction: float | None = None,
    correlation_weight: float = 0.0,
    correlation_eps: float = _STUDY_005_CORRELATION_EPS,
    loss_name: str = "l2",
    huber_delta: float | None = None,
    model: Siren | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Train one model under one fixed batching condition, freshly initialized by default."""
    total_updates = _positive_integer(total_updates, "total_updates")
    report_interval = _positive_integer(report_interval, "report_interval")
    batch_size = _positive_integer(batch_size, "batch_size")
    sample_count = _positive_integer(sample_count, "sample_count")
    prediction_batch_size = _positive_integer(prediction_batch_size, "prediction_batch_size")
    validated_correlation_weight = _nonnegative_finite_float(
        correlation_weight,
        "correlation_weight",
    )
    validated_correlation_eps = _positive_finite_float(
        correlation_eps,
        "correlation_eps",
    )
    uses_trace_correlation_loss = validated_correlation_weight > 0.0
    if loss_name not in ("l2", "huber"):
        raise ValueError(f"unknown loss name: {loss_name!r}")
    if loss_name == "huber":
        if uses_trace_correlation_loss:
            raise ValueError("huber loss does not support the trace correlation loss")
        validated_huber_delta = _positive_finite_float(huber_delta, "huber_delta")
    else:
        if huber_delta is not None:
            raise ValueError("huber_delta requires loss_name 'huber'")
        validated_huber_delta = None
    if total_updates % report_interval != 0:
        raise ValueError("total_updates must be divisible by report_interval")
    validated_traces_per_update = (
        None
        if traces_per_update is None
        else _positive_integer(traces_per_update, "traces_per_update")
    )
    validated_samples_per_trace = (
        None
        if samples_per_trace is None
        else _positive_integer(samples_per_trace, "samples_per_trace")
    )
    validated_patch_starts = None if patch_starts is None else _validated_patch_starts(patch_starts)
    validated_patch_overlap = (
        None
        if temporal_patch_overlap_fraction is None
        else _validated_overlap_fraction(
            temporal_patch_overlap_fraction,
            "temporal_patch_overlap_fraction",
        )
    )
    if batch_mode == "random_replacement":
        if full_batch or replacement is not True:
            raise ValueError("random_replacement requires full_batch=false and replacement=true")
        if validated_traces_per_update is not None:
            raise ValueError("random_replacement does not accept traces_per_update")
        if (
            validated_samples_per_trace is not None
            or validated_patch_starts is not None
            or validated_patch_overlap is not None
        ):
            raise ValueError("random_replacement does not accept temporal-patch parameters")
    elif batch_mode == "random_complete_traces":
        if full_batch or replacement:
            raise ValueError(
                "random_complete_traces requires full_batch=false and replacement=false"
            )
        if validated_traces_per_update is None:
            raise ValueError("random_complete_traces requires traces_per_update")
        if validated_traces_per_update > len(selected_array_rows):
            raise ValueError(
                "traces_per_update must not exceed the number of selected training rows "
                f"({len(selected_array_rows)})"
            )
        expected_batch_size = validated_traces_per_update * sample_count
        if batch_size != expected_batch_size:
            raise ValueError(
                f"batch_size must equal traces_per_update * sample_count ({expected_batch_size})"
            )
        if (
            validated_samples_per_trace is not None
            or validated_patch_starts is not None
            or validated_patch_overlap is not None
        ):
            raise ValueError("random_complete_traces does not accept temporal-patch parameters")
    elif batch_mode == "random_shared_temporal_patch":
        if full_batch or replacement:
            raise ValueError(
                "random_shared_temporal_patch requires full_batch=false and replacement=false"
            )
        if validated_traces_per_update is None:
            raise ValueError("random_shared_temporal_patch requires traces_per_update")
        if validated_samples_per_trace is None:
            raise ValueError("random_shared_temporal_patch requires samples_per_trace")
        if validated_patch_starts is None:
            raise ValueError("random_shared_temporal_patch requires patch_starts")
        if validated_patch_overlap is None:
            raise ValueError(
                "random_shared_temporal_patch requires temporal_patch_overlap_fraction"
            )
        if validated_traces_per_update > len(selected_array_rows):
            raise ValueError(
                "traces_per_update must not exceed the number of selected training rows "
                f"({len(selected_array_rows)})"
            )
        expected_batch_size = validated_traces_per_update * validated_samples_per_trace
        if batch_size != expected_batch_size:
            raise ValueError(
                "batch_size must equal traces_per_update * samples_per_trace "
                f"({expected_batch_size})"
            )
    elif batch_mode == "exact_full_batch":
        if full_batch is not True or replacement:
            raise ValueError("exact_full_batch requires full_batch=true and replacement=false")
        if validated_traces_per_update is not None:
            raise ValueError("exact_full_batch does not accept traces_per_update")
        if (
            validated_samples_per_trace is not None
            or validated_patch_starts is not None
            or validated_patch_overlap is not None
        ):
            raise ValueError("exact_full_batch does not accept temporal-patch parameters")
    else:
        raise ValueError(f"unknown batch mode: {batch_mode!r}")
    if uses_trace_correlation_loss and batch_mode != "random_complete_traces":
        raise ValueError("trace correlation loss requires random_complete_traces batches")

    if (model is None) != (optimizer is None):
        raise ValueError("model and optimizer must either both be provided or both be omitted")
    if model is None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        model = _build_model(config)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=_positive_finite_float(
                get_required_config_value(config, "training.learning_rate"),
                "training.learning_rate",
            ),
        )
    else:
        model.to(device)
    if optimizer is None:
        raise RuntimeError("training optimizer was not initialized")
    sampler = None
    sampler_batch_size = batch_size
    if batch_mode == "random_replacement":
        sampler = RandomPointSampler(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
            selected_array_rows,
            random_seed=random_seed,
        )
    elif batch_mode == "random_complete_traces":
        sampler = RandomTraceBatchSampler(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
            selected_array_rows,
            random_seed=random_seed,
        )
        sampler_batch_size = validated_traces_per_update
    elif batch_mode == "random_shared_temporal_patch":
        sampler = RandomTracePatchSampler(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
            selected_array_rows,
            patch_size=validated_samples_per_trace,
            patch_starts=validated_patch_starts,
            random_seed=random_seed,
        )
        sampler_batch_size = validated_traces_per_update

    interval_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_mse_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_correlation_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    history: list[dict[str, int | float]] = []
    best_row: dict[str, int | float] | None = None
    trace_count = len(selected_array_rows)

    print(
        f"[{label}] start: total_updates={total_updates} batch_mode={batch_mode} "
        f"batch_size={batch_size} traces={trace_count} device={device}",
        flush=True,
    )
    model.train()
    for step in range(1, total_updates + 1):
        if sampler is None:
            if all_coordinate_tensor is None or all_target_tensor is None:
                raise RuntimeError("exact full-batch training tensors are required")
            coordinate_tensor = all_coordinate_tensor
            target_tensor = all_target_tensor
        else:
            batch_coordinates, batch_targets = sampler.sample(sampler_batch_size)
            coordinate_tensor, target_tensor = to_model_tensors(
                batch_coordinates,
                batch_targets,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinate_tensor)
        if validated_huber_delta is None:
            point_loss = torch_functional.mse_loss(prediction, target_tensor)
        else:
            point_loss = torch_functional.huber_loss(
                prediction,
                target_tensor,
                delta=validated_huber_delta,
            )
        if uses_trace_correlation_loss:
            if validated_traces_per_update is None:
                raise RuntimeError("trace correlation loss requires traces_per_update")
            # RandomTraceBatchSampler returns complete traces in trace-major order.
            correlation_loss = trace_correlation_loss(
                prediction.reshape(validated_traces_per_update, sample_count),
                target_tensor.reshape(validated_traces_per_update, sample_count),
                eps=validated_correlation_eps,
            )
            loss = point_loss + validated_correlation_weight * correlation_loss
            interval_mse_loss_sum += point_loss.detach().to(dtype=torch.float64)
            interval_correlation_loss_sum += correlation_loss.detach().to(dtype=torch.float64)
        else:
            loss = point_loss
        loss.backward()
        optimizer.step()
        interval_loss_sum += loss.detach().to(dtype=torch.float64)

        if step % report_interval != 0:
            continue
        mean_loss = float((interval_loss_sum / report_interval).cpu().item())
        if not math.isfinite(mean_loss):
            raise ValueError(f"mean training loss is non-finite at step {step}")
        row: dict[str, int | float] = {
            "step": step,
            "mean_train_loss_since_last_report": mean_loss,
            **_evaluate_training_fit(
                model,
                training_coordinates=training_coordinates,
                training_targets=training_targets,
                trace_count=trace_count,
                sample_count=sample_count,
                prediction_batch_size=prediction_batch_size,
                device=device,
                step=step,
            ),
        }
        if uses_trace_correlation_loss:
            mean_mse_loss = float((interval_mse_loss_sum / report_interval).cpu().item())
            mean_correlation_loss = float(
                (interval_correlation_loss_sum / report_interval).cpu().item()
            )
            if not math.isfinite(mean_mse_loss) or not math.isfinite(mean_correlation_loss):
                raise ValueError(f"mean loss component is non-finite at step {step}")
            row.update(
                {
                    "mean_train_mse_loss_since_last_report": mean_mse_loss,
                    "mean_train_correlation_loss_since_last_report": (mean_correlation_loss),
                }
            )
        history.append(row)
        _print_training_progress(label, total_updates, row)
        if (
            best_row is None
            or row["training_median_trace_snr_db"] > best_row["training_median_trace_snr_db"]
        ):
            best_row = dict(row)
        interval_loss_sum.zero_()
        interval_mse_loss_sum.zero_()
        interval_correlation_loss_sum.zero_()

    if best_row is None or not history:
        raise RuntimeError(f"condition {label!r} produced no report points")
    final_row = history[-1]
    metrics: dict[str, object] = {
        "condition": label,
        "batch_mode": batch_mode,
        "batch_size": batch_size,
        "full_batch": full_batch,
        "replacement": replacement,
        "trace_count": trace_count,
        "sample_count": sample_count,
        "point_count": trace_count * sample_count,
        "point_evaluations": batch_size * total_updates,
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
    if validated_traces_per_update is not None:
        metrics["traces_per_update"] = validated_traces_per_update
    if validated_samples_per_trace is not None:
        metrics.update(
            {
                "samples_per_trace": validated_samples_per_trace,
                "temporal_patch_overlap_fraction": validated_patch_overlap,
                "patch_starts": list(validated_patch_starts),
                "shared_temporal_patch": True,
            }
        )
    if uses_trace_correlation_loss:
        metrics.update(
            {
                "correlation_weight": validated_correlation_weight,
                "correlation_eps": validated_correlation_eps,
                "loss_semantics": _MSE_PLUS_TRACE_CORRELATION,
            }
        )
    metrics["classification"] = classify_condition(metrics)
    return metrics


def _evaluate_training_fit(
    model: Siren,
    *,
    training_coordinates: np.ndarray,
    training_targets: np.ndarray,
    trace_count: int,
    sample_count: int,
    prediction_batch_size: int,
    device: torch.device | str,
    step: int,
) -> dict[str, float]:
    """Evaluate one condition on the complete selected training subset."""
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
    if training_coordinates.shape != (expected_points, len(MODEL_COORDINATE_ORDER)):
        raise ValueError(
            "training_coordinates must have shape "
            f"({expected_points}, {len(MODEL_COORDINATE_ORDER)}), "
            f"got {training_coordinates.shape}"
        )
    trace_targets = training_targets.reshape(trace_count, sample_count)
    trace_predictions = predictions.reshape(trace_count, sample_count)
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
        "training_median_trace_correlation": median_trace_correlation_coefficient(
            trace_targets, trace_predictions
        ),
        "training_prediction_target_rms_ratio": prediction_rms / target_rms,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"training report contains a non-finite value at step {step}")
    return metrics


def classify_condition(metrics: Mapping[str, object]) -> str:
    """Classify one condition using the fixed training-fit thresholds."""
    median_snr = float(metrics["best_training_median_trace_snr_db"])
    rms_ratio = float(metrics["best_training_prediction_target_rms_ratio"])
    if not math.isfinite(median_snr) or not math.isfinite(rms_ratio):
        raise ValueError("classification metrics must be finite")
    if median_snr >= 20.0:
        return "strong_fit"
    if median_snr > 1.0 and rms_ratio > 0.1:
        return "escaped_zero_predictor"
    return "near_zero"


def continuation_summary_decision(
    *,
    anchor_reproduced: bool,
    final_classification: str,
) -> str:
    """Return the staged-continuation outcome after applying its anchor gate."""
    if not isinstance(anchor_reproduced, bool):
        raise ValueError("anchor_reproduced must be a boolean")
    if not anchor_reproduced:
        return "stage8_anchor_failed"
    return full_ffid_summary_decision(final_classification)


def full_ffid_summary_decision(classification: str) -> str:
    """Map one fit classification to the shared full-FFID summary decision."""
    decisions = {
        "strong_fit": "full_ffid_strong_fit",
        "escaped_zero_predictor": "full_ffid_escaped_zero_predictor",
        "near_zero": "full_ffid_near_zero",
    }
    try:
        return decisions[classification]
    except KeyError as error:
        raise ValueError(f"unknown training-fit classification: {classification!r}") from error


def official_siren_summary_decision(
    *,
    legacy_classification: str,
    official_classification: str,
) -> str:
    """Return the paired Study 012 decision after validating its legacy control."""
    known_classifications = {"strong_fit", "escaped_zero_predictor", "near_zero"}
    if legacy_classification not in known_classifications:
        raise ValueError(f"unknown legacy-control classification: {legacy_classification!r}")
    if official_classification not in known_classifications:
        raise ValueError(f"unknown official-SIREN classification: {official_classification!r}")
    if legacy_classification != "near_zero":
        return "legacy_control_not_reproduced"
    decisions = {
        "strong_fit": "official_siren_strong_fit",
        "escaped_zero_predictor": "official_siren_escaped_zero_predictor",
        "near_zero": "official_siren_near_zero",
    }
    return decisions[official_classification]


def amplitude_balancing_summary_decision(
    *,
    control_classification: str,
    per_trace_classification: str,
) -> str:
    """Map the control gate and per-trace-RMS outcome to the summary decision."""
    known_classifications = {"strong_fit", "escaped_zero_predictor", "near_zero"}
    if control_classification not in known_classifications:
        raise ValueError(f"unknown control classification: {control_classification!r}")
    if per_trace_classification not in known_classifications:
        raise ValueError(f"unknown per-trace classification: {per_trace_classification!r}")
    if control_classification != "near_zero":
        return "global_rms_control_not_reproduced"
    decisions = {
        "strong_fit": "per_trace_rms_strong_fit",
        "escaped_zero_predictor": "per_trace_rms_escaped_zero_predictor",
        "near_zero": "per_trace_rms_near_zero",
    }
    return decisions[per_trace_classification]


def full_trace_batch_ablation_summary_decision(
    *,
    control_classification: str,
    full_trace_batch_classification: str,
    reproduction_classification: str,
) -> str:
    """Map the Study 014 gates and full-trace-batch outcome to the summary decision."""
    known_classifications = {"strong_fit", "escaped_zero_predictor", "near_zero"}
    if control_classification not in known_classifications:
        raise ValueError(f"unknown control classification: {control_classification!r}")
    if full_trace_batch_classification not in known_classifications:
        raise ValueError(
            f"unknown full-trace-batch classification: {full_trace_batch_classification!r}"
        )
    if reproduction_classification not in known_classifications:
        raise ValueError(f"unknown reproduction classification: {reproduction_classification!r}")
    if control_classification != "near_zero":
        return "small_batch_control_not_reproduced"
    if reproduction_classification == "near_zero":
        return "combined_escape_not_reproduced"
    decisions = {
        "strong_fit": "full_trace_batch_strong_fit",
        "escaped_zero_predictor": "full_trace_batch_escaped_zero_predictor",
        "near_zero": "full_trace_batch_near_zero",
    }
    return decisions[full_trace_batch_classification]


def strong_fit_budget_extension_summary_decision(
    *,
    baseline_reproduced: bool,
    extension_classification: str,
) -> str:
    """Return the extended-budget outcome after applying its baseline-reproduction gate."""
    if not isinstance(baseline_reproduced, bool):
        raise ValueError("baseline_reproduced must be a boolean")
    known_classifications = {"strong_fit", "escaped_zero_predictor", "near_zero"}
    if extension_classification not in known_classifications:
        raise ValueError(f"unknown extension classification: {extension_classification!r}")
    if not baseline_reproduced:
        return "baseline_not_reproduced"
    decisions = {
        "strong_fit": "extended_budget_strong_fit",
        "escaped_zero_predictor": "extended_budget_escaped_zero_predictor",
        "near_zero": "extended_budget_near_zero",
    }
    return decisions[extension_classification]


def best_median_trace_snr_within(
    history: Sequence[Mapping[str, object]],
    *,
    max_step: int,
) -> float:
    """Return the best median training-trace S/N among reports at or before ``max_step``."""
    values = [
        float(row["training_median_trace_snr_db"])
        for row in history
        if int(row["step"]) <= max_step
    ]
    if not values:
        raise ValueError(f"no history reports at or before step {max_step}")
    return max(values)


def first_step_reaching_median_trace_snr(
    history: Sequence[Mapping[str, object]],
    *,
    threshold_db: float,
) -> int | None:
    """Return the first report step whose median training-trace S/N reaches the threshold."""
    for row in history:
        if float(row["training_median_trace_snr_db"]) >= threshold_db:
            return int(row["step"])
    return None


def _add_continuation_stage_context(
    metrics: dict[str, object],
    *,
    stage_index: int,
    sampler_seed: int,
    cumulative_updates_before_stage: int,
    entry_metrics: Mapping[str, float],
) -> None:
    """Annotate local stage metrics with immutable continuation coordinates."""
    history = metrics.get("history")
    if not isinstance(history, list):
        raise RuntimeError("continuation stage history must be a list")
    best_stage_step = int(metrics["best_step"])
    for row in history:
        if not isinstance(row, dict):
            raise RuntimeError("continuation stage history rows must be mappings")
        stage_step = int(row["step"])
        cumulative_step = cumulative_updates_before_stage + stage_step
        row["stage_step"] = stage_step
        row["cumulative_step"] = cumulative_step
        row["step"] = cumulative_step
    metrics.update(
        {
            "stage_index": stage_index + 1,
            "sampler_seed": sampler_seed,
            "cumulative_updates_start": cumulative_updates_before_stage,
            "cumulative_updates_end": (
                cumulative_updates_before_stage + int(metrics["updates_completed"])
            ),
            "best_stage_step": best_stage_step,
            "best_step": cumulative_updates_before_stage + best_stage_step,
            **{f"entry_{key}": value for key, value in entry_metrics.items()},
        }
    )


def batching_summary_decision(summary_runs: Sequence[Mapping[str, object]]) -> str:
    """Return the fixed paired-condition Study 006 decision."""
    by_condition = {str(run["condition"]): run for run in summary_runs}
    if set(by_condition) != {_EXACT_LABEL, _RANDOM_LABEL}:
        raise ValueError("summary runs must contain both configured conditions")
    exact_classification = by_condition[_EXACT_LABEL]["classification"]
    random_classification = by_condition[_RANDOM_LABEL]["classification"]
    if exact_classification != "strong_fit":
        return "control_failed_unexpected"
    if random_classification == "strong_fit":
        return "random_replacement_succeeds"
    if random_classification == "escaped_zero_predictor":
        return "random_replacement_partially_succeeds"
    if random_classification == "near_zero":
        return "exact_coverage_required"
    raise ValueError(f"unknown random condition classification: {random_classification!r}")


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
        "split_counts": split_counts,
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
    _validate_common_training_fit_config(config)

    total_updates = _positive_integer(
        get_required_config_value(config, "training.updates"),
        "training.updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError("training.updates must be divisible by training.report_interval")
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "conditions": _validated_conditions(
            get_required_config_value(config, "experiment.conditions")
        ),
    }


def _validated_full_ffid_experiment_config(
    config: Mapping[str, object],
    *,
    expected_batch_mode: str = "random_replacement",
    require_trace_correlation_loss: bool = False,
) -> dict[str, object]:
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps"}
    if not require_trace_correlation_loss and correlation_keys.intersection(experiment_config):
        raise ConfigurationError(
            "pure-MSE full-FFID experiments must not define correlation_weight or correlation_eps"
        )
    total_updates = _positive_integer(
        get_required_config_value(config, "training.total_updates"),
        "training.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError(
            "training.total_updates must be divisible by training.report_interval"
        )
    batch_mode = get_required_config_value(config, "experiment.batch_mode")
    if expected_batch_mode not in {
        "random_replacement",
        "random_complete_traces",
        "random_shared_temporal_patch",
    }:
        raise ValueError(f"unsupported expected batch mode: {expected_batch_mode!r}")
    if batch_mode != expected_batch_mode:
        raise ConfigurationError(f"experiment.batch_mode must be {expected_batch_mode!r}")
    replacement = get_required_config_value(config, "experiment.replacement")
    expected_replacement = batch_mode == "random_replacement"
    if replacement is not expected_replacement:
        raise ConfigurationError(
            f"experiment.replacement must be {str(expected_replacement).lower()}"
        )
    traces_per_update = None
    if batch_mode in {"random_complete_traces", "random_shared_temporal_patch"}:
        traces_per_update = _positive_integer(
            get_required_config_value(config, "experiment.traces_per_update"),
            "experiment.traces_per_update",
        )
    samples_per_trace = None
    temporal_patch_overlap_fraction = None
    patch_starts = None
    shared_temporal_patch = None
    if batch_mode == "random_shared_temporal_patch":
        samples_per_trace = _positive_integer(
            get_required_config_value(config, "experiment.samples_per_trace"),
            "experiment.samples_per_trace",
        )
        temporal_patch_overlap_fraction = _validated_overlap_fraction(
            get_required_config_value(config, "experiment.temporal_patch_overlap_fraction"),
            "experiment.temporal_patch_overlap_fraction",
        )
        patch_starts = _validated_patch_starts(
            get_required_config_value(config, "experiment.patch_starts")
        )
        shared_temporal_patch = get_required_config_value(
            config,
            "experiment.shared_temporal_patch",
        )
        if shared_temporal_patch is not True:
            raise ConfigurationError("experiment.shared_temporal_patch must be true")
    correlation_weight = 0.0
    correlation_eps = _STUDY_005_CORRELATION_EPS
    if require_trace_correlation_loss:
        if batch_mode != "random_complete_traces":
            raise ConfigurationError(
                "trace correlation loss requires experiment.batch_mode 'random_complete_traces'"
            )
        correlation_weight = _positive_finite_float(
            get_required_config_value(config, "experiment.correlation_weight"),
            "experiment.correlation_weight",
        )
        correlation_eps = _positive_finite_float(
            get_required_config_value(config, "experiment.correlation_eps"),
            "experiment.correlation_eps",
        )
        if correlation_weight != _STUDY_005_CORRELATION_WEIGHT:
            raise ConfigurationError("experiment.correlation_weight must be 0.1")
        if correlation_eps != _STUDY_005_CORRELATION_EPS:
            raise ConfigurationError("experiment.correlation_eps must be 1.0e-4")
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "batch_mode": batch_mode,
        "replacement": replacement,
        "traces_per_update": traces_per_update,
        "samples_per_trace": samples_per_trace,
        "temporal_patch_overlap_fraction": temporal_patch_overlap_fraction,
        "patch_starts": patch_starts,
        "shared_temporal_patch": shared_temporal_patch,
        "correlation_weight": correlation_weight,
        "correlation_eps": correlation_eps,
    }


def _validated_official_siren_experiment_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the paired legacy/official SIREN full-training contract."""
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    if correlation_keys.intersection(experiment_config) or correlation_keys.intersection(
        training_config
    ):
        raise ConfigurationError("official-SIREN baseline conditions must use pure MSE")
    total_updates = _positive_integer(
        get_required_config_value(config, "training.total_updates"),
        "training.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError(
            "training.total_updates must be divisible by training.report_interval"
        )
    batch_mode = get_required_config_value(config, "experiment.batch_mode")
    if batch_mode != "random_replacement":
        raise ConfigurationError("experiment.batch_mode must be 'random_replacement'")
    replacement = get_required_config_value(config, "experiment.replacement")
    if replacement is not True:
        raise ConfigurationError("experiment.replacement must be true")
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "batch_mode": batch_mode,
        "replacement": replacement,
        "conditions": _validated_official_siren_conditions(
            get_required_config_value(config, "experiment.conditions")
        ),
    }


def _validated_amplitude_balancing_experiment_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the paired amplitude-balancing full-training contract."""
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    if correlation_keys.intersection(experiment_config) or correlation_keys.intersection(
        training_config
    ):
        raise ConfigurationError("amplitude-balancing conditions must not use correlation loss")
    total_updates = _positive_integer(
        get_required_config_value(config, "training.total_updates"),
        "training.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError(
            "training.total_updates must be divisible by training.report_interval"
        )
    batch_mode = get_required_config_value(config, "experiment.batch_mode")
    if batch_mode != "random_replacement":
        raise ConfigurationError("experiment.batch_mode must be 'random_replacement'")
    replacement = get_required_config_value(config, "experiment.replacement")
    if replacement is not True:
        raise ConfigurationError("experiment.replacement must be true")
    huber_delta = _positive_finite_float(
        get_required_config_value(config, "experiment.huber_delta"),
        "experiment.huber_delta",
    )
    if huber_delta != _AMPLITUDE_BALANCING_HUBER_DELTA:
        raise ConfigurationError(
            f"experiment.huber_delta must be {_AMPLITUDE_BALANCING_HUBER_DELTA}"
        )
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "batch_mode": batch_mode,
        "replacement": replacement,
        "huber_delta": huber_delta,
        "conditions": _validated_amplitude_balancing_conditions(
            get_required_config_value(config, "experiment.conditions")
        ),
    }


def _validated_full_trace_batch_ablation_experiment_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the Study 014 full-trace-batch ingredient-ablation contract."""
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    if correlation_keys.intersection(training_config):
        raise ConfigurationError(
            "correlation loss is configured per condition under experiment, not training"
        )
    total_updates = _positive_integer(
        get_required_config_value(config, "training.total_updates"),
        "training.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError(
            "training.total_updates must be divisible by training.report_interval"
        )
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "correlation_weight": _positive_finite_float(
            get_required_config_value(config, "experiment.correlation_weight"),
            "experiment.correlation_weight",
        ),
        "correlation_eps": _positive_finite_float(
            get_required_config_value(config, "experiment.correlation_eps"),
            "experiment.correlation_eps",
        ),
        "conditions": _validated_full_trace_batch_conditions(
            get_required_config_value(config, "experiment.conditions")
        ),
    }


def _validated_strong_fit_budget_extension_experiment_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the Study 015 extended-budget strong-fit contract."""
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    if correlation_keys.intersection(experiment_config) or correlation_keys.intersection(
        training_config
    ):
        raise ConfigurationError("pure-MSE budget extension must not define correlation-loss keys")
    total_updates = _positive_integer(
        get_required_config_value(config, "training.total_updates"),
        "training.total_updates",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if total_updates % report_interval != 0:
        raise ConfigurationError(
            "training.total_updates must be divisible by training.report_interval"
        )
    baseline_window_updates = _positive_integer(
        get_required_config_value(config, "experiment.baseline_window_updates"),
        "experiment.baseline_window_updates",
    )
    if baseline_window_updates % report_interval != 0:
        raise ConfigurationError(
            "experiment.baseline_window_updates must be divisible by training.report_interval"
        )
    if baseline_window_updates >= total_updates:
        raise ConfigurationError(
            "training.total_updates must exceed experiment.baseline_window_updates"
        )
    return {
        "trace_count": _positive_integer(
            get_required_config_value(config, "experiment.trace_count"),
            "experiment.trace_count",
        ),
        "total_updates": total_updates,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "baseline_window_updates": baseline_window_updates,
        "baseline_best_median_trace_snr_db": _positive_finite_float(
            get_required_config_value(config, "experiment.baseline_best_median_trace_snr_db"),
            "experiment.baseline_best_median_trace_snr_db",
        ),
        "baseline_tolerance_db": _positive_finite_float(
            get_required_config_value(config, "experiment.baseline_tolerance_db"),
            "experiment.baseline_tolerance_db",
        ),
        "conditions": _validated_strong_fit_budget_extension_conditions(
            get_required_config_value(config, "experiment.conditions")
        ),
    }


def _validated_continuation_experiment_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the focused nested trace-pool continuation contract."""
    _validate_common_training_fit_config(config)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ConfigurationError("experiment configuration must be a mapping")
    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    if correlation_keys.intersection(experiment_config) or correlation_keys.intersection(
        training_config
    ):
        raise ConfigurationError("pure-MSE continuation must not define correlation-loss keys")

    trace_counts = _validated_continuation_trace_counts(
        get_required_config_value(config, "experiment.trace_counts")
    )
    updates_per_stage = _positive_integer(
        get_required_config_value(config, "training.updates_per_stage"),
        "training.updates_per_stage",
    )
    report_interval = _positive_integer(
        get_required_config_value(config, "training.report_interval"),
        "training.report_interval",
    )
    if updates_per_stage % report_interval != 0:
        raise ConfigurationError(
            "training.updates_per_stage must be divisible by training.report_interval"
        )
    batch_mode = get_required_config_value(config, "experiment.batch_mode")
    if batch_mode != "random_replacement":
        raise ConfigurationError("experiment.batch_mode must be 'random_replacement'")
    replacement = get_required_config_value(config, "experiment.replacement")
    if replacement is not True:
        raise ConfigurationError("experiment.replacement must be true")
    sampler_seed_policy = get_required_config_value(config, "experiment.sampler_seed_policy")
    if sampler_seed_policy != "base_seed_plus_stage_index":
        raise ConfigurationError(
            "experiment.sampler_seed_policy must be 'base_seed_plus_stage_index'"
        )

    required_flags = {
        "carry_model_state": True,
        "carry_optimizer_state": True,
        "reset_optimizer_between_stages": False,
        "rewind_to_best": False,
        "checkpoint": False,
    }
    validated_flags: dict[str, bool] = {}
    for key, expected in required_flags.items():
        value = get_required_config_value(config, f"experiment.{key}")
        if value is not expected:
            raise ConfigurationError(f"experiment.{key} must be {str(expected).lower()}")
        validated_flags[key] = expected

    return {
        "trace_counts": trace_counts,
        "updates_per_stage": updates_per_stage,
        "report_interval": report_interval,
        "batch_size": _positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        "prediction_batch_size": _positive_integer(
            get_required_config_value(config, "training.prediction_batch_size"),
            "training.prediction_batch_size",
        ),
        "batch_mode": batch_mode,
        "replacement": replacement,
        "sampler_seed_policy": sampler_seed_policy,
        "first_stage_final_min_median_trace_snr_db": _finite_float(
            get_required_config_value(
                config,
                "experiment.first_stage_final_min_median_trace_snr_db",
            ),
            "experiment.first_stage_final_min_median_trace_snr_db",
        ),
        **validated_flags,
    }


def _validate_common_training_fit_config(config: Mapping[str, object]) -> None:
    if get_required_config_value(config, "model.name") != "siren":
        raise ConfigurationError("model.name must be 'siren'")
    input_features = get_required_config_value(config, "model.input_features")
    if input_features != len(MODEL_COORDINATE_ORDER):
        raise ConfigurationError(f"model.input_features must be {len(MODEL_COORDINATE_ORDER)}")
    if get_required_config_value(config, "training.loss") != "l2":
        raise ConfigurationError("training.loss must be 'l2'")
    if get_required_config_value(config, "training.optimizer") != "adam":
        raise ConfigurationError("training.optimizer must be 'adam'")
    _positive_finite_float(
        get_required_config_value(config, "training.learning_rate"),
        "training.learning_rate",
    )


def _validated_conditions(value: object) -> tuple[tuple[str, str, bool, bool], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("experiment.conditions must be a sequence")
    conditions: list[tuple[str, str, bool, bool]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ConfigurationError("each experiment condition must be a mapping")
        label = item.get("label")
        batch_mode = item.get("batch_mode")
        full_batch = item.get("full_batch")
        replacement = item.get("replacement")
        if not isinstance(label, str) or not label:
            raise ConfigurationError("condition label must be a non-empty string")
        if not isinstance(batch_mode, str) or not batch_mode:
            raise ConfigurationError("condition batch_mode must be a non-empty string")
        if not isinstance(full_batch, bool) or not isinstance(replacement, bool):
            raise ConfigurationError("condition full_batch and replacement must be booleans")
        conditions.append((label, batch_mode, full_batch, replacement))
    converted = tuple(conditions)
    if converted != _EXPECTED_CONDITIONS:
        raise ConfigurationError(
            "experiment.conditions must contain the fixed exact and random-replacement conditions"
        )
    return converted


def _validated_official_siren_conditions(
    value: object,
) -> tuple[tuple[str, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("experiment.conditions must be a sequence")
    by_label: dict[str, tuple[float, float]] = {}
    expected_keys = {"label", "omega_0", "hidden_omega"}
    for item in value:
        if not isinstance(item, Mapping):
            raise ConfigurationError("each experiment condition must be a mapping")
        if set(item) != expected_keys:
            raise ConfigurationError(
                "each official-SIREN condition must define only label, omega_0, and hidden_omega"
            )
        label = item["label"]
        if not isinstance(label, str) or not label:
            raise ConfigurationError("condition label must be a non-empty string")
        if label in by_label:
            raise ConfigurationError(f"duplicate experiment condition label: {label!r}")
        by_label[label] = (
            _positive_finite_float(item["omega_0"], f"{label}.omega_0"),
            _positive_finite_float(item["hidden_omega"], f"{label}.hidden_omega"),
        )
    expected_by_label = {
        label: (omega_0, hidden_omega)
        for label, omega_0, hidden_omega in _OFFICIAL_SIREN_CONDITIONS
    }
    if by_label != expected_by_label:
        raise ConfigurationError(
            "experiment.conditions must contain exactly legacy_control (300, 1) and "
            "official_siren_30 (30, 30)"
        )
    return _OFFICIAL_SIREN_CONDITIONS


def _validated_amplitude_balancing_conditions(
    value: object,
) -> tuple[tuple[str, str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("experiment.conditions must be a sequence")
    by_label: dict[str, tuple[str, str]] = {}
    expected_keys = {"label", "amplitude_scaling", "loss"}
    for item in value:
        if not isinstance(item, Mapping):
            raise ConfigurationError("each experiment condition must be a mapping")
        if set(item) != expected_keys:
            raise ConfigurationError(
                "each amplitude-balancing condition must define only label, "
                "amplitude_scaling, and loss"
            )
        label = item["label"]
        amplitude_scaling = item["amplitude_scaling"]
        loss_name = item["loss"]
        if not isinstance(label, str) or not label:
            raise ConfigurationError("condition label must be a non-empty string")
        if not isinstance(amplitude_scaling, str) or not isinstance(loss_name, str):
            raise ConfigurationError("condition amplitude_scaling and loss must be strings")
        if label in by_label:
            raise ConfigurationError(f"duplicate experiment condition label: {label!r}")
        by_label[label] = (amplitude_scaling, loss_name)
    expected_by_label = {
        label: (amplitude_scaling, loss_name)
        for label, amplitude_scaling, loss_name in _AMPLITUDE_BALANCING_CONDITIONS
    }
    if by_label != expected_by_label:
        raise ConfigurationError(
            "experiment.conditions must contain exactly global_rms_control (global_rms, l2), "
            "per_trace_rms (per_trace_rms, l2), and huber_global_rms (global_rms, huber)"
        )
    return _AMPLITUDE_BALANCING_CONDITIONS


def _validated_fixed_condition_set(
    value: object,
    *,
    expected: tuple[tuple[str, str, bool, str], ...],
    mismatch_message: str,
) -> tuple[tuple[str, str, bool, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("experiment.conditions must be a sequence")
    by_label: dict[str, tuple[str, bool, str]] = {}
    expected_keys = {"label", "batch_mode", "correlation", "amplitude_scaling"}
    for item in value:
        if not isinstance(item, Mapping):
            raise ConfigurationError("each experiment condition must be a mapping")
        if set(item) != expected_keys:
            raise ConfigurationError(
                "each condition must define only label, batch_mode, "
                "correlation, and amplitude_scaling"
            )
        label = item["label"]
        batch_mode = item["batch_mode"]
        correlation = item["correlation"]
        amplitude_scaling = item["amplitude_scaling"]
        if not isinstance(label, str) or not label:
            raise ConfigurationError("condition label must be a non-empty string")
        if not isinstance(batch_mode, str) or not isinstance(amplitude_scaling, str):
            raise ConfigurationError("condition batch_mode and amplitude_scaling must be strings")
        if not isinstance(correlation, bool):
            raise ConfigurationError("condition correlation must be a boolean")
        if label in by_label:
            raise ConfigurationError(f"duplicate experiment condition label: {label!r}")
        by_label[label] = (batch_mode, correlation, amplitude_scaling)
    expected_by_label = {
        label: (batch_mode, correlation, amplitude_scaling)
        for label, batch_mode, correlation, amplitude_scaling in expected
    }
    if by_label != expected_by_label:
        raise ConfigurationError(mismatch_message)
    return expected


def _validated_full_trace_batch_conditions(
    value: object,
) -> tuple[tuple[str, str, bool, str], ...]:
    return _validated_fixed_condition_set(
        value,
        expected=_FULL_TRACE_BATCH_ABLATION_CONDITIONS,
        mismatch_message=(
            "experiment.conditions must contain exactly the five fixed Study 014 conditions: "
            "small_batch_control, full_trace_batch, full_trace_batch_correlation, "
            "full_trace_batch_per_trace_rms, and full_trace_batch_correlation_per_trace_rms"
        ),
    )


def _validated_strong_fit_budget_extension_conditions(
    value: object,
) -> tuple[tuple[str, str, bool, str], ...]:
    return _validated_fixed_condition_set(
        value,
        expected=_STRONG_FIT_BUDGET_EXTENSION_CONDITIONS,
        mismatch_message=(
            "experiment.conditions must contain exactly the single fixed Study 015 "
            "condition: full_trace_batch_per_trace_rms"
        ),
    )


def _validated_continuation_trace_counts(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ConfigurationError("experiment.trace_counts must be a non-empty sequence")
    counts = tuple(_positive_integer(item, "experiment.trace_counts item") for item in value)
    if tuple(sorted(set(counts))) != counts:
        raise ConfigurationError("experiment.trace_counts must be strictly increasing and unique")
    return counts


def _validate_full_batch_tensors(
    coordinates: torch.Tensor,
    targets: torch.Tensor,
    point_count: int,
) -> None:
    expected_coordinate_shape = (point_count, len(MODEL_COORDINATE_ORDER))
    if coordinates.shape != expected_coordinate_shape:
        raise RuntimeError(
            f"full-batch coordinates have shape {tuple(coordinates.shape)}, "
            f"expected {expected_coordinate_shape}"
        )
    if targets.shape != (point_count, 1):
        raise RuntimeError(
            f"full-batch targets have shape {tuple(targets.shape)}, expected {(point_count, 1)}"
        )


def _build_model(config: Mapping[str, object]) -> Siren:
    return Siren(
        input_features=get_required_config_value(config, "model.input_features"),
        hidden_width=get_required_config_value(config, "model.hidden_width"),
        hidden_layers=get_required_config_value(config, "model.hidden_layers"),
        output_features=1,
        omega_0=get_required_config_value(config, "model.omega_0"),
        hidden_omega=get_required_config_value(config, "model.hidden_omega"),
    )


def _build_inputs_lock(
    *,
    interim_files: Mapping[str, object],
    processed_files: Mapping[str, object],
    preparation_contract: Mapping[str, object],
    selected_array_rows: np.ndarray,
    trace_count: int,
    sample_count: int,
    point_count: int,
    random_seed: int,
    selection_method: str = _SELECTION_METHOD,
    split_counts: Mapping[str, object] | None = None,
    training_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    inputs_lock: dict[str, object] = {
        "interim_files": dict(interim_files),
        "processed_files": dict(processed_files),
        "preparation": dict(preparation_contract),
        "selection": {
            "source_split": TRAIN_SPLIT,
            "method": selection_method,
            "random_seed": random_seed,
            "trace_count": trace_count,
            "sample_count": sample_count,
            "point_count": point_count,
            "full_time_domain": True,
            "selected_array_rows": [int(value) for value in selected_array_rows],
        },
    }
    if training_contract is not None:
        inputs_lock["training"] = dict(training_contract)
    if split_counts is not None:
        inputs_lock["split"] = {
            "counts": dict(split_counts),
            "training_source": TRAIN_SPLIT,
        }
    return inputs_lock


def _resolved_condition_config(
    config: Mapping[str, object],
    *,
    label: str,
    batch_mode: str,
    full_batch: bool,
    replacement: bool,
) -> dict[str, object]:
    resolved = deepcopy(dict(config))
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment configuration must be a mapping")
    experiment["active_condition"] = {
        "label": label,
        "batch_mode": batch_mode,
        "full_batch": full_batch,
        "replacement": replacement,
    }
    return resolved


def _resolved_official_siren_condition_config(
    config: Mapping[str, object],
    *,
    label: str,
    omega_0: float,
    hidden_omega: float,
) -> dict[str, object]:
    resolved = deepcopy(dict(config))
    model = resolved.get("model")
    if not isinstance(model, dict):
        raise ConfigurationError("model configuration must be a mapping")
    model.update({"omega_0": omega_0, "hidden_omega": hidden_omega})
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment configuration must be a mapping")
    experiment["active_condition"] = {
        "label": label,
        "omega_0": omega_0,
        "hidden_omega": hidden_omega,
    }
    return resolved


def _resolved_amplitude_balancing_condition_config(
    config: Mapping[str, object],
    *,
    label: str,
    amplitude_scaling: str,
    loss_name: str,
    huber_delta: float | None,
) -> dict[str, object]:
    resolved = deepcopy(dict(config))
    training = resolved.get("training")
    if not isinstance(training, dict):
        raise ConfigurationError("training configuration must be a mapping")
    training["loss"] = loss_name
    if huber_delta is not None:
        training["huber_delta"] = huber_delta
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment configuration must be a mapping")
    active_condition: dict[str, object] = {
        "label": label,
        "amplitude_scaling": amplitude_scaling,
        "loss": loss_name,
    }
    if huber_delta is not None:
        active_condition["huber_delta"] = huber_delta
    experiment["active_condition"] = active_condition
    return resolved


def _resolved_full_trace_batch_condition_config(
    config: Mapping[str, object],
    *,
    label: str,
    batch_mode: str,
    correlation_weight: float,
    amplitude_scaling: str,
) -> dict[str, object]:
    resolved = deepcopy(dict(config))
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment configuration must be a mapping")
    experiment["active_condition"] = {
        "label": label,
        "batch_mode": batch_mode,
        "correlation_weight": correlation_weight,
        "amplitude_scaling": amplitude_scaling,
    }
    return resolved


def _condition_best_and_final_reports(
    metrics: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the best-step and final report rows recorded in condition metrics."""
    history = metrics.get("history")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence) or not history:
        raise RuntimeError("condition history must be a non-empty sequence")
    history_rows: list[Mapping[str, object]] = []
    for row in history:
        if not isinstance(row, Mapping):
            raise RuntimeError("condition history rows must be mappings")
        history_rows.append(row)
    best_step = int(metrics["best_step"])
    try:
        best_row = next(row for row in history_rows if int(row["step"]) == best_step)
    except StopIteration as error:
        raise RuntimeError("best report step is absent from condition history") from error
    return dict(best_row), dict(history_rows[-1])


def _official_siren_condition_summary(
    run_directory: str,
    *,
    omega_0: float,
    hidden_omega: float,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    best_row, final_row = _condition_best_and_final_reports(metrics)
    return {
        "label": str(metrics["condition"]),
        "run_directory": run_directory,
        "omega_0": omega_0,
        "hidden_omega": hidden_omega,
        "classification": str(metrics["classification"]),
        "best_report": best_row,
        "final_report": final_row,
        "updates_completed": int(metrics["updates_completed"]),
    }


def _amplitude_balancing_condition_summary(
    run_directory: str,
    *,
    amplitude_scaling: str,
    loss_name: str,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    best_row, final_row = _condition_best_and_final_reports(metrics)
    summary: dict[str, object] = {
        "label": str(metrics["condition"]),
        "run_directory": run_directory,
        "amplitude_scaling": amplitude_scaling,
        "loss_name": loss_name,
        "classification": str(metrics["classification"]),
        "best_report": best_row,
        "final_report": final_row,
        "updates_completed": int(metrics["updates_completed"]),
    }
    if "huber_delta" in metrics:
        summary["huber_delta"] = metrics["huber_delta"]
    return summary


def _full_trace_batch_condition_summary(
    run_directory: str,
    *,
    batch_mode: str,
    correlation_weight: float,
    amplitude_scaling: str,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    best_row, final_row = _condition_best_and_final_reports(metrics)
    return {
        "label": str(metrics["condition"]),
        "run_directory": run_directory,
        "batch_mode": batch_mode,
        "batch_size": int(metrics["batch_size"]),
        "correlation_weight": correlation_weight,
        "amplitude_scaling": amplitude_scaling,
        "classification": str(metrics["classification"]),
        "best_report": best_row,
        "final_report": final_row,
        "updates_completed": int(metrics["updates_completed"]),
    }


def _summary_run(run_id: str, metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "condition",
        "batch_mode",
        "batch_size",
        "full_batch",
        "replacement",
        "trace_count",
        "sample_count",
        "point_count",
        "point_evaluations",
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
        "classification",
    )
    summary = {"run_id": run_id, **{key: metrics[key] for key in keys}}
    if "traces_per_update" in metrics:
        summary["traces_per_update"] = metrics["traces_per_update"]
    if "samples_per_trace" in metrics:
        summary.update(
            {
                "samples_per_trace": metrics["samples_per_trace"],
                "temporal_patch_overlap_fraction": metrics["temporal_patch_overlap_fraction"],
                "patch_starts": metrics["patch_starts"],
                "shared_temporal_patch": metrics["shared_temporal_patch"],
            }
        )
    if "correlation_weight" in metrics:
        summary.update(
            {
                "correlation_weight": metrics["correlation_weight"],
                "correlation_eps": metrics["correlation_eps"],
                "loss_semantics": metrics["loss_semantics"],
            }
        )
    return summary


def _build_run_metadata(
    *,
    study_id: str,
    condition: str,
    batch_mode: str,
    batch_size: int,
    trace_count: int,
    sample_count: int,
    point_count: int,
    point_evaluations: int,
    full_batch: bool,
    replacement: bool,
    git_commit: str,
    started_at_utc: str,
    device: str,
    random_seed: int,
    updates_completed: int,
    traces_per_update: int | None = None,
    samples_per_trace: int | None = None,
    temporal_patch_overlap_fraction: float | None = None,
    patch_starts: Sequence[int] | None = None,
    shared_temporal_patch: bool | None = None,
    correlation_weight: float | None = None,
    correlation_eps: float | None = None,
    loss_semantics: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "study_id": study_id,
        "condition": condition,
        "batch_mode": batch_mode,
        "batch_size": batch_size,
        "trace_count": trace_count,
        "sample_count": sample_count,
        "point_count": point_count,
        "point_evaluations": point_evaluations,
        "full_batch": full_batch,
        "replacement": replacement,
        "git_commit": git_commit,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_timestamp(),
        "status": "success",
        "device": device,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "random_seed": random_seed,
        "updates_completed": updates_completed,
    }
    if traces_per_update is not None:
        metadata["traces_per_update"] = traces_per_update
    patch_metadata = (
        samples_per_trace,
        temporal_patch_overlap_fraction,
        patch_starts,
        shared_temporal_patch,
    )
    if any(value is not None for value in patch_metadata):
        if any(value is None for value in patch_metadata):
            raise ValueError(
                "samples_per_trace, temporal_patch_overlap_fraction, patch_starts, and "
                "shared_temporal_patch must be provided together"
            )
        metadata.update(
            {
                "samples_per_trace": samples_per_trace,
                "temporal_patch_overlap_fraction": temporal_patch_overlap_fraction,
                "patch_starts": list(patch_starts),
                "shared_temporal_patch": shared_temporal_patch,
            }
        )
    correlation_metadata = (correlation_weight, correlation_eps, loss_semantics)
    if any(value is not None for value in correlation_metadata):
        if any(value is None for value in correlation_metadata):
            raise ValueError(
                "correlation_weight, correlation_eps, and loss_semantics must be provided together"
            )
        metadata.update(
            {
                "correlation_weight": correlation_weight,
                "correlation_eps": correlation_eps,
                "loss_semantics": loss_semantics,
            }
        )
    return metadata


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


def _validated_patch_starts(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("patch_starts must be a non-empty sequence")
    starts: list[int] = []
    for start in value:
        if isinstance(start, bool) or not isinstance(start, Integral) or int(start) < 0:
            raise ValueError("patch_starts must contain non-negative integers")
        starts.append(int(start))
    if starts != sorted(set(starts)):
        raise ValueError("patch_starts must be strictly increasing and unique")
    return tuple(starts)


def _validated_overlap_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number within [0, 1)")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError(f"{name} must be a finite number within [0, 1)")
    return converted


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _nonnegative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return converted


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
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
