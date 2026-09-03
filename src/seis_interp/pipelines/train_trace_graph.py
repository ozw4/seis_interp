"""Train a trace-node graph interpolator on leakage-safe whole-FFID splits."""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seis_interp import config_values, run_records
from seis_interp.configuration import (
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.data import whole_shot
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.data.trace_store import TRACES_FILE_NAME, canonical_source_files
from seis_interp.evaluation import formal_scope, oracle_trace_snr
from seis_interp.evaluation import whole_shot as whole_shot_evaluation
from seis_interp.models.trace_graph_interpolator import (
    ATTENTION_TIME_RESOLUTIONS,
    GRAPH_MODES,
    NODE_STATIC_FEATURE_NAMES,
    POOLED_ATTENTION_TIME_RESOLUTION,
    TraceGraphInterpolator,
)
from seis_interp.pipelines.train_siren import (
    PROCESSED_INPUT_FILE_NAMES,
    RANDOM_COMPLETE_TRACES_BATCH_MODE,
    _configured_trace_amplitude_filter,
    _load_processed_dataset,
    _split_counts,
    _validate_preparation_data,
    _validate_split_table,
    _validated_preparation_contract,
)
from seis_interp.processing import trace_canonicalization, trace_selection
from seis_interp.processing.c3_receiver_grid import (
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    receiver_grid_offsets,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from seis_interp.training import randomness, whole_shot_training_loop
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
    extract_per_trace_rms_scaled_rows,
)
from seis_interp.training.trace_graph_checkpoints import load_trace_graph_checkpoint
from seis_interp.training.trace_graph_trainer import (
    TraceGraphTrainingResult,
    train_trace_graph_interpolator,
)
from seis_interp.training.whole_shot_batches import (
    RandomWholeShotBatchProvider,
)

MODEL_NAME = "trace_graph_interpolator"
LOSS_NAME = "masked_l2_spectrum_slope_amplitude"
OPTIMIZER_NAME = "adamw"
LEARNING_RATE_SCHEDULE = "cosine"
MIXED_PRECISION = "bfloat16"

ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class _TrainingSettings:
    random_seed: int
    hidden_width: int
    graph_mode: str
    attention_time_resolution: str
    use_gradient_checkpointing: bool
    refinement_passes: int
    message_passing_rounds: int
    time_downsample_factor: int
    stem_kernel_size: int
    temporal_kernel_size: int
    temporal_dilations: tuple[int, ...]
    spatial_kernel_size: int
    attention_width: int
    distance_epsilon: float
    source_gather_count: int
    learning_rate: float
    weight_decay: float
    minimum_learning_rate: float
    total_steps: int
    batch_size: int
    target_sampling: str
    exclude_target_ffid_neighbors: bool
    neighbor_dropout: float
    spectrum_weight: float
    slope_weight: float
    amplitude_weight: float
    evaluation_interval_steps: int
    validation_batch_size: int
    training_audit_count: int
    mixed_precision: str
    device: str
    ffid_range: tuple[int, int] | None
    success_threshold_db: float
    duplicate_physical_coordinate_policy: str
    required_eligible_ffid_count: int
    required_sample_count: int
    required_effective_split_counts: Mapping[str, int]
    required_ffid_split_counts: Mapping[str, int]
    required_fully_excluded_ffids: tuple[int, ...]


def train_trace_graph_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Train one trace-graph interpolator and write an immutable reproducible run."""
    output_directory = Path(output_dir)
    run_records.check_new_output_directory(output_directory)
    started_at_utc = run_records.utc_timestamp()
    git_commit = run_records.current_git_commit()
    config = load_resolved_config(Path(config_path))
    settings = _validated_settings(config, device_override=device_override)
    resolved_config = deepcopy(config)
    resolved_config["training"]["device"] = settings.device

    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    split_table, _normalization, preparation = _load_processed_dataset(processed_directory)
    preliminary_trace_table = pd.read_parquet(interim_directory / TRACES_FILE_NAME)
    split_rows = _validate_split_table(
        split_table,
        len(preliminary_trace_table),
        allow_excluded=True,
    )
    preliminary_joined = trace_selection.join_trace_splits(
        preliminary_trace_table,
        split_table,
        split_rows,
    )
    preliminary_canonical, preliminary_duplicate_audit = (
        trace_canonicalization.canonicalize_eligible_physical_coordinates(preliminary_joined)
    )
    selected_preliminary = trace_selection.select_eligible_traces(
        preliminary_canonical,
        ffid_range=settings.ffid_range,
    )
    amplitude_rows_to_read = selected_preliminary.loc[
        selected_preliminary[SPLIT_COLUMN].isin((TRAIN_SPLIT, VALIDATION_SPLIT)),
        "array_row",
    ].to_numpy(dtype=np.int64)
    dataset = load_interim_trace_dataset(
        interim_directory,
        memory_map_amplitudes=True,
        amplitude_validation_rows=amplitude_rows_to_read,
    )
    interim_files = run_records.file_hashes(interim_directory, INTERIM_FILE_NAMES)
    processed_files = run_records.file_hashes(processed_directory, PROCESSED_INPUT_FILE_NAMES)
    split_counts = _split_counts(split_table)
    _validate_preparation_data(preparation, dataset.metadata, split_counts, interim_files)
    trace_amplitude_filter = _configured_trace_amplitude_filter(resolved_config)
    if trace_amplitude_filter is None:
        raise ConfigurationError("sampling.trace_amplitude_filter is required")
    preparation_contract = _validated_preparation_contract(
        preparation,
        resolved_config,
        split_table=split_table,
        trace_table=dataset.trace_table,
        trace_amplitude_filter=trace_amplitude_filter,
        batch_mode=RANDOM_COMPLETE_TRACES_BATCH_MODE,
        allow_whole_ffid_split=True,
    )
    if preparation_contract.get("split_scope") != "whole_ffid":
        raise ConfigurationError("trace graph training requires sampling.split_scope='whole_ffid'")
    joined_table = trace_selection.join_trace_splits(dataset.trace_table, split_table, split_rows)
    canonical_table, duplicate_audit = (
        trace_canonicalization.canonicalize_eligible_physical_coordinates(joined_table)
    )
    if duplicate_audit != preliminary_duplicate_audit:
        raise RuntimeError("duplicate-coordinate audit changed while loading amplitudes")
    selected_table = trace_selection.select_eligible_traces(
        canonical_table, ffid_range=settings.ffid_range
    )
    trace_selection.validate_selected_split_coverage(selected_table, split_scope="whole_ffid")
    selection_contract = trace_selection.build_trace_selection_contract(
        canonical_table,
        selected_table,
        sample_count=len(dataset.time_s),
        configured_ffid_range=settings.ffid_range,
    )
    configured_scope_audit = formal_scope.build_formal_scope_audit(
        ffid_range=settings.ffid_range,
        exclude_target_ffid_neighbors=settings.exclude_target_ffid_neighbors,
        required_eligible_ffid_count=settings.required_eligible_ffid_count,
        required_sample_count=settings.required_sample_count,
        required_effective_split_counts=settings.required_effective_split_counts,
        required_ffid_split_counts=settings.required_ffid_split_counts,
        required_fully_excluded_ffids=settings.required_fully_excluded_ffids,
        selection_contract=selection_contract,
        preparation_contract=preparation_contract,
    )
    if settings.ffid_range is None and not configured_scope_audit["scope_success"]:
        failed_checks = [
            name for name, passed in configured_scope_audit["checks"].items() if not passed
        ]
        raise ValueError(
            "formal trace-graph run does not match its required survey scope: "
            f"{failed_checks}; configure training.ffid_range for a diagnostic subset"
        )

    receiver_x_offsets, receiver_y_offsets = receiver_grid_offsets(selected_table)
    selected_split = selected_table[SPLIT_COLUMN].to_numpy()
    train_positions = np.flatnonzero(selected_split == TRAIN_SPLIT).astype(np.int64)
    validation_positions = np.flatnonzero(selected_split == VALIDATION_SPLIT).astype(np.int64)
    train_arrays = selected_table.iloc[train_positions]["array_row"].to_numpy(dtype=np.int64)
    validation_arrays = selected_table.iloc[validation_positions]["array_row"].to_numpy(
        dtype=np.int64
    )
    train_amplitudes_host = extract_per_trace_rms_scaled_rows(dataset.amplitudes, train_arrays)
    validation_amplitudes_host = extract_per_trace_rms_scaled_rows(
        dataset.amplitudes, validation_arrays
    )
    device = torch.device(settings.device)
    randomness.seed_global_model_initialization(settings.random_seed, device=device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    train_components = whole_shot.build_gather_tensors(
        selected_table.iloc[train_positions],
        train_amplitudes_host,
        receiver_x_offsets=receiver_x_offsets,
        receiver_y_offsets=receiver_y_offsets,
        device=device,
    )
    validation_components = whole_shot.build_gather_tensors(
        selected_table.iloc[validation_positions],
        validation_amplitudes_host,
        receiver_x_offsets=receiver_x_offsets,
        receiver_y_offsets=receiver_y_offsets,
        device=device,
    )
    del train_amplitudes_host, validation_amplitudes_host
    train_ffids, train_sources, train_gathers, train_availability = train_components
    val_ffids, val_sources, val_gathers, val_availability = validation_components
    source = whole_shot.WholeShotTensorSource(
        train_ffids=train_ffids,
        train_source_coordinates_m=train_sources,
        train_gathers=train_gathers,
        train_availability=train_availability,
        source_gather_count=settings.source_gather_count,
        device=device,
    )
    train_targets = source.build_targets(
        ffids=train_ffids,
        source_coordinates_m=train_sources,
        gathers=train_gathers,
        availability=train_availability,
    )
    validation_targets = source.build_targets(
        ffids=val_ffids,
        source_coordinates_m=val_sources,
        gathers=val_gathers,
        availability=val_availability,
    )
    target_sampling_generator = (
        torch.Generator(device=device).manual_seed(
            randomness.target_sampling_seed(settings.random_seed, settings.target_sampling)
        )
        if settings.target_sampling == randomness.EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        else None
    )
    batch_provider = RandomWholeShotBatchProvider(
        source,
        train_targets,
        target_sampling=settings.target_sampling,
        target_generator=target_sampling_generator,
    )
    validation_evaluator = whole_shot_evaluation.WholeShotGlobalSnrEvaluator(
        source,
        validation_targets,
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    )
    model = TraceGraphInterpolator(
        width=settings.hidden_width,
        graph_mode=settings.graph_mode,
        attention_time_resolution=settings.attention_time_resolution,
        use_gradient_checkpointing=settings.use_gradient_checkpointing,
        refinement_passes=settings.refinement_passes,
        message_passing_rounds=settings.message_passing_rounds,
        time_downsample_factor=settings.time_downsample_factor,
        stem_kernel_size=settings.stem_kernel_size,
        temporal_kernel_size=settings.temporal_kernel_size,
        temporal_dilations=settings.temporal_dilations,
        spatial_kernel_size=settings.spatial_kernel_size,
        attention_width=settings.attention_width,
        distance_epsilon=settings.distance_epsilon,
    )
    generator = torch.Generator(device=device).manual_seed(
        settings.random_seed + randomness.NEIGHBOR_DROPOUT_SEED_OFFSET
    )
    result = train_trace_graph_interpolator(
        model,
        batch_provider,
        validation_evaluator,
        device=device,
        generator=generator,
        checkpoint_path=output_directory / run_records.CHECKPOINT_RELATIVE_PATH,
        total_steps=settings.total_steps,
        batch_size=settings.batch_size,
        neighbor_dropout=settings.neighbor_dropout,
        spectrum_weight=settings.spectrum_weight,
        slope_weight=settings.slope_weight,
        amplitude_weight=settings.amplitude_weight,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        validation_interval=settings.evaluation_interval_steps,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
        training_ffid_count=train_targets.ffid_count,
        training_trace_count=train_targets.trace_count,
        reporter=progress_reporter,
    )
    checkpoint = load_trace_graph_checkpoint(
        output_directory / run_records.CHECKPOINT_RELATIVE_PATH,
        device=device,
    )
    if (
        checkpoint.best_step != result.best_step
        or checkpoint.best_validation_global_snr_db != result.best_validation_global_snr_db
    ):
        raise RuntimeError("best checkpoint metadata does not match trainer result")
    best_validation = validation_evaluator.evaluate(checkpoint.model)
    checkpoint_revalidation_matches = math.isclose(
        best_validation.raw_global_snr_db,
        result.best_validation_global_snr_db,
        rel_tol=formal_scope.CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        abs_tol=formal_scope.CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
    )
    if not checkpoint_revalidation_matches:
        raise RuntimeError("loaded best checkpoint does not reproduce validation S/N")

    training_audit_targets = whole_shot_evaluation.sample_training_audit_targets(
        train_targets,
        trace_count=settings.training_audit_count,
        random_seed=settings.random_seed + randomness.TRAINING_AUDIT_SEED_OFFSET,
    )
    training_audit = whole_shot_evaluation.WholeShotGlobalSnrEvaluator(
        source,
        training_audit_targets,
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    ).evaluate(checkpoint.model)
    availability_contract = {
        TRAIN_SPLIT: source.audit(train_targets),
        VALIDATION_SPLIT: source.audit(validation_targets),
    }
    collision_audit = whole_shot_evaluation.source_collision_audit(
        train_sources,
        val_sources,
        duplicate_audit=duplicate_audit,
    )
    amplitude_access = {
        "value_rows_materialized_by_split": {
            TRAIN_SPLIT: True,
            VALIDATION_SPLIT: True,
            TEST_SPLIT: False,
            EXCLUDED_SPLIT: False,
        },
        "neighbor_amplitude_source_split": TRAIN_SPLIT,
        "validation_targets_used_for_checkpoint_selection": True,
        "test_targets_used_for_checkpoint_selection": False,
        "full_file_bytes_hashed": True,
    }
    scope_audit = formal_scope.complete_whole_shot_formal_scope_audit(
        configured_scope_audit,
        validation_metric_domain=checkpoint.validation_metric_domain,
        availability_contract=availability_contract,
        collision_audit=collision_audit,
        amplitude_access=amplitude_access,
        checkpoint_revalidation_matches=checkpoint_revalidation_matches,
        selected_metric=result.best_validation_global_snr_db,
        recomputed_metric=best_validation.raw_global_snr_db,
    )
    model_contract = _model_contract(checkpoint.model, settings=settings)
    receiver_grid_contract = {
        "shape": [RECEIVER_X_COUNT, RECEIVER_Y_COUNT],
        "relative_receiver_x_m": [float(value) for value in receiver_x_offsets],
        "relative_receiver_y_m": [float(value) for value in receiver_y_offsets],
        "missing_receiver_cells": "zero_filled_and_masked",
    }
    training_contract = _training_contract(settings, result, batch_provider)
    neighborhood_contract = {
        "type": whole_shot.NEIGHBORHOOD_TYPE,
        "distance": whole_shot.SOURCE_DISTANCE,
        "source_gather_count": settings.source_gather_count,
        "source_split": TRAIN_SPLIT,
        "target_ffid_policy": "exclude_exact_ffid",
        "zero_source_distance_policy": "exclude",
        "neighbor_dropout_scope": "whole_source_gather",
    }
    checkpoint_contract = {
        "path": run_records.CHECKPOINT_RELATIVE_PATH.as_posix(),
        "selection_metric": oracle_trace_snr.PRIMARY_METRIC,
        "best_step": result.best_step,
        "stored_validation_global_snr_db": result.best_validation_global_snr_db,
        "recomputed_validation_global_snr_db": best_validation.raw_global_snr_db,
        "revalidation_relative_tolerance": formal_scope.CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        "revalidation_absolute_tolerance": formal_scope.CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
        "revalidation_matches": checkpoint_revalidation_matches,
        "graph_mode": checkpoint.graph_mode,
    }
    base_metrics = asdict(result)
    base_metrics["history"] = [dict(value) for value in result.history]
    metrics = whole_shot_evaluation.build_whole_shot_metrics(
        base_metrics,
        best_validation=best_validation,
        training_audit=training_audit,
        success_threshold_db=settings.success_threshold_db,
        selection_contract=selection_contract,
        duplicate_audit=duplicate_audit,
        collision_audit=collision_audit,
        availability_contract=availability_contract,
        amplitude_access=amplitude_access,
        scope_audit=scope_audit,
        checkpoint_contract=checkpoint_contract,
    )
    inputs_lock = {
        "interim_files": interim_files,
        "processed_files": processed_files,
        "source_files": [dict(value) for value in canonical_source_files(dataset.metadata)],
        "preparation": preparation_contract,
        "selection": selection_contract,
        "prepared_split_counts": {
            **split_counts,
            EXCLUDED_SPLIT: int(split_table[SPLIT_COLUMN].eq(EXCLUDED_SPLIT).sum()),
        },
        "duplicate_physical_coordinates": duplicate_audit,
        "amplitude_access": amplitude_access,
        "model": model_contract,
        "receiver_grid": receiver_grid_contract,
        "neighborhood": neighborhood_contract,
        "target_coordinates": {
            "order": list(whole_shot.TARGET_COORDINATES),
            "fit_split": TRAIN_SPLIT,
            "scaling": whole_shot.TARGET_COORDINATE_SCALING,
            "minimum": list(source.coordinate_min),
            "maximum": list(source.coordinate_max),
            "constant_axis_value": 0.0,
        },
        "training": training_contract,
        "split_counts": selection_contract["split_counts"],
        "neighbor_availability": availability_contract,
        "collision_audit": collision_audit,
        "formal_success_scope": scope_audit,
        "checkpoint": checkpoint_contract,
    }
    run_metadata = {
        "git_commit": git_commit,
        "started_at_utc": started_at_utc,
        "finished_at_utc": run_records.utc_timestamp(),
        "status": "success",
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "random_seed": settings.random_seed,
        "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        "amplitude_scaling": PER_TRACE_RMS_SCALING,
        "model": model_contract,
        "selection": selection_contract,
        "duplicate_physical_coordinates": duplicate_audit,
        "receiver_grid": receiver_grid_contract,
        "neighborhood": neighborhood_contract,
        "training": training_contract,
        "formal_success_scope": scope_audit,
        "checkpoint": checkpoint_contract,
        "environment": run_records.runtime_resource_metadata(device),
    }
    run_records.write_run_outputs(
        output_directory,
        resolved_config,
        inputs_lock,
        metrics,
        run_metadata,
    )
    return metrics


def _validated_settings(
    config: Mapping[str, object],
    *,
    device_override: str | None,
) -> _TrainingSettings:
    config_values.require_exact(config, "model.name", MODEL_NAME)
    config_values.require_exact(
        config,
        "model.target_coordinate_scaling",
        whole_shot.TARGET_COORDINATE_SCALING,
    )
    target_coordinates = config_values.validated_target_coordinate_names(
        get_required_config_value(config, "model.target_coordinates")
    )
    if target_coordinates != whole_shot.TARGET_COORDINATES:
        raise ConfigurationError(
            f"model.target_coordinates must be {list(whole_shot.TARGET_COORDINATES)!r}"
        )
    neighborhood = get_required_config_value(config, "model.neighborhood")
    if not isinstance(neighborhood, Mapping):
        raise ConfigurationError("model.neighborhood must be a mapping")
    if neighborhood.get("type") != whole_shot.NEIGHBORHOOD_TYPE:
        raise ConfigurationError(
            f"model.neighborhood.type must be {whole_shot.NEIGHBORHOOD_TYPE!r}"
        )
    if neighborhood.get("distance") != whole_shot.SOURCE_DISTANCE:
        raise ConfigurationError(
            f"model.neighborhood.distance must be {whole_shot.SOURCE_DISTANCE!r}"
        )
    source_gather_count = config_values.positive_integer(
        neighborhood.get("source_gather_count"),
        "model.neighborhood.source_gather_count",
    )
    graph_mode = get_required_config_value(config, "model.graph_mode")
    if graph_mode not in GRAPH_MODES:
        raise ConfigurationError(f"model.graph_mode must be one of {GRAPH_MODES}")
    model_section = config.get("model")
    if not isinstance(model_section, Mapping):
        raise ConfigurationError("model must be a mapping")
    attention_time_resolution = model_section.get(
        "attention_time_resolution",
        POOLED_ATTENTION_TIME_RESOLUTION,
    )
    if attention_time_resolution not in ATTENTION_TIME_RESOLUTIONS:
        raise ConfigurationError(
            f"model.attention_time_resolution must be one of {ATTENTION_TIME_RESOLUTIONS}"
        )
    use_gradient_checkpointing = model_section.get("use_gradient_checkpointing", False)
    if not isinstance(use_gradient_checkpointing, bool):
        raise ConfigurationError("model.use_gradient_checkpointing must be a boolean")
    refinement_passes = config_values.positive_integer(
        model_section.get("refinement_passes", 1),
        "model.refinement_passes",
    )
    config_values.require_exact(config, "sampling.split_scope", "whole_ffid")
    config_values.require_exact(
        config,
        "sampling.duplicate_physical_coordinate_policy",
        trace_canonicalization.DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    )
    config_values.require_exact(config, "training.amplitude_scaling", PER_TRACE_RMS_SCALING)
    config_values.require_exact(config, "training.loss", LOSS_NAME)
    config_values.require_exact(config, "training.optimizer", OPTIMIZER_NAME)
    config_values.require_exact(config, "training.learning_rate_schedule", LEARNING_RATE_SCHEDULE)
    config_values.require_exact(config, "training.mixed_precision", MIXED_PRECISION)
    config_values.require_exact(
        config, "evaluation.primary_metric", oracle_trace_snr.PRIMARY_METRIC
    )
    config_values.require_exact(
        config, "evaluation.comparison", oracle_trace_snr.SUCCESS_COMPARISON
    )
    exclude_target_ffid_neighbors = get_required_config_value(
        config,
        "training.exclude_target_ffid_neighbors",
    )
    if exclude_target_ffid_neighbors is not True:
        raise ConfigurationError("training.exclude_target_ffid_neighbors must be true")
    learning_rate = config_values.positive_float(
        get_required_config_value(config, "training.learning_rate"),
        "training.learning_rate",
    )
    minimum_learning_rate = config_values.positive_float(
        get_required_config_value(config, "training.minimum_learning_rate"),
        "training.minimum_learning_rate",
    )
    if not math.isclose(
        minimum_learning_rate,
        learning_rate * whole_shot_training_loop.MINIMUM_LEARNING_RATE_FACTOR,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ConfigurationError(
            "training.minimum_learning_rate must equal training.learning_rate * 0.03"
        )
    gradient_clip_norm = config_values.positive_float(
        get_required_config_value(config, "training.gradient_clip_norm"),
        "training.gradient_clip_norm",
    )
    if gradient_clip_norm != whole_shot_training_loop.MAX_GRADIENT_NORM:
        raise ConfigurationError(
            f"training.gradient_clip_norm must be {whole_shot_training_loop.MAX_GRADIENT_NORM:g}"
        )
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigurationError("training must be a mapping")
    target_sampling = training.get("target_sampling", randomness.WITH_REPLACEMENT_TARGET_SAMPLING)
    if target_sampling not in {
        randomness.WITH_REPLACEMENT_TARGET_SAMPLING,
        randomness.EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    }:
        raise ConfigurationError("training.target_sampling is unsupported")
    raw_device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(raw_device, str) or not raw_device:
        raise ConfigurationError("training.device must be a non-empty string")
    temporal_dilations = config_values.validated_positive_integer_list(
        get_required_config_value(config, "model.temporal_dilations"),
        "model.temporal_dilations",
    )
    return _TrainingSettings(
        random_seed=config_values.nonnegative_integer(
            get_required_config_value(config, "project.random_seed"),
            "project.random_seed",
        ),
        hidden_width=config_values.positive_integer(
            get_required_config_value(config, "model.hidden_width"),
            "model.hidden_width",
        ),
        graph_mode=str(graph_mode),
        attention_time_resolution=str(attention_time_resolution),
        use_gradient_checkpointing=use_gradient_checkpointing,
        refinement_passes=refinement_passes,
        message_passing_rounds=config_values.positive_integer(
            get_required_config_value(config, "model.message_passing_rounds"),
            "model.message_passing_rounds",
        ),
        time_downsample_factor=config_values.positive_integer(
            get_required_config_value(config, "model.time_downsample_factor"),
            "model.time_downsample_factor",
        ),
        stem_kernel_size=config_values.odd_positive_integer(
            get_required_config_value(config, "model.stem_kernel_size"),
            "model.stem_kernel_size",
        ),
        temporal_kernel_size=config_values.odd_positive_integer(
            get_required_config_value(config, "model.temporal_kernel_size"),
            "model.temporal_kernel_size",
        ),
        temporal_dilations=temporal_dilations,
        spatial_kernel_size=config_values.odd_positive_integer(
            get_required_config_value(config, "model.spatial_kernel_size"),
            "model.spatial_kernel_size",
        ),
        attention_width=config_values.positive_integer(
            get_required_config_value(config, "model.attention_width"),
            "model.attention_width",
        ),
        distance_epsilon=config_values.positive_float(
            get_required_config_value(config, "model.distance_epsilon"),
            "model.distance_epsilon",
        ),
        source_gather_count=source_gather_count,
        learning_rate=learning_rate,
        weight_decay=config_values.nonnegative_float(
            get_required_config_value(config, "training.weight_decay"),
            "training.weight_decay",
        ),
        minimum_learning_rate=minimum_learning_rate,
        total_steps=config_values.positive_integer(
            get_required_config_value(config, "training.total_steps"),
            "training.total_steps",
        ),
        batch_size=config_values.positive_integer(
            get_required_config_value(config, "training.batch_size"),
            "training.batch_size",
        ),
        target_sampling=str(target_sampling),
        exclude_target_ffid_neighbors=True,
        neighbor_dropout=config_values.probability(
            get_required_config_value(config, "training.neighbor_dropout"),
            "training.neighbor_dropout",
        ),
        spectrum_weight=config_values.nonnegative_float(
            get_required_config_value(config, "training.spectrum_weight"),
            "training.spectrum_weight",
        ),
        slope_weight=config_values.nonnegative_float(
            get_required_config_value(config, "training.slope_weight"),
            "training.slope_weight",
        ),
        amplitude_weight=config_values.nonnegative_float(
            get_required_config_value(config, "training.amplitude_weight"),
            "training.amplitude_weight",
        ),
        evaluation_interval_steps=config_values.positive_integer(
            get_required_config_value(config, "training.evaluation_interval_steps"),
            "training.evaluation_interval_steps",
        ),
        validation_batch_size=config_values.positive_integer(
            get_required_config_value(config, "training.validation_batch_size"),
            "training.validation_batch_size",
        ),
        training_audit_count=config_values.positive_integer(
            get_required_config_value(config, "training.training_audit_count"),
            "training.training_audit_count",
        ),
        mixed_precision=MIXED_PRECISION,
        device=raw_device,
        ffid_range=config_values.optional_ffid_range(config),
        success_threshold_db=config_values.finite_float(
            get_required_config_value(config, "evaluation.success_threshold_db"),
            "evaluation.success_threshold_db",
        ),
        duplicate_physical_coordinate_policy=(
            trace_canonicalization.DUPLICATE_PHYSICAL_COORDINATE_POLICY
        ),
        required_eligible_ffid_count=config_values.positive_integer(
            get_required_config_value(config, "evaluation.required_eligible_ffid_count"),
            "evaluation.required_eligible_ffid_count",
        ),
        required_sample_count=config_values.positive_integer(
            get_required_config_value(config, "evaluation.required_sample_count"),
            "evaluation.required_sample_count",
        ),
        required_effective_split_counts=config_values.validated_effective_split_counts(
            get_required_config_value(config, "evaluation.required_effective_split_counts")
        ),
        required_ffid_split_counts=config_values.validated_ffid_split_counts(
            get_required_config_value(config, "evaluation.required_ffid_split_counts")
        ),
        required_fully_excluded_ffids=config_values.validated_sorted_ffids(
            get_required_config_value(config, "evaluation.required_fully_excluded_ffids"),
            "evaluation.required_fully_excluded_ffids",
        ),
    )


def _model_contract(
    model: TraceGraphInterpolator,
    *,
    settings: _TrainingSettings,
) -> dict[str, object]:
    return {
        "name": MODEL_NAME,
        "hidden_width": model.width,
        "graph_mode": model.graph_mode,
        "attention_time_resolution": model.attention_time_resolution,
        "use_gradient_checkpointing": model.use_gradient_checkpointing,
        "refinement_passes": model.refinement_passes,
        "message_passing_rounds": model.message_passing_rounds,
        "time_downsample_factor": model.time_downsample_factor,
        "stem_kernel_size": model.stem_kernel_size,
        "temporal_kernel_size": model.temporal_kernel_size,
        "temporal_dilations": list(model.temporal_dilations),
        "spatial_kernel_size": model.spatial_kernel_size,
        "attention_width": model.attention_width,
        "distance_epsilon": model.distance_epsilon,
        "node_static_feature_names": list(NODE_STATIC_FEATURE_NAMES),
        "node_definition": "one_trace_per_node_time_as_latent_sequence",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "target_coordinates": list(whole_shot.TARGET_COORDINATES),
        "target_coordinate_scaling": whole_shot.TARGET_COORDINATE_SCALING,
        "receiver_grid_shape": [RECEIVER_X_COUNT, RECEIVER_Y_COUNT],
        "source_gather_count": settings.source_gather_count,
        "prediction_reference": "receiver_wise_inverse_source_distance",
        "residual_decoder_initialization": "zero_final_projection",
    }


def _training_contract(
    settings: _TrainingSettings,
    result: TraceGraphTrainingResult,
    provider: RandomWholeShotBatchProvider,
) -> dict[str, object]:
    return {
        "batch_mode": "random_whole_ffid_gathers",
        "amplitude_scaling": PER_TRACE_RMS_SCALING,
        "training_scale_source": whole_shot_evaluation.TRAINING_SCALE_SOURCE,
        "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        "validation_scale_source": whole_shot_evaluation.VALIDATION_SCALE_SOURCE,
        "loss": LOSS_NAME,
        "loss_target_mask": "eligible_receiver_cells",
        "spectrum_weight": settings.spectrum_weight,
        "slope_weight": settings.slope_weight,
        "amplitude_weight": settings.amplitude_weight,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "learning_rate_schedule": LEARNING_RATE_SCHEDULE,
        "minimum_learning_rate": settings.minimum_learning_rate,
        "minimum_learning_rate_factor": whole_shot_training_loop.MINIMUM_LEARNING_RATE_FACTOR,
        "gradient_clip_norm": whole_shot_training_loop.MAX_GRADIENT_NORM,
        "total_steps": settings.total_steps,
        "batch_size_ffids": settings.batch_size,
        "target_sampling": settings.target_sampling,
        "target_sampling_seed": randomness.target_sampling_seed(
            settings.random_seed,
            settings.target_sampling,
        ),
        "target_sampling_rng_independent_of_neighbor_dropout": (
            settings.target_sampling == randomness.EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        ),
        "exclude_target_ffid_neighbors": settings.exclude_target_ffid_neighbors,
        "neighbor_dropout": settings.neighbor_dropout,
        "neighbor_dropout_seed": settings.random_seed + randomness.NEIGHBOR_DROPOUT_SEED_OFFSET,
        "neighbor_dropout_scope": "whole_source_gather",
        "evaluation_interval_steps": settings.evaluation_interval_steps,
        "validation_batch_size_ffids": settings.validation_batch_size,
        "mixed_precision": settings.mixed_precision,
        "effective_bfloat16": settings.mixed_precision == MIXED_PRECISION
        and torch.device(settings.device).type == "cuda",
        "training_ffid_count": result.training_ffid_count,
        "training_trace_count": result.training_trace_count,
        "drawn_training_ffids": provider.draw_count,
        "unique_training_ffids_seen": provider.unique_target_count,
        "training_audit_trace_count": settings.training_audit_count,
        "training_audit_seed": settings.random_seed + randomness.TRAINING_AUDIT_SEED_OFFSET,
    }
