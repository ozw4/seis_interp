"""Train a joint whole-shot inpainter on leakage-safe whole-FFID splits."""

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
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.data.trace_store import TRACES_FILE_NAME, canonical_source_files
from seis_interp.models.shot_gather_inpainter import (
    DEFAULT_DISTANCE_POWER,
    INVERSE_DISTANCE_SOURCE_WEIGHTING,
    MOMENTS_SOURCE_FEATURE_MODE,
    NO_RECEIVER_POSITION_CONDITIONING,
    ORDERED_RAW_SOURCE_FEATURE_MODE,
    RECEIVER_POSITION_CONDITIONING_MODES,
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    SOURCE_FEATURE_MODES,
    SOURCE_WEIGHTING_MODES,
    ShotGatherInpainter,
)
from seis_interp.pipelines.train_neighbor_inpainter import (
    DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    EPOCH_TARGET_SAMPLING_SEED_OFFSET,
    EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    NEIGHBOR_DROPOUT_SEED_OFFSET,
    PRIMARY_METRIC,
    SUCCESS_COMPARISON,
    TARGET_COORDINATE_SCALING,
    TRAINING_AUDIT_SEED_OFFSET,
    WITH_REPLACEMENT_TARGET_SAMPLING,
    _canonicalize_eligible_physical_coordinates,
    _formal_scope_audit,
    _load_unit_rms_rows,
    _passes_success_threshold,
    _seed_global_model_initialization,
    _snr_db,
)
from seis_interp.pipelines.train_siren import (
    PROCESSED_INPUT_FILE_NAMES,
    RANDOM_COMPLETE_TRACES_BATCH_MODE,
    _configured_trace_amplitude_filter,
    _load_processed_dataset,
    _split_counts,
    _validate_preparation_data,
    _validated_preparation_contract,
)
from seis_interp.processing import trace_selection
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
)
from seis_interp.training.shot_gather_inpainter_checkpoints import (
    SOURCE_WEIGHTING_SCHEMA_VERSION,
    load_shot_gather_inpainter_checkpoint,
)
from seis_interp.training.shot_gather_inpainter_trainer import (
    MAX_GRADIENT_NORM,
    MINIMUM_LEARNING_RATE_FACTOR,
    ShotGatherTrainingResult,
    train_shot_gather_inpainter,
)

MODEL_NAME = "shot_gather_inpainter"
NEIGHBORHOOD_TYPE = "nearest_train_source_gathers"
SOURCE_DISTANCE = "euclidean_source_xy_m"
TARGET_COORDINATES = ("source_x_m", "source_y_m")
LOSS_NAME = "l2_plus_first_difference"
OPTIMIZER_NAME = "adamw"
LEARNING_RATE_SCHEDULE = "cosine"
MIXED_PRECISION = "bfloat16"
TRAINING_SCALE_SOURCE = "training_trace_target_rms"
VALIDATION_SCALE_SOURCE = "validation_trace_target_rms"
CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE = 1.0e-8
CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE = 1.0e-8

ProgressReporter = Callable[[str], None]
_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


@dataclass(frozen=True)
class _TrainingSettings:
    random_seed: int
    hidden_width: int
    stem_kernel_size: int
    residual_kernel_size: int
    temporal_dilations: tuple[int, ...]
    spatial_y_dilations: tuple[int, ...]
    distance_epsilon: float
    distance_power: float
    source_weighting: str
    source_feature_mode: str
    receiver_position_conditioning: str
    source_gather_count: int
    learning_rate: float
    weight_decay: float
    minimum_learning_rate: float
    total_steps: int
    batch_size: int
    target_sampling: str
    exclude_target_ffid_neighbors: bool
    neighbor_dropout: float
    derivative_weight: float
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


@dataclass(frozen=True)
class _GatherTargets:
    ffids: np.ndarray
    source_coordinates_m: np.ndarray
    gathers: torch.Tensor
    availability: torch.Tensor
    neighbor_train_indices: np.ndarray

    @property
    def ffid_count(self) -> int:
        return len(self.ffids)

    @property
    def trace_count(self) -> int:
        return int(torch.count_nonzero(self.availability).cpu())


@dataclass(frozen=True)
class _EvaluationResult:
    raw_global_snr_db: float
    predicted_unit_rms_global_snr_db: float
    signal_energy: float
    error_energy: float
    error_mean_square: float
    ffid_count: int
    trace_count: int


class _ShotGatherTensorSource:
    """Gather nearest source shots exclusively from the compact TRAIN store."""

    def __init__(
        self,
        *,
        train_ffids: np.ndarray,
        train_source_coordinates_m: np.ndarray,
        train_gathers: torch.Tensor,
        train_availability: torch.Tensor,
        source_gather_count: int,
        device: torch.device,
    ) -> None:
        self.train_ffids = np.asarray(train_ffids, dtype=np.int64)
        self.train_source_coordinates_m = np.asarray(
            train_source_coordinates_m,
            dtype=np.float64,
        )
        self.train_gathers = train_gathers
        self.train_availability = train_availability
        self.source_gather_count = source_gather_count
        self.device = device
        if self.train_source_coordinates_m.shape != (len(self.train_ffids), 2):
            raise ValueError("TRAIN source coordinates must have shape (ffids, 2)")
        if len(self.train_ffids) <= source_gather_count:
            raise ValueError(
                "source_gather_count must be smaller than the selected TRAIN FFID count"
            )
        if train_gathers.shape[:3] != (
            len(self.train_ffids),
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        ):
            raise ValueError("TRAIN gathers must use the fixed 8 x 68 receiver grid")
        if train_availability.shape != train_gathers.shape[:3]:
            raise ValueError("TRAIN availability must match the gather receiver grid")
        unique_source_count = len(np.unique(self.train_source_coordinates_m, axis=0))
        if unique_source_count != len(self.train_ffids):
            raise ValueError("selected TRAIN FFIDs must have unique source coordinates")
        self._train_source_coordinates = torch.as_tensor(
            self.train_source_coordinates_m,
            dtype=torch.float32,
            device=device,
        )
        coordinate_min = np.min(self.train_source_coordinates_m, axis=0)
        coordinate_max = np.max(self.train_source_coordinates_m, axis=0)
        self.coordinate_min = tuple(float(value) for value in coordinate_min)
        self.coordinate_max = tuple(float(value) for value in coordinate_max)
        coordinate_range = coordinate_max - coordinate_min
        self._coordinate_min = torch.as_tensor(
            coordinate_min,
            dtype=torch.float32,
            device=device,
        )
        self._coordinate_denominator = torch.as_tensor(
            np.where(coordinate_range > 0.0, coordinate_range, 1.0),
            dtype=torch.float32,
            device=device,
        )

    def build_targets(
        self,
        *,
        ffids: np.ndarray,
        source_coordinates_m: np.ndarray,
        gathers: torch.Tensor,
        availability: torch.Tensor,
    ) -> _GatherTargets:
        target_ffids = np.asarray(ffids, dtype=np.int64)
        target_coordinates = np.asarray(source_coordinates_m, dtype=np.float64)
        neighbor_indices = _nearest_train_source_indices(
            self.train_ffids,
            self.train_source_coordinates_m,
            target_ffids,
            target_coordinates,
            source_gather_count=self.source_gather_count,
        )
        return _GatherTargets(
            ffids=target_ffids,
            source_coordinates_m=target_coordinates,
            gathers=gathers,
            availability=availability,
            neighbor_train_indices=neighbor_indices,
        )

    def inputs(
        self,
        targets: _GatherTargets,
        target_indices: np.ndarray,
        *,
        generator: torch.Generator | None = None,
        neighbor_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(target_indices, dtype=np.int64)
        neighbor_indices = targets.neighbor_train_indices[indices]
        neighbor_index_tensor = torch.as_tensor(
            neighbor_indices,
            dtype=torch.long,
            device=self.device,
        )
        neighbors = self.train_gathers[neighbor_index_tensor]
        availability = self.train_availability[neighbor_index_tensor].clone()
        if neighbor_dropout > 0.0:
            if generator is None:
                raise ValueError("generator is required when neighbor_dropout is positive")
            source_keep = (
                torch.rand(
                    (len(indices), self.source_gather_count, 1, 1),
                    generator=generator,
                    device=self.device,
                )
                >= neighbor_dropout
            )
            availability &= source_keep
        neighbors = neighbors * availability[..., None]
        target_source_coordinates = torch.as_tensor(
            targets.source_coordinates_m[indices],
            dtype=torch.float32,
            device=self.device,
        )
        neighbor_source_coordinates = self._train_source_coordinates[neighbor_index_tensor]
        source_deltas = neighbor_source_coordinates - target_source_coordinates[:, None]
        target_coordinates = (
            target_source_coordinates - self._coordinate_min
        ) / self._coordinate_denominator
        return neighbors, availability, source_deltas, target_coordinates

    def audit(self, targets: _GatherTargets) -> dict[str, object]:
        neighbor_ffids = self.train_ffids[targets.neighbor_train_indices]
        target_ffid_entries = int(np.count_nonzero(neighbor_ffids == targets.ffids[:, None]))
        neighbor_availability = self.train_availability[
            torch.as_tensor(
                targets.neighbor_train_indices,
                dtype=torch.long,
                device=self.device,
            )
        ]
        receiver_coverage = neighbor_availability.any(dim=1)
        target_mask = targets.availability
        uncovered_target_cells = target_mask & ~receiver_coverage
        covered_counts = torch.count_nonzero(receiver_coverage, dim=(1, 2)).cpu().numpy()
        return {
            "target_ffid_count": targets.ffid_count,
            "source_gather_count": self.source_gather_count,
            "neighbor_source_entries": int(neighbor_ffids.size),
            "target_ffid_neighbor_entries": target_ffid_entries,
            "non_train_neighbor_entries": 0,
            "target_trace_count": targets.trace_count,
            "uncovered_target_receiver_cells": int(
                torch.count_nonzero(uncovered_target_cells).cpu()
            ),
            "receiver_cells_with_any_neighbor": {
                "min": int(np.min(covered_counts)),
                "mean": float(np.mean(covered_counts, dtype=np.float64)),
                "max": int(np.max(covered_counts)),
            },
        }


class _RandomTrainGatherProvider:
    """Sample whole TRAIN FFIDs and track exact target coverage."""

    def __init__(
        self,
        source: _ShotGatherTensorSource,
        targets: _GatherTargets,
        *,
        target_sampling: str,
        target_generator: torch.Generator | None,
    ) -> None:
        self.source = source
        self.targets = targets
        self.target_sampling = target_sampling
        self._target_generator = target_generator
        self._epoch_order: torch.Tensor | None = None
        self._epoch_cursor = 0
        self._seen = np.zeros(targets.ffid_count, dtype=bool)
        self.draw_count = 0
        if target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
            if not isinstance(target_generator, torch.Generator):
                raise TypeError("epoch sampling requires a target_generator")
        elif target_sampling != WITH_REPLACEMENT_TARGET_SAMPLING:
            raise ValueError(f"unsupported target sampling mode: {target_sampling!r}")

    @property
    def unique_target_count(self) -> int:
        return int(np.count_nonzero(self._seen))

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        if self.target_sampling == WITH_REPLACEMENT_TARGET_SAMPLING:
            target_indices = torch.randint(
                self.targets.ffid_count,
                (batch_size,),
                generator=generator,
                device=self.source.device,
            )
        else:
            target_indices = self._next_epoch_indices(batch_size)
        target_numpy = target_indices.cpu().numpy()
        self._seen[target_numpy] = True
        self.draw_count += batch_size
        neighbors, availability, source_deltas, target_coordinates = self.source.inputs(
            self.targets,
            target_numpy,
            generator=generator,
            neighbor_dropout=neighbor_dropout,
        )
        return (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            self.targets.gathers[target_indices],
            self.targets.availability[target_indices],
        )

    def _next_epoch_indices(self, batch_size: int) -> torch.Tensor:
        if self._target_generator is None:
            raise AssertionError("validated epoch sampler is missing its generator")
        chunks: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            if self._epoch_order is None or self._epoch_cursor == self.targets.ffid_count:
                self._epoch_order = torch.randperm(
                    self.targets.ffid_count,
                    generator=self._target_generator,
                    device=self.source.device,
                )
                self._epoch_cursor = 0
            take = min(remaining, self.targets.ffid_count - self._epoch_cursor)
            stop = self._epoch_cursor + take
            chunks.append(self._epoch_order[self._epoch_cursor : stop])
            self._epoch_cursor = stop
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class _RawGlobalSnrEvaluator:
    """Evaluate available target traces with float64 energy accumulation."""

    def __init__(
        self,
        source: _ShotGatherTensorSource,
        targets: _GatherTargets,
        *,
        batch_size: int,
        use_bfloat16: bool,
    ) -> None:
        self.source = source
        self.targets = targets
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16 and source.device.type == "cuda"

    def __call__(self, model: ShotGatherInpainter) -> float:
        return self.evaluate(model).raw_global_snr_db

    @torch.inference_mode()
    def evaluate(self, model: ShotGatherInpainter) -> _EvaluationResult:
        model.eval()
        signal_energy = 0.0
        error_energy = 0.0
        predicted_unit_error_energy = 0.0
        trace_count = 0
        for start in range(0, self.targets.ffid_count, self.batch_size):
            stop = min(start + self.batch_size, self.targets.ffid_count)
            batch_indices = np.arange(start, stop, dtype=np.int64)
            neighbors, availability, source_deltas, target_coordinates = self.source.inputs(
                self.targets,
                batch_indices,
            )
            with torch.autocast(
                device_type=self.source.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bfloat16,
            ):
                prediction = model(
                    neighbors,
                    availability,
                    source_deltas,
                    target_coordinates,
                ).float()
            target = self.targets.gathers[start:stop]
            target_mask = self.targets.availability[start:stop]
            target_rows = target[target_mask].double()
            prediction_rows = prediction[target_mask]
            difference = target_rows - prediction_rows.double()
            row_signal = torch.square(target_rows).sum(dim=1)
            row_error = torch.square(difference).sum(dim=1)
            prediction_rms = torch.sqrt(torch.mean(torch.square(prediction_rows), dim=1)).clamp_min(
                1.0e-8
            )
            predicted_unit = prediction_rows / prediction_rms[:, None]
            predicted_unit_error = torch.square(target_rows - predicted_unit.double()).sum(dim=1)
            signal_energy += float(row_signal.sum().cpu())
            error_energy += float(row_error.sum().cpu())
            predicted_unit_error_energy += float(predicted_unit_error.sum().cpu())
            trace_count += len(target_rows)
        point_count = trace_count * self.targets.gathers.shape[-1]
        return _EvaluationResult(
            raw_global_snr_db=_snr_db(signal_energy, error_energy),
            predicted_unit_rms_global_snr_db=_snr_db(
                signal_energy,
                predicted_unit_error_energy,
            ),
            signal_energy=signal_energy,
            error_energy=error_energy,
            error_mean_square=error_energy / point_count,
            ffid_count=self.targets.ffid_count,
            trace_count=trace_count,
        )


def train_shot_gather_inpainter_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Train one whole-shot inpainter and write an immutable reproducible run."""
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
    split_rows = _validate_split_table_for_gathers(split_table, len(preliminary_trace_table))
    preliminary_joined = trace_selection.join_trace_splits(
        preliminary_trace_table,
        split_table,
        split_rows,
    )
    preliminary_canonical, preliminary_duplicate_audit = (
        _canonicalize_eligible_physical_coordinates(preliminary_joined)
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
        raise ConfigurationError("shot gather inpainter requires sampling.split_scope='whole_ffid'")
    joined_table = trace_selection.join_trace_splits(dataset.trace_table, split_table, split_rows)
    canonical_table, duplicate_audit = _canonicalize_eligible_physical_coordinates(joined_table)
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
    configured_scope_audit = _formal_scope_audit(
        settings,
        selection_contract=selection_contract,
        preparation_contract=preparation_contract,
    )
    if settings.ffid_range is None and not configured_scope_audit["scope_success"]:
        failed_checks = [
            name for name, passed in configured_scope_audit["checks"].items() if not passed
        ]
        raise ValueError(
            "formal shot-gather run does not match its required survey scope: "
            f"{failed_checks}; configure training.ffid_range for a diagnostic subset"
        )

    receiver_x_offsets, receiver_y_offsets = _receiver_grid(selected_table)
    selected_split = selected_table[SPLIT_COLUMN].to_numpy()
    train_positions = np.flatnonzero(selected_split == TRAIN_SPLIT).astype(np.int64)
    validation_positions = np.flatnonzero(selected_split == VALIDATION_SPLIT).astype(np.int64)
    train_arrays = selected_table.iloc[train_positions]["array_row"].to_numpy(dtype=np.int64)
    validation_arrays = selected_table.iloc[validation_positions]["array_row"].to_numpy(
        dtype=np.int64
    )
    train_amplitudes_host = _load_unit_rms_rows(dataset.amplitudes, train_arrays)
    validation_amplitudes_host = _load_unit_rms_rows(dataset.amplitudes, validation_arrays)
    device = torch.device(settings.device)
    _seed_global_model_initialization(settings.random_seed, device=device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    train_components = _build_gather_tensors(
        selected_table.iloc[train_positions],
        train_amplitudes_host,
        receiver_x_offsets=receiver_x_offsets,
        receiver_y_offsets=receiver_y_offsets,
        device=device,
    )
    validation_components = _build_gather_tensors(
        selected_table.iloc[validation_positions],
        validation_amplitudes_host,
        receiver_x_offsets=receiver_x_offsets,
        receiver_y_offsets=receiver_y_offsets,
        device=device,
    )
    del train_amplitudes_host, validation_amplitudes_host
    train_ffids, train_sources, train_gathers, train_availability = train_components
    val_ffids, val_sources, val_gathers, val_availability = validation_components
    source = _ShotGatherTensorSource(
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
        torch.Generator(device=device).manual_seed(_target_sampling_seed(settings))
        if settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        else None
    )
    batch_provider = _RandomTrainGatherProvider(
        source,
        train_targets,
        target_sampling=settings.target_sampling,
        target_generator=target_sampling_generator,
    )
    validation_evaluator = _RawGlobalSnrEvaluator(
        source,
        validation_targets,
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    )
    model = ShotGatherInpainter(
        width=settings.hidden_width,
        temporal_dilations=settings.temporal_dilations,
        spatial_y_dilations=settings.spatial_y_dilations,
        stem_kernel_size=settings.stem_kernel_size,
        residual_kernel_size=settings.residual_kernel_size,
        distance_epsilon=settings.distance_epsilon,
        distance_power=settings.distance_power,
        source_weighting=settings.source_weighting,
        source_feature_mode=settings.source_feature_mode,
        source_gather_count=(
            settings.source_gather_count
            if settings.source_feature_mode == ORDERED_RAW_SOURCE_FEATURE_MODE
            else None
        ),
        receiver_position_conditioning=settings.receiver_position_conditioning,
    )
    generator = torch.Generator(device=device).manual_seed(
        settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET
    )
    result = train_shot_gather_inpainter(
        model,
        batch_provider,
        validation_evaluator,
        device=device,
        generator=generator,
        checkpoint_path=output_directory / run_records.CHECKPOINT_RELATIVE_PATH,
        total_steps=settings.total_steps,
        batch_size=settings.batch_size,
        neighbor_dropout=settings.neighbor_dropout,
        derivative_weight=settings.derivative_weight,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        validation_interval=settings.evaluation_interval_steps,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
        training_ffid_count=train_targets.ffid_count,
        training_trace_count=train_targets.trace_count,
        reporter=progress_reporter,
    )
    checkpoint = load_shot_gather_inpainter_checkpoint(
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
        rel_tol=CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        abs_tol=CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
    )
    if not checkpoint_revalidation_matches:
        raise RuntimeError("loaded best checkpoint does not reproduce validation S/N")

    training_audit_targets = _sample_training_audit_targets(
        train_targets,
        trace_count=settings.training_audit_count,
        random_seed=settings.random_seed + TRAINING_AUDIT_SEED_OFFSET,
    )
    training_audit = _RawGlobalSnrEvaluator(
        source,
        training_audit_targets,
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    ).evaluate(checkpoint.model)
    availability_contract = {
        TRAIN_SPLIT: source.audit(train_targets),
        VALIDATION_SPLIT: source.audit(validation_targets),
    }
    collision_audit = _source_collision_audit(
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
    scope_audit = _completed_scope_audit(
        configured_scope_audit,
        availability_contract=availability_contract,
        collision_audit=collision_audit,
        amplitude_access=amplitude_access,
        checkpoint_revalidation_matches=checkpoint_revalidation_matches,
        selected_metric=result.best_validation_global_snr_db,
        recomputed_metric=best_validation.raw_global_snr_db,
    )
    model_contract = _model_contract(
        checkpoint.model,
        settings=settings,
        input_feature_schema_version=checkpoint.input_feature_schema_version,
    )
    receiver_grid_contract = {
        "shape": [RECEIVER_X_COUNT, RECEIVER_Y_COUNT],
        "relative_receiver_x_m": [float(value) for value in receiver_x_offsets],
        "relative_receiver_y_m": [float(value) for value in receiver_y_offsets],
        "missing_receiver_cells": "zero_filled_and_masked",
    }
    training_contract = _training_contract(settings, result, batch_provider)
    neighborhood_contract = {
        "type": NEIGHBORHOOD_TYPE,
        "distance": SOURCE_DISTANCE,
        "source_gather_count": settings.source_gather_count,
        "source_split": TRAIN_SPLIT,
        "target_ffid_policy": "exclude_exact_ffid",
        "zero_source_distance_policy": "exclude",
        "neighbor_dropout_scope": "whole_source_gather",
    }
    checkpoint_contract = {
        "path": run_records.CHECKPOINT_RELATIVE_PATH.as_posix(),
        "selection_metric": PRIMARY_METRIC,
        "best_step": result.best_step,
        "stored_validation_global_snr_db": result.best_validation_global_snr_db,
        "recomputed_validation_global_snr_db": best_validation.raw_global_snr_db,
        "revalidation_relative_tolerance": CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        "revalidation_absolute_tolerance": CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
        "revalidation_matches": checkpoint_revalidation_matches,
        "input_feature_schema_version": checkpoint.input_feature_schema_version,
        "input_feature_names": list(checkpoint.input_feature_names),
        "source_feature_mode": checkpoint.source_feature_mode,
        "source_gather_count": checkpoint.source_gather_count,
        "source_weighting": checkpoint.source_weighting,
        "source_weighting_schema_version": checkpoint.source_weighting_schema_version,
        "source_weighting_input_feature_names": list(
            checkpoint.source_weighting_input_feature_names
        ),
        "receiver_position_conditioning": checkpoint.receiver_position_conditioning,
    }
    metrics = _metrics_payload(
        result,
        best_validation=best_validation,
        training_audit=training_audit,
        settings=settings,
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
            "order": list(TARGET_COORDINATES),
            "fit_split": TRAIN_SPLIT,
            "scaling": TARGET_COORDINATE_SCALING,
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
        TARGET_COORDINATE_SCALING,
    )
    target_coordinates = config_values.validated_target_coordinate_names(
        get_required_config_value(config, "model.target_coordinates")
    )
    if target_coordinates != TARGET_COORDINATES:
        raise ConfigurationError(f"model.target_coordinates must be {list(TARGET_COORDINATES)!r}")
    neighborhood = get_required_config_value(config, "model.neighborhood")
    if not isinstance(neighborhood, Mapping):
        raise ConfigurationError("model.neighborhood must be a mapping")
    if neighborhood.get("type") != NEIGHBORHOOD_TYPE:
        raise ConfigurationError(f"model.neighborhood.type must be {NEIGHBORHOOD_TYPE!r}")
    if neighborhood.get("distance") != SOURCE_DISTANCE:
        raise ConfigurationError(f"model.neighborhood.distance must be {SOURCE_DISTANCE!r}")
    source_gather_count = config_values.positive_integer(
        neighborhood.get("source_gather_count"),
        "model.neighborhood.source_gather_count",
    )
    config_values.require_exact(config, "sampling.split_scope", "whole_ffid")
    config_values.require_exact(
        config,
        "sampling.duplicate_physical_coordinate_policy",
        DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    )
    config_values.require_exact(config, "training.amplitude_scaling", PER_TRACE_RMS_SCALING)
    config_values.require_exact(config, "training.loss", LOSS_NAME)
    config_values.require_exact(config, "training.optimizer", OPTIMIZER_NAME)
    config_values.require_exact(config, "training.learning_rate_schedule", LEARNING_RATE_SCHEDULE)
    config_values.require_exact(config, "training.mixed_precision", MIXED_PRECISION)
    config_values.require_exact(config, "evaluation.primary_metric", PRIMARY_METRIC)
    config_values.require_exact(config, "evaluation.comparison", SUCCESS_COMPARISON)
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
        learning_rate * MINIMUM_LEARNING_RATE_FACTOR,
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
    if gradient_clip_norm != MAX_GRADIENT_NORM:
        raise ConfigurationError(f"training.gradient_clip_norm must be {MAX_GRADIENT_NORM:g}")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigurationError("training must be a mapping")
    target_sampling = training.get("target_sampling", WITH_REPLACEMENT_TARGET_SAMPLING)
    if target_sampling not in {
        WITH_REPLACEMENT_TARGET_SAMPLING,
        EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    }:
        raise ConfigurationError("training.target_sampling is unsupported")
    raw_device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(raw_device, str) or not raw_device:
        raise ConfigurationError("training.device must be a non-empty string")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ConfigurationError("evaluation must be a mapping")
    temporal_dilations = config_values.validated_positive_integer_list(
        get_required_config_value(config, "model.temporal_dilations"),
        "model.temporal_dilations",
    )
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ConfigurationError("model must be a mapping")
    source_feature_mode = model_config.get(
        "source_feature_mode",
        MOMENTS_SOURCE_FEATURE_MODE,
    )
    if source_feature_mode not in SOURCE_FEATURE_MODES:
        raise ConfigurationError(f"model.source_feature_mode must be one of {SOURCE_FEATURE_MODES}")
    source_weighting = model_config.get(
        "source_weighting",
        INVERSE_DISTANCE_SOURCE_WEIGHTING,
    )
    if source_weighting not in SOURCE_WEIGHTING_MODES:
        raise ConfigurationError(f"model.source_weighting must be one of {SOURCE_WEIGHTING_MODES}")
    receiver_position_conditioning = model_config.get(
        "receiver_position_conditioning",
        NO_RECEIVER_POSITION_CONDITIONING,
    )
    if receiver_position_conditioning not in RECEIVER_POSITION_CONDITIONING_MODES:
        raise ConfigurationError(
            "model.receiver_position_conditioning must be one of "
            f"{RECEIVER_POSITION_CONDITIONING_MODES}"
        )
    raw_spatial_y_dilations = model_config.get("spatial_y_dilations")
    spatial_y_dilations = (
        (1,) * len(temporal_dilations)
        if raw_spatial_y_dilations is None
        else config_values.validated_positive_integer_list(
            raw_spatial_y_dilations,
            "model.spatial_y_dilations",
        )
    )
    if len(spatial_y_dilations) != len(temporal_dilations):
        raise ConfigurationError(
            "model.spatial_y_dilations must have the same length as model.temporal_dilations"
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
        stem_kernel_size=config_values.odd_positive_integer(
            get_required_config_value(config, "model.stem_kernel_size"),
            "model.stem_kernel_size",
        ),
        residual_kernel_size=config_values.odd_positive_integer(
            get_required_config_value(config, "model.residual_kernel_size"),
            "model.residual_kernel_size",
        ),
        temporal_dilations=temporal_dilations,
        spatial_y_dilations=spatial_y_dilations,
        distance_epsilon=config_values.positive_float(
            get_required_config_value(config, "model.distance_epsilon"),
            "model.distance_epsilon",
        ),
        distance_power=config_values.positive_float(
            model_config.get("distance_power", DEFAULT_DISTANCE_POWER),
            "model.distance_power",
        ),
        source_weighting=str(source_weighting),
        source_feature_mode=str(source_feature_mode),
        receiver_position_conditioning=str(receiver_position_conditioning),
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
        derivative_weight=config_values.nonnegative_float(
            get_required_config_value(config, "training.derivative_weight"),
            "training.derivative_weight",
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
        duplicate_physical_coordinate_policy=DUPLICATE_PHYSICAL_COORDINATE_POLICY,
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


def _validate_split_table_for_gathers(split_table: pd.DataFrame, row_count: int) -> np.ndarray:
    from seis_interp.pipelines.train_siren import _validate_split_table

    return _validate_split_table(split_table, row_count, allow_excluded=True)


def _receiver_grid(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    relative_x = (table["receiver_x_m"] - table["source_x_m"]).to_numpy(dtype=np.float64)
    relative_y = (table["receiver_y_m"] - table["source_y_m"]).to_numpy(dtype=np.float64)
    x_values = np.unique(relative_x)
    y_values = np.unique(relative_y)
    if len(x_values) != RECEIVER_X_COUNT or len(y_values) != RECEIVER_Y_COUNT:
        raise ValueError(
            "selected data must expose the fixed 8 x 68 relative-receiver grid; "
            f"got {len(x_values)} x {len(y_values)}"
        )
    return np.sort(x_values), np.sort(y_values)


def _build_gather_tensors(
    table: pd.DataFrame,
    amplitudes: np.ndarray,
    *,
    receiver_x_offsets: np.ndarray,
    receiver_y_offsets: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    if len(table) != len(amplitudes):
        raise ValueError("gather rows and amplitudes must have equal length")
    ffids = np.sort(table["ffid"].unique().astype(np.int64))
    ffid_index = {int(ffid): index for index, ffid in enumerate(ffids)}
    x_index = {float(value): index for index, value in enumerate(receiver_x_offsets)}
    y_index = {float(value): index for index, value in enumerate(receiver_y_offsets)}
    source_coordinates = np.empty((len(ffids), 2), dtype=np.float64)
    source_seen = np.zeros(len(ffids), dtype=bool)
    gather_host = np.zeros(
        (len(ffids), RECEIVER_X_COUNT, RECEIVER_Y_COUNT, amplitudes.shape[1]),
        dtype=np.float32,
    )
    availability_host = np.zeros(
        (len(ffids), RECEIVER_X_COUNT, RECEIVER_Y_COUNT),
        dtype=bool,
    )
    for row_index, row in enumerate(table.itertuples(index=False)):
        target_index = ffid_index[int(row.ffid)]
        source = np.asarray((row.source_x_m, row.source_y_m), dtype=np.float64)
        if source_seen[target_index] and not np.array_equal(
            source_coordinates[target_index],
            source,
        ):
            raise ValueError(f"FFID {int(row.ffid)} contains multiple source coordinates")
        source_coordinates[target_index] = source
        source_seen[target_index] = True
        relative_x = float(row.receiver_x_m - row.source_x_m)
        relative_y = float(row.receiver_y_m - row.source_y_m)
        try:
            receiver_x_index = x_index[relative_x]
            receiver_y_index = y_index[relative_y]
        except KeyError as error:
            raise ValueError("trace is outside the validated receiver grid") from error
        if availability_host[target_index, receiver_x_index, receiver_y_index]:
            raise ValueError(f"FFID {int(row.ffid)} has a duplicate receiver cell")
        gather_host[target_index, receiver_x_index, receiver_y_index] = amplitudes[row_index]
        availability_host[target_index, receiver_x_index, receiver_y_index] = True
    if not np.all(source_seen):
        raise AssertionError("a selected FFID was not populated")
    return (
        ffids,
        source_coordinates,
        torch.from_numpy(gather_host).to(device),
        torch.from_numpy(availability_host).to(device),
    )


def _nearest_train_source_indices(
    train_ffids: np.ndarray,
    train_source_coordinates_m: np.ndarray,
    target_ffids: np.ndarray,
    target_source_coordinates_m: np.ndarray,
    *,
    source_gather_count: int,
) -> np.ndarray:
    result = np.empty((len(target_ffids), source_gather_count), dtype=np.int64)
    for target_index, (target_ffid, target_source) in enumerate(
        zip(target_ffids, target_source_coordinates_m, strict=True)
    ):
        squared_distance = np.sum(
            np.square(train_source_coordinates_m - target_source),
            axis=1,
        )
        eligible = (train_ffids != target_ffid) & (squared_distance > 0.0)
        candidates = np.flatnonzero(eligible)
        if len(candidates) < source_gather_count:
            raise ValueError(
                f"FFID {int(target_ffid)} has only {len(candidates)} non-colliding TRAIN sources; "
                f"{source_gather_count} required"
            )
        order = np.lexsort((train_ffids[candidates], squared_distance[candidates]))
        result[target_index] = candidates[order[:source_gather_count]]
    return result


def _sample_training_audit_targets(
    targets: _GatherTargets,
    *,
    trace_count: int,
    random_seed: int,
) -> _GatherTargets:
    available_flat = (
        torch.nonzero(targets.availability.flatten(), as_tuple=False).flatten().cpu().numpy()
    )
    if trace_count > len(available_flat):
        raise ConfigurationError(
            "training.training_audit_count must not exceed selected training trace count"
        )
    generator = np.random.default_rng(random_seed)
    selected_flat = generator.choice(available_flat, size=trace_count, replace=False)
    receiver_cell_count = RECEIVER_X_COUNT * RECEIVER_Y_COUNT
    ffid_indices = np.unique(selected_flat // receiver_cell_count)
    compact_by_ffid = {int(value): index for index, value in enumerate(ffid_indices)}
    audit_mask = torch.zeros(
        (len(ffid_indices), RECEIVER_X_COUNT, RECEIVER_Y_COUNT),
        dtype=torch.bool,
        device=targets.availability.device,
    )
    for flat_index in selected_flat:
        ffid_index, receiver_flat = divmod(int(flat_index), receiver_cell_count)
        receiver_x, receiver_y = divmod(receiver_flat, RECEIVER_Y_COUNT)
        audit_mask[compact_by_ffid[ffid_index], receiver_x, receiver_y] = True
    gather_indices = torch.as_tensor(
        ffid_indices,
        dtype=torch.long,
        device=targets.gathers.device,
    )
    return _GatherTargets(
        ffids=targets.ffids[ffid_indices],
        source_coordinates_m=targets.source_coordinates_m[ffid_indices],
        gathers=targets.gathers[gather_indices],
        availability=audit_mask,
        neighbor_train_indices=targets.neighbor_train_indices[ffid_indices],
    )


def _source_collision_audit(
    train_sources: np.ndarray,
    validation_sources: np.ndarray,
    *,
    duplicate_audit: Mapping[str, object],
) -> dict[str, int]:
    train_keys = {tuple(value) for value in train_sources.tolist()}
    validation_keys = {tuple(value) for value in validation_sources.tolist()}
    return {
        "canonical_remaining_duplicate_physical_cells": int(
            duplicate_audit["remaining_duplicate_physical_cell_count"]
        ),
        "train_duplicate_source_coordinates": len(train_sources) - len(train_keys),
        "train_validation_source_coordinate_overlap": len(train_keys & validation_keys),
    }


def _completed_scope_audit(
    configured: Mapping[str, object],
    *,
    availability_contract: Mapping[str, Mapping[str, object]],
    collision_audit: Mapping[str, int],
    amplitude_access: Mapping[str, object],
    checkpoint_revalidation_matches: bool,
    selected_metric: float,
    recomputed_metric: float,
) -> dict[str, object]:
    completed = deepcopy(dict(configured))
    checks = dict(completed["checks"])
    materialized = amplitude_access["value_rows_materialized_by_split"]
    checks.update(
        {
            "validation_metric_domain_matches": (
                ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN == "oracle_per_trace_unit_rms"
            ),
            "checkpoint_raw_metric_reproduced": checkpoint_revalidation_matches,
            "selected_metric_matches_recomputed_raw_metric": math.isclose(
                selected_metric,
                recomputed_metric,
                rel_tol=CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
                abs_tol=CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
            ),
            "canonical_duplicate_physical_cells_remaining_zero": (
                collision_audit["canonical_remaining_duplicate_physical_cells"] == 0
            ),
            "train_source_coordinate_collisions_zero": (
                collision_audit["train_duplicate_source_coordinates"] == 0
            ),
            "train_validation_source_coordinate_overlap_zero": (
                collision_audit["train_validation_source_coordinate_overlap"] == 0
            ),
            "target_ffid_neighbor_entries_zero": all(
                availability_contract[split]["target_ffid_neighbor_entries"] == 0
                for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
            ),
            "neighbor_sources_train_only": all(
                availability_contract[split]["non_train_neighbor_entries"] == 0
                for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
            ),
            "test_value_rows_not_materialized": materialized[TEST_SPLIT] is False,
            "excluded_value_rows_not_materialized": materialized[EXCLUDED_SPLIT] is False,
        }
    )
    completed["checks"] = checks
    completed["scope_success"] = all(checks.values())
    return completed


def _model_contract(
    model: ShotGatherInpainter,
    *,
    settings: _TrainingSettings,
    input_feature_schema_version: int,
) -> dict[str, object]:
    input_feature_names = tuple(model.input_feature_names)
    if len(input_feature_names) != model.input_channels:
        raise RuntimeError("model input feature names do not match its input channel count")
    return {
        "name": MODEL_NAME,
        "hidden_width": model.width,
        "input_channels": model.input_channels,
        "input_feature_schema_version": input_feature_schema_version,
        "input_feature_names": list(input_feature_names),
        "source_feature_mode": model.source_feature_mode,
        "source_weighting": model.source_weighting,
        "source_weighting_schema_version": SOURCE_WEIGHTING_SCHEMA_VERSION,
        "source_weighting_input_feature_names": list(model.source_weighting_input_feature_names),
        "receiver_position_conditioning": model.receiver_position_conditioning,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temporal_dilations": list(model.temporal_dilations),
        "spatial_y_dilations": list(model.spatial_y_dilations),
        "stem_kernel_size": model.stem_kernel_size,
        "residual_kernel_size": model.residual_kernel_size,
        "distance_epsilon": model.distance_epsilon,
        "distance_power": model.distance_power,
        "dynamic_attention_width": model.dynamic_attention_width,
        "dynamic_attention_kernel_size": model.dynamic_attention_kernel_size,
        "target_coordinates": list(TARGET_COORDINATES),
        "target_coordinate_scaling": TARGET_COORDINATE_SCALING,
        "receiver_grid_shape": [RECEIVER_X_COUNT, RECEIVER_Y_COUNT],
        "source_gather_count": settings.source_gather_count,
        "prediction_reference": (
            "receiver_time_dynamic_attention_over_inverse_distance_logits"
            if model.source_weighting != INVERSE_DISTANCE_SOURCE_WEIGHTING
            else "receiver_wise_inverse_source_distance"
        ),
        "residual_decoder_initialization": "zero_final_projection",
    }


def _training_contract(
    settings: _TrainingSettings,
    result: ShotGatherTrainingResult,
    provider: _RandomTrainGatherProvider,
) -> dict[str, object]:
    return {
        "batch_mode": "random_whole_ffid_gathers",
        "amplitude_scaling": PER_TRACE_RMS_SCALING,
        "training_scale_source": TRAINING_SCALE_SOURCE,
        "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        "validation_scale_source": VALIDATION_SCALE_SOURCE,
        "loss": LOSS_NAME,
        "loss_target_mask": "eligible_receiver_cells",
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "learning_rate_schedule": LEARNING_RATE_SCHEDULE,
        "minimum_learning_rate": settings.minimum_learning_rate,
        "minimum_learning_rate_factor": MINIMUM_LEARNING_RATE_FACTOR,
        "gradient_clip_norm": MAX_GRADIENT_NORM,
        "total_steps": settings.total_steps,
        "batch_size_ffids": settings.batch_size,
        "target_sampling": settings.target_sampling,
        "target_sampling_seed": _target_sampling_seed(settings),
        "target_sampling_rng_independent_of_neighbor_dropout": (
            settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        ),
        "exclude_target_ffid_neighbors": settings.exclude_target_ffid_neighbors,
        "neighbor_dropout": settings.neighbor_dropout,
        "neighbor_dropout_seed": settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET,
        "neighbor_dropout_scope": "whole_source_gather",
        "derivative_weight": settings.derivative_weight,
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
        "training_audit_seed": settings.random_seed + TRAINING_AUDIT_SEED_OFFSET,
    }


def _target_sampling_seed(settings: _TrainingSettings) -> int:
    if settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
        return settings.random_seed + EPOCH_TARGET_SAMPLING_SEED_OFFSET
    return settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET


def _metrics_payload(
    result: ShotGatherTrainingResult,
    *,
    best_validation: _EvaluationResult,
    training_audit: _EvaluationResult,
    settings: _TrainingSettings,
    selection_contract: Mapping[str, object],
    duplicate_audit: Mapping[str, object],
    collision_audit: Mapping[str, object],
    availability_contract: Mapping[str, object],
    amplitude_access: Mapping[str, object],
    scope_audit: Mapping[str, object],
    checkpoint_contract: Mapping[str, object],
) -> dict[str, object]:
    metrics = asdict(result)
    metrics["history"] = [dict(value) for value in result.history]
    accepted_metric = best_validation.raw_global_snr_db
    metric_success = _passes_success_threshold(accepted_metric, settings.success_threshold_db)
    scope_success = bool(scope_audit["scope_success"])
    metrics.update(
        {
            "amplitude_scaling": PER_TRACE_RMS_SCALING,
            "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
            "validation_scale_source": VALIDATION_SCALE_SOURCE,
            "primary_metric_prediction": "raw_model_output",
            "primary_metric_prediction_self_normalized": False,
            "best_validation_raw_global_snr_db": accepted_metric,
            "best_validation_signal_energy": best_validation.signal_energy,
            "best_validation_error_energy": best_validation.error_energy,
            "best_validation_error_mean_square": best_validation.error_mean_square,
            "best_validation_predicted_unit_rms_global_snr_db": (
                best_validation.predicted_unit_rms_global_snr_db
            ),
            "best_validation_global_snr_db": accepted_metric,
            PRIMARY_METRIC: accepted_metric,
            "success_threshold_db": settings.success_threshold_db,
            "success_comparison": SUCCESS_COMPARISON,
            "metric_success": metric_success,
            "scope_success": scope_success,
            "success": metric_success and scope_success,
            "validation_ffid_count": best_validation.ffid_count,
            "validation_trace_count": best_validation.trace_count,
            "training_audit_trace_count": training_audit.trace_count,
            "training_audit_global_snr_db": training_audit.raw_global_snr_db,
            "training_audit_predicted_unit_rms_global_snr_db": (
                training_audit.predicted_unit_rms_global_snr_db
            ),
            "split_counts": dict(selection_contract["split_counts"]),
            "sample_count": selection_contract["sample_count"],
            "effective_eligible_trace_count": selection_contract["effective_eligible_trace_count"],
            "duplicate_physical_coordinates": dict(duplicate_audit),
            "collision_audit": dict(collision_audit),
            "amplitude_access": dict(amplitude_access),
            "neighbor_availability": dict(availability_contract),
            "formal_success_scope": dict(scope_audit),
            "checkpoint": dict(checkpoint_contract),
        }
    )
    return metrics
