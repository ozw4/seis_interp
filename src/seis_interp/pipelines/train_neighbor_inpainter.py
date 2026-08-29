"""Train the geometry-neighbor trace inpainter on prepared survey splits."""

from __future__ import annotations

import math
import platform
import resource
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seis_interp.configuration import (
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import (
    OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES,
)
from seis_interp.data.trace_store import (
    TRACES_FILE_NAME,
    canonical_source_files,
)
from seis_interp.models.neighbor_trace_inpainter import (
    DEFAULT_COORDINATE_CONDITIONING,
    DEFAULT_NEIGHBOR_GATING,
    TARGET_COORDINATE_MASKED_SOFTMAX_GATING,
    NeighborTraceInpainter,
)
from seis_interp.pipelines.train_siren import (
    CHECKPOINT_RELATIVE_PATH,
    PROCESSED_INPUT_FILE_NAMES,
    RANDOM_COMPLETE_TRACES_BATCH_MODE,
    _check_new_output_directory,
    _configured_trace_amplitude_filter,
    _file_hashes,
    _git_commit,
    _load_processed_dataset,
    _split_counts,
    _utc_timestamp,
    _validate_preparation_data,
    _validate_split_table,
    _validated_preparation_contract,
    _write_run_outputs,
)
from seis_interp.processing.multiline_neighbor_geometry import (
    SOURCE_X_LINE_SPACING_M,
    SOURCE_Y_HALF_SHOT_SPACING_M,
    MultilineNeighborGeometryLookup,
)
from seis_interp.processing.multiline_neighbor_geometry import (
    TARGET_COORDINATE_ORDER as MULTILINE_TARGET_COORDINATE_ORDER,
)
from seis_interp.processing.neighbor_geometry import (
    RECEIVER_SPACING_M,
    SOURCE_SHOT_SPACING_M,
    TARGET_COORDINATE_ORDER,
    NeighborGeometryLookup,
)
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
    per_trace_rms_scaled_amplitudes,
)
from seis_interp.training.neighbor_inpainter_checkpoints import (
    load_neighbor_inpainter_checkpoint,
)
from seis_interp.training.neighbor_inpainter_trainer import (
    MAX_GRADIENT_NORM,
    MINIMUM_LEARNING_RATE_FACTOR,
    NeighborInpainterTrainingResult,
    train_neighbor_trace_inpainter,
)

MODEL_NAME = "neighbor_trace_inpainter"
LOSS_NAME = "l2_plus_first_difference"
OPTIMIZER_NAME = "adamw"
LEARNING_RATE_SCHEDULE = "cosine"
MIXED_PRECISION = "bfloat16"
VALIDATION_SCALE_SOURCE = "validation_trace_target_rms"
TRAINING_SCALE_SOURCE = "training_trace_target_rms"
PRIMARY_METRIC = "oracle_per_trace_unit_rms_global_snr_db"
SUCCESS_COMPARISON = "strictly_greater_than"
DUPLICATE_PHYSICAL_COORDINATE_POLICY = "keep_lowest_array_row"
TARGET_COORDINATE_SCALING = "train_minmax"
STEM_KERNEL_SIZE = 15
RESIDUAL_KERNEL_SIZE = 7
SINGLE_LINE_GEOMETRY = "single_source_line"
MULTILINE_GEOMETRY = "multiline_staggered_source"
CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE = 1.0e-8
CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE = 1.0e-8
WITH_REPLACEMENT_TARGET_SAMPLING = "with_replacement"
EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING = "epoch_without_replacement"
NEIGHBOR_DROPOUT_SEED_OFFSET = 1
TRAINING_AUDIT_SEED_OFFSET = 2
EPOCH_TARGET_SAMPLING_SEED_OFFSET = 3

ProgressReporter = Callable[[str], None]

_EXPECTED_NEIGHBORHOOD = {
    "relative_receiver_x_radius": 1,
    "source_shot_radius": 2,
    "relative_receiver_y_radius": 3,
    "relative_receiver_spacing_m": RECEIVER_SPACING_M,
    "source_shot_spacing_m": SOURCE_SHOT_SPACING_M,
    "same_source_x_only": True,
}
_MULTILINE_NEIGHBORHOOD_KEYS = {
    "type",
    "relative_receiver_x_radius",
    "source_x_line_radius",
    "source_y_half_shot_radius",
    "relative_receiver_y_radius",
    "relative_receiver_spacing_m",
    "source_x_line_spacing_m",
    "source_y_half_shot_spacing_m",
}
_PHYSICAL_COORDINATE_COLUMNS = (
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
_AMPLITUDE_ROW_CHUNK_SIZE = 4096
_AVAILABILITY_CHUNK_SIZE = 65536


@dataclass(frozen=True)
class _TrainingSettings:
    random_seed: int
    hidden_width: int
    target_coordinates: tuple[str, ...]
    stem_kernel_size: int
    residual_kernel_size: int
    temporal_dilations: tuple[int, ...]
    coordinate_conditioning: str
    neighbor_gating: str
    neighbor_geometry: str
    relative_receiver_x_radius: int
    source_x_line_radius: int
    source_y_half_shot_radius: int
    relative_receiver_y_radius: int
    learning_rate: float
    weight_decay: float
    minimum_learning_rate: float
    total_steps: int
    batch_size: int
    target_sampling: str
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
    required_fully_excluded_ffids: tuple[int, ...]


@dataclass(frozen=True)
class _EvaluationResult:
    raw_global_snr_db: float
    predicted_unit_rms_global_snr_db: float
    signal_energy: float
    error_energy: float
    error_mean_square: float
    clean_trace_count: int
    clean_raw_global_snr_db: float


class _NeighborTensorSource:
    """Gather train-only physical neighbors from compact device tensors."""

    def __init__(
        self,
        lookup: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
        *,
        train_positions: np.ndarray,
        train_amplitudes: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.lookup = lookup
        self.train_positions = np.asarray(train_positions, dtype=np.int64)
        self.train_amplitudes = train_amplitudes
        self.device = device
        self._train_index_by_position = np.full(lookup.row_count, -1, dtype=np.int64)
        self._train_index_by_position[self.train_positions] = np.arange(
            len(self.train_positions), dtype=np.int64
        )

    def gather(
        self,
        target_positions: np.ndarray,
        *,
        generator: torch.Generator | None = None,
        neighbor_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = np.asarray(target_positions, dtype=np.int64)
        neighbor_positions = self.lookup.neighbor_positions(positions)
        availability = neighbor_positions >= 0
        safe_positions = np.maximum(neighbor_positions, 0)
        compact_indices = self._train_index_by_position[safe_positions]
        if np.any(compact_indices[availability] < 0):
            raise RuntimeError("neighbor lookup returned a row outside the TRAIN amplitude store")
        safe_compact = np.maximum(compact_indices, 0)
        compact_tensor = torch.as_tensor(safe_compact, dtype=torch.long, device=self.device)
        availability_tensor = torch.as_tensor(availability, dtype=torch.bool, device=self.device)
        if neighbor_dropout > 0.0:
            if generator is None:
                raise ValueError("generator is required when neighbor_dropout is positive")
            keep = (
                torch.rand(
                    availability_tensor.shape,
                    generator=generator,
                    device=self.device,
                )
                >= neighbor_dropout
            )
            availability_tensor &= keep
        neighbors = self.train_amplitudes[compact_tensor]
        neighbors = neighbors * availability_tensor[..., None]
        coordinates = torch.as_tensor(
            self.lookup.target_coordinates(positions),
            dtype=torch.float32,
            device=self.device,
        )
        return neighbors, availability_tensor, coordinates


class _RandomTrainBatchProvider:
    """Sample target traces while tracking exact draw coverage for provenance."""

    def __init__(
        self,
        source: _NeighborTensorSource,
        *,
        target_sampling: str = WITH_REPLACEMENT_TARGET_SAMPLING,
        target_generator: torch.Generator | None = None,
    ) -> None:
        if target_sampling not in {
            WITH_REPLACEMENT_TARGET_SAMPLING,
            EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
        }:
            raise ValueError(f"unsupported target sampling mode: {target_sampling!r}")
        if target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
            if not isinstance(target_generator, torch.Generator):
                raise TypeError(
                    "target_generator is required for epoch_without_replacement sampling"
                )
        elif target_generator is not None:
            raise ValueError(
                "target_generator must be omitted for legacy with_replacement sampling"
            )
        self.source = source
        self.target_sampling = target_sampling
        self._target_generator = target_generator
        self._epoch_order: torch.Tensor | None = None
        self._epoch_cursor = 0
        self._seen = np.zeros(len(source.train_positions), dtype=bool)
        self.draw_count = 0

    @property
    def unique_target_count(self) -> int:
        return int(np.count_nonzero(self._seen))

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.target_sampling == WITH_REPLACEMENT_TARGET_SAMPLING:
            # Keep this legacy path on the caller's generator: target draws and
            # neighbor dropout retain their established interleaved RNG sequence
            # when training.target_sampling is absent.
            compact_indices = torch.randint(
                len(self.source.train_positions),
                (batch_size,),
                generator=generator,
                device=self.source.device,
            )
        else:
            compact_indices = self._next_epoch_indices(batch_size)
        compact_numpy = compact_indices.cpu().numpy()
        self._seen[compact_numpy] = True
        self.draw_count += batch_size
        target_positions = self.source.train_positions[compact_numpy]
        neighbors, availability, coordinates = self.source.gather(
            target_positions,
            generator=generator,
            neighbor_dropout=neighbor_dropout,
        )
        targets = self.source.train_amplitudes[compact_indices]
        return neighbors, availability, coordinates, targets

    def _next_epoch_indices(self, batch_size: int) -> torch.Tensor:
        """Draw from consecutive deterministic permutations, wrapping as needed."""
        if self._target_generator is None:
            raise AssertionError("validated epoch sampler is missing its generator")
        trace_count = len(self.source.train_positions)
        chunks: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            if self._epoch_order is None or self._epoch_cursor == trace_count:
                self._epoch_order = torch.randperm(
                    trace_count,
                    generator=self._target_generator,
                    device=self.source.device,
                )
                self._epoch_cursor = 0
            take = min(remaining, trace_count - self._epoch_cursor)
            stop = self._epoch_cursor + take
            chunks.append(self._epoch_order[self._epoch_cursor : stop])
            self._epoch_cursor = stop
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class _RawGlobalSnrEvaluator:
    """Evaluate point-weighted global S/N with float64 energy accumulation."""

    def __init__(
        self,
        source: _NeighborTensorSource,
        *,
        target_positions: np.ndarray,
        target_amplitudes: torch.Tensor,
        clean_mask: np.ndarray,
        batch_size: int,
        use_bfloat16: bool,
    ) -> None:
        self.source = source
        self.target_positions = np.asarray(target_positions, dtype=np.int64)
        self.target_amplitudes = target_amplitudes
        self.clean_mask = np.asarray(clean_mask, dtype=bool)
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16 and source.device.type == "cuda"
        if self.clean_mask.shape != (len(self.target_positions),):
            raise ValueError("clean validation mask must match target positions")

    def __call__(self, model: NeighborTraceInpainter) -> float:
        return self.evaluate(model).raw_global_snr_db

    @torch.inference_mode()
    def evaluate(self, model: NeighborTraceInpainter) -> _EvaluationResult:
        model.eval()
        signal_energy = 0.0
        error_energy = 0.0
        predicted_unit_error_energy = 0.0
        clean_signal_energy = 0.0
        clean_error_energy = 0.0
        for start in range(0, len(self.target_positions), self.batch_size):
            stop = min(start + self.batch_size, len(self.target_positions))
            neighbors, availability, coordinates = self.source.gather(
                self.target_positions[start:stop]
            )
            with torch.autocast(
                device_type=self.source.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bfloat16,
            ):
                prediction = model(neighbors, availability, coordinates).float()
            target = self.target_amplitudes[start:stop]
            target_double = target.double()
            difference = target_double - prediction.double()
            row_signal = torch.square(target_double).sum(dim=1)
            row_error = torch.square(difference).sum(dim=1)
            prediction_rms = torch.sqrt(torch.mean(torch.square(prediction), dim=1)).clamp_min(
                1.0e-8
            )
            predicted_unit = prediction / prediction_rms[:, None]
            predicted_unit_error = torch.square(target_double - predicted_unit.double()).sum(dim=1)
            signal_energy += float(row_signal.sum().cpu())
            error_energy += float(row_error.sum().cpu())
            predicted_unit_error_energy += float(predicted_unit_error.sum().cpu())
            clean = torch.as_tensor(
                self.clean_mask[start:stop], dtype=torch.bool, device=target.device
            )
            clean_signal_energy += float(row_signal[clean].sum().cpu())
            clean_error_energy += float(row_error[clean].sum().cpu())

        point_count = len(self.target_positions) * self.target_amplitudes.shape[1]
        return _EvaluationResult(
            raw_global_snr_db=_snr_db(signal_energy, error_energy),
            predicted_unit_rms_global_snr_db=_snr_db(signal_energy, predicted_unit_error_energy),
            signal_energy=signal_energy,
            error_energy=error_energy,
            error_mean_square=error_energy / point_count,
            clean_trace_count=int(np.count_nonzero(self.clean_mask)),
            clean_raw_global_snr_db=_snr_db(clean_signal_energy, clean_error_energy),
        )


def train_neighbor_inpainter_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Train one neighbor inpainter and write immutable, reproducible run artifacts."""
    output_directory = Path(output_dir)
    _check_new_output_directory(output_directory)
    started_at_utc = _utc_timestamp()
    git_commit = _git_commit()
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
        allow_excluded="trace_amplitude_filter" in preparation,
    )
    preliminary_joined = _joined_trace_table(
        preliminary_trace_table,
        split_table,
        split_rows,
    )
    preliminary_canonical, preliminary_duplicate_audit = (
        _canonicalize_eligible_physical_coordinates(preliminary_joined)
    )
    selected_preliminary = _selected_trace_table(
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
    interim_files = _file_hashes(interim_directory, INTERIM_FILE_NAMES)
    processed_files = _file_hashes(processed_directory, PROCESSED_INPUT_FILE_NAMES)
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
    )
    joined_table = _joined_trace_table(
        dataset.trace_table,
        split_table,
        split_rows,
    )
    canonical_table, duplicate_audit = _canonicalize_eligible_physical_coordinates(joined_table)
    if duplicate_audit != preliminary_duplicate_audit:
        raise RuntimeError("duplicate-coordinate audit changed while loading the interim dataset")
    selected_table = _selected_trace_table(
        canonical_table,
        ffid_range=settings.ffid_range,
    )
    selection_contract = _selection_contract(
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
            "formal neighbor-inpainter run does not match its required survey scope: "
            f"{failed_checks}; configure training.ffid_range for a diagnostic subset"
        )
    selected_split = selected_table[SPLIT_COLUMN].to_numpy()
    train_positions = np.flatnonzero(selected_split == TRAIN_SPLIT).astype(np.int64)
    validation_positions = np.flatnonzero(selected_split == VALIDATION_SPLIT).astype(np.int64)
    _validate_selected_split_coverage(selected_table)
    if settings.training_audit_count > len(train_positions):
        raise ConfigurationError(
            "training.training_audit_count must not exceed selected training trace count"
        )

    available = selected_split == TRAIN_SPLIT
    geometry = _build_neighbor_geometry(selected_table, available, settings=settings)
    if geometry.collision_count != 0:
        raise RuntimeError(
            "canonicalized TRAIN geometry still contains physical coordinate collisions"
        )
    geometry_contract = _geometry_contract(geometry, settings=settings)
    availability_contract = {
        TRAIN_SPLIT: _neighbor_availability(geometry, train_positions),
        VALIDATION_SPLIT: _neighbor_availability(geometry, validation_positions),
    }
    overlap_mask, overlap_train_positions = _train_validation_coordinate_overlap(
        selected_table,
        train_positions,
        validation_positions,
    )
    if np.any(overlap_mask):
        raise RuntimeError(
            "canonicalized validation geometry still overlaps a TRAIN physical coordinate"
        )
    clean_validation_mask = ~overlap_mask

    train_array_rows = selected_table.iloc[train_positions]["array_row"].to_numpy(dtype=np.int64)
    validation_array_rows = selected_table.iloc[validation_positions]["array_row"].to_numpy(
        dtype=np.int64
    )
    train_amplitudes_host = _load_unit_rms_rows(dataset.amplitudes, train_array_rows)
    validation_amplitudes_host = _load_unit_rms_rows(dataset.amplitudes, validation_array_rows)
    exact_overlap_count = _exact_overlap_amplitude_count(
        train_amplitudes_host,
        validation_amplitudes_host,
        train_positions,
        overlap_train_positions,
        overlap_mask,
    )

    device = torch.device(settings.device)
    _seed_global_model_initialization(settings.random_seed, device=device)
    if device.type == "cuda":
        # Some supported PyTorch/CUDA builds reject an explicit device argument
        # here even though the no-argument form works for the current device.
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    train_amplitudes = torch.from_numpy(train_amplitudes_host).to(device)
    validation_amplitudes = torch.from_numpy(validation_amplitudes_host).to(device)
    del train_amplitudes_host, validation_amplitudes_host
    tensor_source = _NeighborTensorSource(
        geometry,
        train_positions=train_positions,
        train_amplitudes=train_amplitudes,
        device=device,
    )
    target_sampling_seed = _target_sampling_seed(settings)
    target_sampling_generator = (
        torch.Generator(device=device).manual_seed(target_sampling_seed)
        if settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        else None
    )
    batch_provider = _RandomTrainBatchProvider(
        tensor_source,
        target_sampling=settings.target_sampling,
        target_generator=target_sampling_generator,
    )
    validation_evaluator = _RawGlobalSnrEvaluator(
        tensor_source,
        target_positions=validation_positions,
        target_amplitudes=validation_amplitudes,
        clean_mask=clean_validation_mask,
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    )
    model = NeighborTraceInpainter(
        neighbor_count=geometry.neighbor_count,
        width=settings.hidden_width,
        target_coordinate_count=len(settings.target_coordinates),
        stem_kernel_size=settings.stem_kernel_size,
        residual_kernel_size=settings.residual_kernel_size,
        temporal_dilations=settings.temporal_dilations,
        coordinate_conditioning=settings.coordinate_conditioning,
        neighbor_gating=settings.neighbor_gating,
    )
    generator = torch.Generator(device=device).manual_seed(
        settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET
    )
    result = train_neighbor_trace_inpainter(
        model,
        batch_provider,
        validation_evaluator,
        device=device,
        generator=generator,
        checkpoint_path=output_directory / CHECKPOINT_RELATIVE_PATH,
        total_steps=settings.total_steps,
        batch_size=settings.batch_size,
        neighbor_dropout=settings.neighbor_dropout,
        derivative_weight=settings.derivative_weight,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        validation_interval=settings.evaluation_interval_steps,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
        training_trace_count=len(train_positions),
        reporter=progress_reporter,
    )
    checkpoint = load_neighbor_inpainter_checkpoint(
        output_directory / CHECKPOINT_RELATIVE_PATH,
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
        raise RuntimeError("loaded best checkpoint does not reproduce its validation S/N")

    audit_generator = np.random.default_rng(settings.random_seed + TRAINING_AUDIT_SEED_OFFSET)
    audit_compact = np.sort(
        audit_generator.choice(
            len(train_positions),
            size=settings.training_audit_count,
            replace=False,
        )
    )
    training_audit_evaluator = _RawGlobalSnrEvaluator(
        tensor_source,
        target_positions=train_positions[audit_compact],
        target_amplitudes=train_amplitudes[
            torch.as_tensor(audit_compact, dtype=torch.long, device=device)
        ],
        clean_mask=np.ones(settings.training_audit_count, dtype=bool),
        batch_size=settings.validation_batch_size,
        use_bfloat16=settings.mixed_precision == MIXED_PRECISION,
    )
    training_audit = training_audit_evaluator.evaluate(checkpoint.model)

    collision_audit = {
        "canonical_remaining_duplicate_physical_cells": duplicate_audit[
            "remaining_duplicate_physical_cell_count"
        ],
        "train_coordinate_collision_rows": geometry.collision_count,
        "train_coordinate_collision_cells": geometry.collision_cell_count,
        "train_validation_coordinate_overlap_rows": int(np.count_nonzero(overlap_mask)),
        "train_validation_exact_unit_amplitude_duplicate_rows": exact_overlap_count,
    }
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
    scope_audit = _completed_formal_scope_audit(
        configured_scope_audit,
        collision_audit=collision_audit,
        geometry_contract=geometry_contract,
        amplitude_access=amplitude_access,
        checkpoint_revalidation_matches=checkpoint_revalidation_matches,
        selected_metric=result.best_validation_global_snr_db,
        recomputed_metric=best_validation.raw_global_snr_db,
    )
    training_contract = _training_contract(settings, result, batch_provider)
    model_contract = {
        "name": MODEL_NAME,
        "hidden_width": settings.hidden_width,
        "neighbor_count": geometry.neighbor_count,
        "input_channels": checkpoint.model.input_channels,
        "parameter_count": sum(parameter.numel() for parameter in checkpoint.model.parameters()),
        "temporal_dilations": list(checkpoint.model.temporal_dilations),
        "stem_kernel_size": checkpoint.model.stem.kernel_size[0],
        "residual_kernel_size": checkpoint.model.blocks[0].kernel_size,
        "target_coordinates": list(settings.target_coordinates),
        "target_coordinate_scaling": TARGET_COORDINATE_SCALING,
        "coordinate_conditioning": checkpoint.model.coordinate_conditioning,
        "neighbor_gating": checkpoint.model.neighbor_gating,
    }
    checkpoint_contract = {
        "path": CHECKPOINT_RELATIVE_PATH.as_posix(),
        "selection_metric": PRIMARY_METRIC,
        "best_step": result.best_step,
        "stored_validation_global_snr_db": result.best_validation_global_snr_db,
        "recomputed_validation_global_snr_db": best_validation.raw_global_snr_db,
        "revalidation_relative_tolerance": CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        "revalidation_absolute_tolerance": CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
        "revalidation_matches": checkpoint_revalidation_matches,
    }
    metrics = _metrics_payload(
        result,
        best_validation=best_validation,
        training_audit=training_audit,
        settings=settings,
        selection_contract=selection_contract,
        collision_audit=collision_audit,
        availability_contract=availability_contract,
        duplicate_audit=duplicate_audit,
        scope_audit=scope_audit,
        checkpoint_contract=checkpoint_contract,
        amplitude_access=amplitude_access,
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
        "neighborhood": geometry_contract,
        "target_coordinates": {
            "order": list(settings.target_coordinates),
            "fit_split": TRAIN_SPLIT,
            "scaling": TARGET_COORDINATE_SCALING,
            "minimum": list(geometry.coordinate_min),
            "maximum": list(geometry.coordinate_max),
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
        "finished_at_utc": _utc_timestamp(),
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
        "neighborhood": geometry_contract,
        "training": training_contract,
        "formal_success_scope": scope_audit,
        "checkpoint": checkpoint_contract,
        "environment": _runtime_resource_metadata(device),
    }
    _write_run_outputs(
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
    _require_exact(config, "model.name", MODEL_NAME)
    _require_exact(config, "model.target_coordinate_scaling", TARGET_COORDINATE_SCALING)
    (
        neighbor_geometry,
        relative_receiver_x_radius,
        source_x_line_radius,
        source_y_half_shot_radius,
        relative_receiver_y_radius,
        expected_target_coordinates,
    ) = _validated_neighborhood(get_required_config_value(config, "model.neighborhood"))
    target_coordinates = _validated_target_coordinate_names(
        get_required_config_value(config, "model.target_coordinates")
    )
    if target_coordinates != expected_target_coordinates:
        raise ConfigurationError(
            "model.target_coordinates must match model.neighborhood geometry: "
            f"{list(expected_target_coordinates)!r}"
        )
    stem_kernel_size = _odd_positive_integer(
        get_required_config_value(config, "model.stem_kernel_size"),
        "model.stem_kernel_size",
    )
    residual_kernel_size = _odd_positive_integer(
        get_required_config_value(config, "model.residual_kernel_size"),
        "model.residual_kernel_size",
    )
    temporal_dilations = _validated_positive_integer_list(
        get_required_config_value(config, "model.temporal_dilations"),
        "model.temporal_dilations",
    )
    coordinate_conditioning = _validated_coordinate_conditioning(config)
    neighbor_gating = _validated_neighbor_gating(config)
    hidden_width = _positive_integer(
        get_required_config_value(config, "model.hidden_width"), "model.hidden_width"
    )
    _require_exact(
        config,
        "sampling.duplicate_physical_coordinate_policy",
        DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    )
    _require_exact(config, "training.amplitude_scaling", PER_TRACE_RMS_SCALING)
    _require_exact(config, "training.loss", LOSS_NAME)
    _require_exact(config, "training.optimizer", OPTIMIZER_NAME)
    _require_exact(config, "training.learning_rate_schedule", LEARNING_RATE_SCHEDULE)
    _require_exact(config, "training.mixed_precision", MIXED_PRECISION)
    _require_exact(config, "evaluation.primary_metric", PRIMARY_METRIC)
    _require_exact(config, "evaluation.comparison", SUCCESS_COMPARISON)
    success_threshold_db = _finite_float(
        get_required_config_value(config, "evaluation.success_threshold_db"),
        "evaluation.success_threshold_db",
    )
    required_effective_split_counts = _validated_effective_split_counts(
        get_required_config_value(config, "evaluation.required_effective_split_counts")
    )
    required_fully_excluded_ffids = _validated_sorted_ffids(
        get_required_config_value(config, "evaluation.required_fully_excluded_ffids"),
        "evaluation.required_fully_excluded_ffids",
    )
    required_eligible_ffid_count = _positive_integer(
        get_required_config_value(config, "evaluation.required_eligible_ffid_count"),
        "evaluation.required_eligible_ffid_count",
    )
    required_sample_count = _positive_integer(
        get_required_config_value(config, "evaluation.required_sample_count"),
        "evaluation.required_sample_count",
    )
    learning_rate = _positive_float(
        get_required_config_value(config, "training.learning_rate"), "training.learning_rate"
    )
    minimum_learning_rate = _positive_float(
        get_required_config_value(config, "training.minimum_learning_rate"),
        "training.minimum_learning_rate",
    )
    expected_minimum = learning_rate * MINIMUM_LEARNING_RATE_FACTOR
    if not math.isclose(minimum_learning_rate, expected_minimum, rel_tol=1.0e-12, abs_tol=0.0):
        raise ConfigurationError(
            "training.minimum_learning_rate must equal "
            f"training.learning_rate * {MINIMUM_LEARNING_RATE_FACTOR:g}"
        )
    gradient_clip_norm = _positive_float(
        get_required_config_value(config, "training.gradient_clip_norm"),
        "training.gradient_clip_norm",
    )
    if gradient_clip_norm != MAX_GRADIENT_NORM:
        raise ConfigurationError(f"training.gradient_clip_norm must be {MAX_GRADIENT_NORM:g}")
    random_seed = _nonnegative_integer(
        get_required_config_value(config, "project.random_seed"), "project.random_seed"
    )
    raw_device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(raw_device, str) or not raw_device:
        raise ConfigurationError("training.device must be a non-empty string")
    return _TrainingSettings(
        random_seed=random_seed,
        hidden_width=hidden_width,
        target_coordinates=target_coordinates,
        stem_kernel_size=stem_kernel_size,
        residual_kernel_size=residual_kernel_size,
        temporal_dilations=temporal_dilations,
        coordinate_conditioning=coordinate_conditioning,
        neighbor_gating=neighbor_gating,
        neighbor_geometry=neighbor_geometry,
        relative_receiver_x_radius=relative_receiver_x_radius,
        source_x_line_radius=source_x_line_radius,
        source_y_half_shot_radius=source_y_half_shot_radius,
        relative_receiver_y_radius=relative_receiver_y_radius,
        learning_rate=learning_rate,
        weight_decay=_nonnegative_float(
            get_required_config_value(config, "training.weight_decay"),
            "training.weight_decay",
        ),
        minimum_learning_rate=minimum_learning_rate,
        total_steps=_positive_integer(
            get_required_config_value(config, "training.total_steps"),
            "training.total_steps",
        ),
        batch_size=_positive_integer(
            get_required_config_value(config, "training.batch_size"), "training.batch_size"
        ),
        target_sampling=_validated_target_sampling(config),
        neighbor_dropout=_probability(
            get_required_config_value(config, "training.neighbor_dropout"),
            "training.neighbor_dropout",
        ),
        derivative_weight=_nonnegative_float(
            get_required_config_value(config, "training.derivative_weight"),
            "training.derivative_weight",
        ),
        evaluation_interval_steps=_positive_integer(
            get_required_config_value(config, "training.evaluation_interval_steps"),
            "training.evaluation_interval_steps",
        ),
        validation_batch_size=_positive_integer(
            get_required_config_value(config, "training.validation_batch_size"),
            "training.validation_batch_size",
        ),
        training_audit_count=_positive_integer(
            get_required_config_value(config, "training.training_audit_count"),
            "training.training_audit_count",
        ),
        mixed_precision=MIXED_PRECISION,
        device=raw_device,
        ffid_range=_optional_ffid_range(config),
        success_threshold_db=success_threshold_db,
        duplicate_physical_coordinate_policy=DUPLICATE_PHYSICAL_COORDINATE_POLICY,
        required_eligible_ffid_count=required_eligible_ffid_count,
        required_sample_count=required_sample_count,
        required_effective_split_counts=required_effective_split_counts,
        required_fully_excluded_ffids=required_fully_excluded_ffids,
    )


def _validated_neighborhood(
    value: object,
) -> tuple[str, int, int, int, int, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("model.neighborhood must be a mapping")
    neighborhood = dict(value)
    if neighborhood == _EXPECTED_NEIGHBORHOOD:
        return (
            SINGLE_LINE_GEOMETRY,
            int(_EXPECTED_NEIGHBORHOOD["relative_receiver_x_radius"]),
            0,
            2 * int(_EXPECTED_NEIGHBORHOOD["source_shot_radius"]),
            int(_EXPECTED_NEIGHBORHOOD["relative_receiver_y_radius"]),
            tuple(TARGET_COORDINATE_ORDER),
        )

    if set(neighborhood) != _MULTILINE_NEIGHBORHOOD_KEYS:
        raise ConfigurationError(
            "model.neighborhood must be the legacy single-line contract or contain exactly "
            f"{sorted(_MULTILINE_NEIGHBORHOOD_KEYS)}"
        )
    if neighborhood["type"] != MULTILINE_GEOMETRY:
        raise ConfigurationError(f"model.neighborhood.type must be {MULTILINE_GEOMETRY!r}")
    expected_spacings = {
        "relative_receiver_spacing_m": RECEIVER_SPACING_M,
        "source_x_line_spacing_m": SOURCE_X_LINE_SPACING_M,
        "source_y_half_shot_spacing_m": SOURCE_Y_HALF_SHOT_SPACING_M,
    }
    for name, expected in expected_spacings.items():
        actual = neighborhood[name]
        if isinstance(actual, bool) or not isinstance(actual, Real) or float(actual) != expected:
            raise ConfigurationError(f"model.neighborhood.{name} must be {expected:g}")
    radii = tuple(
        _nonnegative_integer(neighborhood[name], f"model.neighborhood.{name}")
        for name in (
            "relative_receiver_x_radius",
            "source_x_line_radius",
            "source_y_half_shot_radius",
            "relative_receiver_y_radius",
        )
    )
    if not any(radii):
        raise ConfigurationError("model.neighborhood must include at least one non-zero radius")
    return (MULTILINE_GEOMETRY, *radii, tuple(MULTILINE_TARGET_COORDINATE_ORDER))


def _validated_target_coordinate_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("model.target_coordinates must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in value):
        raise ConfigurationError("model.target_coordinates must contain non-empty strings")
    converted = tuple(value)
    if len(set(converted)) != len(converted):
        raise ConfigurationError("model.target_coordinates must not contain duplicates")
    return converted


def _validated_positive_integer_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty list")
    return tuple(_positive_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def _validated_coordinate_conditioning(config: Mapping[str, object]) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("model configuration must be a mapping")
    value = model.get("coordinate_conditioning", DEFAULT_COORDINATE_CONDITIONING)
    supported = {DEFAULT_COORDINATE_CONDITIONING, "film"}
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(
            f"model.coordinate_conditioning must be one of {sorted(supported)}, got {value!r}"
        )
    return value


def _validated_neighbor_gating(config: Mapping[str, object]) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("model configuration must be a mapping")
    value = model.get("neighbor_gating", DEFAULT_NEIGHBOR_GATING)
    supported = {DEFAULT_NEIGHBOR_GATING, TARGET_COORDINATE_MASKED_SOFTMAX_GATING}
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(
            f"model.neighbor_gating must be one of {sorted(supported)}, got {value!r}"
        )
    return value


def _validated_target_sampling(config: Mapping[str, object]) -> str:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigurationError("training configuration must be a mapping")
    value = training.get("target_sampling", WITH_REPLACEMENT_TARGET_SAMPLING)
    supported = {
        WITH_REPLACEMENT_TARGET_SAMPLING,
        EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    }
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(
            f"training.target_sampling must be one of {sorted(supported)}, got {value!r}"
        )
    return value


def _odd_positive_integer(value: object, name: str) -> int:
    converted = _positive_integer(value, name)
    if converted % 2 == 0:
        raise ConfigurationError(f"{name} must be odd")
    return converted


def _build_neighbor_geometry(
    table: pd.DataFrame,
    available: np.ndarray,
    *,
    settings: _TrainingSettings,
) -> NeighborGeometryLookup | MultilineNeighborGeometryLookup:
    if settings.neighbor_geometry == SINGLE_LINE_GEOMETRY:
        return NeighborGeometryLookup(table, available)
    if settings.neighbor_geometry != MULTILINE_GEOMETRY:
        raise AssertionError(
            f"unsupported validated neighbor geometry: {settings.neighbor_geometry}"
        )
    return MultilineNeighborGeometryLookup(
        table,
        available,
        relative_receiver_x_radius=settings.relative_receiver_x_radius,
        source_x_line_radius=settings.source_x_line_radius,
        source_y_half_shot_radius=settings.source_y_half_shot_radius,
        relative_receiver_y_radius=settings.relative_receiver_y_radius,
    )


def _joined_trace_table(
    trace_table: pd.DataFrame,
    split_table: pd.DataFrame,
    split_rows: np.ndarray,
) -> pd.DataFrame:
    split_by_array_row = np.empty(len(trace_table), dtype=object)
    split_by_array_row[split_rows] = split_table[SPLIT_COLUMN].to_numpy()
    trace_rows = trace_table["array_row"].to_numpy(dtype=np.int64)
    joined = trace_table.copy()
    joined[SPLIT_COLUMN] = split_by_array_row[trace_rows]
    return joined


def _canonicalize_eligible_physical_coordinates(
    joined_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep the lowest array row per eligible physical cell, before selection.

    Winner selection deliberately uses only the exact physical key and
    ``array_row``. Split labels are retained solely for the removal audit and
    never influence which row wins.
    """
    keys = list(_PHYSICAL_COORDINATE_COLUMNS)
    eligible = joined_table.loc[joined_table[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)].copy()
    group_sizes = eligible.groupby(keys, sort=False, dropna=False)["array_row"].transform("size")
    duplicate_rows = eligible.loc[group_sizes.gt(1)]
    winners = (
        eligible.sort_values("array_row", kind="stable")
        .drop_duplicates(keys, keep="first")
        .sort_index()
    )
    removed = eligible.loc[~eligible.index.isin(winners.index)].copy()
    winner_lookup = winners[keys + ["array_row", "ffid", SPLIT_COLUMN]].rename(
        columns={
            "array_row": "kept_array_row",
            "ffid": "kept_ffid",
            SPLIT_COLUMN: "kept_split",
        }
    )
    removed_details = removed.merge(
        winner_lookup,
        how="left",
        on=keys,
        sort=False,
        validate="many_to_one",
    ).sort_values("array_row")
    canonical = joined_table.drop(index=removed.index).reset_index(drop=True)
    canonical_eligible = canonical.loc[canonical[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)]
    remaining_duplicate_mask = canonical_eligible.duplicated(keys, keep=False)
    remaining_duplicate_rows = canonical_eligible.loc[remaining_duplicate_mask]

    removed_counts_by_split = {
        split: int(removed[SPLIT_COLUMN].eq(split).sum()) for split in _EFFECTIVE_SPLITS
    }
    removed_ffid_counts = removed["ffid"].value_counts().sort_index()
    removed_records = [
        {
            "array_row": int(row.array_row),
            "ffid": int(row.ffid),
            "split": str(row.split),
            "kept_array_row": int(row.kept_array_row),
            "kept_ffid": int(row.kept_ffid),
            "kept_split": str(row.kept_split),
        }
        for row in removed_details[
            ["array_row", "ffid", SPLIT_COLUMN, "kept_array_row", "kept_ffid", "kept_split"]
        ].itertuples(index=False)
    ]
    audit: dict[str, object] = {
        "policy": DUPLICATE_PHYSICAL_COORDINATE_POLICY,
        "physical_coordinate_key": keys,
        "scope": "all_amplitude_eligible_splits_before_ffid_selection",
        "winner_rule": "lowest_array_row",
        "winner_selection_uses_split": False,
        "winner_selection_uses_amplitude": False,
        "input_eligible_trace_count": len(eligible),
        "duplicate_physical_cell_count": int(duplicate_rows.drop_duplicates(keys).shape[0]),
        "duplicate_physical_row_count": len(duplicate_rows),
        "removed_trace_count": len(removed),
        "removed_counts_by_split": removed_counts_by_split,
        "removed_counts_by_ffid": {
            str(int(ffid)): int(count) for ffid, count in removed_ffid_counts.items()
        },
        "removed_rows": removed_records,
        "retained_eligible_trace_count": len(canonical_eligible),
        "remaining_duplicate_physical_cell_count": int(
            remaining_duplicate_rows.drop_duplicates(keys).shape[0]
        ),
        "remaining_duplicate_physical_row_count": len(remaining_duplicate_rows),
    }
    return canonical, audit


def _selected_trace_table(
    canonical_table: pd.DataFrame,
    *,
    ffid_range: tuple[int, int] | None,
) -> pd.DataFrame:
    selected = canonical_table[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)
    if ffid_range is not None:
        selected &= canonical_table["ffid"].between(*ffid_range)
    result = canonical_table.loc[selected].reset_index(drop=True)
    if result.empty:
        raise ValueError("configured FFID selection contains no eligible traces")
    return result


def _validate_selected_split_coverage(table: pd.DataFrame) -> None:
    for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT):
        missing = sorted(
            set(int(value) for value in table["ffid"].unique())
            - set(int(value) for value in table.loc[table[SPLIT_COLUMN].eq(split), "ffid"].unique())
        )
        if missing:
            raise ValueError(f"selected eligible FFIDs contain no {split} rows: {missing}")


def _load_unit_rms_rows(amplitudes: np.ndarray, array_rows: np.ndarray) -> np.ndarray:
    scaled = np.empty((len(array_rows), amplitudes.shape[1]), dtype=np.float32)
    for start in range(0, len(array_rows), _AMPLITUDE_ROW_CHUNK_SIZE):
        stop = min(start + _AMPLITUDE_ROW_CHUNK_SIZE, len(array_rows))
        rows = array_rows[start:stop]
        scaled[start:stop] = per_trace_rms_scaled_amplitudes(
            amplitudes[rows],
            array_rows=rows,
        )
    return scaled


def _neighbor_availability(
    geometry: NeighborGeometryLookup,
    positions: np.ndarray,
) -> dict[str, object]:
    counts = np.empty(len(positions), dtype=np.int16)
    for start in range(0, len(positions), _AVAILABILITY_CHUNK_SIZE):
        stop = min(start + _AVAILABILITY_CHUNK_SIZE, len(positions))
        counts[start:stop] = np.count_nonzero(
            geometry.neighbor_positions(positions[start:stop]) >= 0,
            axis=1,
        )
    quantile_values = np.quantile(
        counts,
        (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0),
    )
    labels = ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")
    histogram = np.bincount(counts, minlength=geometry.neighbor_count + 1)
    return {
        "row_count": len(positions),
        "mean": float(np.mean(counts, dtype=np.float64)),
        "zero_neighbor_rows": int(histogram[0]),
        "quantiles": {
            label: float(value) for label, value in zip(labels, quantile_values, strict=True)
        },
        "histogram": {
            str(count): int(frequency) for count, frequency in enumerate(histogram) if frequency
        },
    }


def _physical_coordinate_frame(table: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    selected = table.iloc[positions]
    return pd.DataFrame(
        {
            "source_x_m": selected["source_x_m"].to_numpy(dtype=np.float64),
            "source_y_m": selected["source_y_m"].to_numpy(dtype=np.float64),
            "relative_receiver_x_m": (selected["receiver_x_m"] - selected["source_x_m"]).to_numpy(
                dtype=np.float64
            ),
            "relative_receiver_y_m": (selected["receiver_y_m"] - selected["source_y_m"]).to_numpy(
                dtype=np.float64
            ),
        }
    )


def _train_validation_coordinate_overlap(
    table: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = list(_physical_coordinate_frame(table, train_positions).columns)
    train = _physical_coordinate_frame(table, train_positions)
    train["train_position"] = train_positions
    train["array_row"] = table.iloc[train_positions]["array_row"].to_numpy(dtype=np.int64)
    train = train.sort_values("array_row").drop_duplicates(keys, keep="first")
    validation = _physical_coordinate_frame(table, validation_positions)
    validation["validation_index"] = np.arange(len(validation_positions), dtype=np.int64)
    joined = validation.merge(train[keys + ["train_position"]], how="left", on=keys, sort=False)
    joined = joined.sort_values("validation_index")
    overlap = joined["train_position"].notna().to_numpy()
    train_position = np.full(len(validation_positions), -1, dtype=np.int64)
    train_position[overlap] = joined.loc[overlap, "train_position"].to_numpy(dtype=np.int64)
    return overlap, train_position


def _exact_overlap_amplitude_count(
    train_amplitudes: np.ndarray,
    validation_amplitudes: np.ndarray,
    train_positions: np.ndarray,
    overlap_train_positions: np.ndarray,
    overlap_mask: np.ndarray,
) -> int:
    compact_by_position = {int(position): index for index, position in enumerate(train_positions)}
    count = 0
    for validation_index in np.flatnonzero(overlap_mask):
        train_index = compact_by_position[int(overlap_train_positions[validation_index])]
        count += int(
            np.array_equal(train_amplitudes[train_index], validation_amplitudes[validation_index])
        )
    return count


def _selection_contract(
    canonical_table: pd.DataFrame,
    selected_table: pd.DataFrame,
    *,
    sample_count: int,
    configured_ffid_range: tuple[int, int] | None,
) -> dict[str, object]:
    split_counts = {
        split: int(selected_table[SPLIT_COLUMN].eq(split).sum())
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    in_range = np.ones(len(canonical_table), dtype=bool)
    if configured_ffid_range is not None:
        in_range = canonical_table["ffid"].between(*configured_ffid_range).to_numpy()
    full_split = canonical_table[SPLIT_COLUMN].to_numpy()
    excluded_count = int(np.count_nonzero(in_range & (full_split == EXCLUDED_SPLIT)))
    ffids = sorted(int(value) for value in selected_table["ffid"].unique())
    contract: dict[str, object] = {
        "configured_ffid_range": (
            list(configured_ffid_range) if configured_ffid_range is not None else None
        ),
        "selected_ffid_count": len(ffids),
        "selected_ffid_range": [ffids[0], ffids[-1]],
        "selected_ffids": ffids,
        "sample_count": sample_count,
        "effective_eligible_trace_count": sum(split_counts.values()),
        "split_counts": {**split_counts, EXCLUDED_SPLIT: excluded_count},
    }
    return contract


def _geometry_contract(
    geometry: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
    *,
    settings: _TrainingSettings,
) -> dict[str, object]:
    if settings.neighbor_geometry == SINGLE_LINE_GEOMETRY:
        condition: dict[str, object] = {
            "type": SINGLE_LINE_GEOMETRY,
            **_EXPECTED_NEIGHBORHOOD,
        }
        axes = [
            "relative_receiver_x_index",
            "source_shot_index",
            "relative_receiver_y_index",
        ]
    else:
        condition = {
            "type": MULTILINE_GEOMETRY,
            "relative_receiver_x_radius": settings.relative_receiver_x_radius,
            "source_x_line_radius": settings.source_x_line_radius,
            "source_y_half_shot_radius": settings.source_y_half_shot_radius,
            "relative_receiver_y_radius": settings.relative_receiver_y_radius,
            "relative_receiver_spacing_m": RECEIVER_SPACING_M,
            "source_x_line_spacing_m": SOURCE_X_LINE_SPACING_M,
            "source_y_half_shot_spacing_m": SOURCE_Y_HALF_SHOT_SPACING_M,
            "same_source_x_only": settings.source_x_line_radius == 0,
        }
        axes = [
            "relative_receiver_x_index",
            "source_x_line_index",
            "source_y_half_shot_index",
            "relative_receiver_y_index",
        ]
    offsets = geometry.offsets
    return {
        **condition,
        "neighbor_count": geometry.neighbor_count,
        "offset_order": [list(value) for value in offsets],
        "offset_order_axes": axes,
        "available_training_rows": geometry.available_count,
        "indexed_training_cells": geometry.indexed_available_count,
        "center_offset_count": sum(not any(offset) for offset in offsets),
    }


def _training_contract(
    settings: _TrainingSettings,
    result: NeighborInpainterTrainingResult,
    provider: _RandomTrainBatchProvider,
) -> dict[str, object]:
    return {
        "amplitude_scaling": PER_TRACE_RMS_SCALING,
        "training_scale_source": TRAINING_SCALE_SOURCE,
        "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        "validation_scale_source": VALIDATION_SCALE_SOURCE,
        "loss": LOSS_NAME,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "learning_rate_schedule": LEARNING_RATE_SCHEDULE,
        "minimum_learning_rate": settings.minimum_learning_rate,
        "minimum_learning_rate_factor": MINIMUM_LEARNING_RATE_FACTOR,
        "gradient_clip_norm": MAX_GRADIENT_NORM,
        "total_steps": settings.total_steps,
        "batch_size": settings.batch_size,
        "target_sampling": settings.target_sampling,
        "target_sampling_seed": _target_sampling_seed(settings),
        "neighbor_dropout_seed": settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET,
        "target_sampling_rng_independent_of_neighbor_dropout": (
            settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        ),
        "neighbor_dropout": settings.neighbor_dropout,
        "derivative_weight": settings.derivative_weight,
        "evaluation_interval_steps": settings.evaluation_interval_steps,
        "validation_batch_size": settings.validation_batch_size,
        "mixed_precision": settings.mixed_precision,
        "effective_bfloat16": settings.mixed_precision == MIXED_PRECISION
        and torch.device(settings.device).type == "cuda",
        "cudnn_benchmark": (
            torch.device(settings.device).type == "cuda" and torch.backends.cudnn.benchmark
        ),
        "cudnn_deterministic": (
            torch.device(settings.device).type == "cuda" and torch.backends.cudnn.deterministic
        ),
        "training_trace_count": result.training_trace_count,
        "drawn_training_targets": provider.draw_count,
        "unique_training_targets_seen": provider.unique_target_count,
        "training_audit_count": settings.training_audit_count,
        "training_audit_seed": settings.random_seed + TRAINING_AUDIT_SEED_OFFSET,
    }


def _target_sampling_seed(settings: _TrainingSettings) -> int:
    if settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
        return settings.random_seed + EPOCH_TARGET_SAMPLING_SEED_OFFSET
    return settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET


def _formal_scope_audit(
    settings: _TrainingSettings,
    *,
    selection_contract: Mapping[str, object],
    preparation_contract: Mapping[str, object],
) -> dict[str, object]:
    trace_quality = preparation_contract.get("trace_quality")
    if not isinstance(trace_quality, Mapping):
        raise RuntimeError("validated preparation contract is missing trace_quality")
    fully_excluded_ffids = trace_quality.get("fully_excluded_ffids")
    if not isinstance(fully_excluded_ffids, list):
        raise RuntimeError("validated preparation trace_quality has invalid fully_excluded_ffids")
    split_counts = selection_contract["split_counts"]
    if not isinstance(split_counts, Mapping):
        raise RuntimeError("selection split_counts must be an object")

    required_split_counts = dict(settings.required_effective_split_counts)
    actual_split_counts = {split: int(split_counts[split]) for split in _EFFECTIVE_SPLITS}
    checks = {
        "ffid_range_not_configured": settings.ffid_range is None,
        "eligible_ffid_count_matches": (
            selection_contract["selected_ffid_count"] == settings.required_eligible_ffid_count
        ),
        "sample_count_matches": (
            selection_contract["sample_count"] == settings.required_sample_count
        ),
        "effective_split_counts_match": actual_split_counts == required_split_counts,
        "fully_excluded_ffids_match": (
            fully_excluded_ffids == list(settings.required_fully_excluded_ffids)
        ),
    }
    return {
        "requirements": {
            "ffid_range": None,
            "eligible_ffid_count": settings.required_eligible_ffid_count,
            "sample_count": settings.required_sample_count,
            "effective_split_counts": required_split_counts,
            "fully_excluded_ffids": list(settings.required_fully_excluded_ffids),
        },
        "actual": {
            "ffid_range": (list(settings.ffid_range) if settings.ffid_range is not None else None),
            "eligible_ffid_count": selection_contract["selected_ffid_count"],
            "sample_count": selection_contract["sample_count"],
            "effective_split_counts": actual_split_counts,
            "fully_excluded_ffids": list(fully_excluded_ffids),
        },
        "checks": checks,
        "scope_success": all(checks.values()),
    }


def _completed_formal_scope_audit(
    configured_scope_audit: Mapping[str, object],
    *,
    collision_audit: Mapping[str, object],
    geometry_contract: Mapping[str, object],
    amplitude_access: Mapping[str, object],
    checkpoint_revalidation_matches: bool,
    selected_metric: float,
    recomputed_metric: float,
) -> dict[str, object]:
    completed = deepcopy(dict(configured_scope_audit))
    raw_checks = completed.get("checks")
    if not isinstance(raw_checks, Mapping):
        raise RuntimeError("formal scope audit checks must be an object")
    materialized = amplitude_access.get("value_rows_materialized_by_split")
    if not isinstance(materialized, Mapping):
        raise RuntimeError("amplitude materialization audit must be an object")
    checks = dict(raw_checks)
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
            "train_geometry_collision_cells_zero": (
                collision_audit["train_coordinate_collision_cells"] == 0
            ),
            "train_validation_coordinate_overlap_zero": (
                collision_audit["train_validation_coordinate_overlap_rows"] == 0
            ),
            "neighbor_center_offset_count_zero": geometry_contract["center_offset_count"] == 0,
            "test_value_rows_not_materialized": materialized.get(TEST_SPLIT) is False,
            "excluded_value_rows_not_materialized": materialized.get(EXCLUDED_SPLIT) is False,
        }
    )
    completed["checks"] = checks
    completed["scope_success"] = all(checks.values())
    return completed


def _metrics_payload(
    result: NeighborInpainterTrainingResult,
    *,
    best_validation: _EvaluationResult,
    training_audit: _EvaluationResult,
    settings: _TrainingSettings,
    selection_contract: Mapping[str, object],
    collision_audit: Mapping[str, object],
    availability_contract: Mapping[str, object],
    duplicate_audit: Mapping[str, object],
    scope_audit: Mapping[str, object],
    checkpoint_contract: Mapping[str, object],
    amplitude_access: Mapping[str, object],
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
            "best_validation_raw_global_snr_db": best_validation.raw_global_snr_db,
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
            "clean_validation_trace_count": best_validation.clean_trace_count,
            "clean_validation_raw_global_snr_db": best_validation.clean_raw_global_snr_db,
            "clean_validation_global_snr_db": best_validation.clean_raw_global_snr_db,
            "training_audit_trace_count": settings.training_audit_count,
            "training_audit_seed": settings.random_seed + TRAINING_AUDIT_SEED_OFFSET,
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


def _runtime_resource_metadata(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cudnn_benchmark": device.type == "cuda" and torch.backends.cudnn.benchmark,
        "cudnn_deterministic": device.type == "cuda" and torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return result


def _seed_global_model_initialization(seed: int, *, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _snr_db(signal_energy: float, error_energy: float) -> float:
    if not math.isfinite(signal_energy) or signal_energy <= 0.0:
        raise ValueError("validation signal energy must be positive and finite")
    if not math.isfinite(error_energy) or error_energy <= 0.0:
        raise ValueError("validation error energy must be positive and finite")
    return 10.0 * math.log10(signal_energy / error_energy)


def _passes_success_threshold(metric_db: float, threshold_db: float) -> bool:
    return bool(metric_db > threshold_db)


def _require_exact(config: Mapping[str, object], dotted_path: str, expected: object) -> None:
    actual = get_required_config_value(config, dotted_path)
    if actual != expected:
        raise ConfigurationError(f"{dotted_path} must be {expected!r}, got {actual!r}")


def _optional_ffid_range(config: Mapping[str, object]) -> tuple[int, int] | None:
    training = config.get("training")
    if not isinstance(training, Mapping) or "ffid_range" not in training:
        return None
    value = training["ffid_range"]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in value)
        or int(value[0]) < 0
        or int(value[0]) > int(value[1])
    ):
        raise ConfigurationError("training.ffid_range must be [minimum, maximum] integers")
    return int(value[0]), int(value[1])


def _validated_effective_split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_EFFECTIVE_SPLITS):
        raise ConfigurationError(
            "evaluation.required_effective_split_counts must contain exactly "
            f"{list(_EFFECTIVE_SPLITS)}"
        )
    return {
        split: _positive_integer(
            value[split],
            f"evaluation.required_effective_split_counts.{split}",
        )
        for split in _EFFECTIVE_SPLITS
    }


def _validated_sorted_ffids(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(ffid, bool) or not isinstance(ffid, Integral) or int(ffid) < 0 for ffid in value
    ):
        raise ConfigurationError(f"{name} must be a sorted unique list of non-negative integers")
    converted = [int(ffid) for ffid in value]
    if converted != sorted(set(converted)):
        raise ConfigurationError(f"{name} must be a sorted unique list of non-negative integers")
    return tuple(converted)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigurationError(f"{name} must be a finite number")
    return converted


def _positive_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return converted


def _nonnegative_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0:
        raise ConfigurationError(f"{name} must be non-negative")
    return converted


def _probability(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0 or converted >= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1)")
    return converted
