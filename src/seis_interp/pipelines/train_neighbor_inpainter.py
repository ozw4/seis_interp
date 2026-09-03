"""Train the geometry-neighbor trace inpainter on prepared survey splits."""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Real
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
from seis_interp.data.trace_store import (
    OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES,
)
from seis_interp.data.trace_store import (
    TRACES_FILE_NAME,
    canonical_source_files,
)
from seis_interp.evaluation import formal_scope, oracle_trace_snr
from seis_interp.models.neighbor_trace_inpainter import (
    DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX as DEFAULT_LEGACY_COARSE_SHIFT,
)
from seis_interp.models.neighbor_trace_inpainter import (
    DEFAULT_COORDINATE_CONDITIONING,
    DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE,
    DEFAULT_NEIGHBOR_GATING,
    DEFAULT_PREDICTION_REFERENCE,
    MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE,
    SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
    TARGET_COORDINATE_MASKED_SOFTMAX_GATING,
    NeighborTraceInpainter,
)
from seis_interp.models.shared_offset_attention_inpainter import (
    DEFAULT_ATTENTION_GEOMETRY_PRIOR_SCALE,
    DEFAULT_ATTENTION_WIDTH,
    DEFAULT_NEIGHBOR_FEATURE_WIDTH,
    DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE,
    OFFSET_ORDER_AXES,
    OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING,
    SharedOffsetAttentionInpainter,
)
from seis_interp.models.shared_offset_attention_inpainter import (
    DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX as DEFAULT_SHARED_COARSE_SHIFT,
)
from seis_interp.models.shared_offset_attention_inpainter import (
    MODEL_NAME as SHARED_OFFSET_ATTENTION_MODEL_NAME,
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
from seis_interp.processing.source_bracketing import SameLineReceiverBracketingLookup
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
    extract_per_trace_rms_scaled_rows,
)
from seis_interp.training.neighbor_inpainter_checkpoints import (
    NeighborInpainterModel,
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
TARGET_COORDINATE_SCALING = "train_minmax"
STEM_KERNEL_SIZE = 15
RESIDUAL_KERNEL_SIZE = 7
SINGLE_LINE_GEOMETRY = "single_source_line"
MULTILINE_GEOMETRY = "multiline_staggered_source"
WITH_REPLACEMENT_TARGET_SAMPLING = "with_replacement"
EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING = "epoch_without_replacement"
_SOURCE_BRACKETING_REFERENCE_MODES = frozenset(
    (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    )
)
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
_AVAILABILITY_CHUNK_SIZE = 65536


@dataclass(frozen=True)
class _TrainingSettings:
    random_seed: int
    model_name: str
    hidden_width: int
    neighbor_feature_width: int
    attention_width: int
    coarse_shift_samples_per_relative_receiver_y_index: int
    attention_geometry_prior_scale: float
    target_coordinates: tuple[str, ...]
    stem_kernel_size: int
    residual_kernel_size: int
    temporal_dilations: tuple[int, ...]
    coordinate_conditioning: str
    neighbor_gating: str
    neighbor_alignment_kernel_size: int
    prediction_reference: str
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
    required_ffid_split_counts: Mapping[str, int] | None
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
        exclude_target_ffid_neighbors: bool = False,
        ffids_by_position: np.ndarray | None = None,
        source_bracketing: SameLineReceiverBracketingLookup | None = None,
        source_bracketing_reference: str = (SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE),
    ) -> None:
        self.lookup = lookup
        self.train_positions = np.asarray(train_positions, dtype=np.int64)
        self.train_amplitudes = train_amplitudes
        self.device = device
        self.exclude_target_ffid_neighbors = exclude_target_ffid_neighbors
        self.ffids_by_position = (
            np.asarray(ffids_by_position, dtype=np.int64) if ffids_by_position is not None else None
        )
        self.source_bracketing = source_bracketing
        self.source_bracketing_reference = source_bracketing_reference
        if (
            self.source_bracketing is not None
            and self.source_bracketing_reference not in _SOURCE_BRACKETING_REFERENCE_MODES
        ):
            raise ValueError(
                "source_bracketing_reference must select a supported source bracketing mode"
            )
        if self.exclude_target_ffid_neighbors and (
            self.ffids_by_position is None or self.ffids_by_position.shape != (lookup.row_count,)
        ):
            raise ValueError(
                "ffids_by_position must match lookup rows when target-FFID neighbors are excluded"
            )
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
        neighbor_positions = _neighbor_positions(
            self.lookup,
            positions,
            exclude_target_ffid_neighbors=self.exclude_target_ffid_neighbors,
            ffids_by_position=self.ffids_by_position,
        )
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
        if self.source_bracketing is not None:
            bracket = self.source_bracketing.batch(positions)
            bracket_available = bracket.positions >= 0
            safe_bracket_positions = np.maximum(bracket.positions, 0)
            bracket_compact = self._train_index_by_position[safe_bracket_positions]
            if np.any(bracket_compact[bracket_available] < 0):
                raise RuntimeError(
                    "source bracket returned a row outside the TRAIN amplitude store"
                )
            safe_bracket_compact = np.maximum(bracket_compact, 0)
            bracket_indices = torch.as_tensor(
                safe_bracket_compact,
                dtype=torch.long,
                device=self.device,
            )
            bracket_weights = torch.as_tensor(
                bracket.weights,
                dtype=self.train_amplitudes.dtype,
                device=self.device,
            )
            if (
                self.source_bracketing_reference
                == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
            ):
                bracket_available_tensor = torch.as_tensor(
                    bracket_available,
                    dtype=torch.bool,
                    device=self.device,
                )
                bracket_traces = self.train_amplitudes[bracket_indices]
                bracket_traces = bracket_traces * bracket_available_tensor[..., None]
                neighbors = torch.cat((neighbors, bracket_traces), dim=1)
                availability_tensor = torch.cat(
                    (availability_tensor, bracket_weights),
                    dim=1,
                )
            else:
                reference = (
                    self.train_amplitudes[bracket_indices] * bracket_weights[..., None]
                ).sum(dim=1)
                reference_available = torch.as_tensor(
                    np.any(bracket_available, axis=1),
                    dtype=torch.bool,
                    device=self.device,
                )
                neighbors = torch.cat((neighbors, reference[:, None, :]), dim=1)
                availability_tensor = torch.cat(
                    (availability_tensor, reference_available[:, None]),
                    dim=1,
                )
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

    def __call__(self, model: NeighborInpainterModel) -> float:
        return self.evaluate(model).raw_global_snr_db

    @torch.inference_mode()
    def evaluate(self, model: NeighborInpainterModel) -> _EvaluationResult:
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
            raw_global_snr_db=oracle_trace_snr.global_snr_db_from_energies(
                signal_energy, error_energy
            ),
            predicted_unit_rms_global_snr_db=oracle_trace_snr.global_snr_db_from_energies(
                signal_energy, predicted_unit_error_energy
            ),
            signal_energy=signal_energy,
            error_energy=error_energy,
            error_mean_square=error_energy / point_count,
            clean_trace_count=int(np.count_nonzero(self.clean_mask)),
            clean_raw_global_snr_db=oracle_trace_snr.global_snr_db_from_energies(
                clean_signal_energy, clean_error_energy
            ),
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
        allow_excluded="trace_amplitude_filter" in preparation,
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
    joined_table = trace_selection.join_trace_splits(
        dataset.trace_table,
        split_table,
        split_rows,
    )
    canonical_table, duplicate_audit = (
        trace_canonicalization.canonicalize_eligible_physical_coordinates(joined_table)
    )
    if duplicate_audit != preliminary_duplicate_audit:
        raise RuntimeError("duplicate-coordinate audit changed while loading the interim dataset")
    selected_table = trace_selection.select_eligible_traces(
        canonical_table,
        ffid_range=settings.ffid_range,
    )
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
            "formal neighbor-inpainter run does not match its required survey scope: "
            f"{failed_checks}; configure training.ffid_range for a diagnostic subset"
        )
    selected_split = selected_table[SPLIT_COLUMN].to_numpy()
    selected_ffids = selected_table["ffid"].to_numpy(dtype=np.int64)
    train_positions = np.flatnonzero(selected_split == TRAIN_SPLIT).astype(np.int64)
    validation_positions = np.flatnonzero(selected_split == VALIDATION_SPLIT).astype(np.int64)
    trace_selection.validate_selected_split_coverage(
        selected_table,
        split_scope=str(preparation_contract["split_scope"]),
    )
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
        TRAIN_SPLIT: _neighbor_availability(
            geometry,
            train_positions,
            exclude_target_ffid_neighbors=settings.exclude_target_ffid_neighbors,
            ffids_by_position=selected_ffids,
        ),
        VALIDATION_SPLIT: _neighbor_availability(
            geometry,
            validation_positions,
            exclude_target_ffid_neighbors=settings.exclude_target_ffid_neighbors,
            ffids_by_position=selected_ffids,
        ),
    }
    source_bracketing: SameLineReceiverBracketingLookup | None = None
    source_bracketing_contract: dict[str, object] | None = None
    if settings.prediction_reference in _SOURCE_BRACKETING_REFERENCE_MODES:
        source_bracketing = SameLineReceiverBracketingLookup(
            selected_table,
            available,
            ffids_by_position=selected_ffids,
        )
        source_bracketing_contract = _source_bracketing_contract(
            source_bracketing,
            train_positions=train_positions,
            validation_positions=validation_positions,
            prediction_reference=settings.prediction_reference,
        )
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
    train_amplitudes_host = extract_per_trace_rms_scaled_rows(dataset.amplitudes, train_array_rows)
    validation_amplitudes_host = extract_per_trace_rms_scaled_rows(
        dataset.amplitudes, validation_array_rows
    )
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
        exclude_target_ffid_neighbors=settings.exclude_target_ffid_neighbors,
        ffids_by_position=selected_ffids,
        source_bracketing=source_bracketing,
        source_bracketing_reference=settings.prediction_reference,
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
    model = _build_inpainter_model(settings, geometry)
    generator = torch.Generator(device=device).manual_seed(
        settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET
    )
    result = train_neighbor_trace_inpainter(
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
        training_trace_count=len(train_positions),
        reporter=progress_reporter,
    )
    checkpoint = load_neighbor_inpainter_checkpoint(
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
    scope_audit = formal_scope.complete_neighbor_formal_scope_audit(
        configured_scope_audit,
        collision_audit=collision_audit,
        geometry_contract=geometry_contract,
        availability_contract=availability_contract,
        source_bracketing_contract=source_bracketing_contract,
        amplitude_access=amplitude_access,
        checkpoint_revalidation_matches=checkpoint_revalidation_matches,
        selected_metric=result.best_validation_global_snr_db,
        recomputed_metric=best_validation.raw_global_snr_db,
    )
    training_contract = _training_contract(settings, result, batch_provider)
    model_contract = _model_contract(
        checkpoint.model,
        settings=settings,
        geometry=geometry,
    )
    checkpoint_contract = {
        "path": run_records.CHECKPOINT_RELATIVE_PATH.as_posix(),
        "selection_metric": oracle_trace_snr.PRIMARY_METRIC,
        "best_step": result.best_step,
        "stored_validation_global_snr_db": result.best_validation_global_snr_db,
        "recomputed_validation_global_snr_db": best_validation.raw_global_snr_db,
        "revalidation_relative_tolerance": formal_scope.CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
        "revalidation_absolute_tolerance": formal_scope.CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
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
        source_bracketing_contract=source_bracketing_contract,
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
        "neighborhood": geometry_contract,
        "training": training_contract,
        "formal_success_scope": scope_audit,
        "checkpoint": checkpoint_contract,
        "environment": run_records.runtime_resource_metadata(device),
    }
    if source_bracketing_contract is not None:
        inputs_lock["source_bracketing"] = source_bracketing_contract
        run_metadata["source_bracketing"] = source_bracketing_contract
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
    model_name = _validated_model_name(config)
    config_values.require_exact(
        config,
        "model.target_coordinate_scaling",
        TARGET_COORDINATE_SCALING,
    )
    (
        neighbor_geometry,
        relative_receiver_x_radius,
        source_x_line_radius,
        source_y_half_shot_radius,
        relative_receiver_y_radius,
        expected_target_coordinates,
    ) = _validated_neighborhood(get_required_config_value(config, "model.neighborhood"))
    target_coordinates = config_values.validated_target_coordinate_names(
        get_required_config_value(config, "model.target_coordinates")
    )
    if target_coordinates != expected_target_coordinates:
        raise ConfigurationError(
            "model.target_coordinates must match model.neighborhood geometry: "
            f"{list(expected_target_coordinates)!r}"
        )
    stem_kernel_size = config_values.odd_positive_integer(
        get_required_config_value(config, "model.stem_kernel_size"),
        "model.stem_kernel_size",
    )
    residual_kernel_size = config_values.odd_positive_integer(
        get_required_config_value(config, "model.residual_kernel_size"),
        "model.residual_kernel_size",
    )
    temporal_dilations = config_values.validated_positive_integer_list(
        get_required_config_value(config, "model.temporal_dilations"),
        "model.temporal_dilations",
    )
    coordinate_conditioning = _validated_coordinate_conditioning(config)
    neighbor_gating = _validated_neighbor_gating(config)
    neighbor_alignment_kernel_size = _validated_neighbor_alignment_kernel_size(config)
    prediction_reference = _validated_prediction_reference(config, model_name=model_name)
    hidden_width = config_values.positive_integer(
        get_required_config_value(config, "model.hidden_width"), "model.hidden_width"
    )
    (
        neighbor_feature_width,
        attention_width,
        coarse_shift_samples_per_relative_receiver_y_index,
        attention_geometry_prior_scale,
    ) = _validated_shared_offset_attention_settings(config, model_name=model_name)
    if model_name == SHARED_OFFSET_ATTENTION_MODEL_NAME:
        if neighbor_geometry != MULTILINE_GEOMETRY:
            raise ConfigurationError(
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r} requires "
                f"model.neighborhood.type={MULTILINE_GEOMETRY!r}"
            )
        if tuple(expected_target_coordinates) != tuple(MULTILINE_TARGET_COORDINATE_ORDER):
            raise ConfigurationError(
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r} requires four-axis "
                "multiline target coordinates"
            )
        if coordinate_conditioning != "film":
            raise ConfigurationError(
                f"model.coordinate_conditioning must be 'film' for "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}"
            )
        if neighbor_gating != OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING:
            raise ConfigurationError(
                f"model.neighbor_gating must be "
                f"{OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING!r} for "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}"
            )
        if neighbor_alignment_kernel_size != DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE:
            raise ConfigurationError(
                "model.neighbor_alignment_kernel_size must be 1 for "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}; coarse alignment "
                "is controlled by model.coarse_shift_samples_per_relative_receiver_y_index"
            )
    else:
        if neighbor_gating == OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING:
            raise ConfigurationError(
                f"model.neighbor_gating={OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING!r} requires "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}"
            )
        if (
            coarse_shift_samples_per_relative_receiver_y_index > 0
            and neighbor_geometry != MULTILINE_GEOMETRY
        ):
            raise ConfigurationError(
                "model.coarse_shift_samples_per_relative_receiver_y_index > 0 requires "
                f"model.neighborhood.type={MULTILINE_GEOMETRY!r}"
            )
        if (
            prediction_reference in _SOURCE_BRACKETING_REFERENCE_MODES
            and coarse_shift_samples_per_relative_receiver_y_index > 0
        ):
            raise ConfigurationError(
                "same-line exact-receiver bracketing cannot be combined with legacy "
                "coarse alignment"
            )
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
    success_threshold_db = config_values.finite_float(
        get_required_config_value(config, "evaluation.success_threshold_db"),
        "evaluation.success_threshold_db",
    )
    required_effective_split_counts = config_values.validated_effective_split_counts(
        get_required_config_value(config, "evaluation.required_effective_split_counts")
    )
    evaluation = config.get("evaluation")
    required_ffid_split_counts = (
        config_values.validated_ffid_split_counts(evaluation["required_ffid_split_counts"])
        if isinstance(evaluation, Mapping) and "required_ffid_split_counts" in evaluation
        else None
    )
    sampling = config.get("sampling")
    split_scope = (
        sampling.get("split_scope", "global") if isinstance(sampling, Mapping) else "global"
    )
    exclude_target_ffid_neighbors = _validated_exclude_target_ffid_neighbors(config)
    if split_scope == "whole_ffid" and not exclude_target_ffid_neighbors:
        raise ConfigurationError(
            "sampling.split_scope='whole_ffid' requires training.exclude_target_ffid_neighbors=true"
        )
    if split_scope == "whole_ffid" and required_ffid_split_counts is None:
        raise ConfigurationError(
            "sampling.split_scope='whole_ffid' requires evaluation.required_ffid_split_counts"
        )
    required_fully_excluded_ffids = config_values.validated_sorted_ffids(
        get_required_config_value(config, "evaluation.required_fully_excluded_ffids"),
        "evaluation.required_fully_excluded_ffids",
    )
    required_eligible_ffid_count = config_values.positive_integer(
        get_required_config_value(config, "evaluation.required_eligible_ffid_count"),
        "evaluation.required_eligible_ffid_count",
    )
    required_sample_count = config_values.positive_integer(
        get_required_config_value(config, "evaluation.required_sample_count"),
        "evaluation.required_sample_count",
    )
    learning_rate = config_values.positive_float(
        get_required_config_value(config, "training.learning_rate"), "training.learning_rate"
    )
    minimum_learning_rate = config_values.positive_float(
        get_required_config_value(config, "training.minimum_learning_rate"),
        "training.minimum_learning_rate",
    )
    expected_minimum = learning_rate * MINIMUM_LEARNING_RATE_FACTOR
    if not math.isclose(minimum_learning_rate, expected_minimum, rel_tol=1.0e-12, abs_tol=0.0):
        raise ConfigurationError(
            "training.minimum_learning_rate must equal "
            f"training.learning_rate * {MINIMUM_LEARNING_RATE_FACTOR:g}"
        )
    gradient_clip_norm = config_values.positive_float(
        get_required_config_value(config, "training.gradient_clip_norm"),
        "training.gradient_clip_norm",
    )
    if gradient_clip_norm != MAX_GRADIENT_NORM:
        raise ConfigurationError(f"training.gradient_clip_norm must be {MAX_GRADIENT_NORM:g}")
    random_seed = config_values.nonnegative_integer(
        get_required_config_value(config, "project.random_seed"), "project.random_seed"
    )
    raw_device = device_override or get_required_config_value(config, "training.device")
    if not isinstance(raw_device, str) or not raw_device:
        raise ConfigurationError("training.device must be a non-empty string")
    return _TrainingSettings(
        random_seed=random_seed,
        model_name=model_name,
        hidden_width=hidden_width,
        neighbor_feature_width=neighbor_feature_width,
        attention_width=attention_width,
        coarse_shift_samples_per_relative_receiver_y_index=(
            coarse_shift_samples_per_relative_receiver_y_index
        ),
        attention_geometry_prior_scale=attention_geometry_prior_scale,
        target_coordinates=target_coordinates,
        stem_kernel_size=stem_kernel_size,
        residual_kernel_size=residual_kernel_size,
        temporal_dilations=temporal_dilations,
        coordinate_conditioning=coordinate_conditioning,
        neighbor_gating=neighbor_gating,
        neighbor_alignment_kernel_size=neighbor_alignment_kernel_size,
        prediction_reference=prediction_reference,
        neighbor_geometry=neighbor_geometry,
        relative_receiver_x_radius=relative_receiver_x_radius,
        source_x_line_radius=source_x_line_radius,
        source_y_half_shot_radius=source_y_half_shot_radius,
        relative_receiver_y_radius=relative_receiver_y_radius,
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
            get_required_config_value(config, "training.batch_size"), "training.batch_size"
        ),
        target_sampling=_validated_target_sampling(config),
        exclude_target_ffid_neighbors=exclude_target_ffid_neighbors,
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
        success_threshold_db=success_threshold_db,
        duplicate_physical_coordinate_policy=(
            trace_canonicalization.DUPLICATE_PHYSICAL_COORDINATE_POLICY
        ),
        required_eligible_ffid_count=required_eligible_ffid_count,
        required_sample_count=required_sample_count,
        required_effective_split_counts=required_effective_split_counts,
        required_ffid_split_counts=required_ffid_split_counts,
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
        config_values.nonnegative_integer(neighborhood[name], f"model.neighborhood.{name}")
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


def _validated_model_name(config: Mapping[str, object]) -> str:
    value = get_required_config_value(config, "model.name")
    supported = {MODEL_NAME, SHARED_OFFSET_ATTENTION_MODEL_NAME}
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(f"model.name must be one of {sorted(supported)}, got {value!r}")
    return value


def _validated_shared_offset_attention_settings(
    config: Mapping[str, object],
    *,
    model_name: str,
) -> tuple[int, int, int, float]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("model configuration must be a mapping")
    if model_name != SHARED_OFFSET_ATTENTION_MODEL_NAME:
        shared_only_fields = {
            "neighbor_feature_width",
            "attention_width",
            "attention_geometry_prior_scale",
        }
        unexpected = sorted(shared_only_fields.intersection(model))
        if unexpected:
            raise ConfigurationError(
                f"shared offset attention fields require "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}: {unexpected}"
            )
        return (
            DEFAULT_NEIGHBOR_FEATURE_WIDTH,
            DEFAULT_ATTENTION_WIDTH,
            config_values.nonnegative_integer(
                model.get(
                    "coarse_shift_samples_per_relative_receiver_y_index",
                    DEFAULT_LEGACY_COARSE_SHIFT,
                ),
                "model.coarse_shift_samples_per_relative_receiver_y_index",
            ),
            DEFAULT_ATTENTION_GEOMETRY_PRIOR_SCALE,
        )
    return (
        config_values.positive_integer(
            model.get("neighbor_feature_width", DEFAULT_NEIGHBOR_FEATURE_WIDTH),
            "model.neighbor_feature_width",
        ),
        config_values.positive_integer(
            model.get("attention_width", DEFAULT_ATTENTION_WIDTH),
            "model.attention_width",
        ),
        config_values.nonnegative_integer(
            model.get(
                "coarse_shift_samples_per_relative_receiver_y_index",
                DEFAULT_SHARED_COARSE_SHIFT,
            ),
            "model.coarse_shift_samples_per_relative_receiver_y_index",
        ),
        config_values.nonnegative_float(
            model.get(
                "attention_geometry_prior_scale",
                DEFAULT_ATTENTION_GEOMETRY_PRIOR_SCALE,
            ),
            "model.attention_geometry_prior_scale",
        ),
    )


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
    supported = {
        DEFAULT_NEIGHBOR_GATING,
        TARGET_COORDINATE_MASKED_SOFTMAX_GATING,
        OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING,
    }
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(
            f"model.neighbor_gating must be one of {sorted(supported)}, got {value!r}"
        )
    return value


def _validated_neighbor_alignment_kernel_size(config: Mapping[str, object]) -> int:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("model configuration must be a mapping")
    return config_values.odd_positive_integer(
        model.get(
            "neighbor_alignment_kernel_size",
            DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE,
        ),
        "model.neighbor_alignment_kernel_size",
    )


def _validated_prediction_reference(
    config: Mapping[str, object],
    *,
    model_name: str,
) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("model configuration must be a mapping")
    if model_name == SHARED_OFFSET_ATTENTION_MODEL_NAME:
        value = model.get(
            "prediction_reference",
            DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE,
        )
        if value != DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE:
            raise ConfigurationError(
                f"model.prediction_reference must be "
                f"{DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE!r} for "
                f"model.name={SHARED_OFFSET_ATTENTION_MODEL_NAME!r}"
            )
        return DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE
    value = model.get("prediction_reference", DEFAULT_PREDICTION_REFERENCE)
    supported = {
        DEFAULT_PREDICTION_REFERENCE,
        MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    }
    if not isinstance(value, str) or value not in supported:
        raise ConfigurationError(
            f"model.prediction_reference must be one of {sorted(supported)}, got {value!r}"
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


def _build_inpainter_model(
    settings: _TrainingSettings,
    geometry: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
) -> NeighborTraceInpainter | SharedOffsetAttentionInpainter:
    if settings.model_name == MODEL_NAME:
        reference_neighbor_count = {
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE: 1,
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE: 2,
        }.get(settings.prediction_reference, 0)
        return NeighborTraceInpainter(
            neighbor_count=geometry.neighbor_count + reference_neighbor_count,
            width=settings.hidden_width,
            target_coordinate_count=len(settings.target_coordinates),
            stem_kernel_size=settings.stem_kernel_size,
            residual_kernel_size=settings.residual_kernel_size,
            temporal_dilations=settings.temporal_dilations,
            coordinate_conditioning=settings.coordinate_conditioning,
            neighbor_gating=settings.neighbor_gating,
            neighbor_alignment_kernel_size=settings.neighbor_alignment_kernel_size,
            prediction_reference=settings.prediction_reference,
            coarse_shift_samples_per_relative_receiver_y_index=(
                settings.coarse_shift_samples_per_relative_receiver_y_index
            ),
            neighbor_offsets=(
                geometry.offsets
                if settings.coarse_shift_samples_per_relative_receiver_y_index > 0
                else None
            ),
        )
    if settings.model_name != SHARED_OFFSET_ATTENTION_MODEL_NAME:
        raise AssertionError(f"unsupported validated model: {settings.model_name}")
    offsets = geometry.offsets
    if any(len(offset) != len(OFFSET_ORDER_AXES) for offset in offsets):
        raise AssertionError("shared offset attention requires four-axis multiline offsets")
    return SharedOffsetAttentionInpainter(
        neighbor_offsets=offsets,
        width=settings.hidden_width,
        neighbor_feature_width=settings.neighbor_feature_width,
        attention_width=settings.attention_width,
        target_coordinate_count=len(settings.target_coordinates),
        stem_kernel_size=settings.stem_kernel_size,
        residual_kernel_size=settings.residual_kernel_size,
        temporal_dilations=settings.temporal_dilations,
        coarse_shift_samples_per_relative_receiver_y_index=(
            settings.coarse_shift_samples_per_relative_receiver_y_index
        ),
        attention_geometry_prior_scale=settings.attention_geometry_prior_scale,
    )


def _neighbor_positions(
    geometry: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
    positions: np.ndarray,
    *,
    exclude_target_ffid_neighbors: bool,
    ffids_by_position: np.ndarray | None,
) -> np.ndarray:
    neighbors = geometry.neighbor_positions(positions)
    if exclude_target_ffid_neighbors:
        if ffids_by_position is None:
            raise AssertionError("validated target-FFID masking is missing FFID values")
        available = neighbors >= 0
        safe_neighbors = np.maximum(neighbors, 0)
        same_ffid = available & (
            ffids_by_position[safe_neighbors] == ffids_by_position[positions, None]
        )
        neighbors[same_ffid] = -1
    return neighbors


def _neighbor_availability(
    geometry: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
    positions: np.ndarray,
    *,
    exclude_target_ffid_neighbors: bool,
    ffids_by_position: np.ndarray,
) -> dict[str, object]:
    counts = np.empty(len(positions), dtype=np.int16)
    target_ffid_neighbor_entries = 0
    for start in range(0, len(positions), _AVAILABILITY_CHUNK_SIZE):
        stop = min(start + _AVAILABILITY_CHUNK_SIZE, len(positions))
        batch_positions = positions[start:stop]
        batch_neighbors = _neighbor_positions(
            geometry,
            batch_positions,
            exclude_target_ffid_neighbors=exclude_target_ffid_neighbors,
            ffids_by_position=ffids_by_position,
        )
        available = batch_neighbors >= 0
        counts[start:stop] = np.count_nonzero(available, axis=1)
        safe_neighbors = np.maximum(batch_neighbors, 0)
        target_ffid_neighbor_entries += int(
            np.count_nonzero(
                available
                & (ffids_by_position[safe_neighbors] == ffids_by_position[batch_positions, None])
            )
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
        "target_ffid_neighbor_entries": target_ffid_neighbor_entries,
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
        "target_ffid_neighbor_policy": (
            "exclude_exact_ffid"
            if settings.exclude_target_ffid_neighbors
            else "allow_same_ffid_except_target_center"
        ),
    }


def _source_bracketing_contract(
    lookup: SameLineReceiverBracketingLookup,
    *,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    prediction_reference: str,
) -> dict[str, object]:
    if prediction_reference not in _SOURCE_BRACKETING_REFERENCE_MODES:
        raise ValueError("prediction_reference must select a source bracketing mode")
    reference_contract: dict[str, object]
    if prediction_reference == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE:
        reference_contract = {"reference_channel": "last"}
    else:
        reference_contract = {
            "reference_channels": "last_two",
            "reference_channel_order": [
                "strict_lower_source_y",
                "strict_upper_source_y",
            ],
            "reference_trace_values": "raw_train_amplitudes",
            "availability_values": "linear_interpolation_weights",
            "prediction_reference_combination": "weighted_sum",
        }
    return {
        "enabled": True,
        "type": prediction_reference,
        "key": [
            "source_x_m",
            "relative_receiver_x_m",
            "relative_receiver_y_m",
        ],
        "source_y_selection": "strict_nearest_lower_and_upper",
        "two_sided_rule": "linear_source_y_distance",
        "one_sided_rule": "nearest",
        "source_split": TRAIN_SPLIT,
        "target_ffid_policy": "exclude_exact_ffid",
        **reference_contract,
        "neighbor_dropout_applied": False,
        TRAIN_SPLIT: lookup.audit(train_positions),
        VALIDATION_SPLIT: lookup.audit(validation_positions),
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
        "exclude_target_ffid_neighbors": settings.exclude_target_ffid_neighbors,
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


def _model_contract(
    model: NeighborTraceInpainter | SharedOffsetAttentionInpainter,
    *,
    settings: _TrainingSettings,
    geometry: NeighborGeometryLookup | MultilineNeighborGeometryLookup,
) -> dict[str, object]:
    common = {
        "name": settings.model_name,
        "hidden_width": settings.hidden_width,
        "neighbor_count": model.neighbor_count,
        "input_channels": model.input_channels,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temporal_dilations": list(model.temporal_dilations),
        "stem_kernel_size": model.stem.kernel_size[0],
        "residual_kernel_size": model.blocks[0].kernel_size,
        "target_coordinates": list(settings.target_coordinates),
        "target_coordinate_scaling": TARGET_COORDINATE_SCALING,
        "coordinate_conditioning": model.coordinate_conditioning,
        "neighbor_gating": model.neighbor_gating,
        "prediction_reference": model.prediction_reference,
    }
    if isinstance(model, NeighborTraceInpainter):
        contract = {
            **common,
            "neighbor_alignment_kernel_size": model.neighbor_alignment_kernel_size,
            "neighbor_alignment": _neighbor_alignment_contract(model),
        }
        if settings.prediction_reference == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE:
            contract.update(
                {
                    "local_neighbor_count": geometry.neighbor_count,
                    "reference_neighbor_count": 1,
                    "reference_channel_index": model.neighbor_count - 1,
                    "reference_source": "raw_last_neighbor_channel",
                    "residual_decoder_initialization": "zero_final_projection",
                }
            )
        elif (
            settings.prediction_reference
            == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
        ):
            contract.update(
                {
                    "local_neighbor_count": geometry.neighbor_count,
                    "reference_neighbor_count": 2,
                    "reference_channel_indices": [
                        model.neighbor_count - 2,
                        model.neighbor_count - 1,
                    ],
                    "reference_channel_order": [
                        "strict_lower_source_y",
                        "strict_upper_source_y",
                    ],
                    "reference_source": "raw_last_two_neighbor_channels",
                    "reference_weight_source": "last_two_availability_channels",
                    "reference_combination": "weighted_sum",
                    "residual_decoder_initialization": "zero_final_projection",
                }
            )
        if model.coarse_shift_samples_per_relative_receiver_y_index == 0:
            return contract
        if model.neighbor_offsets is None or model.coarse_sample_shifts is None:
            raise RuntimeError("coarse-aligned checkpoint model is missing derived buffers")
        exact_offsets = tuple(
            tuple(int(component) for component in offset)
            for offset in model.neighbor_offsets.detach().cpu().tolist()
        )
        if exact_offsets != geometry.offsets:
            raise RuntimeError("checkpoint model offsets do not match pipeline geometry order")
        contract["coarse_alignment"] = {
            "type": "zero_padded_integer_shift",
            "offset_order": [list(offset) for offset in exact_offsets],
            "offset_order_axes": list(OFFSET_ORDER_AXES),
            "offset_order_source": "pipeline_geometry_exact",
            "samples_per_relative_receiver_y_index": (
                model.coarse_shift_samples_per_relative_receiver_y_index
            ),
            "sample_shifts": [
                int(value) for value in model.coarse_sample_shifts.detach().cpu().tolist()
            ],
            "source_sample_index": "output_sample_index_minus_shift",
            "circular_wrap": False,
            "valid_sample_availability_channels": "time_dependent",
            "applied_before_target_gate_fir_and_stem": True,
        }
        return contract
    if not isinstance(model, SharedOffsetAttentionInpainter):
        raise TypeError(f"unsupported neighbor inpainter model: {type(model).__name__}")
    exact_offsets = tuple(
        tuple(int(component) for component in offset)
        for offset in model.neighbor_offsets.detach().cpu().tolist()
    )
    if exact_offsets != geometry.offsets:
        raise RuntimeError("checkpoint model offsets do not match pipeline geometry order")
    return {
        **common,
        "neighbor_feature_width": model.neighbor_feature_width,
        "attention_width": model.attention_width,
        "offset_order": [list(offset) for offset in exact_offsets],
        "offset_order_axes": list(OFFSET_ORDER_AXES),
        "offset_order_source": "pipeline_geometry_exact",
        "coarse_alignment": {
            "type": "zero_padded_integer_shift",
            "samples_per_relative_receiver_y_index": (
                model.coarse_shift_samples_per_relative_receiver_y_index
            ),
            "source_sample_index": "output_sample_index_minus_shift",
            "circular_wrap": False,
        },
        "attention": {
            "type": "offset_target_time_content_masked_softmax",
            "complexity": "O(B*K*T)",
            "geometry_prior": "-(drx^2+16*dsx^2+dsy^2+dry^2)",
            "geometry_prior_scale": model.attention_geometry_prior_scale,
            "unavailable_weight": 0.0,
            "all_unavailable_weights": "finite_zero",
        },
        "shared_neighbor_encoder": {
            "shared_across_offsets": True,
            "feature_width": model.neighbor_feature_width,
            "offset_conditioning": "film",
        },
        "residual_decoder_initialization": "zero_final_projection",
    }


def _neighbor_alignment_contract(model: NeighborTraceInpainter) -> dict[str, object]:
    alignment = model.neighbor_alignment
    contract = {
        "enabled": alignment is not None,
        "type": "depthwise_fir" if alignment is not None else "none",
        "kernel_size": model.neighbor_alignment_kernel_size,
        "groups": alignment.groups if alignment is not None else None,
        "bias": alignment.bias is not None if alignment is not None else None,
        "initialization": "identity_center_tap" if alignment is not None else None,
        "applied_after_time_invariant_neighbor_gating": (
            model.coarse_shift_samples_per_relative_receiver_y_index == 0
        ),
        "unavailable_channels_zeroed_before_fir": alignment is not None,
    }
    if model.coarse_shift_samples_per_relative_receiver_y_index > 0:
        contract["applied_after_time_dependent_target_gating"] = True
        contract["coarse_alignment_applied_before_fir"] = True
    return contract


def _target_sampling_seed(settings: _TrainingSettings) -> int:
    if settings.target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
        return settings.random_seed + EPOCH_TARGET_SAMPLING_SEED_OFFSET
    return settings.random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET


def _validated_exclude_target_ffid_neighbors(config: Mapping[str, object]) -> bool:
    training = config.get("training")
    if not isinstance(training, Mapping) or "exclude_target_ffid_neighbors" not in training:
        return False
    value = training["exclude_target_ffid_neighbors"]
    if not isinstance(value, bool):
        raise ConfigurationError("training.exclude_target_ffid_neighbors must be a boolean")
    return value


def _metrics_payload(
    result: NeighborInpainterTrainingResult,
    *,
    best_validation: _EvaluationResult,
    training_audit: _EvaluationResult,
    settings: _TrainingSettings,
    selection_contract: Mapping[str, object],
    collision_audit: Mapping[str, object],
    availability_contract: Mapping[str, object],
    source_bracketing_contract: Mapping[str, object] | None,
    duplicate_audit: Mapping[str, object],
    scope_audit: Mapping[str, object],
    checkpoint_contract: Mapping[str, object],
    amplitude_access: Mapping[str, object],
) -> dict[str, object]:
    metrics = asdict(result)
    metrics["history"] = [dict(value) for value in result.history]
    accepted_metric = best_validation.raw_global_snr_db
    metric_success = oracle_trace_snr.passes_success_threshold(
        accepted_metric, settings.success_threshold_db
    )
    scope_success = bool(scope_audit["scope_success"])
    metrics.update(
        {
            "amplitude_scaling": PER_TRACE_RMS_SCALING,
            "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
            "validation_scale_source": VALIDATION_SCALE_SOURCE,
            "primary_metric_prediction": "raw_model_output",
            "primary_metric_prediction_self_normalized": False,
            "prediction_reference": settings.prediction_reference,
            "best_validation_raw_global_snr_db": best_validation.raw_global_snr_db,
            "best_validation_signal_energy": best_validation.signal_energy,
            "best_validation_error_energy": best_validation.error_energy,
            "best_validation_error_mean_square": best_validation.error_mean_square,
            "best_validation_predicted_unit_rms_global_snr_db": (
                best_validation.predicted_unit_rms_global_snr_db
            ),
            "best_validation_global_snr_db": accepted_metric,
            oracle_trace_snr.PRIMARY_METRIC: accepted_metric,
            "success_threshold_db": settings.success_threshold_db,
            "success_comparison": oracle_trace_snr.SUCCESS_COMPARISON,
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
    if source_bracketing_contract is not None:
        metrics["source_bracketing"] = dict(source_bracketing_contract)
    return metrics


def _seed_global_model_initialization(seed: int, *, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
