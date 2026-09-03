"""Whole-shot S/N evaluation, training audits, and shared metrics payload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from seis_interp.configuration import ConfigurationError
from seis_interp.data import whole_shot
from seis_interp.evaluation import oracle_trace_snr
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
)

TRAINING_SCALE_SOURCE = "training_trace_target_rms"
VALIDATION_SCALE_SOURCE = "validation_trace_target_rms"


@dataclass(frozen=True)
class WholeShotEvaluationResult:
    raw_global_snr_db: float
    predicted_unit_rms_global_snr_db: float
    signal_energy: float
    error_energy: float
    error_mean_square: float
    ffid_count: int
    trace_count: int


class WholeShotGlobalSnrEvaluator:
    """Evaluate available target traces with float64 energy accumulation."""

    def __init__(
        self,
        source: whole_shot.WholeShotTensorSource,
        targets: whole_shot.WholeShotTargets,
        *,
        batch_size: int,
        use_bfloat16: bool,
    ) -> None:
        self.source = source
        self.targets = targets
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16 and source.device.type == "cuda"

    def __call__(self, model: torch.nn.Module) -> float:
        return self.evaluate(model).raw_global_snr_db

    @torch.inference_mode()
    def evaluate(self, model: torch.nn.Module) -> WholeShotEvaluationResult:
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
        return WholeShotEvaluationResult(
            raw_global_snr_db=oracle_trace_snr.global_snr_db_from_energies(
                signal_energy, error_energy
            ),
            predicted_unit_rms_global_snr_db=oracle_trace_snr.global_snr_db_from_energies(
                signal_energy,
                predicted_unit_error_energy,
            ),
            signal_energy=signal_energy,
            error_energy=error_energy,
            error_mean_square=error_energy / point_count,
            ffid_count=self.targets.ffid_count,
            trace_count=trace_count,
        )


def sample_training_audit_targets(
    targets: whole_shot.WholeShotTargets,
    *,
    trace_count: int,
    random_seed: int,
) -> whole_shot.WholeShotTargets:
    """Select a fixed-seed subset of available training traces for auditing."""
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
    return whole_shot.WholeShotTargets(
        ffids=targets.ffids[ffid_indices],
        source_coordinates_m=targets.source_coordinates_m[ffid_indices],
        gathers=targets.gathers[gather_indices],
        availability=audit_mask,
        neighbor_train_indices=targets.neighbor_train_indices[ffid_indices],
    )


def source_collision_audit(
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


def build_whole_shot_metrics(
    base_training_metrics: Mapping[str, object],
    *,
    best_validation: WholeShotEvaluationResult,
    training_audit: WholeShotEvaluationResult,
    success_threshold_db: float,
    selection_contract: Mapping[str, object],
    duplicate_audit: Mapping[str, object],
    collision_audit: Mapping[str, object],
    availability_contract: Mapping[str, object],
    amplitude_access: Mapping[str, object],
    scope_audit: Mapping[str, object],
    checkpoint_contract: Mapping[str, object],
) -> dict[str, object]:
    """Extend caller-built training metrics with the shared whole-shot payload."""
    metrics = dict(base_training_metrics)
    accepted_metric = best_validation.raw_global_snr_db
    metric_success = oracle_trace_snr.passes_success_threshold(
        accepted_metric, success_threshold_db
    )
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
            oracle_trace_snr.PRIMARY_METRIC: accepted_metric,
            "success_threshold_db": success_threshold_db,
            "success_comparison": oracle_trace_snr.SUCCESS_COMPARISON,
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
