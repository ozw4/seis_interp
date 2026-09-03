from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from seis_interp.configuration import ConfigurationError
from seis_interp.data.whole_shot import WholeShotTargets, WholeShotTensorSource
from seis_interp.evaluation.whole_shot import (
    TRAINING_SCALE_SOURCE,
    VALIDATION_SCALE_SOURCE,
    WholeShotEvaluationResult,
    WholeShotGlobalSnrEvaluator,
    build_whole_shot_metrics,
    sample_training_audit_targets,
    source_collision_audit,
)
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT

DEVICE = torch.device("cpu")
TIME_SAMPLES = 4
TARGET_VALUE = 2.0
PREDICTION_VALUE = 1.5


class _ConstantModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        source_deltas: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = neighbors.shape[0]
        return torch.full(
            (batch_size, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, neighbors.shape[-1]),
            self.value,
        )


def _tensor_source() -> WholeShotTensorSource:
    gathers = torch.zeros(4, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES)
    for index in range(4):
        gathers[index] = float(index + 1)
    return WholeShotTensorSource(
        train_ffids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        train_source_coordinates_m=np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        train_gathers=gathers,
        train_availability=torch.ones(4, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
        source_gather_count=2,
        device=DEVICE,
    )


def _validation_targets(source: WholeShotTensorSource) -> WholeShotTargets:
    availability = torch.zeros(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    availability[0, 0, 0] = True
    availability[0, 3, 40] = True
    availability[0, 7, 67] = True
    availability[1, 1, 5] = True
    availability[1, 6, 30] = True
    gathers = torch.full((2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES), 100.0)
    gathers[availability] = TARGET_VALUE
    return source.build_targets(
        ffids=np.asarray([20, 21], dtype=np.int64),
        source_coordinates_m=np.asarray([[1.0, 0.0], [29.0, 0.0]]),
        gathers=gathers,
        availability=availability,
    )


def _evaluate(batch_size: int = 2, use_bfloat16: bool = False) -> WholeShotEvaluationResult:
    source = _tensor_source()
    evaluator = WholeShotGlobalSnrEvaluator(
        source,
        _validation_targets(source),
        batch_size=batch_size,
        use_bfloat16=use_bfloat16,
    )
    return evaluator.evaluate(_ConstantModel(PREDICTION_VALUE))


def _audit_targets() -> WholeShotTargets:
    availability = torch.zeros(3, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    availability[0, :, :10] = True
    availability[1, 2, :] = True
    availability[2, :, ::2] = True
    gathers = torch.stack(
        [
            torch.full((RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES), float(index + 1))
            for index in range(3)
        ]
    )
    return WholeShotTargets(
        ffids=np.asarray([5, 9, 11], dtype=np.int64),
        source_coordinates_m=np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        gathers=gathers,
        availability=availability,
        neighbor_train_indices=np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64),
    )


def _metrics_arguments() -> dict[str, object]:
    best_validation = WholeShotEvaluationResult(
        raw_global_snr_db=12.0,
        predicted_unit_rms_global_snr_db=6.0,
        signal_energy=80.0,
        error_energy=5.0,
        error_mean_square=0.25,
        ffid_count=2,
        trace_count=5,
    )
    training_audit = WholeShotEvaluationResult(
        raw_global_snr_db=14.0,
        predicted_unit_rms_global_snr_db=7.0,
        signal_energy=40.0,
        error_energy=1.6,
        error_mean_square=0.1,
        ffid_count=3,
        trace_count=4,
    )
    return {
        "best_validation": best_validation,
        "training_audit": training_audit,
        "success_threshold_db": 12.0,
        "selection_contract": {
            "split_counts": {"train": 4, "validation": 2, "test": 1},
            "sample_count": TIME_SAMPLES,
            "effective_eligible_trace_count": 7,
        },
        "duplicate_audit": {"remaining_duplicate_physical_cell_count": 0},
        "collision_audit": {
            "canonical_remaining_duplicate_physical_cells": 0,
            "train_duplicate_source_coordinates": 0,
            "train_validation_source_coordinate_overlap": 0,
        },
        "availability_contract": {"train": {"target_ffid_count": 4}},
        "amplitude_access": {"full_file_bytes_hashed": True},
        "scope_audit": {"checks": {"sample_count_matches": True}, "scope_success": True},
        "checkpoint_contract": {"best_step": 1},
    }


def test_scale_source_constants_are_fixed() -> None:
    assert TRAINING_SCALE_SOURCE == "training_trace_target_rms"
    assert VALIDATION_SCALE_SOURCE == "validation_trace_target_rms"


def test_evaluator_excludes_unavailable_cells_and_reports_known_energies() -> None:
    result = _evaluate()

    trace_count = 5
    assert result.trace_count == trace_count
    assert result.ffid_count == 2
    assert result.signal_energy == trace_count * TIME_SAMPLES * TARGET_VALUE**2
    assert result.error_energy == trace_count * TIME_SAMPLES * 0.25
    assert result.error_mean_square == 0.25
    assert result.raw_global_snr_db == pytest.approx(10.0 * math.log10(16.0))


def test_evaluator_reports_predicted_unit_rms_metric() -> None:
    result = _evaluate()

    assert result.predicted_unit_rms_global_snr_db == pytest.approx(10.0 * math.log10(4.0))


def test_evaluator_is_independent_of_batch_partition() -> None:
    assert _evaluate(batch_size=1) == _evaluate(batch_size=64)


def test_bfloat16_option_is_disabled_on_cpu() -> None:
    source = _tensor_source()
    evaluator = WholeShotGlobalSnrEvaluator(
        source,
        _validation_targets(source),
        batch_size=2,
        use_bfloat16=True,
    )

    assert evaluator.use_bfloat16 is False
    assert evaluator.evaluate(_ConstantModel(PREDICTION_VALUE)) == _evaluate(use_bfloat16=False)


def test_training_audit_selects_exactly_requested_traces() -> None:
    targets = _audit_targets()

    audit = sample_training_audit_targets(targets, trace_count=6, random_seed=77)

    assert int(torch.count_nonzero(audit.availability)) == 6
    for position, ffid in enumerate(audit.ffids):
        original = int(np.flatnonzero(targets.ffids == ffid)[0])
        outside = audit.availability[position] & ~targets.availability[original]
        assert not bool(outside.any())


def test_training_audit_is_deterministic_for_a_seed() -> None:
    first = sample_training_audit_targets(_audit_targets(), trace_count=6, random_seed=77)
    second = sample_training_audit_targets(_audit_targets(), trace_count=6, random_seed=77)

    np.testing.assert_array_equal(first.ffids, second.ffids)
    assert torch.equal(first.availability, second.availability)


def test_training_audit_keeps_ffid_gather_and_neighbor_alignment() -> None:
    targets = _audit_targets()

    audit = sample_training_audit_targets(targets, trace_count=8, random_seed=3)

    assert np.all(np.diff(np.searchsorted(targets.ffids, audit.ffids)) > 0)
    for position, ffid in enumerate(audit.ffids):
        original = int(np.flatnonzero(targets.ffids == ffid)[0])
        assert torch.equal(audit.gathers[position], targets.gathers[original])
        np.testing.assert_array_equal(
            audit.neighbor_train_indices[position],
            targets.neighbor_train_indices[original],
        )
        np.testing.assert_array_equal(
            audit.source_coordinates_m[position],
            targets.source_coordinates_m[original],
        )


def test_training_audit_rejects_excessive_trace_count() -> None:
    targets = _audit_targets()
    available = int(torch.count_nonzero(targets.availability))

    with pytest.raises(ConfigurationError, match="must not exceed selected training trace count"):
        sample_training_audit_targets(targets, trace_count=available + 1, random_seed=1)


def test_source_collision_audit_reports_three_exact_keys() -> None:
    audit = source_collision_audit(
        np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        duplicate_audit={"remaining_duplicate_physical_cell_count": 5},
    )

    assert audit == {
        "canonical_remaining_duplicate_physical_cells": 5,
        "train_duplicate_source_coordinates": 1,
        "train_validation_source_coordinate_overlap": 1,
    }


def test_metrics_builder_adds_exact_shared_keys_with_strict_threshold() -> None:
    base_metrics = {"total_steps": 2, "best_step": 1, "history": [{"step": 0, "loss": 1.0}]}
    arguments = _metrics_arguments()

    metrics = build_whole_shot_metrics(base_metrics, **arguments)

    assert set(metrics) == set(base_metrics) | {
        "amplitude_scaling",
        "validation_metric_domain",
        "validation_scale_source",
        "primary_metric_prediction",
        "primary_metric_prediction_self_normalized",
        "best_validation_raw_global_snr_db",
        "best_validation_signal_energy",
        "best_validation_error_energy",
        "best_validation_error_mean_square",
        "best_validation_predicted_unit_rms_global_snr_db",
        "best_validation_global_snr_db",
        "oracle_per_trace_unit_rms_global_snr_db",
        "success_threshold_db",
        "success_comparison",
        "metric_success",
        "scope_success",
        "success",
        "validation_ffid_count",
        "validation_trace_count",
        "training_audit_trace_count",
        "training_audit_global_snr_db",
        "training_audit_predicted_unit_rms_global_snr_db",
        "split_counts",
        "sample_count",
        "effective_eligible_trace_count",
        "duplicate_physical_coordinates",
        "collision_audit",
        "amplitude_access",
        "neighbor_availability",
        "formal_success_scope",
        "checkpoint",
    }
    assert metrics["amplitude_scaling"] == "per_trace_rms"
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert metrics["validation_scale_source"] == "validation_trace_target_rms"
    assert metrics["oracle_per_trace_unit_rms_global_snr_db"] == 12.0
    assert metrics["best_validation_global_snr_db"] == 12.0
    assert metrics["success_comparison"] == "strictly_greater_than"
    assert metrics["metric_success"] is False, "12.0 dB must not pass the strict 12.0 dB threshold"
    assert metrics["success"] is False
    assert metrics["training_audit_trace_count"] == 4

    passing = build_whole_shot_metrics(
        base_metrics,
        **{**arguments, "success_threshold_db": 11.9},
    )
    assert passing["metric_success"] is True
    assert passing["success"] is True


def test_metrics_builder_applies_scope_success_and() -> None:
    arguments = _metrics_arguments()
    arguments["scope_audit"] = {
        "checks": {"sample_count_matches": False},
        "scope_success": False,
    }
    arguments["success_threshold_db"] = 11.0

    metrics = build_whole_shot_metrics({"history": []}, **arguments)

    assert metrics["metric_success"] is True
    assert metrics["scope_success"] is False
    assert metrics["success"] is False


def test_metrics_builder_does_not_mutate_inputs() -> None:
    base_metrics = {"total_steps": 2, "history": [{"step": 0, "loss": 1.0}]}
    arguments = _metrics_arguments()
    base_snapshot = deepcopy(base_metrics)
    arguments_snapshot = deepcopy(arguments)

    build_whole_shot_metrics(base_metrics, **arguments)

    assert base_metrics == base_snapshot
    assert arguments == arguments_snapshot


def test_metrics_builder_preserves_caller_history() -> None:
    history = [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.5}]

    metrics = build_whole_shot_metrics({"history": history}, **_metrics_arguments())

    assert metrics["history"] == [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.5}]
