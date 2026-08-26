from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_interp.pipelines.batching_ablation import (
    _build_model,
    batching_summary_decision,
    classify_condition,
)
from seis_interp.pipelines.domain_scaling import deterministic_nested_trace_subsets
from seis_interp.training.point_sampler import RandomPointSampler, build_trace_points


def _point_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    spatial = np.arange(15, dtype=np.float64).reshape(3, 5)
    amplitudes = np.asarray(
        [[row * 100 + sample for sample in range(4)] for row in range(3)],
        dtype=np.float32,
    )
    return time, spatial, amplitudes


def _model_config() -> dict[str, object]:
    return {
        "model": {
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 2,
            "omega_0": 30.0,
        }
    }


def test_study_006_uses_the_study_004_and_005_nested_eight_row_contract() -> None:
    training_rows = np.array([2, 4, 7, 9, 11, 15, 18, 22, 25, 31, 35, 40])

    selected = deterministic_nested_trace_subsets(training_rows, (8,), random_seed=42)[8]

    np.testing.assert_array_equal(selected, [2, 22, 18, 31, 40, 9, 15, 7])


def test_same_seed_produces_identical_initial_model_outputs() -> None:
    coordinates = torch.linspace(-1.0, 1.0, 18).reshape(3, 6)

    outputs: list[torch.Tensor] = []
    for _ in range(2):
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        outputs.append(_build_model(_model_config())(coordinates).detach())

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)


def test_exact_full_batch_contains_every_point_once() -> None:
    time, spatial, amplitudes = _point_arrays()

    coordinates, targets = build_trace_points(time, spatial, amplitudes, np.array([2, 0]))

    assert coordinates.shape == (8, 6)
    assert len(np.unique(coordinates, axis=0)) == 8
    np.testing.assert_array_equal(targets, [200, 201, 202, 203, 0, 1, 2, 3])


def test_random_replacement_is_sized_deterministic_and_contains_duplicates() -> None:
    time, spatial, amplitudes = _point_arrays()
    first = RandomPointSampler(time, spatial, amplitudes, np.array([0, 2]), random_seed=7)
    second = RandomPointSampler(time, spatial, amplitudes, np.array([0, 2]), random_seed=7)

    first_coordinates, first_targets = first.sample(8)
    second_coordinates, second_targets = second.sample(8)

    assert first_coordinates.shape == (8, 6)
    assert first_targets.shape == (8,)
    np.testing.assert_array_equal(first_coordinates, second_coordinates)
    np.testing.assert_array_equal(first_targets, second_targets)
    assert len(np.unique(first_coordinates, axis=0)) < 8


@pytest.mark.parametrize(
    ("snr", "rms_ratio", "expected"),
    [
        (20.0, 0.0, "strong_fit"),
        (1.01, 0.101, "escaped_zero_predictor"),
        (1.0, 1.0, "near_zero"),
        (19.0, 0.1, "near_zero"),
    ],
)
def test_condition_classification_uses_fixed_thresholds(
    snr: float,
    rms_ratio: float,
    expected: str,
) -> None:
    metrics = {
        "best_training_median_trace_snr_db": snr,
        "best_training_prediction_target_rms_ratio": rms_ratio,
    }

    assert classify_condition(metrics) == expected


@pytest.mark.parametrize(
    ("exact", "random", "expected"),
    [
        ("strong_fit", "strong_fit", "random_replacement_succeeds"),
        (
            "strong_fit",
            "escaped_zero_predictor",
            "random_replacement_partially_succeeds",
        ),
        ("strong_fit", "near_zero", "exact_coverage_required"),
        ("escaped_zero_predictor", "strong_fit", "control_failed_unexpected"),
        ("near_zero", "near_zero", "control_failed_unexpected"),
    ],
)
def test_summary_decision_uses_exact_control_and_random_classifications(
    exact: str,
    random: str,
    expected: str,
) -> None:
    runs = [
        {"condition": "exact_full_batch", "classification": exact},
        {"condition": "random_replacement_5000", "classification": random},
    ]

    assert batching_summary_decision(runs) == expected
