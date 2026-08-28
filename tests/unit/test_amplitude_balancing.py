from __future__ import annotations

import numpy as np
import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.pipelines import batching_ablation as pipeline


@pytest.mark.parametrize(
    ("control", "per_trace", "expected"),
    [
        ("strong_fit", "strong_fit", "global_rms_control_not_reproduced"),
        ("escaped_zero_predictor", "near_zero", "global_rms_control_not_reproduced"),
        ("near_zero", "strong_fit", "per_trace_rms_strong_fit"),
        ("near_zero", "escaped_zero_predictor", "per_trace_rms_escaped_zero_predictor"),
        ("near_zero", "near_zero", "per_trace_rms_near_zero"),
    ],
)
def test_amplitude_balancing_decision_applies_the_control_gate(
    control: str,
    per_trace: str,
    expected: str,
) -> None:
    assert (
        pipeline.amplitude_balancing_summary_decision(
            control_classification=control,
            per_trace_classification=per_trace,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("control", "per_trace"),
    [
        ("unknown", "near_zero"),
        ("near_zero", "unknown"),
    ],
)
def test_amplitude_balancing_decision_rejects_unknown_classifications(
    control: str,
    per_trace: str,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        pipeline.amplitude_balancing_summary_decision(
            control_classification=control,
            per_trace_classification=per_trace,
        )


def test_amplitude_balancing_conditions_are_exact_and_canonicalized() -> None:
    reverse_order = [
        {"label": "huber_global_rms", "amplitude_scaling": "global_rms", "loss": "huber"},
        {"label": "per_trace_rms", "amplitude_scaling": "per_trace_rms", "loss": "l2"},
        {"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"},
    ]

    assert pipeline._validated_amplitude_balancing_conditions(reverse_order) == (
        ("global_rms_control", "global_rms", "l2"),
        ("per_trace_rms", "per_trace_rms", "l2"),
        ("huber_global_rms", "global_rms", "huber"),
    )


@pytest.mark.parametrize(
    "conditions",
    [
        [{"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"}],
        [
            {"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"},
            {"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"},
            {"label": "huber_global_rms", "amplitude_scaling": "global_rms", "loss": "huber"},
        ],
        [
            {"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"},
            {"label": "per_trace_rms", "amplitude_scaling": "global_rms", "loss": "l2"},
            {"label": "huber_global_rms", "amplitude_scaling": "global_rms", "loss": "huber"},
        ],
        [
            {"label": "global_rms_control", "amplitude_scaling": "global_rms", "loss": "l2"},
            {"label": "per_trace_rms", "amplitude_scaling": "per_trace_rms", "loss": "l2"},
            {
                "label": "huber_global_rms",
                "amplitude_scaling": "global_rms",
                "loss": "huber",
                "extra": True,
            },
        ],
    ],
)
def test_amplitude_balancing_conditions_reject_contract_drift(
    conditions: list[dict[str, object]],
) -> None:
    with pytest.raises(ConfigurationError):
        pipeline._validated_amplitude_balancing_conditions(conditions)


def test_per_trace_rms_scaling_normalizes_only_selected_rows() -> None:
    amplitudes = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.5, 0.5, 0.5],
            [7.0, 7.0, 7.0],
        ],
        dtype=np.float32,
    )
    selected = np.array([0, 1], dtype=np.int64)

    scaled, scales = pipeline.per_trace_rms_scaled_amplitudes(amplitudes, selected)

    expected_scales = np.sqrt(np.mean(np.square(amplitudes[:2].astype(np.float64)), axis=1))
    np.testing.assert_allclose(scales, expected_scales)
    scaled_rms = np.sqrt(np.mean(np.square(scaled[:2].astype(np.float64)), axis=1))
    np.testing.assert_allclose(scaled_rms, np.ones(2), rtol=1e-6)
    np.testing.assert_array_equal(scaled[2], amplitudes[2])
    assert scaled.dtype == amplitudes.dtype
    np.testing.assert_array_equal(amplitudes[0], np.array([3.0, 4.0, 0.0], dtype=np.float32))


def test_per_trace_rms_scaling_rejects_invalid_inputs() -> None:
    amplitudes = np.zeros((2, 3), dtype=np.float32)
    amplitudes[1] = 1.0

    with pytest.raises(ValueError, match="positive and finite"):
        pipeline.per_trace_rms_scaled_amplitudes(amplitudes, np.array([0]))
    with pytest.raises(ValueError, match="row range"):
        pipeline.per_trace_rms_scaled_amplitudes(amplitudes, np.array([2]))
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.per_trace_rms_scaled_amplitudes(amplitudes, np.array([], dtype=np.int64))


def _loss_validation_arguments() -> dict[str, object]:
    time = np.zeros(2, dtype=np.float64)
    spatial = np.zeros((2, 5), dtype=np.float64)
    amplitudes = np.ones((2, 2), dtype=np.float64)
    rows = np.array([0, 1], dtype=np.int64)
    return {
        "config": {},
        "label": "probe",
        "batch_mode": "random_replacement",
        "full_batch": False,
        "replacement": True,
        "total_updates": 1,
        "report_interval": 1,
        "batch_size": 1,
        "normalized_time": time,
        "normalized_spatial_by_array_row": spatial,
        "normalized_amplitudes": amplitudes,
        "selected_array_rows": rows,
        "all_coordinate_tensor": None,
        "all_target_tensor": None,
        "training_coordinates": np.zeros((4, 6), dtype=np.float64),
        "training_targets": np.ones(4, dtype=np.float64),
        "sample_count": 2,
        "prediction_batch_size": 1,
        "device": "cpu",
        "random_seed": 0,
    }


def test_training_fit_condition_rejects_invalid_loss_contracts() -> None:
    arguments = _loss_validation_arguments()

    with pytest.raises(ValueError, match="unknown loss name"):
        pipeline.run_training_fit_condition(**arguments, loss_name="l1")
    with pytest.raises(ValueError, match="huber_delta requires"):
        pipeline.run_training_fit_condition(**arguments, loss_name="l2", huber_delta=1.0)
    with pytest.raises(ValueError, match="huber_delta must be"):
        pipeline.run_training_fit_condition(**arguments, loss_name="huber")
    with pytest.raises(ValueError, match="does not support the trace correlation loss"):
        pipeline.run_training_fit_condition(
            **arguments,
            loss_name="huber",
            huber_delta=1.0,
            correlation_weight=0.1,
        )
