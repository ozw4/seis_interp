from __future__ import annotations

import pytest

from seis_interp.pipelines.correlation_loss_ablation import _ablation_decision


def _run(
    label: str,
    *,
    snr: float,
    correlation: float,
    rms_ratio: float,
) -> dict[str, object]:
    return {
        "label": label,
        "best_training_median_trace_snr_db": snr,
        "best_training_median_trace_correlation": correlation,
        "best_training_prediction_target_rms_ratio": rms_ratio,
    }


@pytest.mark.parametrize(
    ("control", "correlation", "expected"),
    [
        ((1.01, 0.101, 0.101), (2.0, 0.2, 0.2), "full_batch_control_succeeds"),
        ((1.0, 0.2, 0.2), (1.01, 0.101, 0.101), "correlation_loss_promising"),
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.1, 0.101),
            "correlation_loss_inflates_amplitude_without_alignment",
        ),
        ((0.0, 0.0, 0.0), (1.0, 0.101, 0.101), "correlation_loss_not_effective"),
        ((0.0, 0.0, 0.0), (1.01, 0.1, 0.101), "correlation_loss_not_effective"),
        ((0.0, 0.0, 0.0), (1.01, 0.101, 0.1), "correlation_loss_not_effective"),
    ],
)
def test_ablation_decision_uses_strict_thresholds_and_control_priority(
    control: tuple[float, float, float],
    correlation: tuple[float, float, float],
    expected: str,
) -> None:
    runs = [
        _run(
            "mse_control",
            snr=control[0],
            correlation=control[1],
            rms_ratio=control[2],
        ),
        _run(
            "mse_corr_0p1",
            snr=correlation[0],
            correlation=correlation[1],
            rms_ratio=correlation[2],
        ),
    ]

    assert _ablation_decision(runs) == expected
