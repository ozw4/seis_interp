from __future__ import annotations

import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.pipelines import batching_ablation as pipeline


@pytest.mark.parametrize(
    ("control", "full_trace_batch", "reproduction", "expected"),
    [
        ("strong_fit", "near_zero", "strong_fit", "small_batch_control_not_reproduced"),
        (
            "escaped_zero_predictor",
            "strong_fit",
            "strong_fit",
            "small_batch_control_not_reproduced",
        ),
        ("near_zero", "strong_fit", "near_zero", "combined_escape_not_reproduced"),
        ("near_zero", "strong_fit", "strong_fit", "full_trace_batch_strong_fit"),
        (
            "near_zero",
            "escaped_zero_predictor",
            "escaped_zero_predictor",
            "full_trace_batch_escaped_zero_predictor",
        ),
        ("near_zero", "near_zero", "escaped_zero_predictor", "full_trace_batch_near_zero"),
    ],
)
def test_full_trace_batch_decision_applies_both_gates(
    control: str,
    full_trace_batch: str,
    reproduction: str,
    expected: str,
) -> None:
    assert (
        pipeline.full_trace_batch_ablation_summary_decision(
            control_classification=control,
            full_trace_batch_classification=full_trace_batch,
            reproduction_classification=reproduction,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("control", "full_trace_batch", "reproduction"),
    [
        ("unknown", "near_zero", "near_zero"),
        ("near_zero", "unknown", "near_zero"),
        ("near_zero", "near_zero", "unknown"),
    ],
)
def test_full_trace_batch_decision_rejects_unknown_classifications(
    control: str,
    full_trace_batch: str,
    reproduction: str,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        pipeline.full_trace_batch_ablation_summary_decision(
            control_classification=control,
            full_trace_batch_classification=full_trace_batch,
            reproduction_classification=reproduction,
        )


def _canonical_conditions() -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "batch_mode": batch_mode,
            "correlation": correlation,
            "amplitude_scaling": amplitude_scaling,
        }
        for label, batch_mode, correlation, amplitude_scaling in (
            pipeline._FULL_TRACE_BATCH_ABLATION_CONDITIONS
        )
    ]


def test_full_trace_batch_conditions_are_exact_and_canonicalized() -> None:
    reverse_order = list(reversed(_canonical_conditions()))

    assert (
        pipeline._validated_full_trace_batch_conditions(reverse_order)
        == pipeline._FULL_TRACE_BATCH_ABLATION_CONDITIONS
    )


def _drifted_condition_sets() -> list[list[dict[str, object]]]:
    missing = _canonical_conditions()[:-1]
    duplicated = _canonical_conditions()
    duplicated[1] = dict(duplicated[0])
    wrong_flag = _canonical_conditions()
    wrong_flag[1]["correlation"] = True
    extra_key = _canonical_conditions()
    extra_key[0]["extra"] = True
    return [missing, duplicated, wrong_flag, extra_key]


@pytest.mark.parametrize("conditions", _drifted_condition_sets())
def test_full_trace_batch_conditions_reject_contract_drift(
    conditions: list[dict[str, object]],
) -> None:
    with pytest.raises(ConfigurationError):
        pipeline._validated_full_trace_batch_conditions(conditions)
