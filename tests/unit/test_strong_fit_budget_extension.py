from __future__ import annotations

import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.pipelines import batching_ablation as pipeline


@pytest.mark.parametrize(
    ("baseline_reproduced", "classification", "expected"),
    [
        (False, "strong_fit", "baseline_not_reproduced"),
        (False, "escaped_zero_predictor", "baseline_not_reproduced"),
        (False, "near_zero", "baseline_not_reproduced"),
        (True, "strong_fit", "extended_budget_strong_fit"),
        (True, "escaped_zero_predictor", "extended_budget_escaped_zero_predictor"),
        (True, "near_zero", "extended_budget_near_zero"),
    ],
)
def test_budget_extension_decision_applies_the_baseline_gate(
    baseline_reproduced: bool,
    classification: str,
    expected: str,
) -> None:
    assert (
        pipeline.strong_fit_budget_extension_summary_decision(
            baseline_reproduced=baseline_reproduced,
            extension_classification=classification,
        )
        == expected
    )


def test_budget_extension_decision_rejects_unknown_classifications() -> None:
    with pytest.raises(ValueError, match="unknown"):
        pipeline.strong_fit_budget_extension_summary_decision(
            baseline_reproduced=True,
            extension_classification="unknown",
        )


def test_budget_extension_decision_rejects_non_boolean_gate() -> None:
    with pytest.raises(ValueError, match="boolean"):
        pipeline.strong_fit_budget_extension_summary_decision(
            baseline_reproduced="yes",
            extension_classification="strong_fit",
        )


_HISTORY = [
    {"step": 500, "training_median_trace_snr_db": 3.0},
    {"step": 1000, "training_median_trace_snr_db": 16.5},
    {"step": 1500, "training_median_trace_snr_db": 12.0},
    {"step": 2000, "training_median_trace_snr_db": 21.0},
    {"step": 2500, "training_median_trace_snr_db": 20.5},
]


def test_best_median_trace_snr_within_uses_only_the_window() -> None:
    assert pipeline.best_median_trace_snr_within(_HISTORY, max_step=1500) == 16.5
    assert pipeline.best_median_trace_snr_within(_HISTORY, max_step=2500) == 21.0


def test_best_median_trace_snr_within_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="no history reports"):
        pipeline.best_median_trace_snr_within(_HISTORY, max_step=499)


def test_first_step_reaching_median_trace_snr_finds_the_first_crossing() -> None:
    assert pipeline.first_step_reaching_median_trace_snr(_HISTORY, threshold_db=20.0) == 2000
    assert pipeline.first_step_reaching_median_trace_snr(_HISTORY, threshold_db=16.5) == 1000
    assert pipeline.first_step_reaching_median_trace_snr(_HISTORY, threshold_db=25.0) is None


def _canonical_conditions() -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "batch_mode": batch_mode,
            "correlation": correlation,
            "amplitude_scaling": amplitude_scaling,
        }
        for label, batch_mode, correlation, amplitude_scaling in (
            pipeline._STRONG_FIT_BUDGET_EXTENSION_CONDITIONS
        )
    ]


def test_budget_extension_conditions_are_exact_and_canonicalized() -> None:
    assert (
        pipeline._validated_strong_fit_budget_extension_conditions(_canonical_conditions())
        == pipeline._STRONG_FIT_BUDGET_EXTENSION_CONDITIONS
    )


def _drifted_condition_sets() -> list[list[dict[str, object]]]:
    empty: list[dict[str, object]] = []
    wrong_label = _canonical_conditions()
    wrong_label[0]["label"] = "full_trace_batch"
    wrong_scaling = _canonical_conditions()
    wrong_scaling[0]["amplitude_scaling"] = "global_rms"
    with_correlation = _canonical_conditions()
    with_correlation[0]["correlation"] = True
    extra_condition = _canonical_conditions() + [
        {
            "label": "small_batch_control",
            "batch_mode": "random_replacement",
            "correlation": False,
            "amplitude_scaling": "global_rms",
        }
    ]
    extra_key = _canonical_conditions()
    extra_key[0]["extra"] = True
    return [empty, wrong_label, wrong_scaling, with_correlation, extra_condition, extra_key]


@pytest.mark.parametrize("conditions", _drifted_condition_sets())
def test_budget_extension_conditions_reject_contract_drift(
    conditions: list[dict[str, object]],
) -> None:
    with pytest.raises(ConfigurationError):
        pipeline._validated_strong_fit_budget_extension_conditions(conditions)
