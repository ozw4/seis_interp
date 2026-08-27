from __future__ import annotations

import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.pipelines import batching_ablation as pipeline


@pytest.mark.parametrize(
    ("legacy", "official", "expected"),
    [
        ("strong_fit", "strong_fit", "legacy_control_not_reproduced"),
        ("escaped_zero_predictor", "near_zero", "legacy_control_not_reproduced"),
        ("near_zero", "strong_fit", "official_siren_strong_fit"),
        (
            "near_zero",
            "escaped_zero_predictor",
            "official_siren_escaped_zero_predictor",
        ),
        ("near_zero", "near_zero", "official_siren_near_zero"),
    ],
)
def test_official_siren_decision_applies_the_legacy_control_gate(
    legacy: str,
    official: str,
    expected: str,
) -> None:
    assert (
        pipeline.official_siren_summary_decision(
            legacy_classification=legacy,
            official_classification=official,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("legacy", "official"),
    [
        ("unknown", "near_zero"),
        ("near_zero", "unknown"),
    ],
)
def test_official_siren_decision_rejects_unknown_classifications(
    legacy: str,
    official: str,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        pipeline.official_siren_summary_decision(
            legacy_classification=legacy,
            official_classification=official,
        )


def test_official_siren_conditions_are_exact_and_canonicalized() -> None:
    reverse_order = [
        {"label": "official_siren_30", "omega_0": 30.0, "hidden_omega": 30.0},
        {"label": "legacy_control", "omega_0": 300.0, "hidden_omega": 1.0},
    ]

    assert pipeline._validated_official_siren_conditions(reverse_order) == (
        ("legacy_control", 300.0, 1.0),
        ("official_siren_30", 30.0, 30.0),
    )


@pytest.mark.parametrize(
    "conditions",
    [
        [{"label": "legacy_control", "omega_0": 300.0, "hidden_omega": 1.0}],
        [
            {"label": "legacy_control", "omega_0": 300.0, "hidden_omega": 1.0},
            {"label": "legacy_control", "omega_0": 300.0, "hidden_omega": 1.0},
        ],
        [
            {"label": "legacy_control", "omega_0": 300.0, "hidden_omega": 1.0},
            {"label": "official_siren_30", "omega_0": 30.0, "hidden_omega": 1.0},
        ],
        [
            {
                "label": "legacy_control",
                "omega_0": 300.0,
                "hidden_omega": 1.0,
                "extra": True,
            },
            {"label": "official_siren_30", "omega_0": 30.0, "hidden_omega": 30.0},
        ],
    ],
)
def test_official_siren_conditions_reject_contract_drift(
    conditions: list[dict[str, object]],
) -> None:
    with pytest.raises(ConfigurationError):
        pipeline._validated_official_siren_conditions(conditions)


def test_batching_model_builder_forwards_hidden_omega() -> None:
    model = pipeline._build_model(
        {
            "model": {
                "input_features": 6,
                "hidden_width": 8,
                "hidden_layers": 2,
                "omega_0": 30.0,
                "hidden_omega": 30.0,
            }
        }
    )

    assert model.omega_0 == 30.0
    assert model.hidden_omega == 30.0
    assert model.network[1].omega == 30.0
