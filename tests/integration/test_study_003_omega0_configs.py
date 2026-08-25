from __future__ import annotations

from pathlib import Path

import pytest

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_003_omega0_sensitivity"
CONTROL_CONFIG = STUDY_DIRECTORY / "config.yaml"
VARIANT_CONFIGS = tuple(sorted((STUDY_DIRECTORY / "variants").glob("*.yaml")))
STUDY_CONFIGS = (CONTROL_CONFIG, *VARIANT_CONFIGS)

SHARED_CONDITIONS = {
    "project.random_seed": 42,
    "sampling.random_trace_holdout_fraction": 0.20,
    "sampling.validation_fraction_of_holdout": 0.25,
    "normalization.coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
    "normalization.amplitude": "train_global_rms",
    "model.name": "siren",
    "model.input_features": 6,
    "model.hidden_layers": 4,
    "model.hidden_width": 256,
    "training.loss": "l2",
    "training.optimizer": "adam",
    "training.batch_size": 1024,
    "training.steps_per_epoch": 500,
    "training.max_epochs": 100,
    "training.early_stopping_patience": 100,
    "training.validation_batch_size": 65536,
    "training.device": "cuda",
}

EXPECTED_OMEGA_LEARNING_RATE_CONDITIONS = {
    (10.0, 1.0e-4),
    (100.0, 3.0e-4),
    (100.0, 1.0e-3),
    (300.0, 3.0e-4),
    (300.0, 1.0e-3),
    (600.0, 3.0e-4),
    (600.0, 1.0e-3),
}


def _resolved(config_path: Path) -> dict[str, object]:
    return load_resolved_config(config_path, repository_root=REPOSITORY_ROOT)


@pytest.mark.parametrize("config_path", STUDY_CONFIGS, ids=[path.name for path in STUDY_CONFIGS])
def test_every_condition_shares_the_fixed_training_setup(config_path: Path) -> None:
    resolved = _resolved(config_path)

    assert {
        dotted_path: get_required_config_value(resolved, dotted_path)
        for dotted_path in SHARED_CONDITIONS
    } == SHARED_CONDITIONS


def test_control_and_variants_cover_the_expected_omega0_learning_rates() -> None:
    conditions = [
        (
            get_required_config_value(_resolved(config_path), "model.omega_0"),
            get_required_config_value(_resolved(config_path), "training.learning_rate"),
        )
        for config_path in STUDY_CONFIGS
    ]

    assert len(conditions) == len(EXPECTED_OMEGA_LEARNING_RATE_CONDITIONS)
    assert set(conditions) == EXPECTED_OMEGA_LEARNING_RATE_CONDITIONS
