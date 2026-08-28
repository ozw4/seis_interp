from __future__ import annotations

from pathlib import Path

import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)

FORMAL_STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_016_all_ffid_siren"
TEMP_STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_all_ffid_temp"


def test_temp_study_preserves_the_existing_processed_data_contract() -> None:
    formal_config = load_resolved_config(
        FORMAL_STUDY_DIRECTORY / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    temp_config = load_resolved_config(
        TEMP_STUDY_DIRECTORY / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    fixed_paths = (
        "project.random_seed",
        "sampling.random_trace_holdout_fraction",
        "sampling.validation_fraction_of_holdout",
        "sampling.split_scope",
        "sampling.trace_amplitude_filter.exclude_all_zero",
        "sampling.trace_amplitude_filter.max_abs_amplitude",
        "normalization.coordinates",
        "normalization.amplitude",
    )

    assert {path: get_required_config_value(temp_config, path) for path in fixed_paths} == {
        path: get_required_config_value(formal_config, path) for path in fixed_paths
    }
    assert get_required_config_value(formal_config, "training.amplitude_scaling") == (
        "train_global_rms"
    )
    assert get_required_config_value(temp_config, "training.amplitude_scaling") == "per_trace_rms"
    assert get_required_config_value(formal_config, "training.loss") == "l2"
    assert "correlation_weight" not in formal_config["training"]
    assert "correlation_eps" not in formal_config["training"]
    assert get_required_config_value(temp_config, "training.loss") == "l2"
    assert float(get_required_config_value(temp_config, "training.correlation_weight")) >= 0.0
    assert float(get_required_config_value(temp_config, "training.correlation_eps")) > 0.0


def test_temp_study_reuses_the_all_ffid_input_contract() -> None:
    formal_inputs = yaml.safe_load(
        (FORMAL_STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8")
    )
    temp_inputs = yaml.safe_load((TEMP_STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))

    assert temp_inputs == formal_inputs
    (dataset,) = temp_inputs["datasets"]
    processed = Path(dataset["processed"])
    assert not processed.is_absolute()
    assert (TEMP_STUDY_DIRECTORY / processed).resolve() == (
        REPOSITORY_ROOT
        / "data"
        / "processed"
        / "c3_na"
        / "all_ffids_per_ffid_random_split_amplitude_qc"
    )
