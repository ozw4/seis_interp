from __future__ import annotations

import re
from pathlib import Path

import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_016_all_ffid_siren"


def test_all_ffid_study_resolves_the_training_contract() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected = {
        "study.status": "draft",
        "project.random_seed": 42,
        "sampling.random_trace_holdout_fraction": 0.20,
        "sampling.validation_fraction_of_holdout": 0.25,
        "sampling.split_scope": "per_ffid",
        "normalization.coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "normalization.amplitude": "train_global_rms",
        "model.name": "siren",
        "model.input_features": 6,
        "model.hidden_width": 256,
        "model.hidden_layers": 4,
        "model.omega_0": 30.0,
        "model.hidden_omega": 30.0,
        "training.batch_mode": "full_ffid_epoch",
        "training.loss": "l2",
        "training.optimizer": "adam",
        "training.learning_rate": 1.0e-4,
        "training.max_epochs": 10,
        "training.early_stopping_patience": 3,
        "training.validation_batch_size": 65536,
        "training.device": "cuda:0",
    }

    assert {path: get_required_config_value(config, path) for path in expected} == expected


def test_all_ffid_inputs_lock_the_manifest_sources_without_reading_raw_data() -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))
    manifest = yaml.safe_load(
        (REPOSITORY_ROOT / "data" / "external" / "seg_c3_na" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    (dataset,) = inputs["datasets"]

    assert [item["name"] for item in dataset["files"]] == [
        item["name"] for item in manifest["files"]
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in dataset["files"])
    assert dataset["subset"] == {
        "ffid_min": 2,
        "ffid_max": 4781,
        "include_incomplete_ffids": True,
        "expected_complete_trace_count": 544,
        "time_window_s": None,
    }
    for key in ("manifest", "interim", "processed"):
        value = Path(dataset[key])
        assert not value.is_absolute()
        assert (STUDY_DIRECTORY / value).resolve().is_relative_to(REPOSITORY_ROOT)
