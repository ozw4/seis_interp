from __future__ import annotations

from pathlib import Path

import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.pipelines.train_neighbor_inpainter import _validated_settings

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_019_all_ffid_75pct_neighbor_inpainter"


def test_study_019_locks_the_75pct_per_ffid_contract() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert config["sampling"] == {
        "random_trace_holdout_fraction": 0.25,
        "validation_fraction_of_holdout": 0.25,
        "split_scope": "per_ffid",
        "trace_amplitude_filter": {
            "exclude_all_zero": True,
            "max_abs_amplitude": 10000.0,
        },
        "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
    }
    assert settings.success_threshold_db == 25.0
    assert settings.required_eligible_ffid_count == 4780
    assert dict(settings.required_effective_split_counts) == {
        "train": 1727598,
        "validation": 143978,
        "test": 431889,
    }
    assert settings.required_fully_excluded_ffids == (1746,)


def test_study_019_inputs_lock_split_and_canonical_counts() -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))
    (dataset,) = inputs["datasets"]

    assert dataset["subset"]["split"] == {
        "scope": "per_ffid",
        "random_seed": 42,
        "train_fraction": 0.75,
        "validation_fraction": 0.0625,
        "test_fraction": 0.1875,
        "prepared_counts": {
            "train": 1727610,
            "validation": 143980,
            "test": 431890,
        },
    }
    canonical = dataset["subset"]["physical_coordinate_canonicalization"]
    assert canonical["removed_split_counts"] == {
        "train": 12,
        "validation": 2,
        "test": 1,
    }
    assert canonical["effective_split_counts"] == {
        "train": 1727598,
        "validation": 143978,
        "test": 431889,
    }
    for key in ("manifest", "interim", "processed"):
        path = Path(dataset[key])
        assert not path.is_absolute()
        assert (STUDY_DIRECTORY / path).resolve().is_relative_to(REPOSITORY_ROOT)
