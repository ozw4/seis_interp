from __future__ import annotations

from pathlib import Path

import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.pipelines.train_neighbor_inpainter import _validated_settings

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_018_all_ffid_50pct_neighbor_inpainter"


def test_study_018_locks_the_50pct_per_ffid_baseline_contract() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert config["sampling"] == {
        "random_trace_holdout_fraction": 0.5,
        "validation_fraction_of_holdout": 0.25,
        "split_scope": "per_ffid",
        "trace_amplitude_filter": {
            "exclude_all_zero": True,
            "max_abs_amplitude": 10000.0,
        },
        "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
    }
    assert settings.success_threshold_db == 20.0
    assert settings.required_eligible_ffid_count == 4780
    assert dict(settings.required_effective_split_counts) == {
        "train": 1151731,
        "validation": 287933,
        "test": 863801,
    }
    assert settings.required_fully_excluded_ffids == (1746,)


def test_study_018_inputs_lock_split_and_canonical_counts() -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))
    (dataset,) = inputs["datasets"]

    assert dataset["subset"]["split"] == {
        "scope": "per_ffid",
        "random_seed": 42,
        "train_fraction": 0.5,
        "validation_fraction": 0.125,
        "test_fraction": 0.375,
        "prepared_counts": {
            "train": 1151740,
            "validation": 287935,
            "test": 863805,
        },
    }
    canonical = dataset["subset"]["physical_coordinate_canonicalization"]
    assert canonical["removed_split_counts"] == {
        "train": 9,
        "validation": 2,
        "test": 4,
    }
    assert canonical["effective_split_counts"] == {
        "train": 1151731,
        "validation": 287933,
        "test": 863801,
    }
    for key in ("manifest", "interim", "processed"):
        path = Path(dataset[key])
        assert not path.is_absolute()
        assert (STUDY_DIRECTORY / path).resolve().is_relative_to(REPOSITORY_ROOT)


def test_stage02_changes_only_to_four_coordinate_equivalent_aperture() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "variants" / "stage02_source_x_coordinate.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert settings.target_coordinates == (
        "source_x_m",
        "source_y_m",
        "relative_receiver_x_m",
        "relative_receiver_y_m",
    )
    assert settings.neighbor_geometry == "multiline_staggered_source"
    assert settings.relative_receiver_x_radius == 1
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 3


def test_stage06_enables_film_conditioning() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage06_receiver_aperture_film.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.coordinate_conditioning == "film"
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.relative_receiver_y_radius == 5
