from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import ConfigurationError, load_resolved_config
from seis_interp.pipelines.train_neighbor_inpainter import (
    EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    WITH_REPLACEMENT_TARGET_SAMPLING,
    _validated_settings,
    train_neighbor_inpainter_run,
)
from tests.integration.test_train_neighbor_inpainter_pipeline import (
    _build_neighbor_training_fixture,
)


def test_target_sampling_config_defaults_to_legacy_and_accepts_epoch_mode(
    tmp_path: Path,
) -> None:
    config_path, _interim, _processed = _build_neighbor_training_fixture(tmp_path)
    config = load_resolved_config(config_path)

    assert (
        _validated_settings(config, device_override="cpu").target_sampling
        == WITH_REPLACEMENT_TARGET_SAMPLING
    )

    config["training"]["target_sampling"] = EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
    assert (
        _validated_settings(config, device_override="cpu").target_sampling
        == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
    )


@pytest.mark.parametrize("invalid", ["random_per_epoch", 1, None])
def test_target_sampling_config_rejects_unsupported_values(
    tmp_path: Path,
    invalid: object,
) -> None:
    config_path, _interim, _processed = _build_neighbor_training_fixture(tmp_path)
    config = load_resolved_config(config_path)
    config["training"]["target_sampling"] = invalid

    with pytest.raises(ConfigurationError, match="training.target_sampling must be one of"):
        _validated_settings(config, device_override="cpu")


def test_epoch_target_sampling_provenance_records_independent_seeds(tmp_path: Path) -> None:
    config_path, interim, processed = _build_neighbor_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["target_sampling"] = EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    train_neighbor_inpainter_run(
        config_path=config_path,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    for training in (inputs_lock["training"], run["training"]):
        assert training["target_sampling"] == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
        assert training["target_sampling_seed"] == 8
        assert training["neighbor_dropout_seed"] == 6
        assert training["target_sampling_rng_independent_of_neighbor_dropout"] is True
        assert training["drawn_training_targets"] == 8
        assert training["unique_training_targets_seen"] == 8
    assert resolved["training"]["target_sampling"] == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING
