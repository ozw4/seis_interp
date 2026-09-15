from copy import deepcopy

import pytest

from seis_interp.ccnet5d_poc_config import validate_ccnet5d_poc_config
from seis_interp.configuration import ConfigurationError


def _config() -> dict[str, object]:
    return {
        "model": {
            "name": "ccnet5d",
            "hidden_channels": 2,
            "intermediate_channels": 3,
            "kernel_size": 3,
            "output_activation": "linear",
        },
        "patches": {
            "shape": [384, 2, 3, 1, 4],
            "inner_mask_fraction": 0.5,
            "mask_kind": "random_trace",
            "placement_seed": 17,
            "inner_mask_seed": 401,
        },
        "training": {
            "model_initialization_seed": 19,
            "optimizer": "adam",
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": 1.0e-3,
            "max_steps": 5,
            "report_interval": 2,
            "device": "cpu",
        },
        "prediction": {"core_shape": [64, 2, 3, 1, 4]},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def test_validates_single_fixed_step_observed_only_configuration() -> None:
    settings = validate_ccnet5d_poc_config(_config())

    assert settings.model == {
        "hidden_channels": 2,
        "intermediate_channels": 3,
        "kernel_size": 3,
        "output_activation": "linear",
    }
    assert settings.patches.shape == (384, 2, 3, 1, 4)
    assert settings.patches.inner_mask_fraction == 0.5
    assert settings.training.max_steps == 5
    assert settings.prediction_core_shape == (64, 2, 3, 1, 4)


@pytest.mark.parametrize("decay", [None, 0.9, 0.999])
def test_optional_ema(decay):
    config = _config()
    config["training"]["ema_decay"] = decay
    assert validate_ccnet5d_poc_config(config).training.ema_decay == decay


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimizer", "sgd"),
        ("weight_decay", -1),
        ("weight_decay", True),
        ("weight_decay", float("nan")),
        ("supervised_traces_per_update", 0),
        ("supervised_traces_per_update", True),
        ("supervised_traces_per_update", 1.5),
    ],
)
def test_invalid_batched_optimizer_settings(key, value):
    config = _config()
    config["training"][key] = value
    with pytest.raises(ConfigurationError):
        validate_ccnet5d_poc_config(config)


@pytest.mark.parametrize("decay", [0, 1, -0.1, True, "0.999", float("nan"), float("inf")])
def test_rejects_invalid_ema(decay):
    config = _config()
    config["training"]["ema_decay"] = decay
    with pytest.raises(ConfigurationError, match="ema_decay"):
        validate_ccnet5d_poc_config(config)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "match"),
    [
        ("training", "loss", "mse", "loss"),
        ("training", "model_initialization_seed", True, "model_initialization_seed"),
        ("patches", "placement_seed", -1, "placement_seed"),
        ("patches", "inner_mask_seed", 1.5, "inner_mask_seed"),
        ("training", "amplitude_scaling", "per_patch_rms", "amplitude_scaling"),
        ("patches", "inner_mask_fraction", 1.0, "strictly less than 1"),
        ("patches", "mask_kind", "random_sample", "mask_kind"),
        ("model", "kernel_size", 2, "odd"),
        ("model", "output_activation", "relu", "output_activation must be 'linear'"),
        ("model", "output_activation", [], "output_activation"),
    ],
)
def test_rejects_non_poc_training_choices(
    section: str,
    field: str,
    replacement: object,
    match: str,
) -> None:
    config = deepcopy(_config())
    config[section][field] = replacement  # type: ignore[index]

    with pytest.raises(ConfigurationError, match=match):
        validate_ccnet5d_poc_config(config)
