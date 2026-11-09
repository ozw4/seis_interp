"""The ``inr_real`` method honours its declared loss, optimizer and trace budget."""

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.forge_mvp_config import validate_field_method_config
from seis_interp.models.trace_coordinate_field import TraceCoordinateField
from seis_interp.training import forge_mvp_methods
from seis_interp.training.forge_mvp_methods import _fit_inr

COLUMNS = [
    f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
]
TIME_COUNT = 8
TRACES = 2
STEPS = 5
BUDGET = {
    "max_steps": STEPS,
    "learning_rate": 0.001,
    "loss_traces_per_update": TRACES,
    "parameter_count_target": 1000,
    "parameter_count_tolerance": 0.05,
}


def _inputs():
    rng = np.random.default_rng(3)
    geometry = pd.DataFrame(rng.uniform(0, 1, size=(6, 4)) * [0, 1, 0, 2], columns=COLUMNS)
    geometry.insert(0, "cell_id", np.arange(6))
    return ForgeModelInputs(
        geometry=geometry,
        observed_cells=np.array([0, 2, 3, 5]),
        test_cells=np.array([1, 4]),
        observed_values=rng.normal(size=(4, TIME_COUNT)).astype(np.float32),
        spatial_shape=(1, 2, 1, 3),
        time_s=np.arange(TIME_COUNT) * 0.002,
        global_rms=1.0,
        binding={},
    )


def _fit_config(**training):
    return {
        "device": "cpu",
        "model_seed": 7301,
        "loss": "masked_trace_mse",
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "max_steps": STEPS,
            "batch_size": TRACES,
            "weight_decay": 0.01,
            "ema_decay": None,
            "report_interval": 1,
        }
        | training,
        "prediction": {"batch_size": 2},
    }


def _method_config(**overrides):
    config = {
        "method_id": "inr_real",
        "geometry_mode": "real",
        "device": "cpu",
        "loss": "masked_trace_mse",
        "checkpoint_rule": "fixed_final_ema",
        "model_seed": 7301,
        "model": {
            "input_features": 4,
            "fourier_components": 2,
            "frequency_base": 1.1,
            "hidden_width": 4,
            "hidden_layers": 2,
            "activation": "gelu",
            "output_activation": "linear",
            "time_sample_count": TIME_COUNT,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "max_steps": STEPS,
            "weight_decay": 0.0,
            "report_interval": 1,
            "learning_rate_schedule": "constant",
            "mixed_precision": "off",
            "ema_decay": 0.9,
            "batch_size": TRACES,
        },
        "prediction": {"batch_size": 2},
    }
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(section), dict):
            config[section].update(value)
        else:
            config[section] = value
    return config


def _contract():
    return {
        "model_seed": 7301,
        "time_sample_count": TIME_COUNT,
        "matched_training": BUDGET,
        "observed_trace_count": 4,
    }


def _model():
    torch.manual_seed(7301)
    return TraceCoordinateField(
        time_sample_count=TIME_COUNT, fourier_components=2, hidden_width=4, hidden_layers=2
    )


def _fit(config, decays=(), snapshots=()):
    return _fit_inr(_model(), _inputs(), config, lambda _: None, decays, snapshots)


@pytest.mark.parametrize(
    "optimizer,used,unused",
    [("adamw", torch.optim.AdamW, torch.optim.Adam), ("adam", torch.optim.Adam, torch.optim.AdamW)],
)
def test_the_declared_optimizer_is_the_one_used(monkeypatch, optimizer, used, unused):
    steps = []
    for kind in (used, unused):
        original = kind.step
        monkeypatch.setattr(
            kind,
            "step",
            lambda self, *a, _k=kind, _o=original, **kw: (steps.append(_k), _o(self))[1],
        )
    _fit(_fit_config(optimizer=optimizer))
    assert [kind for kind in steps if kind is used] == [used] * STEPS
    assert unused not in steps


def test_an_invalid_optimizer_is_rejected():
    with pytest.raises(ValueError, match="adam or adamw"):
        _fit(_fit_config(optimizer="sgd"))


def test_the_declared_loss_is_the_one_used(monkeypatch):
    seen = []
    original = forge_mvp_methods.masked_trace_loss

    def recording(prediction, target, *args, **kwargs):
        seen.append((kwargs.get("loss_name"), tuple(prediction.shape), tuple(target.shape)))
        return original(prediction, target, *args, **kwargs)

    monkeypatch.setattr(forge_mvp_methods, "masked_trace_loss", recording)
    _fit(_fit_config())
    assert {name for name, _, _ in seen} == {"masked_trace_mse"}
    assert {prediction for _, prediction, target in seen if prediction == target} == {
        (TRACES, TIME_COUNT)
    }


def test_relative_mse_keeps_the_direct_objective(monkeypatch):
    called = []
    monkeypatch.setattr(
        forge_mvp_methods,
        "masked_trace_loss",
        lambda *a, **k: called.append(k.get("loss_name")) or torch.zeros(()),
    )
    config = _fit_config()
    config["loss"] = "masked_trace_relative_mse"
    _fit(config)
    assert called == []


def test_the_fit_returns_test_predictions_shadows_and_the_zero_predictor_gate():
    decays = (0.5, 0.9)
    config = _fit_config(ema_decay=0.9)
    prediction, history, shadows, records, states = _fit(config, decays=decays)
    assert prediction.shape == (2, TIME_COUNT) and prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    assert len(history) == STEPS
    # ``_fit_inr`` returns the non-primary shadows; the primary weights are the prediction.
    assert sorted(shadows) == ["0.5"]
    assert records["g0"]["updates"] == STEPS
    assert records["g0"]["zero_predictor_loss"] > 0
    assert records["would_clip_threshold"] == 1.0
    assert records["final_training_loss_window_updates"] == STEPS
    assert set(states) <= {"0.5", "0.9"}


def test_the_declared_method_config_is_accepted():
    validate_field_method_config(_method_config(), _contract(), loss_name="masked_trace_mse")


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"model": {"time_sample_count": TIME_COUNT + 1}}, "complete traces"),
        ({"geometry_mode": "grid"}, "unsupported field method or geometry"),
        ({"method_id": "nersi_real"}, "unsupported field method or geometry"),
        ({"training": {"batch_size": TRACES + 1}}, "budgeted loss traces"),
        ({"loss": "masked_trace_relative_mse"}, "declared training loss"),
        ({"training": {"optimizer": "sgd"}}, "optimizer contract"),
        ({"training": {"weight_decay_after": {"step": 2, "value": 0.1}}}, "constant weight decay"),
        ({"prediction": {"batch_size": 0}}, "prediction.batch_size"),
        ({"alignment": {"velocity_m_per_s": 500.0}}, "shared preprocessing condition"),
    ],
)
def test_structural_mistakes_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_field_method_config(
            _method_config(**overrides), _contract(), loss_name="masked_trace_mse"
        )
