import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.models.nersi_trace import NersiTrace
from seis_interp.training.forge_mvp_methods import _fit_nersi

COLUMNS = [
    f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
]


def _inputs():
    rng = np.random.default_rng(3)
    shape = (1, 2, 1, 3)
    geometry = pd.DataFrame(rng.uniform(0, 1, size=(6, 4)) * [0, 1, 0, 2], columns=COLUMNS)
    geometry.insert(0, "cell_id", np.arange(6))
    return ForgeModelInputs(
        geometry=geometry,
        observed_cells=np.array([0, 2, 3, 5]),
        test_cells=np.array([1, 4]),
        observed_values=rng.normal(size=(4, 8)).astype(np.float32),
        spatial_shape=shape,
        time_s=np.arange(8) * 0.002,
        global_rms=1.0,
        binding={},
    )


def _config(schedule):
    return {
        "device": "cpu",
        "model_seed": 7301,
        "training": {
            "learning_rate": 0.001,
            "max_steps": 5,
            "batch_size": 2,
            "weight_decay": 0.0,
            "weight_decay_after": schedule,
            "ema_decay": None,
            "report_interval": 1,
        },
        "prediction": {"batch_size": 2},
    }


def _fit(monkeypatch, schedule):
    recorded = []
    original_step = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        recorded.append([group["weight_decay"] for group in self.param_groups])
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    torch.manual_seed(7301)
    model = NersiTrace(
        time_sample_count=8,
        fourier_components=2,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
    )
    prediction, history, *_ = _fit_nersi(
        model, _inputs(), _config(schedule), lambda _: None, (), ()
    )
    return recorded, prediction, history, model


def test_weight_decay_switches_exactly_at_the_scheduled_step(monkeypatch):
    recorded, _, _, _ = _fit(monkeypatch, {"step": 3, "value": 2.0e-4})
    assert recorded == [[0.0], [0.0], [2.0e-4], [2.0e-4], [2.0e-4]]


def test_absent_schedule_keeps_the_constant_decay_and_the_same_fit(monkeypatch):
    recorded, prediction, history, model = _fit(monkeypatch, None)
    assert recorded == [[0.0]] * 5
    # A schedule that sets the same value is a no-op on the fit.
    _, same_prediction, same_history, same_model = _fit(monkeypatch, {"step": 2, "value": 0.0})
    np.testing.assert_array_equal(prediction, same_prediction)
    assert history == same_history
    for name, value in model.state_dict().items():
        assert torch.equal(same_model.state_dict()[name], value)


@pytest.mark.parametrize(
    "schedule",
    [
        {"step": 0, "value": 1.0e-4},
        {"step": 6, "value": 1.0e-4},
        {"step": 2, "value": -1.0},
        {"step": 2},
    ],
)
def test_invalid_schedule_is_rejected(monkeypatch, schedule):
    with pytest.raises(ValueError, match="weight_decay_after"):
        _fit(monkeypatch, schedule)
