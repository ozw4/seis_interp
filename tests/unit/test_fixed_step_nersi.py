from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.models.nersi import Nersi
from seis_interp.training import fixed_step_nersi as training_module
from seis_interp.training.c3_volume_nersi_data import C3VolumeNersiData
from seis_interp.training.fixed_step_nersi import (
    observed_profile_mse,
    train_nersi_fixed_steps,
)


def _model(*, seed: int = 7) -> Nersi:
    torch.manual_seed(seed)
    return Nersi(
        fourier_components=2,
        frequency_base=1.25,
        encoder_width=8,
        latent_channels=2,
        decoder_channels=(2, 2, 2),
        profile_shape=(8, 8),
        kernel_size=1,
    )


def _data() -> C3VolumeNersiData:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    profiles = np.empty((4, 1, 8, 8), dtype=np.float32)
    for index in range(4):
        profiles[index] = index + np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 64
    mask = np.array(
        [
            [True, False, True, False, False, False, False, False],
            [False, True, True, True, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ],
        dtype=np.bool_,
    )
    return C3VolumeNersiData(
        normalized_coordinates=coordinates,
        normalized_profiles=profiles,
        observed_trace_mask=mask,
        training_profile_indices=np.array([0, 1, 3], dtype=np.int64),
        amplitude_scale=2.5,
        spatial_shape=(1, 2, 2, 8),
        profile_shape=(8, 8),
    )


def _train(model: Nersi, data: C3VolumeNersiData, **changes):
    arguments = {
        "device": "cpu",
        "learning_rate": 1e-3,
        "profiles_per_step": 2,
        "max_steps": 4,
        "report_interval": 2,
        "random_seed": 0,
    }
    arguments.update(changes)
    return train_nersi_fixed_steps(model, data, **arguments)


def test_observed_profile_mse_uses_observed_sample_weighting() -> None:
    prediction = torch.zeros((2, 1, 2, 3))
    target = torch.tensor(
        [
            [[[1.0, 10.0, 20.0], [2.0, 30.0, 40.0]]],
            [[[3.0, 4.0, 50.0], [5.0, 6.0, 60.0]]],
        ]
    )
    mask = torch.tensor([[True, False, False], [True, True, False]])

    loss = observed_profile_mse(prediction, target, mask)

    # Six observed samples contribute equally. The two profiles have unequal
    # trace counts, so an equal mean of per-profile losses would differ.
    assert loss.item() == pytest.approx((1 + 4 + 9 + 16 + 25 + 36) / 6)


def test_masked_targets_do_not_affect_loss_or_gradient_even_when_nan() -> None:
    base_prediction = torch.linspace(-1.0, 1.0, 12).reshape(2, 1, 2, 3)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    baseline_target = torch.zeros_like(base_prediction)
    poisoned_target = baseline_target.clone()
    poisoned_target[:, :, :, 2] = torch.nan
    poisoned_target[0, :, :, 1] = 1.0e30
    losses = []
    gradients = []
    for target in (baseline_target, poisoned_target):
        prediction = base_prediction.clone().requires_grad_()
        loss = observed_profile_mse(prediction, target, mask)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(prediction.grad.detach().clone())

    torch.testing.assert_close(losses[1], losses[0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
    assert torch.isfinite(losses[1])
    assert torch.isfinite(gradients[1]).all()
    expanded = mask[:, None, None, :].expand_as(gradients[1])
    assert torch.count_nonzero(gradients[1][~expanded]) == 0


def test_fixed_steps_history_reporting_and_cpu_reproducibility() -> None:
    first_model = _model()
    second_model = copy.deepcopy(first_model)
    messages: list[str] = []

    first = _train(first_model, _data(), max_steps=5, report_interval=2, reporter=messages.append)
    second = _train(second_model, _data(), max_steps=5, report_interval=2)

    assert first == second
    assert first.steps_completed == 5
    assert [record["step"] for record in first.history] == [2, 4, 5]
    assert math.isfinite(first.final_batch_loss)
    assert len(messages) == 3
    assert messages[-1] == (
        f"nersi_volume step 5/5: train_loss={first.history[-1]['train_loss']:.8g}"
    )
    for name, expected in first_model.state_dict().items():
        assert torch.equal(second_model.state_dict()[name], expected), name
    assert first_model.training
    assert all(parameter.device.type == "cpu" for parameter in first_model.parameters())


def test_different_sampling_seed_changes_a_partial_profile_update() -> None:
    first_model = _model()
    second_model = copy.deepcopy(first_model)

    _train(first_model, _data(), max_steps=1, random_seed=0)
    _train(second_model, _data(), max_steps=1, random_seed=1)

    assert any(
        not torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )


def test_full_candidate_batch_uses_stable_order_without_rng_draw(monkeypatch) -> None:
    class NoDrawGenerator:
        def choice(self, *args, **kwargs):
            pytest.fail("a full-candidate batch must not draw from the RNG")

    monkeypatch.setattr(training_module.np.random, "default_rng", lambda seed: NoDrawGenerator())

    result = _train(_model(), _data(), profiles_per_step=3, max_steps=1)

    assert result.steps_completed == 1


def test_all_missing_profiles_are_not_valid_training_candidates() -> None:
    data = _data()
    assert not data.observed_trace_mask[2].any()
    assert data.training_profile_indices.tolist() == [0, 1, 3]

    invalid = replace(
        data,
        training_profile_indices=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="training_profile_indices"):
        _train(_model(), invalid)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"profiles_per_step": 0}, "profiles_per_step"),
        ({"profiles_per_step": 4}, "profiles_per_step"),
        ({"max_steps": True}, "max_steps"),
        ({"report_interval": -1}, "report_interval"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_invalid_training_settings_are_rejected(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _train(_model(), _data(), **changes)


def test_nonfinite_observed_loss_reports_the_step_number() -> None:
    data = _data()
    profiles = data.normalized_profiles.copy()
    profiles[0, 0, 0, 0] = np.nan
    poisoned = replace(data, normalized_profiles=profiles)

    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        _train(
            _model(),
            poisoned,
            profiles_per_step=len(poisoned.training_profile_indices),
            max_steps=1,
        )
