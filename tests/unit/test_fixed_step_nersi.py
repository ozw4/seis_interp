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
    observed_profile_trace_loss,
    train_nersi_fixed_steps,
)
from seis_interp.training.trace_relative_loss import masked_trace_relative_mse


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


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
@pytest.mark.parametrize("logical_batch", [2, 3])
@pytest.mark.parametrize(
    "augmentation", [{}, {"coordinate_jitter_cells": 0.1}, {"profile_mixup_max_fraction": 0.2}]
)
def test_accumulation_matches_full_batch_gradients_with_unequal_trace_counts(
    monkeypatch, loss_name, augmentation, logical_batch
):
    gradients = []
    step = torch.optim.Adam.step

    def capture(optimizer, *args, **kwargs):
        gradients.append(
            [p.grad.clone() for group in optimizer.param_groups for p in group["params"]]
        )
        return step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture)
    full, micro = _model(), _model()
    options = dict(
        max_steps=2,
        loss_name=loss_name,
        **augmentation,
        optimization={
            "learning_rate_schedule": "cosine",
            "minimum_learning_rate": 1e-4,
            "ema_decay": 0.9,
        },
    )
    a = _train(full, _data(), profiles_per_step=logical_batch, **options)
    b = _train(
        micro, _data(), profiles_per_step=1, gradient_accumulation_steps=logical_batch, **options
    )
    assert len(gradients) == 4
    for i in range(2):
        for x, y in zip(gradients[i], gradients[i + 2], strict=True):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    for x, y in zip(full.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    assert a.final_batch_loss == pytest.approx(b.final_batch_loss, rel=2e-6)
    assert a.supervised_trace_presentations == b.supervised_trace_presentations
    assert a.steps_completed == b.steps_completed == 2


def test_accumulation_one_preserves_sampling_and_updates_exactly():
    a, b = _model(), _model()
    first = _train(a, _data())
    second = _train(b, _data(), gradient_accumulation_steps=1)
    assert first == second
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 4])
def test_invalid_accumulation_rejected(value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        _train(_model(), _data(), profiles_per_step=1, gradient_accumulation_steps=value)


def test_profile_loss_selects_observed_complete_traces_for_shared_loss() -> None:
    prediction = torch.zeros((2, 1, 2, 3))
    target = torch.tensor(
        [
            [[[1.0, 10.0, 20.0], [2.0, 30.0, 40.0]]],
            [[[3.0, 4.0, 50.0], [5.0, 6.0, 60.0]]],
        ]
    )
    mask = torch.tensor([[True, False, False], [False, True, False]])

    loss = observed_profile_trace_loss(prediction, target, mask)
    prediction_traces = prediction[:, 0].transpose(1, 2)[mask]
    target_traces = target[:, 0].transpose(1, 2)[mask]

    assert prediction_traces.shape == target_traces.shape == (2, 2)
    torch.testing.assert_close(loss, masked_trace_relative_mse(prediction_traces, target_traces))


@pytest.mark.parametrize("profiles_per_step", [2, 3])
def test_supervised_trace_presentations_count_actual_selected_observed_rows(profiles_per_step):
    data = _data()
    rng = np.random.default_rng(201)
    expected = 0
    for _ in range(4):
        selected = (
            data.training_profile_indices
            if profiles_per_step == 3
            else rng.choice(data.training_profile_indices, size=profiles_per_step, replace=False)
        )
        expected += int(data.observed_trace_mask[selected].sum())
    result = _train(_model(), data, random_seed=201, profiles_per_step=profiles_per_step)
    assert result.supervised_trace_presentations == expected


def test_profiles_with_different_observed_counts_reduce_over_selected_traces() -> None:
    prediction = torch.zeros((3, 1, 2, 3))
    target = torch.tensor(
        [
            [[[1.0, 20.0, 30.0], [2.0, 40.0, 50.0]]],
            [[[3.0, 4.0, 60.0], [5.0, 6.0, 70.0]]],
            [[[7.0, 80.0, 90.0], [8.0, 100.0, 110.0]]],
        ]
    )
    mask = torch.tensor([[True, False, False], [True, True, False], [True, False, False]])
    actual = observed_profile_trace_loss(prediction, target, mask)
    selected_targets = target[:, 0].transpose(1, 2)[mask]

    torch.testing.assert_close(
        actual,
        masked_trace_relative_mse(torch.zeros_like(selected_targets), selected_targets),
    )


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
        loss = observed_profile_trace_loss(prediction, target, mask)
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


def test_disabled_augmentation_preserves_exact_training():
    first, second = _model(), _model()
    assert _train(first, _data()) == _train(
        second, _data(), coordinate_jitter_cells=0.0, augmentation_seed=501
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_jitter_keeps_sampling_and_observed_labels_independent(monkeypatch):
    original = training_module.masked_trace_loss
    captured_targets = []

    def record_loss(prediction, target, **kwargs):
        captured_targets.append(target.detach().clone())
        return original(prediction, target, **kwargs)

    monkeypatch.setattr(training_module, "masked_trace_loss", record_loss)
    runs = []
    for radius, seed in [(0, 501), (0.1, 501), (0.1, 502), (0.1, 501)]:
        model = _model()
        coordinates = []
        handle = model.register_forward_pre_hook(
            lambda module, args, storage=coordinates: storage.append(args[0].detach().clone())
        )
        captured_targets.clear()
        result = _train(model, _data(), coordinate_jitter_cells=radius, augmentation_seed=seed)
        handle.remove()
        runs.append((coordinates, list(captured_targets), result, model.state_dict()))
    for run in runs[1:]:
        for first, changed in zip(runs[0][1], run[1], strict=True):
            assert torch.equal(first, changed)
        assert run[2].supervised_trace_presentations == runs[0][2].supervised_trace_presentations
    assert not torch.equal(runs[0][0][0], runs[1][0][0])
    assert not torch.equal(runs[1][0][0], runs[2][0][0])
    assert runs[1][2] == runs[3][2]
    for name, value in runs[1][3].items():
        assert torch.equal(value, runs[3][3][name])


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
@pytest.mark.parametrize("profiles_per_step", [2, 3])
def test_fixed_steps_call_shared_loss_with_only_selected_time_last_traces(
    monkeypatch, loss_name, profiles_per_step
) -> None:
    data = _data()
    selected_shapes: list[tuple[int, int]] = []
    original = training_module.masked_trace_loss
    requested_loss = loss_name
    expected_rows = []
    rng = np.random.default_rng(0)
    for _ in range(4):
        selected = (
            data.training_profile_indices
            if profiles_per_step == 3
            else rng.choice(data.training_profile_indices, size=profiles_per_step, replace=False)
        )
        expected_rows.append(
            torch.from_numpy(
                data.normalized_profiles[selected, 0].transpose(0, 2, 1)[
                    data.observed_trace_mask[selected]
                ]
            )
        )

    def recording_loss(prediction, target, trace_mask=None, *, loss_name):
        assert loss_name == requested_loss
        expected = expected_rows[len(selected_shapes)]
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        assert trace_mask is None
        assert prediction.shape == target.shape
        selected_shapes.append(tuple(prediction.shape))
        return original(prediction, target, loss_name=loss_name)

    monkeypatch.setattr(training_module, "masked_trace_loss", recording_loss)

    _train(
        _model(),
        data,
        profiles_per_step=profiles_per_step,
        max_steps=4,
        loss_name=loss_name,
    )

    assert selected_shapes == [tuple(row.shape) for row in expected_rows]


def test_different_sampling_seed_changes_a_partial_profile_update() -> None:
    first_model = _model()
    second_model = copy.deepcopy(first_model)

    _train(first_model, _data(), max_steps=1, random_seed=0)
    _train(second_model, _data(), max_steps=1, random_seed=1)

    assert any(
        not torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )


def test_mixup_keeps_original_sampling_and_counts_synthetic_supervision(monkeypatch):
    data = _data()
    sampling_rng = np.random.default_rng(0)
    expected = [
        sampling_rng.choice(data.training_profile_indices, size=2, replace=False) for _ in range(4)
    ]
    original_loss = training_module.observed_profile_trace_loss
    counts = []

    def capture(prediction, target, mask, *, loss_name):
        selected = expected[len(counts)]
        np.testing.assert_array_equal(target[:2].numpy(), data.normalized_profiles[selected])
        np.testing.assert_array_equal(mask[:2].numpy(), data.observed_trace_mask[selected])
        assert loss_name == "masked_trace_mse"
        counts.append(int(mask.sum()))
        assert len(mask) > 2
        return original_loss(prediction, target, mask, loss_name=loss_name)

    monkeypatch.setattr(training_module, "observed_profile_trace_loss", capture)
    result = _train(
        _model(),
        data,
        loss_name="masked_trace_mse",
        profile_mixup_max_fraction=0.2,
        augmentation_seed=501,
    )
    assert len(counts) == result.steps_completed == 4
    assert result.supervised_trace_presentations == sum(counts)
    assert sum(counts) > sum(int(data.observed_trace_mask[selected].sum()) for selected in expected)


def test_disabled_mixup_preserves_baseline_training_exactly():
    first = _model()
    second = copy.deepcopy(first)
    before = _train(first, _data())
    after = _train(second, _data(), profile_mixup_max_fraction=0.0, augmentation_seed=501)
    assert before == after
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


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
        ({"profile_mixup_max_fraction": 0.6}, "profile_mixup"),
        ({"profile_mixup_max_fraction": 0.1, "coordinate_jitter_cells": 0.1}, "combined"),
    ],
)
def test_invalid_training_settings_are_rejected(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _train(_model(), _data(), **changes)


def test_nonfinite_selected_observed_target_is_rejected_by_shared_loss() -> None:
    data = _data()
    profiles = data.normalized_profiles.copy()
    profiles[0, 0, 0, 0] = np.nan
    poisoned = replace(data, normalized_profiles=profiles)

    with pytest.raises(ValueError, match="selected prediction and target traces must be finite"):
        _train(
            _model(),
            poisoned,
            profiles_per_step=len(poisoned.training_profile_indices),
            max_steps=1,
        )
