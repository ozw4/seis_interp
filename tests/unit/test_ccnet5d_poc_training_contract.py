from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training import ccnet5d_observed_training as observed_training
from seis_interp.training.ccnet5d_observed_patches import (
    CCNet5DObservedPatch,
    CCNet5DObservedPatchSource,
)
from seis_interp.training.ccnet5d_poc_checkpoints import (
    CCNET5D_POC_CHECKPOINT_ROLE,
    CCNET5D_POC_METHOD_VARIANT,
    load_ccnet5d_poc_checkpoint,
    save_ccnet5d_poc_checkpoint,
)


def _model() -> CCNet5D:
    torch.manual_seed(17)
    return CCNet5D(
        hidden_channels=1,
        intermediate_channels=1,
        kernel_size=1,
        output_activation="linear",
    )


def test_exact_64_traces_use_one_forward_backward_and_adamw_step(monkeypatch):
    volume = _observed_volume()
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=2.0,
        patch_shape=volume.values.shape,
        inner_mask_fraction=0.5,
        placement_seed=7,
        inner_mask_seed=401,
    )
    model = _model()
    events = []
    model.register_forward_hook(lambda *args: events.append("forward"))
    next(model.parameters()).register_hook(lambda grad: events.append("backward"))
    step = torch.optim.AdamW.step

    def update(optimizer, *args, **kwargs):
        assert optimizer.param_groups[0]["weight_decay"] == 0
        events.append("step")
        return step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", update)
    loss_fn = observed_training.masked_trace_loss

    def loss(prediction, target, **kwargs):
        assert prediction.shape == target.shape == (64, 4)
        assert torch.isfinite(target).all()
        result = loss_fn(prediction, target, **kwargs)
        torch.testing.assert_close(result, ((prediction.double() - target.double()) ** 2).mean())
        return result

    monkeypatch.setattr(observed_training, "masked_trace_loss", loss)
    result = observed_training.train_ccnet5d_observed_steps(
        model,
        source,
        device="cpu",
        optimizer_updates=3,
        learning_rate=0.001,
        report_every_steps=1,
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        loss_name="masked_trace_mse",
        ema_decay=0.999,
    )
    assert events == ["forward", "backward", "step"] * 3
    assert result.supervised_trace_presentations == 192
    assert [h["supervised_trace_presentations"] for h in result.history] == [64, 128, 192]


def test_stacked_batch_selection_is_exact_reproducible_and_observed_only():
    volume = _observed_volume()

    def source():
        return CCNet5DObservedPatchSource(
            volume,
            amplitude_scale=2.0,
            patch_shape=volume.values.shape,
            inner_mask_fraction=0.5,
            placement_seed=7,
            inner_mask_seed=401,
        )

    first, second = source(), source()
    for _ in range(3):
        a = first.sample_batch(5)
        b = second.sample_batch(5)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
        inputs, targets, mask = a
        assert inputs.shape == (3, *volume.values.shape)
        assert mask.sum() == 5
        assert not mask[..., -1].any()
        assert np.all(np.moveaxis(inputs, 1, -1)[mask] == 0)
        assert np.all(np.isfinite(np.moveaxis(targets, 1, -1)[mask]))


def test_ema_matches_post_update_average_without_changing_training(monkeypatch):
    snapshots = []
    real_step = torch.optim.Adam.step

    def capture_step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        snapshots.append([p.detach().clone() for g in optimizer.param_groups for p in g["params"]])
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    results = []
    for decay in (None, 0.9):
        volume = _observed_volume()
        source = CCNet5DObservedPatchSource(
            volume,
            amplitude_scale=2.0,
            patch_shape=volume.values.shape,
            inner_mask_fraction=0.5,
            placement_seed=7,
            inner_mask_seed=401,
        )
        model = _model()
        results.append(
            observed_training.train_ccnet5d_observed_steps(
                model,
                source,
                device="cpu",
                optimizer_updates=3,
                learning_rate=0.001,
                report_every_steps=1,
                ema_decay=decay,
            )
        )
    assert results[0] == results[1]
    for raw, ema in zip(snapshots[:3], snapshots[3:], strict=True):
        for a, b in zip(raw, ema, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for i, parameter in enumerate(model.parameters()):
        expected = snapshots[0][i] * 0.81 + snapshots[1][i] * 0.09 + snapshots[2][i] * 0.1
        torch.testing.assert_close(parameter, expected)


def _observed_volume() -> ObservedC3Volume:
    shape = (4, 1, 1, 1, 4)
    observed_mask = np.array([[[[True, True, True, False]]]], dtype=np.bool_)
    target_mask = ~observed_mask
    values = np.zeros(shape, dtype=np.float32)
    values[:, observed_mask] = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    return ObservedC3Volume(
        values=values,
        time_s=np.arange(shape[0], dtype=np.float64) * 0.004,
        array_rows=np.arange(4, dtype=np.int64).reshape(shape[1:]),
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=target_mask,
    )


def _fixed_training_batch(source: CCNet5DObservedPatchSource) -> CCNet5DObservedPatch:
    sampled = source.sample()
    poisoned_targets = sampled.pseudo_target.copy()
    poisoned_targets[:, ~sampled.pseudo_target_mask] = np.nan
    return CCNet5DObservedPatch(
        model_input=sampled.model_input,
        pseudo_target=poisoned_targets,
        pseudo_target_mask=sampled.pseudo_target_mask,
        visible_observed_mask=sampled.visible_observed_mask,
        patch_slices=sampled.patch_slices,
    )


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
def test_fixed_step_training_calls_shared_loss_with_only_hidden_complete_traces(
    loss_name,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = _observed_volume()
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=2.0,
        patch_shape=volume.values.shape,
        inner_mask_fraction=0.5,
        placement_seed=7,
        inner_mask_seed=401,
    )
    batch = _fixed_training_batch(source)
    hidden_count = int(np.count_nonzero(batch.pseudo_target_mask))
    expected_targets = torch.from_numpy(batch.pseudo_target[:, batch.pseudo_target_mask].T.copy())
    sample_count = 0

    def fixed_sample() -> CCNet5DObservedPatch:
        nonlocal sample_count
        sample_count += 1
        return batch

    monkeypatch.setattr(source, "sample", fixed_sample)
    requested_loss = loss_name
    real_shared_loss = observed_training.masked_trace_loss
    loss_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recorded_shared_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        trace_mask: torch.Tensor | None = None,
        *,
        loss_name: str,
    ) -> torch.Tensor:
        assert loss_name == requested_loss
        assert trace_mask is None
        assert prediction.shape == target.shape == (hidden_count, volume.values.shape[0])
        assert torch.isfinite(prediction).all()
        assert torch.isfinite(target).all()
        torch.testing.assert_close(target.cpu(), expected_targets, rtol=0, atol=0)
        loss_calls.append((prediction.detach().cpu(), target.detach().cpu()))
        return real_shared_loss(prediction, target, loss_name=loss_name)

    monkeypatch.setattr(
        observed_training,
        "masked_trace_loss",
        recorded_shared_loss,
    )
    model = _model()

    def unexpected_validation() -> None:
        pytest.fail("fixed-step PoC training must not enter a validation/eval path")

    monkeypatch.setattr(model, "eval", unexpected_validation)
    reports: list[str] = []

    result = observed_training.train_ccnet5d_observed_steps(
        model,
        source,
        device="cpu",
        optimizer_updates=3,
        learning_rate=1.0e-3,
        report_every_steps=2,
        reporter=reports.append,
        loss_name=loss_name,
    )

    assert sample_count == 3
    assert len(loss_calls) == 3
    assert result.steps_completed == 3
    assert result.supervised_trace_presentations == 3 * hidden_count
    assert np.isfinite(result.final_loss)
    assert [entry["step"] for entry in result.history] == [2, 3]
    assert all(set(entry) == {"step", "loss", "learning_rate"} for entry in result.history)
    assert len(reports) == 2
    parameter_names = inspect.signature(observed_training.train_ccnet5d_observed_steps).parameters
    assert not {"validation", "early_stopping", "best_checkpoint"} & set(parameter_names)


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
@pytest.mark.parametrize("decay", [None, 0.999])
def test_final_checkpoint_round_trip_binds_observed_only_training_metadata(
    loss_name,
    decay,
    tmp_path: Path,
) -> None:
    model = _model()
    values = torch.randn(1, 1, 4, 1, 1, 1, 4)
    expected_prediction = model(values).detach().clone()
    saved_state = copy.deepcopy(model.state_dict())
    path = tmp_path / "final.pt"

    save_ccnet5d_poc_checkpoint(
        path,
        model,
        loss=loss_name,
        ema_decay=decay,
        amplitude_scale=2.75,
        patch_shape=(4, 1, 1, 1, 4),
        inner_mask_fraction=0.5,
        placement_seed=23,
        inner_mask_seed=401,
        model_initialization_seed=101,
        optimizer_updates=3,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) - ({"ema"} if decay is not None else set()) == {
        "model_type",
        "method_variant",
        "model_config",
        "state_dict",
        "axis_order",
        "normalization",
        "patches",
        "optimizer_updates",
        "model_initialization_seed",
        "loss",
        "checkpoint_role",
    }
    assert payload["model_type"] == "ccnet5d"
    assert payload["method_variant"] == CCNET5D_POC_METHOD_VARIANT
    assert payload["axis_order"] == list(VOLUME_AXIS_ORDER)
    assert payload["normalization"] == {
        "type": "global_rms",
        "source": "O_only",
        "scale": 2.75,
    }
    assert payload["patches"] == {
        "shape": [4, 1, 1, 1, 4],
        "inner_mask_fraction": 0.5,
        "placement_seed": 23,
        "inner_mask_seed": 401,
    }
    assert payload["optimizer_updates"] == 3
    assert payload["loss"] == loss_name
    assert payload["checkpoint_role"] == CCNET5D_POC_CHECKPOINT_ROLE == "final"
    assert "optimizer_state" not in payload
    assert "optimizer_state_dict" not in payload
    for name, tensor in payload["state_dict"].items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, saved_state[name], rtol=0, atol=0)

    loaded = load_ccnet5d_poc_checkpoint(path, device="cpu")
    assert loaded.ema_decay == decay
    if decay is not None:
        payload["ema"]["prediction_weights"] = "final_raw"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="ema"):
            load_ccnet5d_poc_checkpoint(path)

    assert loaded.loss == loss_name
    assert loaded.amplitude_scale == 2.75
    assert loaded.patch_shape == (4, 1, 1, 1, 4)
    assert loaded.inner_mask_fraction == 0.5
    assert loaded.placement_seed == 23
    assert loaded.optimizer_updates == 3
    assert loaded.model.constructor_config() == model.constructor_config()
    torch.testing.assert_close(loaded.model(values), expected_prediction, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("checkpoint_role", "best", "checkpoint_role"),
        ("loss", "unknown", "loss"),
        (
            "ema",
            {
                "decay": 1,
                "initialization": "first_post_update_weights",
                "prediction_weights": "final_ema",
            },
            "ema_decay",
        ),
        (
            "ema",
            {
                "decay": None,
                "initialization": "first_post_update_weights",
                "prediction_weights": "final_ema",
            },
            "ema",
        ),
        (
            "normalization",
            {"type": "global_rms", "source": "all_traces", "scale": 2.75},
            "O-only global RMS",
        ),
    ],
)
def test_checkpoint_rejects_nonfinal_or_non_observed_normalization_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
    match: str,
) -> None:
    path = tmp_path / "final.pt"
    save_ccnet5d_poc_checkpoint(
        path,
        _model(),
        amplitude_scale=2.75,
        patch_shape=(4, 1, 1, 1, 4),
        inner_mask_fraction=0.5,
        placement_seed=23,
        inner_mask_seed=401,
        model_initialization_seed=101,
        optimizer_updates=3,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        load_ccnet5d_poc_checkpoint(path)


@pytest.mark.parametrize("key", ["model_initialization_seed", "placement_seed", "inner_mask_seed"])
@pytest.mark.parametrize("value", [None, True, -1, 1.5])
def test_checkpoint_requires_explicit_nonnegative_seed_integers(tmp_path, key, value):
    path = tmp_path / "final.pt"
    save_ccnet5d_poc_checkpoint(
        path,
        _model(),
        amplitude_scale=2.75,
        patch_shape=(4, 1, 1, 1, 4),
        inner_mask_fraction=0.5,
        placement_seed=301,
        inner_mask_seed=401,
        model_initialization_seed=101,
        optimizer_updates=3,
    )
    payload = torch.load(path, weights_only=True)
    section = payload if key == "model_initialization_seed" else payload["patches"]
    section[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=key):
        load_ccnet5d_poc_checkpoint(path)
