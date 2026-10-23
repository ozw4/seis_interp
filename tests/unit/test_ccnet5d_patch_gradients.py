from copy import deepcopy
from itertools import islice

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training import ccnet5d_observed_training as training
from seis_interp.training import ccnet5d_patch_gradients as patch_gradients
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.trace_relative_loss import masked_trace_loss


def _source():
    shape = (7, 1, 1, 1, 669)
    mask = np.ones(shape[1:], dtype=bool)
    mask[..., -1] = False
    volume = ObservedC3Volume(
        values=np.random.default_rng(41).uniform(0.1, 2, size=shape).astype(np.float32),
        time_s=np.arange(7) * 0.004,
        array_rows=np.arange(mask.size).reshape(mask.shape),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    return CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=1,
        patch_shape=shape,
        inner_mask_fraction=0.5,
        placement_seed=7301,
        inner_mask_seed=7301,
    )


@pytest.mark.parametrize("updates", [3, 7])
def test_one_forward_saves_all_gradients_before_sequential_adamw_and_ema(monkeypatch, updates):
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=1).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.2)
    reference = deepcopy(model)
    events, patch_parameters = [], []

    def record_forward(module, args, output):
        assert args[0].shape == (1, 1, 7, 1, 1, 1, 669)
        events.append("forward")
        patch_parameters[:] = [p.detach().clone() for p in model.parameters()]

    def record_gradient(gradient):
        events.append("gradient")
        for parameter, before in zip(model.parameters(), patch_parameters, strict=True):
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        return gradient

    model.register_forward_hook(record_forward)
    next(model.parameters()).register_hook(record_gradient)
    step, advance_average = torch.optim.AdamW.step, training._advance_average

    def record_step(optimizer, *args, **kwargs):
        events.append("adamw")
        return step(optimizer, *args, **kwargs)

    def record_average(*args, **kwargs):
        events.append("ema")
        return advance_average(*args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    monkeypatch.setattr(training, "_advance_average", record_average)
    source = _source()
    result = training.train_ccnet5d_observed_steps(
        model,
        source,
        device="cpu",
        optimizer_updates=updates,
        learning_rate=0.01,
        report_every_steps=100,
        loss_name="masked_trace_mse",
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        reuse_patch=True,
        cache_patch_gradients=True,
        ema_decay=0.9,
    )
    group_counts = [updates] if updates < 5 else [5, updates - 5]
    expected_events = []
    for groups in group_counts:
        expected_events += ["forward"] + ["gradient"] * groups + ["adamw", "ema"] * groups
    assert events == expected_events
    assert result.steps_completed == updates
    assert result.supervised_trace_presentations == updates * 64
    assert source.patches_drawn == len(group_counts)
    assert result.history[-1]["patch_forwards_completed"] == len(group_counts)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0)
    expected_ema, remaining = None, updates
    reference_source = _source()
    while remaining:
        groups = list(islice(reference_source.sample_reused_batches(64), remaining))
        saved = []
        for x, y, mask in groups:
            frozen = deepcopy(reference)
            prediction = frozen(torch.as_tensor(x, dtype=torch.float64)[:, None])
            loss = masked_trace_loss(
                prediction[:, 0].movedim(1, -1)[mask],
                torch.as_tensor(y, dtype=torch.float64).movedim(1, -1)[mask],
                loss_name="masked_trace_mse",
            )
            loss.backward()
            saved.append(tuple(p.grad.clone() for p in frozen.parameters()))
        for gradients in saved:
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(reference.parameters(), gradients, strict=True):
                parameter.grad = gradient
            optimizer.step()
            if expected_ema is None:
                expected_ema = [p.detach().clone() for p in reference.parameters()]
            else:
                for average, p in zip(expected_ema, reference.parameters(), strict=True):
                    average.lerp_(p, 0.1)
        remaining -= len(groups)
    for actual, expected in zip(model.parameters(), expected_ema, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_bf16_forward_runs_once_and_preserves_fp32_optimizer_ema(monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA required")
    model = CCNet5D(hidden_channels=2, intermediate_channels=3, kernel_size=1)
    forward_dtypes, losses, events = [], [], []
    model.register_forward_hook(lambda m, args, output: forward_dtypes.append(output.dtype))
    loss_fn = patch_gradients.masked_trace_loss

    def record_loss(prediction, target, **kwargs):
        assert prediction.shape == target.shape == (64, 7)
        assert prediction.dtype == target.dtype == torch.float32
        assert not torch.is_autocast_enabled()
        loss = loss_fn(prediction, target, **kwargs)
        losses.append(loss.dtype)
        events.append("loss")
        return loss

    step = torch.optim.AdamW.step

    def record_step(optimizer, *args, **kwargs):
        events.append("step")
        assert all(p.grad.dtype == torch.float32 for p in model.parameters())
        result = step(optimizer, *args, **kwargs)
        assert all(
            value.dtype == torch.float32
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )
        return result

    monkeypatch.setattr(patch_gradients, "masked_trace_loss", record_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    result = training.train_ccnet5d_observed_steps(
        model,
        _source(),
        device="cuda",
        optimizer_updates=5,
        learning_rate=0.001,
        report_every_steps=1,
        loss_name="masked_trace_mse",
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        reuse_patch=True,
        cache_patch_gradients=True,
        mixed_precision="bf16",
        ema_decay=0.9,
        ema_eval_decays=(0.99,),
        ema_snapshot_updates=(3,),
    )
    assert forward_dtypes == [torch.bfloat16]
    assert losses == [torch.float64] * 5
    assert events == ["loss"] * 5 + ["step"] * 5
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in result.shadow_parameters[0.99])
    assert all(p.dtype == torch.float32 for p in result.shadow_snapshots[3][0.99])


@pytest.mark.parametrize(
    "options",
    [
        {"reuse_patch": False},
        {"reuse_patch": True, "spatial_query_chunk_size": 64},
    ],
)
def test_cached_gradients_require_shared_patch_and_full_model_forward(options):
    with pytest.raises(ValueError, match="requires reuse_patch and the full-patch forward"):
        training.train_ccnet5d_observed_steps(
            CCNet5D(),
            _source(),
            device="cpu",
            optimizer_updates=1,
            learning_rate=0.001,
            report_every_steps=1,
            supervised_traces_per_update=64,
            cache_patch_gradients=True,
            **options,
        )


def test_small_patches_share_one_forward_and_exact_64_trace_gradient(monkeypatch):
    shape = (11, 4, 6, 4, 12)
    observed = np.ones(shape[1:], dtype=bool)
    volume = ObservedC3Volume(
        values=np.random.default_rng(79).normal(size=shape).astype(np.float32),
        time_s=np.arange(shape[0]) * 0.004,
        array_rows=np.arange(observed.size).reshape(observed.shape),
        observed_trace_mask=observed,
        evaluation_target_trace_mask=~observed,
    )

    def source():
        return CCNet5DObservedPatchSource(
            volume,
            amplitude_scale=1,
            patch_shape=(11, 2, 3, 2, 4),
            inner_mask_fraction=0.5,
            placement_seed=17,
            inner_mask_seed=41,
        )

    x, y, mask = next(source().sample_reused_batches(64))
    assert x.shape == (3, 11, 2, 3, 2, 4) and mask.sum() == 64
    assert np.all(np.moveaxis(x, 1, -1)[mask] == 0)
    torch.manual_seed(101)
    model = CCNet5D(
        hidden_channels=2,
        intermediate_channels=3,
        kernel_size=3,
        time_dilation=2,
    ).double()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.2)
    reference = deepcopy(model)
    batches_seen, steps = [], []
    model.register_forward_hook(lambda m, args, output: batches_seen.append(args[0].shape))
    step = torch.optim.AdamW.step

    def record_step(optimizer, *args, **kwargs):
        steps.append(tuple(p.grad.detach().clone() for p in model.parameters()))
        return step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    training_source = source()
    result = training.train_ccnet5d_observed_steps(
        model,
        training_source,
        device="cpu",
        optimizer_updates=1,
        learning_rate=0.001,
        report_every_steps=1,
        loss_name="masked_trace_mse",
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        reuse_patch=True,
        cache_patch_gradients=True,
        ema_decay=0.9,
    )
    assert batches_seen == [(3, 1, 11, 2, 3, 2, 4)]
    assert training_source.patches_drawn == result.history[-1]["patch_forwards_completed"] == 3
    assert result.steps_completed == 1 and result.supervised_trace_presentations == 64
    assert len(steps) == 1

    expected = torch.zeros((), dtype=torch.float64)
    for index in range(3):
        prediction = reference(torch.as_tensor(x[index : index + 1], dtype=torch.float64)[:, None])
        rows = prediction[:, 0].movedim(1, -1)[mask[index : index + 1]]
        target = torch.as_tensor(y[index : index + 1], dtype=torch.float64).movedim(1, -1)[
            mask[index : index + 1]
        ]
        expected = expected + (rows - target).square().sum() / (64 * 11)
    expected.backward()
    assert result.final_loss == pytest.approx(expected.item(), rel=1e-12)
    for actual, parameter in zip(steps[0], reference.parameters(), strict=True):
        torch.testing.assert_close(actual, parameter.grad, rtol=1e-10, atol=1e-12)
