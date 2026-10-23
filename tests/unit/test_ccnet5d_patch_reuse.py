from copy import deepcopy

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training import ccnet5d_observed_training as training
from seis_interp.training import ccnet5d_query_training as query_training
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.trace_relative_loss import masked_trace_loss


def _source(observed_count=668):
    spatial = (1, 1, 1, observed_count + 1)
    mask = np.ones(spatial, dtype=bool)
    mask[..., -1] = False
    values = np.random.default_rng(17).uniform(0.1, 2, size=(7, *spatial)).astype(np.float32)
    values[:, ~mask] = np.nan
    volume = ObservedC3Volume(
        values=values,
        time_s=np.arange(7) * 0.004,
        array_rows=np.arange(mask.size).reshape(spatial),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    return CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=1,
        patch_shape=values.shape,
        inner_mask_fraction=0.5,
        placement_seed=7301,
        inner_mask_seed=7301,
    )


def test_patch_reuse_emits_five_disjoint_groups_from_334_hidden_traces():
    first, second = _source(), _source()
    groups = list(first.sample_reused_batches(64))
    replicas = list(second.sample_reused_batches(64))
    assert len(groups) == 5 and first.patches_drawn == second.patches_drawn == 1
    inputs, targets, _ = groups[0]
    hidden = (targets != 0).any(axis=1)
    assert hidden.sum() == 334
    used = np.zeros(hidden.shape, dtype=bool)
    for (x, y, mask), replica in zip(groups, replicas, strict=True):
        assert x is inputs and y is targets
        assert mask.sum() == 64 and not (used & mask).any()
        assert not mask[..., -1].any()
        assert np.all(np.moveaxis(inputs, 1, -1)[hidden] == 0)
        assert np.all(np.isfinite(np.moveaxis(targets, 1, -1)[mask]))
        used |= mask
        for actual, expected in zip((x, y, mask), replica, strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert used.sum() == 320 and (hidden & ~used).sum() == 14


def test_patch_reuse_stacks_small_patches_until_a_full_group_is_available():
    source = _source(3)
    groups = list(source.sample_reused_batches(64))
    assert len(groups) == 1 and groups[0][2].sum() == 64
    assert groups[0][0].shape == (32, 7, 1, 1, 1, 4)
    assert source.patches_drawn == 32


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_reused_groups_each_use_new_weights_and_one_optimizer_and_ema_step(monkeypatch, chunk_size):
    torch.manual_seed(7)
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=1)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.2)
    reference = deepcopy(model)
    actual_source, reference_source = _source(), _source()
    snapshots = []
    step = torch.optim.AdamW.step

    def record(optimizer, *args, **kwargs):
        result = step(optimizer, *args, **kwargs)
        snapshots.append([p.detach().clone() for p in model.parameters()])
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", record)
    result = training.train_ccnet5d_observed_steps(
        model,
        actual_source,
        device="cpu",
        optimizer_updates=5,
        learning_rate=0.001,
        report_every_steps=100,
        loss_name="masked_trace_mse",
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        reuse_patch=True,
        spatial_query_chunk_size=chunk_size,
        ema_decay=0.9,
    )
    assert len(snapshots) == 5 and actual_source.patches_drawn == 1
    assert result.steps_completed == 5 and result.supervised_trace_presentations == 320
    assert [entry["step"] for entry in result.history] == [1, 5]
    for before, after in zip(snapshots[:-1], snapshots[1:], strict=True):
        assert not torch.equal(before[-1], after[-1])
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0)
    expected_average = None
    for x, y, mask in reference_source.sample_reused_batches(64):
        optimizer.zero_grad(set_to_none=True)
        prediction = reference(torch.from_numpy(x)[:, None])[:, 0].movedim(1, -1)[mask]
        target = torch.from_numpy(y).movedim(1, -1)[mask]
        loss = masked_trace_loss(prediction, target, loss_name="masked_trace_mse")
        loss.backward()
        optimizer.step()
        if expected_average is None:
            expected_average = [p.detach().clone() for p in reference.parameters()]
        else:
            for average, p in zip(expected_average, reference.parameters(), strict=True):
                average.lerp_(p, 0.1)
    for actual, expected in zip(model.parameters(), expected_average, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_query_reuse_keeps_complete_time_fp32_weights_and_float64_loss(monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA required")
    model = CCNet5D(hidden_channels=2, intermediate_channels=3, kernel_size=1)
    original = query_training.masked_trace_loss
    forward = query_training.forward_ccnet5d_trace_queries
    dtypes = []
    forward_dtypes = []

    def record_forward(*args, **kwargs):
        prediction = forward(*args, **kwargs)
        forward_dtypes.append(prediction.dtype)
        return prediction

    def record(prediction, target, **kwargs):
        assert not torch.is_autocast_enabled()
        assert prediction.shape == target.shape == (64, 7)
        assert prediction.dtype == target.dtype == torch.float32
        objective = original(prediction, target, **kwargs)
        dtypes.append(objective.dtype)
        return objective

    monkeypatch.setattr(query_training, "masked_trace_loss", record)
    monkeypatch.setattr(query_training, "forward_ccnet5d_trace_queries", record_forward)
    result = training.train_ccnet5d_observed_steps(
        model,
        _source(),
        device="cuda",
        optimizer_updates=2,
        learning_rate=0.001,
        report_every_steps=1,
        loss_name="masked_trace_mse",
        optimizer_name="adamw",
        weight_decay=0,
        supervised_traces_per_update=64,
        reuse_patch=True,
        spatial_query_chunk_size=32,
        mixed_precision="bf16",
        ema_decay=0.9,
        ema_eval_decays=(0.99,),
    )
    assert dtypes == [torch.float64, torch.float64]
    assert forward_dtypes == [torch.bfloat16, torch.bfloat16]
    assert np.isfinite(result.final_loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in result.shadow_parameters[0.99])


def test_patch_reuse_requires_an_explicit_trace_budget():
    with pytest.raises(ValueError, match="requires supervised_traces_per_update"):
        training.train_ccnet5d_observed_steps(
            CCNet5D(),
            _source(),
            device="cpu",
            optimizer_updates=1,
            learning_rate=0.001,
            report_every_steps=1,
            reuse_patch=True,
        )
