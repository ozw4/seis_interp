from copy import deepcopy

import numpy as np
import pytest
import torch

from seis_interp.models import ccnet5d_trace_queries as queries_module
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.models.ccnet5d_trace_queries import (
    forward_ccnet5d_trace_queries,
    prepare_ccnet5d_trace_queries,
)
from seis_interp.training.ccnet5d_query_training import backward_ccnet5d_trace_queries
from seis_interp.training.trace_relative_loss import masked_trace_loss


@pytest.mark.parametrize("dilation,time", [(1, 7), (2, 9), (3, 5)])
@pytest.mark.parametrize("recompute", [False, True])
def test_spatial_queries_match_full_patch_outputs_and_all_gradients(dilation, time, recompute):
    torch.manual_seed(123)
    model = CCNet5D(
        hidden_channels=2,
        intermediate_channels=3,
        kernel_size=3,
        time_kernel_size=3,
        time_dilation=dilation,
        output_activation="linear",
    ).double()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("bias"):
                value.fill_(0.25)
    reference = deepcopy(model)
    spatial = (2, 3, 2, 3)
    inputs = torch.randn(2, 1, time, *spatial, dtype=torch.float64, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    mask = np.zeros((2, *spatial), dtype=bool)
    mask[0, 0, 0, 0, 0] = mask[0, -1, 1, 1, 1] = mask[1, -1, -1, -1, -1] = True
    plan = prepare_ccnet5d_trace_queries(model, mask, device="cpu")
    cells = inputs[:, 0].movedim(1, -1).reshape(-1, time)[plan.input_cells, None]
    prediction = forward_ccnet5d_trace_queries(
        model,
        cells,
        plan,
        spatial_chunk_size=13,
        recompute=recompute,
    )
    expected = reference(other)[:, 0].movedim(1, -1)[mask]
    torch.testing.assert_close(prediction, expected, rtol=1e-11, atol=1e-12)
    prediction.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, other.grad, rtol=1e-10, atol=1e-12)
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
def test_query_backward_uses_complete_time_for_every_spatial_chunk(monkeypatch, loss_name):
    torch.manual_seed(41)
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=3).double()
    reference = deepcopy(model)
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(1, 11, 2, 2, 2, 2))
    targets = rng.normal(size=inputs.shape)
    mask = np.zeros((1, 2, 2, 2, 2), dtype=bool)
    mask.reshape(-1)[[0, 7, 15]] = True
    targets[np.broadcast_to(~mask[:, None], targets.shape)] = np.nan
    times = []
    original = queries_module._convolve_cells

    def record(values, *args):
        times.append(values.shape[-1])
        return original(values, *args)

    monkeypatch.setattr(queries_module, "_convolve_cells", record)
    loss = backward_ccnet5d_trace_queries(
        model,
        inputs,
        targets,
        mask,
        device="cpu",
        spatial_chunk_size=3,
        loss_name=loss_name,
    )
    prediction = reference(torch.as_tensor(inputs)[:, None])[:, 0].movedim(1, -1)[mask]
    target = torch.as_tensor(targets).movedim(1, -1)[mask]
    expected = masked_trace_loss(prediction, target, loss_name=loss_name)
    expected.backward()
    assert times and set(times) == {11}
    assert loss == pytest.approx(expected.item(), rel=1e-12)
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "mask",
    [np.zeros((1, 2, 2, 2, 2), dtype=bool), np.ones((2, 2), dtype=bool), np.ones((1, 2, 2, 2, 2))],
)
def test_query_plan_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="nonempty Boolean"):
        prepare_ccnet5d_trace_queries(CCNet5D(), mask, device="cpu")
