"""Linear edge-value delays: sign, zero boundaries, derivatives and bandwidth."""

from __future__ import annotations

import math

import pytest
import torch

from seis_interp.models.trace_graph_time_shift import shift_trace_graph_values


@pytest.mark.parametrize(
    "shift,expected",
    [
        (0.0, [1.0, 2.0, 4.0, 8.0]),
        (1.0, [2.0, 4.0, 8.0, 0.0]),
        (-1.0, [0.0, 1.0, 2.0, 4.0]),
        (0.5, [1.5, 3.0, 6.0, 4.0]),
        (-0.5, [0.5, 1.5, 3.0, 6.0]),
        (4.0, [0.0, 0.0, 0.0, 0.0]),
        (-4.0, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_signed_integer_fractional_and_outside_shifts_have_no_wrap(shift, expected) -> None:
    values = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
    original = values.clone()
    actual = shift_trace_graph_values(values, torch.tensor([shift]))
    assert torch.equal(actual, torch.tensor([[expected]]))
    assert torch.equal(values, original)


def test_every_channel_uses_its_own_edge_shift() -> None:
    values = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]]).repeat(2, 1, 1)
    actual = shift_trace_graph_values(values, torch.tensor([1.0, -1.0]))
    expected = torch.tensor(
        [[[2.0, 3.0, 0.0], [20.0, 30.0, 0.0]], [[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]]]
    )
    assert torch.equal(actual, expected)


def test_zero_shift_preserves_input_gradient_and_has_finite_nonzero_lag_gradient() -> None:
    values = torch.tensor([[[2.0, 4.0, 8.0]]], dtype=torch.float64, requires_grad=True)
    shifts = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    output = shift_trace_graph_values(values, shifts)
    weights = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    assert torch.equal(output, values)
    (output * weights).sum().backward()
    assert torch.equal(values.grad, weights)
    # Right-hand derivative: (4-2)*1 + (8-4)*2 + (0-8)*3.
    assert torch.equal(shifts.grad, torch.tensor([-14.0], dtype=torch.float64))


def test_fractional_shifts_pass_gradcheck() -> None:
    values = torch.arange(20, dtype=torch.float64).reshape(2, 2, 5).requires_grad_()
    shifts = torch.tensor([0.25, -0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(shift_trace_graph_values, (values, shifts))


@pytest.mark.parametrize("frequency_hz", [5.0, 21.0, 27.34375])
def test_half_frame_shift_has_the_documented_frequency_attenuation(frequency_hz) -> None:
    # Current factor-two latent frames are 16 ms apart. This checks the
    # interpolation operator on pure sinusoids, not measured learned latents.
    omega = 2 * math.pi * frequency_hz * 0.016
    frame = torch.arange(128, dtype=torch.float64)
    values = torch.sin(omega * frame + 0.2)[None, None]
    actual = shift_trace_graph_values(values, torch.tensor([0.5], dtype=torch.float64))
    attenuation = math.cos(omega / 2)
    ideal_delay = torch.sin(omega * (frame + 0.5) + 0.2)
    torch.testing.assert_close(actual[0, 0, :-1], attenuation * ideal_delay[:-1])
    assert 0 < attenuation < 1
    if frequency_hz == 27.34375:
        assert math.isclose(attenuation, 0.19509032201612833)


def test_one_frame_and_empty_edge_batches_are_supported() -> None:
    result = shift_trace_graph_values(torch.tensor([[[8.0]]]), torch.tensor([0.5]))
    assert torch.equal(result, torch.tensor([[[4.0]]]))
    empty = shift_trace_graph_values(torch.empty(0, 3, 4), torch.empty(0))
    assert empty.shape == (0, 3, 4)


def test_edge_order_batch_splits_and_positive_gain_preserve_the_operator() -> None:
    values = torch.arange(30, dtype=torch.float64).reshape(3, 2, 5)
    shifts = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
    output = shift_trace_graph_values(values, shifts)
    order = torch.tensor([2, 0, 1])
    assert torch.equal(shift_trace_graph_values(values[order], shifts[order]), output[order])
    separate = torch.cat(
        [shift_trace_graph_values(values[i : i + 1], shifts[i : i + 1]) for i in range(3)]
    )
    assert torch.equal(separate, output)
    assert torch.equal(shift_trace_graph_values(3 * values, shifts), 3 * output)


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_shifts_are_rejected(shift) -> None:
    with pytest.raises(ValueError, match="finite"):
        shift_trace_graph_values(torch.ones(1, 2, 4), torch.tensor([shift]))


@pytest.mark.parametrize(
    "values,shifts,error,match",
    [
        (None, torch.zeros(1), TypeError, "Tensor"),
        (torch.ones(1, 3), torch.zeros(1), ValueError, "shape"),
        (torch.ones(1, 2, 0), torch.zeros(1), ValueError, "shape"),
        (torch.ones(1, 2, 3), torch.zeros(1, 1), ValueError, "shape"),
        (torch.ones(1, 2, 3), torch.zeros(2), ValueError, "shape"),
        (torch.ones(1, 2, 3, dtype=torch.int64), torch.zeros(1), TypeError, "floating"),
        (torch.ones(1, 2, 3), torch.zeros(1, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(1, 2, 3), torch.zeros(1, dtype=torch.float64), ValueError, "dtype"),
    ],
)
def test_invalid_tensor_contracts_are_rejected(values, shifts, error, match) -> None:
    with pytest.raises(error, match=match):
        shift_trace_graph_values(values, shifts)
