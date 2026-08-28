from __future__ import annotations

import pytest
import torch

from seis_interp.training.correlation_loss import trace_correlation_loss


def test_trace_correlation_loss_is_zero_for_matching_traces() -> None:
    target = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0]])

    assert trace_correlation_loss(target, target).item() == pytest.approx(0.0, abs=1.0e-4)


def test_trace_correlation_loss_ignores_positive_scale_and_offset() -> None:
    target = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0]])
    scale = torch.tensor([[2.5], [4.0]])
    offset = torch.tensor([[7.0], [-3.0]])
    prediction = scale * target + offset

    assert trace_correlation_loss(prediction, target).item() == pytest.approx(0.0, abs=1.0e-4)


def test_trace_correlation_loss_is_two_for_sign_reversal() -> None:
    target = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0]])

    assert trace_correlation_loss(-target, target).item() == pytest.approx(2.0, abs=1.0e-4)


def test_zero_prediction_has_finite_nonzero_gradient() -> None:
    target = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0]])
    prediction = torch.zeros_like(target, requires_grad=True)

    loss = trace_correlation_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.all(torch.isfinite(prediction.grad))
    assert torch.any(prediction.grad != 0.0)


def test_trace_correlation_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        trace_correlation_loss(torch.ones((2, 3)), torch.ones((2, 4)))


@pytest.mark.parametrize("shape", [(3,), (0, 3), (2, 0)])
def test_trace_correlation_loss_rejects_non_matrix_or_empty_input(
    shape: tuple[int, ...],
) -> None:
    values = torch.ones(shape)

    with pytest.raises(ValueError, match="two-dimensional|must not be empty"):
        trace_correlation_loss(values, values)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_trace_correlation_loss_rejects_invalid_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        trace_correlation_loss(torch.ones((1, 2)), torch.ones((1, 2)), eps=eps)
