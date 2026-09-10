from __future__ import annotations

import pytest
import torch

from seis_interp.training.trace_relative_loss import masked_trace_relative_mse


def test_matches_mean_of_per_trace_relative_mse() -> None:
    target = torch.tensor([[1.0, -1.0], [2.0, 2.0]], dtype=torch.float64)
    prediction = torch.tensor([[0.0, -2.0], [1.0, 4.0]], dtype=torch.float64)

    actual = masked_trace_relative_mse(prediction, target)

    first_trace = (1.0 + 1.0) / 2.0 / 1.0**2
    second_trace = (1.0 + 4.0) / 2.0 / 2.0**2
    assert actual.item() == pytest.approx((first_trace + second_trace) / 2.0)


def test_trace_energy_does_not_change_equal_relative_error_weighting() -> None:
    target = torch.tensor([[1.0, -1.0], [1.0e6, -1.0e6]])
    prediction = target * 0.5

    assert masked_trace_relative_mse(prediction, target).item() == pytest.approx(0.25)


@pytest.mark.parametrize("scale", [0.1, 3.0, 1.0e-30, 1.0e30])
def test_nonzero_traces_are_invariant_to_joint_scaling(scale: float) -> None:
    target = torch.tensor([[1.0, -2.0], [4.0, -3.0]])
    prediction = target * 0.3
    expected = masked_trace_relative_mse(prediction, target)

    actual = masked_trace_relative_mse(prediction * scale, target * scale)

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-8)


def test_boolean_mask_selects_complete_traces_before_subtraction() -> None:
    target = torch.tensor(
        [
            [[1.0, -1.0], [float("nan"), float("nan")]],
            [[2.0, -2.0], [float("nan"), float("nan")]],
        ]
    )
    prediction = torch.tensor(
        [
            [[0.5, -0.5], [float("nan"), float("nan")]],
            [[1.0, -1.0], [float("nan"), float("nan")]],
        ]
    )
    trace_mask = torch.tensor([[True, False], [True, False]])

    actual = masked_trace_relative_mse(prediction, target, trace_mask)

    assert actual.item() == pytest.approx(0.25)


def test_zero_energy_target_uses_divisor_one() -> None:
    target = torch.zeros(2, 2)
    prediction = torch.tensor([[2.0, -2.0], [1.0, -1.0]])

    actual = masked_trace_relative_mse(prediction, target)

    assert actual.item() == pytest.approx((4.0 + 1.0) / 2.0)


def test_only_target_rms_is_detached_from_gradient() -> None:
    target = torch.tensor([[2.0, -2.0], [3.0, -3.0]], requires_grad=True)
    prediction = torch.zeros_like(target, requires_grad=True)

    loss = masked_trace_relative_mse(prediction, target)
    loss.backward()

    expected_target_gradient = 2 * target.detach() / torch.tensor([[4.0], [9.0]]) / target.numel()
    torch.testing.assert_close(target.grad, expected_target_gradient)
    torch.testing.assert_close(prediction.grad, -expected_target_gradient)
    assert torch.isfinite(prediction.grad).all()


def test_extreme_and_subnormal_float32_values_remain_finite() -> None:
    maximum = torch.finfo(torch.float32).max
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = torch.tensor([[maximum, -maximum], [tiny, -tiny]])
    prediction = -target

    actual = masked_trace_relative_mse(prediction, target)

    assert torch.isfinite(actual)
    assert actual.item() == pytest.approx(4.0)


@pytest.mark.parametrize(
    "prediction,target,error,match",
    [
        (None, torch.zeros(2), TypeError, "prediction"),
        (torch.zeros(2), torch.zeros(2, dtype=torch.int64), TypeError, "target"),
        (torch.zeros(2), torch.zeros(1, 2), ValueError, "matching shapes"),
        (torch.zeros(()), torch.zeros(()), ValueError, "time-last"),
        (torch.zeros(2, 0), torch.zeros(2, 0), ValueError, "time-last"),
        (torch.tensor([float("nan")]), torch.zeros(1), ValueError, "finite"),
    ],
)
def test_rejects_invalid_trace_pairs(prediction, target, error, match) -> None:
    with pytest.raises(error, match=match):
        masked_trace_relative_mse(prediction, target)


@pytest.mark.parametrize(
    "trace_mask,error,match",
    [
        ([[True, False]], TypeError, "torch.Tensor"),
        (torch.ones(1, 2), TypeError, "torch.bool"),
        (torch.ones(2, dtype=torch.bool), ValueError, "shape"),
        (torch.zeros(1, 2, dtype=torch.bool), ValueError, "at least one trace"),
    ],
)
def test_rejects_invalid_or_empty_trace_masks(trace_mask, error, match) -> None:
    values = torch.zeros(1, 2, 3)
    with pytest.raises(error, match=match):
        masked_trace_relative_mse(values, values, trace_mask)


def test_one_dimensional_input_is_one_complete_trace() -> None:
    target = torch.tensor([1.0, -1.0])
    prediction = torch.zeros_like(target)

    assert masked_trace_relative_mse(prediction, target).item() == pytest.approx(1.0)
