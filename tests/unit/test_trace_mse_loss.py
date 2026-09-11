import pytest
import torch

from seis_interp.training.trace_relative_loss import (
    masked_trace_loss,
    masked_trace_mse,
    masked_trace_relative_mse,
)


def test_hand_computed_time_then_trace_mean_and_gradient():
    prediction = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    objective = masked_trace_mse(prediction, target)
    assert objective.dtype == torch.float64
    assert objective.item() == ((1 + 4) / 2 + (9 + 16) / 2) / 2
    objective.backward()
    torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - target) / 4)


def test_whole_trace_selection_ignores_nonfinite_unselected_rows_and_gradients():
    prediction = torch.tensor([[[1.0, 3.0], [torch.nan, torch.inf]]], requires_grad=True)
    target = torch.tensor([[[0.0, 1.0], [torch.inf, torch.nan]]])
    objective = masked_trace_mse(prediction, target, torch.tensor([[True, False]]))
    assert objective.item() == 2.5
    objective.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("side", ["prediction", "target"])
def test_selected_nonfinite_traces_are_rejected(value, side):
    pair = {"prediction": torch.zeros(2, 3), "target": torch.zeros(2, 3)}
    pair[side][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        masked_trace_mse(**pair, trace_mask=torch.tensor([True, False]))


@pytest.mark.parametrize(
    "prediction,target,mask,error",
    [
        (None, torch.zeros(2), None, TypeError),
        (torch.zeros(2), torch.ones(2, dtype=torch.int64), None, TypeError),
        (torch.zeros(2), torch.zeros(2, dtype=torch.float64), None, TypeError),
        (torch.zeros(2), torch.zeros(3), None, ValueError),
        (torch.zeros(2), torch.zeros(2, device="meta"), None, ValueError),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool), ValueError),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2), TypeError),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(3, dtype=torch.bool), ValueError),
        (
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.ones(2, dtype=torch.bool, device="meta"),
            ValueError,
        ),
        (torch.zeros(0, 3), torch.zeros(0, 3), None, ValueError),
    ],
)
def test_rejects_invalid_pair_or_mask(prediction, target, mask, error):
    with pytest.raises(error):
        masked_trace_mse(prediction, target, mask)


def test_float64_residual_avoids_float32_square_overflow():
    target = torch.tensor([[torch.finfo(torch.float32).max, 0.0]])
    objective = masked_trace_mse(-target, target)
    assert objective.dtype == torch.float64 and torch.isfinite(objective)


def test_mse_does_not_reweight_traces_by_target_energy():
    target = torch.tensor([[1.0, -1.0], [10.0, -10.0]])
    prediction = target / 2
    assert masked_trace_mse(prediction, target).item() == 12.625
    assert masked_trace_relative_mse(prediction, target).item() == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["masked_mse", "mse", "unknown"])
def test_dispatch_rejects_unknown_loss(name):
    with pytest.raises(ValueError, match="loss_name"):
        masked_trace_loss(torch.ones(2, 3), torch.ones(2, 3), loss_name=name)
