import json

import pytest
import torch

from seis_interp.training.nersi_loss_support import recorded_trace_samples
from seis_interp.training.neural_diagnostics import NeuralTrainingDiagnostics


@pytest.mark.parametrize("scale,status", [(0.5, "passed"), (0.0, "failed"), (0.1, "failed")])
def test_g0_uses_exact_zero_loss_and_only_updates_901_to_1000(scale, status):
    diagnostics = NeuralTrainingDiagnostics(1100, "masked_trace_mse")
    target = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])
    for step in range(1, 1101):
        prediction = target * (scale if 901 <= step <= 1000 else 20)
        loss = (prediction.double() - target.double()).square().mean()
        diagnostics.observe(step, prediction, target, loss)
    record = diagnostics.records()["g0"]
    exact = target.double().square().mean().item()
    assert record["zero_predictor_loss"] == pytest.approx(exact, rel=1e-15)
    assert record["output_rms_over_target_rms"] == pytest.approx(scale)
    assert record["loss_over_L0"] == pytest.approx((1 - scale) ** 2)
    assert (record["first_update"], record["last_update"], record["updates"]) == (901, 1000, 100)
    assert record["status"] == status
    json.dumps(record, allow_nan=False)


def test_short_probe_is_not_a_g0_pass_and_gradient_diagnostic_never_clips():
    diagnostics = NeuralTrainingDiagnostics(2, "masked_trace_mse")
    model = torch.nn.Linear(1, 1)
    for step, gradient in enumerate((0.1, 2.0), 1):
        model.weight.grad = torch.full_like(model.weight, gradient)
        before = model.weight.grad.clone()
        diagnostics.gradients(model)
        assert torch.equal(model.weight.grad, before)
        diagnostics.observe(step, torch.ones(1, 2), torch.ones(1, 2), torch.tensor(0.0))
    record = diagnostics.records()
    assert record["would_clip_fraction"] == 0.5
    assert record["g0"]["status"] == "insufficient_updates"


def test_recorded_samples_ignore_padding_and_keep_zero_samples_inside_support():
    prediction = torch.full((2, 7), 100.0, requires_grad=True)
    target = torch.zeros_like(prediction)
    selected, teacher = recorded_trace_samples(prediction, target, [1, 3], 3)
    assert selected.shape == teacher.shape == (2, 3)
    (selected - teacher).square().mean().backward()
    assert torch.equal(
        prediction.grad != 0,
        torch.tensor(
            [
                [False, True, True, True, False, False, False],
                [False, False, False, True, True, True, False],
            ]
        ),
    )
    whole, whole_target = recorded_trace_samples(prediction, target)
    assert whole is prediction and whole_target is target
