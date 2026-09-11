"""Unit tests for the composite trace-graph reconstruction losses."""

from __future__ import annotations

import pytest
import torch

from seis_interp.training.trace_graph_losses import (
    amplitude_envelope_loss,
    masked_mean_square,
    slope_consistency_loss,
    spectrum_loss,
    trace_graph_training_errors_and_loss,
)

BATCH = 2
TIME = 64


def _gather(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.randn(BATCH, 8, 68, TIME, generator=generator)
    availability = torch.rand(BATCH, 8, 68, generator=generator) > 0.2
    return target, availability


def test_masked_mean_square_matches_manual_average() -> None:
    generator = torch.Generator().manual_seed(0)
    target, availability = _gather(generator)
    prediction = target + 1.0
    value = masked_mean_square(prediction, target, availability)
    expected = torch.square(prediction[availability] - target[availability]).mean()
    assert torch.allclose(value, expected)


def test_masked_mean_square_ignores_masked_rows() -> None:
    generator = torch.Generator().manual_seed(1)
    target, availability = _gather(generator)
    prediction = target.clone()
    corrupted = prediction.clone()
    corrupted[~availability] = 1.0e6
    assert masked_mean_square(prediction, target, availability).item() == 0.0
    assert masked_mean_square(corrupted, target, availability).item() == 0.0


def test_spectrum_loss_is_zero_for_identical_gathers() -> None:
    generator = torch.Generator().manual_seed(2)
    target, availability = _gather(generator)
    assert spectrum_loss(target, target, availability).item() == pytest.approx(0.0, abs=1.0e-6)


def test_spectrum_loss_penalizes_time_shift_through_phase() -> None:
    generator = torch.Generator().manual_seed(3)
    target, availability = _gather(generator)
    shifted = torch.roll(target, shifts=5, dims=-1)
    value = spectrum_loss(shifted, target, availability)
    assert value.item() > 0.1


def test_slope_consistency_loss_is_zero_for_identical_gathers() -> None:
    generator = torch.Generator().manual_seed(4)
    target, availability = _gather(generator)
    value = slope_consistency_loss(target, target, availability)
    assert value.item() == pytest.approx(0.0, abs=1.0e-9)


def test_slope_consistency_loss_penalizes_wrong_moveout() -> None:
    time_axis = torch.arange(TIME, dtype=torch.float32)
    receiver_axis = torch.arange(68, dtype=torch.float32)
    plane_wave = torch.sin(0.4 * (time_axis[None, :] - 0.8 * receiver_axis[:, None]))
    target = plane_wave[None, None].expand(1, 8, 68, TIME).contiguous()
    flipped = target.flip(dims=(2,))
    availability = torch.ones(1, 8, 68, dtype=torch.bool)
    matched = slope_consistency_loss(target, target, availability)
    mismatched = slope_consistency_loss(flipped, target, availability)
    assert mismatched.item() > 10.0 * matched.item()
    assert mismatched.item() > 0.01


def test_amplitude_envelope_loss_penalizes_scale_not_phase() -> None:
    generator = torch.Generator().manual_seed(5)
    target, availability = _gather(generator)
    scaled = 2.0 * target
    shifted = torch.roll(target, shifts=3, dims=-1)
    scaled_value = amplitude_envelope_loss(scaled, target, availability)
    shifted_value = amplitude_envelope_loss(shifted, target, availability)
    assert scaled_value.item() > 0.1
    assert shifted_value.item() < 0.5 * scaled_value.item()


def test_amplitude_envelope_loss_is_zero_for_identical_gathers() -> None:
    generator = torch.Generator().manual_seed(6)
    target, availability = _gather(generator)
    value = amplitude_envelope_loss(target, target, availability)
    assert value.item() == pytest.approx(0.0, abs=1.0e-9)


def test_losses_ignore_masked_rows() -> None:
    generator = torch.Generator().manual_seed(7)
    target, availability = _gather(generator)
    prediction = target + 0.1
    corrupted_prediction = prediction.clone()
    corrupted_prediction[~availability] = 1.0e4
    corrupted_target = target.clone()
    corrupted_target[~availability] = -1.0e4
    for loss in (masked_mean_square, spectrum_loss, amplitude_envelope_loss):
        clean = loss(prediction, target, availability)
        corrupted = loss(corrupted_prediction, corrupted_target, availability)
        assert torch.allclose(clean, corrupted), loss.__name__


def test_slope_loss_requires_receiver_pairs() -> None:
    generator = torch.Generator().manual_seed(8)
    target, _availability = _gather(generator)
    sparse = torch.zeros(BATCH, 8, 68, dtype=torch.bool)
    sparse[:, :, ::2] = True
    value = slope_consistency_loss(target + 1.0, target, sparse)
    assert value.item() == 0.0


def test_rejects_shape_mismatch() -> None:
    generator = torch.Generator().manual_seed(9)
    target, availability = _gather(generator)
    with pytest.raises(ValueError, match="target shape"):
        masked_mean_square(target[..., :-1], target, availability)
    with pytest.raises(ValueError, match="target_availability"):
        masked_mean_square(target, target, availability[:, :-1])


def test_rejects_non_boolean_mask() -> None:
    generator = torch.Generator().manual_seed(10)
    target, availability = _gather(generator)
    with pytest.raises(TypeError, match="torch.bool"):
        spectrum_loss(target, target, availability.float())


def test_rejects_empty_selection() -> None:
    generator = torch.Generator().manual_seed(11)
    target, _availability = _gather(generator)
    empty = torch.zeros(BATCH, 8, 68, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one trace"):
        masked_mean_square(target, target, empty)
    with pytest.raises(ValueError, match="at least one trace"):
        spectrum_loss(target, target, empty)


def test_trace_training_default_preserves_exact_errors_objective_gradient_and_rng() -> None:
    prediction = torch.tensor([[1.3, -0.4], [2.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.4, -0.1], [1.0, 0.2]])
    expected_errors = (prediction - target).square()
    expected = expected_errors.mean()
    expected.backward()
    gradient = prediction.grad.clone()
    prediction.grad = None
    rng = torch.get_rng_state().clone()
    errors, actual = trace_graph_training_errors_and_loss(prediction, target)
    assert errors.dtype == actual.dtype == torch.float32
    assert torch.equal(errors, expected_errors) and torch.equal(actual, expected)
    actual.backward()
    assert torch.equal(prediction.grad, gradient)
    assert torch.equal(torch.get_rng_state(), rng)


def test_relative_trace_loss_weights_teacher_rms_but_retains_input_unit_errors() -> None:
    target = torch.tensor([[1.0, -1.0], [10.0, -10.0], [0.0, 0.0]])
    prediction = torch.tensor([[0.5, -0.5], [5.0, -5.0], [2.0, -2.0]], requires_grad=True)
    errors, objective = trace_graph_training_errors_and_loss(
        prediction, target, loss="masked_trace_relative_mse"
    )
    expected_errors = (prediction.double() - target.double()).square()
    assert torch.equal(errors, expected_errors)
    assert objective.item() == pytest.approx((0.25 + 0.25 + 4) / 3)
    assert errors.mean().item() == pytest.approx((0.25 + 25 + 4) / 3)
    objective.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == prediction.numel()


def test_relative_trace_loss_checks_finiteness_only_in_shared_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = []
    torch_isfinite = torch.isfinite

    def counted_isfinite(value: torch.Tensor) -> torch.Tensor:
        checked.append(value)
        return torch_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    prediction = torch.tensor([[0.5, -0.5]])
    target = torch.tensor([[1.0, -1.0]])

    trace_graph_training_errors_and_loss(
        prediction,
        target,
        loss="masked_trace_relative_mse",
    )
    assert 1 <= len(checked) <= 2


def test_relative_trace_weights_are_detached_from_teacher_gradient() -> None:
    target = torch.tensor([[2.0, -2.0], [3.0, -3.0]], requires_grad=True)
    prediction = torch.zeros_like(target, requires_grad=True)
    _, objective = trace_graph_training_errors_and_loss(
        prediction, target, loss="masked_trace_relative_mse"
    )
    objective.backward()
    expected = 2 * target.detach() / torch.tensor([[4.0], [9.0]]) / target.numel()
    torch.testing.assert_close(target.grad, expected)
    torch.testing.assert_close(prediction.grad, -expected)


@pytest.mark.parametrize("scale", [0.1, 3.0, 1e-30, 1e30])
def test_relative_trace_objective_is_scale_invariant_for_nonzero_teachers(scale) -> None:
    target = torch.tensor([[1.0, -2.0], [4.0, -3.0]])
    prediction = target * 0.3
    _, base = trace_graph_training_errors_and_loss(
        prediction, target, loss="masked_trace_relative_mse"
    )
    scaled = (prediction * scale).requires_grad_()
    errors, actual = trace_graph_training_errors_and_loss(
        scaled, target * scale, loss="masked_trace_relative_mse"
    )
    assert torch.isfinite(errors).all() and torch.isfinite(actual)
    torch.testing.assert_close(actual, base, rtol=1e-6, atol=1e-8)
    actual.backward()
    assert torch.isfinite(scaled.grad).all()


def test_relative_trace_loss_handles_subnormal_float32_teacher_without_rms_underflow() -> None:
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = torch.tensor([[tiny, -tiny]])
    errors, objective = trace_graph_training_errors_and_loss(
        torch.zeros_like(target), target, loss="masked_trace_relative_mse"
    )
    assert (errors > 0).all()
    assert objective.item() == pytest.approx(1.0)


@pytest.mark.parametrize("loss", [None, True, 0, "relative", "masked_trace_relative_mse "])
def test_trace_training_loss_rejects_unknown_modes_before_tensor_operations(loss) -> None:
    with pytest.raises(ValueError, match="loss must be"):
        trace_graph_training_errors_and_loss(None, None, loss=loss)


@pytest.mark.parametrize(
    "prediction,target,error,match",
    [
        (None, torch.zeros(1, 2), TypeError, "floating"),
        (torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.int64), TypeError, "floating"),
        (torch.zeros(2), torch.zeros(2), ValueError, "shape"),
        (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), ValueError, "two-dimensional"),
        (torch.zeros(0, 2), torch.zeros(0, 2), ValueError, "at least one trace"),
        (torch.zeros(1, 2), torch.zeros(2, 2), ValueError, "shape"),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1), ValueError, "finite"),
        (torch.zeros(1, 1), torch.tensor([[float("inf")]]), ValueError, "finite"),
    ],
)
def test_relative_trace_loss_validates_unpadded_query_tensors(prediction, target, error, match):
    with pytest.raises(error, match=match):
        trace_graph_training_errors_and_loss(prediction, target, loss="masked_trace_relative_mse")
