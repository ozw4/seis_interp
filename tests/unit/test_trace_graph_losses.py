"""Unit tests for the composite trace-graph reconstruction losses."""

from __future__ import annotations

import pytest
import torch

from seis_interp.training.trace_graph_losses import (
    amplitude_envelope_loss,
    masked_mean_square,
    slope_consistency_loss,
    spectrum_loss,
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
