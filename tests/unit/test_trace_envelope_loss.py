from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from seis_interp.training.trace_envelope_loss import (
    gaussian_envelope_kernels,
    local_rms_envelope,
    trace_envelope_mse,
    trace_envelope_weight,
    validate_trace_envelope_loss_options,
)


def _options():
    return {"weight": 1.0, "sigma_samples": [4, 8], "decay_steps": 2500}


def _reference_envelope(waveforms, sigma):
    radius = int(np.ceil(4 * sigma))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= kernel.sum()
    return np.stack(
        [
            np.sqrt(
                np.convolve(np.pad(row**2, (radius, radius), mode="reflect"), kernel, mode="valid")
                + 1e-6
            )
            for row in waveforms
        ]
    )


def test_options_are_detached_and_none_preserves_absent_configuration():
    options = _options()
    original = copy.deepcopy(options)
    actual = validate_trace_envelope_loss_options(options, time_count=384, microbatch_size=262144)
    assert actual == {"weight": 1.0, "sigma_samples": [4.0, 8.0], "decay_steps": 2500}
    actual["sigma_samples"].append(1.0)
    assert options == original
    assert validate_trace_envelope_loss_options(None, time_count=0, microbatch_size=0) is None


@pytest.mark.parametrize("value", [False, [], {}, {**_options(), "epsilon": 1e-6}, {"weight": 1}])
def test_options_require_the_exact_mapping(value):
    with pytest.raises(ValueError, match="envelope_loss"):
        validate_trace_envelope_loss_options(value)


@pytest.mark.parametrize("invalid", [True, 0, -1, "1", np.nan, np.inf, 10**400])
@pytest.mark.parametrize("field", ["weight", "sigma_samples"])
def test_options_require_positive_finite_nonboolean_weight_and_sigmas(invalid, field):
    options = _options()
    options[field] = [invalid] if field == "sigma_samples" else invalid
    with pytest.raises(ValueError, match=field):
        validate_trace_envelope_loss_options(options)


@pytest.mark.parametrize("sigmas", [[], "4", None, 4.0, [1e308]])
def test_options_require_a_nonempty_sigma_sequence_with_finite_radii(sigmas):
    with pytest.raises(ValueError, match="sigma_samples"):
        validate_trace_envelope_loss_options({**_options(), "sigma_samples": sigmas})


@pytest.mark.parametrize("decay", [True, 0, 1, 2.0, np.nan, "2500"])
def test_options_require_at_least_two_integer_decay_steps(decay):
    with pytest.raises(ValueError, match="decay_steps"):
        validate_trace_envelope_loss_options({**_options(), "decay_steps": decay})


@pytest.mark.parametrize(
    "dimensions,match",
    [
        ({"time_count": 32}, "radius"),
        ({"time_count": 384, "microbatch_size": 383}, "complete trace"),
        ({"time_count": 0}, "time_count"),
        ({"time_count": True}, "time_count"),
        ({"microbatch_size": 0}, "microbatch_size"),
        ({"microbatch_size": True}, "microbatch_size"),
    ],
)
def test_options_reject_unsupported_trace_and_microbatch_dimensions(dimensions, match):
    with pytest.raises(ValueError, match=match):
        validate_trace_envelope_loss_options(_options(), **dimensions)


def test_weight_decay_has_fixed_endpoints_independent_of_short_execution():
    weights = [trace_envelope_weight(step, weight=1.0, decay_steps=2500) for step in range(1, 11)]
    assert weights[0] == 1.0
    assert weights[-1] == pytest.approx(1.0 - 9.0 / 2499)
    assert trace_envelope_weight(2499, weight=1.0, decay_steps=2500) > 0
    assert trace_envelope_weight(2500, weight=1.0, decay_steps=2500) == 0
    assert trace_envelope_weight(5000, weight=1.0, decay_steps=2500) == 0
    assert trace_envelope_weight(1, weight=2.0, decay_steps=2) == 2
    assert trace_envelope_weight(2, weight=2.0, decay_steps=2) == 0


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_weight_rejects_invalid_step(step):
    with pytest.raises(ValueError, match="step"):
        trace_envelope_weight(step, weight=1.0, decay_steps=2500)


@pytest.mark.parametrize("sigma", [4.0, 8.0, 1.1])
def test_envelope_matches_independent_reflected_impulses_and_constants(sigma):
    waveforms = np.zeros((4, 37), dtype=np.float64)
    waveforms[0, 0] = 2.0
    waveforms[1, -1] = -3.0
    waveforms[2] = 0.75
    (kernel,) = gaussian_envelope_kernels([sigma], device="cpu", dtype=torch.float64)
    assert kernel.shape == (1, 1, 2 * int(np.ceil(4 * sigma)) + 1)
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-15)
    actual = local_rms_envelope(torch.tensor(waveforms), kernel)
    np.testing.assert_allclose(
        actual.numpy(), _reference_envelope(waveforms, sigma), rtol=1e-13, atol=1e-15
    )
    np.testing.assert_array_equal(actual[3].numpy(), np.full(37, 0.001))


def test_loss_equally_averages_both_sigmas_and_does_not_mix_trace_values():
    prediction = torch.tensor(
        np.linspace(-1.0, 1.0, 74).reshape(2, 37), dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        np.sin(np.arange(74)).reshape(2, 37), dtype=torch.float64, requires_grad=True
    )
    kernels = gaussian_envelope_kernels([4, 8], device="cpu", dtype=torch.float64)
    actual = trace_envelope_mse(prediction, target, kernels=kernels)
    reference = np.mean(
        [
            np.mean(
                (
                    _reference_envelope(prediction.detach().numpy(), sigma)
                    - _reference_envelope(target.detach().numpy(), sigma)
                )
                ** 2
            )
            for sigma in (4, 8)
        ]
    )
    assert float(actual.detach()) == pytest.approx(reference, rel=1e-13)
    actual.backward()
    assert target.grad is None
    original_gradient = prediction.grad[0].clone()
    changed = prediction.detach().clone()
    changed[1] = 100.0
    changed.requires_grad_()
    trace_envelope_mse(changed, target, kernels=kernels).backward()
    torch.testing.assert_close(changed.grad[0], original_gradient, rtol=0, atol=0)


def test_reflected_envelope_loss_passes_gradcheck_and_zero_has_finite_gradients():
    kernels = gaussian_envelope_kernels([0.5, 1.25], device="cpu", dtype=torch.float64)
    prediction = torch.tensor(
        [[0.2, -0.5, 0.9, 0.3, -0.7, 0.4, 0.1]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[0.3, 0.2, -0.6, 0.7, 0.1, -0.4, 0.8]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: trace_envelope_mse(value, target, kernels=kernels), (prediction,)
    )
    zero = torch.zeros_like(prediction, requires_grad=True)
    zero_loss = trace_envelope_mse(zero, zero.detach(), kernels=kernels)
    zero_loss.backward()
    assert zero_loss.item() == 0.0
    assert torch.isfinite(zero.grad).all()
    assert torch.count_nonzero(zero.grad) == 0


def test_loss_rejects_mismatched_shapes_and_short_trace():
    kernels = gaussian_envelope_kernels([4, 8], device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="shapes"):
        trace_envelope_mse(torch.zeros(1, 37), torch.zeros(2, 37), kernels=kernels)
    with pytest.raises(ValueError, match="radius"):
        local_rms_envelope(torch.zeros(1, 32), kernels[1])
