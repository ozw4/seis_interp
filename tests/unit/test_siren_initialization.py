from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from seis_interp.models.siren import Siren
from seis_interp.training.siren_initialization import apply_siren_time_weight_initialization


def _model() -> Siren:
    torch.manual_seed(20260908)
    return Siren(
        input_features=5,
        hidden_width=8,
        hidden_layers=3,
        omega_0=30.0,
        hidden_omega=30.0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_only_first_layer_time_weights_change_without_rng_or_model_state_changes(dtype):
    model = _model().to(dtype=dtype).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = dict(model.named_parameters())
    rng_before = torch.random.get_rng_state().clone()

    assert apply_siren_time_weight_initialization(model, 3.0) is None

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not model.training
    assert dict(model.named_parameters()).keys() == parameters.keys()
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        assert parameter.dtype == dtype
        assert parameter.device == torch.device("cpu")
        assert parameter.requires_grad
        assert parameter.is_leaf
        if name == "network.0.linear.weight":
            assert torch.equal(parameter[:, 0], before[name][:, 0] * 3.0)
            assert torch.equal(parameter[:, 1:], before[name][:, 1:])
        else:
            assert torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("kwargs", [{}, {"scale": 1}, {"scale": 1.0}])
def test_default_and_unit_scale_are_strict_noops(kwargs):
    model = _model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    rng_before = torch.random.get_rng_state().clone()

    apply_siren_time_weight_initialization(model, **kwargs)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
        assert parameter._version == versions[name], name


@pytest.mark.parametrize("scale", [3, np.float32(3), np.float64(3), Fraction(3, 1)])
def test_positive_real_scales_are_accepted(scale):
    model = _model()
    time_weights = model.network[0].linear.weight[:, 0].detach().clone()

    apply_siren_time_weight_initialization(model, scale)

    assert torch.equal(model.network[0].linear.weight[:, 0], 3.0 * time_weights)


@pytest.mark.parametrize(
    "scale",
    [True, False, None, "3", 3j, [], 0, -1, np.nan, np.inf, -np.inf, 10**1000],
)
def test_invalid_scales_reject_before_mutating_parameters_or_rng(scale):
    model = _model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(ValueError, match="scale must be a positive finite number"):
        apply_siren_time_weight_initialization(model, scale)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
        assert parameter._version == versions[name], name


def test_non_siren_model_is_rejected_without_mutation():
    model = torch.nn.Linear(5, 1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(TypeError, match="model must be a Siren"):
        apply_siren_time_weight_initialization(model, 3.0)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_scaled_initialization_has_expected_time_dependence_and_restores_as_plain_siren():
    model = _model().double()
    coordinates = torch.tensor(
        [[-0.75, 0.5, -0.2, 0.1, 0.8], [0.25, -0.1, 0.6, 0.4, -0.3]],
        dtype=torch.float64,
    )
    rescaled_time = coordinates.clone()
    rescaled_time[:, 0] *= 3.0
    expected = model(rescaled_time).detach()

    apply_siren_time_weight_initialization(model, 3.0)
    actual = model(coordinates)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    restored = _model().double()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored(coordinates), actual.detach())
