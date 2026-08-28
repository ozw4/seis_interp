from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models import SineLayer, Siren
from seis_interp.pipelines.train_siren import _build_model


def test_sine_layer_forward_matches_formula() -> None:
    layer = SineLayer(2, 3, omega=4.0)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]))
        layer.linear.bias.copy_(torch.tensor([0.05, -0.15, 0.25]))
    x = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])

    actual = layer(x)
    expected = torch.sin(layer.omega * layer.linear(x))

    torch.testing.assert_close(actual, expected)


def test_siren_output_shape() -> None:
    torch.manual_seed(0)
    model = Siren(input_features=5, hidden_width=16, hidden_layers=2, output_features=1)

    output = model(torch.randn(8, 5))

    assert output.shape == (8, 1)


def test_siren_default_input_width_matches_model_coordinate_schema() -> None:
    model = Siren(hidden_width=16, hidden_layers=2)
    first_layer = model.network[0]

    assert isinstance(first_layer, SineLayer)
    assert len(MODEL_COORDINATE_ORDER) == 6
    assert first_layer.linear.in_features == len(MODEL_COORDINATE_ORDER)
    assert model(torch.randn(3, len(MODEL_COORDINATE_ORDER))).shape == (3, 1)


def test_siren_final_layer_is_linear() -> None:
    model = Siren(input_features=5, hidden_width=16, hidden_layers=2)

    final_layer = model.network[-1]

    assert isinstance(final_layer, nn.Linear)
    assert not isinstance(final_layer, SineLayer)


def test_same_seed_gives_same_initial_parameters() -> None:
    torch.manual_seed(42)
    model_a = Siren(input_features=5, hidden_width=16, hidden_layers=2)
    torch.manual_seed(42)
    model_b = Siren(input_features=5, hidden_width=16, hidden_layers=2)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    assert state_a.keys() == state_b.keys()
    for key, value in state_a.items():
        assert torch.equal(value, state_b[key]), key


def test_first_sine_layer_weight_bound() -> None:
    torch.manual_seed(0)
    model = Siren(input_features=5, hidden_width=16, hidden_layers=2, omega_0=10.0)

    weight = model.network[0].linear.weight
    bound = 1.0 / 5 + 1e-6

    assert weight.abs().max().item() <= bound


def test_hidden_sine_layer_weight_bound() -> None:
    torch.manual_seed(0)
    model = Siren(
        input_features=5,
        hidden_width=16,
        hidden_layers=4,
        omega_0=10.0,
        hidden_omega=30.0,
    )

    hidden_layers = model.network[1:-1]
    bound = math.sqrt(6.0 / 16) / 30.0 + 1e-6

    assert len(hidden_layers) == 3
    assert all(isinstance(layer, SineLayer) for layer in hidden_layers)
    assert all(layer.omega == 30.0 for layer in hidden_layers)
    assert all(layer.linear.weight.abs().max().item() <= bound for layer in hidden_layers)


def test_exponential_layer_omega_schedule_uses_geometric_progression() -> None:
    torch.manual_seed(0)
    model = Siren(
        input_features=5,
        hidden_width=16,
        hidden_layers=4,
        omega_0=5.0,
        hidden_omega=50.0,
        layer_omega_schedule="exponential",
    )

    expected_omegas = tuple(5.0 * 10.0 ** (index / 3.0) for index in range(4))
    sine_layers = [layer for layer in model.network if isinstance(layer, SineLayer)]

    assert model.layer_omega_schedule == "exponential"
    assert model.layer_omegas == pytest.approx(expected_omegas)
    assert [layer.omega for layer in sine_layers] == pytest.approx(expected_omegas)
    for layer, omega in zip(sine_layers[1:], expected_omegas[1:], strict=True):
        bound = math.sqrt(6.0 / 16) / omega + 1e-6
        assert layer.linear.weight.abs().max().item() <= bound


def test_dense_skips_concatenate_all_prior_activations_with_scheduled_omegas() -> None:
    torch.manual_seed(0)
    model = Siren(
        input_features=5,
        hidden_width=8,
        hidden_layers=4,
        output_features=1,
        omega_0=5.0,
        hidden_omega=50.0,
        layer_omega_schedule="exponential",
        skip_connections="dense",
    )
    expected_omegas = tuple(5.0 * 10.0 ** (index / 3.0) for index in range(4))

    assert model.skip_connections == "dense"
    assert model.sine_layer_input_features == (5, 8, 16, 24)
    assert model.final_input_features == 32
    for layer_index, (in_features, omega) in enumerate(
        zip(model.sine_layer_input_features, expected_omegas, strict=True)
    ):
        layer = model.network[layer_index]
        assert isinstance(layer, SineLayer)
        assert layer.linear.in_features == in_features
        assert layer.omega == pytest.approx(omega)
        if layer_index > 0:
            bound = math.sqrt(6.0 / in_features) / omega + 1e-6
            assert layer.linear.weight.abs().max().item() <= bound
    final_layer = model.network[-1]
    assert isinstance(final_layer, nn.Linear)
    assert final_layer.in_features == 32
    assert final_layer.weight.abs().max().item() <= math.sqrt(6.0 / 32) / 50.0 + 1e-6

    coordinates = torch.randn(3, 5)
    activations: list[torch.Tensor] = []
    with torch.no_grad():
        for layer_index in range(model.hidden_layers):
            layer_input = coordinates if not activations else torch.cat(activations, dim=-1)
            activations.append(model.network[layer_index](layer_input))
        expected = final_layer(torch.cat(activations, dim=-1))
    actual = model(coordinates)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    expected_parameter_count = 8 * 6 + 8 * 9 + 8 * 17 + 8 * 25 + 33
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameter_count
    assert (
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        == expected_parameter_count * 4
    )


def test_final_linear_uses_official_hidden_omega_weight_bound() -> None:
    torch.manual_seed(0)
    model = Siren(
        input_features=5,
        hidden_width=16,
        hidden_layers=2,
        hidden_omega=30.0,
    )

    final_layer = model.network[-1]
    bound = math.sqrt(6.0 / 16) / 30.0 + 1e-6

    assert isinstance(final_layer, nn.Linear)
    assert final_layer.weight.abs().max().item() <= bound


def test_hidden_omega_default_preserves_legacy_parameters_and_layer_count() -> None:
    torch.manual_seed(42)
    default_model = Siren(input_features=5, hidden_width=16, hidden_layers=4)
    torch.manual_seed(42)
    explicit_legacy_model = Siren(
        input_features=5,
        hidden_width=16,
        hidden_layers=4,
        hidden_omega=1.0,
        layer_omega_schedule=None,
        skip_connections=None,
    )
    torch.manual_seed(42)
    legacy_layers: list[nn.Module] = [
        SineLayer(5, 16, omega=10.0, is_first=True),
        *(SineLayer(16, 16, omega=1.0) for _ in range(3)),
    ]
    legacy_final = nn.Linear(16, 1)
    with torch.no_grad():
        legacy_final.weight.uniform_(-math.sqrt(6.0 / 16), math.sqrt(6.0 / 16))
    legacy_network = nn.Sequential(*legacy_layers, legacy_final)

    assert default_model.hidden_omega == 1.0
    assert default_model.layer_omega_schedule is None
    assert default_model.layer_omegas == (10.0, 1.0, 1.0, 1.0)
    assert default_model.skip_connections is None
    assert default_model.sine_layer_input_features == (5, 16, 16, 16)
    assert default_model.final_input_features == 16
    assert len(default_model.network) == 5
    assert sum(isinstance(layer, SineLayer) for layer in default_model.network) == 4
    for name, value in default_model.state_dict().items():
        assert torch.equal(value, explicit_legacy_model.state_dict()[name]), name
    for name, value in default_model.network.state_dict().items():
        assert torch.equal(value, legacy_network.state_dict()[name]), name
    coordinates = torch.randn(3, 5)
    torch.testing.assert_close(default_model(coordinates), legacy_network(coordinates))


def test_train_siren_builder_forwards_hidden_omega() -> None:
    config = {
        "model": {
            "name": "siren",
            "input_features": len(MODEL_COORDINATE_ORDER),
            "hidden_width": 8,
            "hidden_layers": 2,
            "omega_0": 10.0,
            "hidden_omega": 30.0,
            "skip_connections": None,
        }
    }

    model = _build_model(config)

    assert model.hidden_omega == 30.0
    assert model.skip_connections is None


def test_train_siren_builder_forwards_exponential_layer_omega_schedule() -> None:
    config = {
        "model": {
            "name": "siren",
            "input_features": len(MODEL_COORDINATE_ORDER),
            "hidden_width": 8,
            "hidden_layers": 4,
            "omega_0": 5.0,
            "hidden_omega": 50.0,
            "layer_omega_schedule": "exponential",
        }
    }

    model = _build_model(config)

    assert model.layer_omega_schedule == "exponential"
    assert model.layer_omegas == pytest.approx(
        tuple(5.0 * 10.0 ** (index / 3.0) for index in range(4))
    )


def test_train_siren_builder_forwards_dense_skip_connections() -> None:
    config = {
        "model": {
            "name": "siren",
            "input_features": len(MODEL_COORDINATE_ORDER),
            "hidden_width": 8,
            "hidden_layers": 4,
            "omega_0": 10.0,
            "hidden_omega": 30.0,
            "skip_connections": "dense",
        }
    }

    model = _build_model(config)

    assert model.skip_connections == "dense"
    assert model.sine_layer_input_features == (6, 8, 16, 24)
    assert model.final_input_features == 32


@pytest.mark.parametrize(
    ("coordinate_features", "input_features", "wrong_input_features"),
    [
        ("cmp_cartesian_half_offset", 5, 6),
        ("cmp_cartesian_half_offset_radius", 6, 5),
    ],
)
def test_train_siren_builder_validates_coordinate_feature_width(
    coordinate_features: str,
    input_features: int,
    wrong_input_features: int,
) -> None:
    config = {
        "model": {
            "name": "siren",
            "coordinate_features": coordinate_features,
            "input_features": input_features,
            "hidden_width": 8,
            "hidden_layers": 2,
            "omega_0": 10.0,
            "hidden_omega": 30.0,
        }
    }

    assert _build_model(config).input_features == input_features
    config["model"]["input_features"] = wrong_input_features
    with pytest.raises(ValueError, match=f"must be {input_features}"):
        _build_model(config)


def test_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    model = Siren(input_features=5, hidden_width=16, hidden_layers=2)

    loss = model(torch.randn(8, 5)).pow(2).mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_features": 0},
        {"in_features": -1},
        {"out_features": 0},
        {"out_features": -2},
    ],
)
def test_sine_layer_rejects_invalid_features(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SineLayer(**{"in_features": 2, "out_features": 3, **kwargs})


@pytest.mark.parametrize("omega", [0.0, -1.0, math.nan])
def test_sine_layer_rejects_invalid_omega(omega: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        SineLayer(2, 3, omega=omega)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"output_features": -1},
        {"hidden_width": 0},
        {"hidden_layers": -3},
    ],
)
def test_siren_rejects_invalid_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Siren(**kwargs)


@pytest.mark.parametrize("omega_0", [0.0, -10.0, math.nan])
def test_siren_rejects_invalid_omega_0(omega_0: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        Siren(omega_0=omega_0)


@pytest.mark.parametrize("hidden_omega", [0.0, -10.0, math.nan, math.inf, True])
def test_siren_rejects_invalid_hidden_omega(hidden_omega: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        Siren(hidden_omega=hidden_omega)


@pytest.mark.parametrize("schedule", ["linear", "constant", 1, True])
def test_siren_rejects_unknown_layer_omega_schedule(schedule: object) -> None:
    with pytest.raises(ValueError, match="layer_omega_schedule must be 'exponential'"):
        Siren(layer_omega_schedule=schedule)


def test_siren_rejects_exponential_schedule_with_one_sine_layer() -> None:
    with pytest.raises(ValueError, match="requires hidden_layers >= 2"):
        Siren(hidden_layers=1, layer_omega_schedule="exponential")


@pytest.mark.parametrize("skip_connections", ["residual", "all", 1, True])
def test_siren_rejects_unknown_skip_connections(skip_connections: object) -> None:
    with pytest.raises(ValueError, match="skip_connections must be 'dense'"):
        Siren(skip_connections=skip_connections)
