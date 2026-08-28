"""Minimal SIREN model mapping coordinates to signal values.

By default, the network follows the established repository contract: a first
sine layer scaled by ``omega_0``, later sine layers scaled by
``hidden_omega``, and a final linear layer. An optional exponential schedule
geometrically interpolates the sine-layer frequencies between those two
endpoints.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

EXPONENTIAL_LAYER_OMEGA_SCHEDULE = "exponential"
DENSE_SKIP_CONNECTIONS = "dense"


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_positive_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


def _validate_layer_omega_schedule(value: str | None, *, hidden_layers: int) -> None:
    if value is None:
        return
    if value != EXPONENTIAL_LAYER_OMEGA_SCHEDULE:
        raise ValueError(f"layer_omega_schedule must be 'exponential' or None, got {value!r}.")
    if hidden_layers < 2:
        raise ValueError(
            "layer_omega_schedule='exponential' requires hidden_layers >= 2, "
            f"got {hidden_layers!r}."
        )


def _validate_skip_connections(value: str | None) -> None:
    if value is None:
        return
    if value != DENSE_SKIP_CONNECTIONS:
        raise ValueError(f"skip_connections must be 'dense' or None, got {value!r}.")


def _sine_layer_omegas(
    *,
    omega_0: float,
    hidden_omega: float,
    hidden_layers: int,
    layer_omega_schedule: str | None,
) -> tuple[float, ...]:
    """Return the activation frequency assigned to each sine layer."""
    if layer_omega_schedule is None:
        return (omega_0, *((hidden_omega,) * (hidden_layers - 1)))

    log_start = math.log(omega_0)
    log_step = (math.log(hidden_omega) - log_start) / (hidden_layers - 1)
    intermediate = tuple(
        math.exp(log_start + layer_index * log_step) for layer_index in range(1, hidden_layers - 1)
    )
    return (omega_0, *intermediate, hidden_omega)


class SineLayer(nn.Module):
    """Linear layer followed by ``sin(omega * (W x + b))``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega: float = 1.0,
        is_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("in_features", in_features)
        _validate_positive_int("out_features", out_features)
        _validate_positive_finite("omega", omega)
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        weight_bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-weight_bound, weight_bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(coordinates))


class Siren(nn.Module):
    """Sine-activated MLP with a linear output layer.

    ``hidden_layers`` counts every sine-activated layer. Without a layer
    schedule, the first sine layer applies ``omega_0`` and later sine layers
    apply ``hidden_omega``. With ``layer_omega_schedule='exponential'``, the
    sine-layer frequencies form a geometric progression from ``omega_0`` to
    ``hidden_omega``. An exponential schedule therefore requires at least two
    sine layers. With ``skip_connections='dense'``, each later sine layer
    consumes all preceding sine-layer activations and the final linear layer
    consumes all sine-layer activations. This makes dense parameter count grow
    quadratically with the number of sine layers.

    The constructor arguments define the function together with the
    weights but are not part of ``state_dict()``. Loading weights into a
    model built with a different ``omega_0``, ``hidden_omega``, layer schedule,
    or skip architecture therefore either fails on incompatible tensor shapes
    or yields a different function, so whatever saves these weights must store
    the constructor arguments next to them.
    """

    def __init__(
        self,
        input_features: int = 6,
        hidden_width: int = 256,
        hidden_layers: int = 4,
        output_features: int = 1,
        omega_0: float = 10.0,
        hidden_omega: float = 1.0,
        layer_omega_schedule: str | None = None,
        skip_connections: str | None = None,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_features", input_features)
        _validate_positive_int("hidden_width", hidden_width)
        _validate_positive_int("hidden_layers", hidden_layers)
        _validate_positive_int("output_features", output_features)
        _validate_positive_finite("omega_0", omega_0)
        _validate_positive_finite("hidden_omega", hidden_omega)
        _validate_layer_omega_schedule(
            layer_omega_schedule,
            hidden_layers=hidden_layers,
        )
        _validate_skip_connections(skip_connections)

        self.input_features = input_features
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.output_features = output_features
        self.omega_0 = omega_0
        self.hidden_omega = hidden_omega
        self.layer_omega_schedule = layer_omega_schedule
        self.skip_connections = skip_connections
        self.layer_omegas = _sine_layer_omegas(
            omega_0=omega_0,
            hidden_omega=hidden_omega,
            hidden_layers=hidden_layers,
            layer_omega_schedule=layer_omega_schedule,
        )
        self.sine_layer_input_features = (
            (input_features, *((hidden_width,) * (hidden_layers - 1)))
            if skip_connections is None
            else (
                input_features,
                *(hidden_width * layer_index for layer_index in range(1, hidden_layers)),
            )
        )
        self.final_input_features = (
            hidden_width if skip_connections is None else hidden_width * hidden_layers
        )

        layers: list[nn.Module] = [
            SineLayer(
                input_features,
                hidden_width,
                omega=self.layer_omegas[0],
                is_first=True,
            )
        ]
        layers.extend(
            SineLayer(in_features, hidden_width, omega=omega)
            for in_features, omega in zip(
                self.sine_layer_input_features[1:],
                self.layer_omegas[1:],
                strict=True,
            )
        )
        final_layer = nn.Linear(self.final_input_features, output_features)
        final_bound = math.sqrt(6.0 / self.final_input_features) / hidden_omega
        with torch.no_grad():
            final_layer.weight.uniform_(-final_bound, final_bound)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.skip_connections is None:
            return self.network(coordinates)

        activations: list[torch.Tensor] = []
        for layer_index in range(self.hidden_layers):
            layer_input = coordinates if not activations else torch.cat(activations, dim=-1)
            activations.append(self.network[layer_index](layer_input))
        return self.network[-1](torch.cat(activations, dim=-1))
