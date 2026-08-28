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
    sine layers.

    The constructor arguments define the function together with the
    weights but are not part of ``state_dict()``. Loading weights into a
    model built with a different ``omega_0``, ``hidden_omega``, or layer
    schedule therefore succeeds silently and yields a different function, so
    whatever saves these weights must store the constructor arguments next to
    them.
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

        self.input_features = input_features
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.output_features = output_features
        self.omega_0 = omega_0
        self.hidden_omega = hidden_omega
        self.layer_omega_schedule = layer_omega_schedule
        self.layer_omegas = _sine_layer_omegas(
            omega_0=omega_0,
            hidden_omega=hidden_omega,
            hidden_layers=hidden_layers,
            layer_omega_schedule=layer_omega_schedule,
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
            SineLayer(hidden_width, hidden_width, omega=omega) for omega in self.layer_omegas[1:]
        )
        final_layer = nn.Linear(hidden_width, output_features)
        final_bound = math.sqrt(6.0 / hidden_width) / hidden_omega
        with torch.no_grad():
            final_layer.weight.uniform_(-final_bound, final_bound)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates)
