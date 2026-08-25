"""Minimal SIREN model mapping coordinates to signal values.

The network follows the paper equations: a first sine layer scaled by
``omega_0``, unscaled hidden sine layers, and a final linear layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


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

    ``hidden_layers`` counts every sine-activated layer. Only the first
    sine layer applies ``omega_0``; later sine layers use ``omega=1.0``.

    The constructor arguments define the function together with the
    weights but are not part of ``state_dict()``. Loading weights into a
    model built with a different ``omega_0`` therefore succeeds silently
    and yields a different function, so whatever saves these weights must
    store the constructor arguments next to them.
    """

    def __init__(
        self,
        input_features: int = 6,
        hidden_width: int = 256,
        hidden_layers: int = 4,
        output_features: int = 1,
        omega_0: float = 10.0,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_features", input_features)
        _validate_positive_int("hidden_width", hidden_width)
        _validate_positive_int("hidden_layers", hidden_layers)
        _validate_positive_int("output_features", output_features)
        _validate_positive_finite("omega_0", omega_0)

        self.input_features = input_features
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.output_features = output_features
        self.omega_0 = omega_0

        layers: list[nn.Module] = [
            SineLayer(input_features, hidden_width, omega=omega_0, is_first=True)
        ]
        layers.extend(
            SineLayer(hidden_width, hidden_width, omega=1.0) for _ in range(hidden_layers - 1)
        )
        final_layer = nn.Linear(hidden_width, output_features)
        final_bound = math.sqrt(6.0 / hidden_width)
        with torch.no_grad():
            final_layer.weight.uniform_(-final_bound, final_bound)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates)
