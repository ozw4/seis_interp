from __future__ import annotations

import math

import torch

from seis_interp.models import Siren


def _target_signal(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3 * math.pi * x) + 0.25 * torch.sin(7 * math.pi * x)


def _mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def test_siren_reconstructs_held_out_points_of_a_1d_signal() -> None:
    torch.manual_seed(42)
    x = torch.linspace(-1.0, 1.0, 257).unsqueeze(-1)
    y = _target_signal(x)
    observed = torch.arange(len(x)) % 2 == 0
    held_out = ~observed

    model = Siren(
        input_features=1,
        hidden_width=32,
        hidden_layers=3,
        output_features=1,
        omega_0=10.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(750):
        optimizer.zero_grad()
        loss = _mean_squared_error(model(x[observed]), y[observed])
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    with torch.no_grad():
        observed_mse = _mean_squared_error(model(x[observed]), y[observed])
        held_out_mse = _mean_squared_error(model(x[held_out]), y[held_out])

    assert torch.isfinite(observed_mse)
    assert torch.isfinite(held_out_mse)
    assert observed_mse.item() < 5e-4
    assert held_out_mse.item() < 5e-4
