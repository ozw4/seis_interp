"""Fixed-step SIREN training on sampled points, without target validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.models.siren import Siren
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import RandomPointSampler

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class FixedStepSirenResult:
    """Completed updates, the final batch loss, and interval-mean loss history."""

    steps_completed: int
    final_batch_loss: float
    history: tuple[dict[str, int | float], ...]


def train_siren_fixed_steps(
    model: Siren,
    sampler: RandomPointSampler,
    *,
    device: torch.device | str,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    report_interval: int,
    reporter: Reporter | None = None,
) -> FixedStepSirenResult:
    """Perform exactly ``max_steps`` Adam/MSE updates without reseeding or evaluation.

    Losses are measured before each update. History contains float64 means since
    the previous report, including a final partial interval. A missing reporter
    is silent, and the trained model remains on the requested device.
    """
    if not isinstance(model, Siren):
        raise TypeError("model must be a Siren")
    if not isinstance(sampler, RandomPointSampler):
        raise TypeError("sampler must be a RandomPointSampler")
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    batch_size_value = _positive_integer(batch_size, "batch_size")
    max_steps_value = _positive_integer(max_steps, "max_steps")
    report_interval_value = _positive_integer(report_interval, "report_interval")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    loss_function = torch.nn.MSELoss()
    history: list[dict[str, int | float]] = []
    interval_losses: list[float] = []

    for step in range(1, max_steps_value + 1):
        batch_coordinates, batch_targets = sampler.sample(batch_size_value)
        coordinate_tensor, target_tensor = to_model_tensors(
            batch_coordinates, batch_targets, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinate_tensor)
        if prediction.shape != target_tensor.shape:
            raise ValueError("model output shape must match the target shape")
        loss = loss_function(prediction, target_tensor)
        final_batch_loss = float(loss.detach().cpu().item())
        if not math.isfinite(final_batch_loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        optimizer.step()
        interval_losses.append(final_batch_loss)

        if step % report_interval_value == 0 or step == max_steps_value:
            train_loss = float(np.mean(interval_losses, dtype=np.float64))
            history.append({"step": step, "train_loss": train_loss})
            interval_losses.clear()
            if reporter is not None:
                reporter(f"siren_volume step {step}/{max_steps_value}: train_loss={train_loss:.8g}")

    return FixedStepSirenResult(
        steps_completed=max_steps_value,
        final_batch_loss=final_batch_loss,
        history=tuple(history),
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
