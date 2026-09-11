"""Fixed-step CCNet-5D fitting on hidden traces from the observed set."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import torch

from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.trace_relative_loss import masked_trace_relative_mse

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class CCNet5DObservedTrainingResult:
    """Final fixed-step training counters and bounded loss history."""

    steps_completed: int
    supervised_trace_presentations: int
    final_loss: float
    history: tuple[dict[str, object], ...]


def train_ccnet5d_observed_steps(
    model: CCNet5D,
    source: CCNet5DObservedPatchSource,
    *,
    device: torch.device | str,
    optimizer_updates: int,
    learning_rate: float,
    report_every_steps: int,
    reporter: Reporter | None = None,
) -> CCNet5DObservedTrainingResult:
    """Run Adam for exactly the configured updates using pseudo-target traces only."""
    if not isinstance(model, CCNet5D):
        raise TypeError("model must be a CCNet5D")
    if not isinstance(source, CCNet5DObservedPatchSource):
        raise TypeError("source must be a CCNet5DObservedPatchSource")
    update_count = _positive_integer(optimizer_updates, "optimizer_updates")
    report_interval = _positive_integer(report_every_steps, "report_every_steps")
    rate = _positive_finite_float(learning_rate, "learning_rate")
    device_value = torch.device(device)
    model.to(device_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    parameter = next(model.parameters())
    history: list[dict[str, object]] = []
    interval_loss = 0.0
    interval_count = 0
    final_loss = float("nan")
    supervised_trace_presentations = 0

    for step in range(1, update_count + 1):
        batch = source.sample()
        inputs = torch.as_tensor(
            batch.model_input,
            dtype=parameter.dtype,
            device=device_value,
        )[None, None]
        targets = torch.as_tensor(
            batch.pseudo_target,
            dtype=parameter.dtype,
            device=device_value,
        )
        target_mask = torch.as_tensor(
            batch.pseudo_target_mask,
            dtype=torch.bool,
            device=device_value,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        if prediction.shape != inputs.shape:
            raise ValueError("CCNet-5D output shape must match the pseudo-mask input patch")
        prediction_traces = prediction[0, 0].movedim(0, -1)[target_mask]
        target_traces = targets.movedim(0, -1)[target_mask]
        loss = masked_trace_relative_mse(prediction_traces, target_traces)
        final_loss = float(loss.detach().cpu().item())
        if not math.isfinite(final_loss):
            raise RuntimeError(f"non-finite CCNet-5D training loss at step {step}")
        loss.backward()
        optimizer.step()
        supervised_trace_presentations += int(batch.pseudo_target_mask.sum())
        interval_loss += final_loss
        interval_count += 1

        if step % report_interval == 0 or step == update_count:
            mean_loss = interval_loss / interval_count
            history.append(
                {
                    "step": step,
                    "trace_relative_loss": mean_loss,
                    "learning_rate": rate,
                }
            )
            if reporter is not None:
                reporter(
                    f"ccnet5d observed-only step {step}/{update_count}: "
                    f"trace_relative_loss={mean_loss:.8g}"
                )
            interval_loss = 0.0
            interval_count = 0

    return CCNet5DObservedTrainingResult(
        steps_completed=update_count,
        supervised_trace_presentations=supervised_trace_presentations,
        final_loss=final_loss,
        history=tuple(history),
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted
