"""Fixed-step CCNet-5D fitting on hidden traces from the observed set."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import torch

from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training.ccnet5d_ema import validate_ccnet5d_ema_decay
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.trace_relative_loss import masked_trace_loss

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
    loss_name: str = "masked_trace_relative_mse",
    ema_decay: float | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    supervised_traces_per_update: int | None = None,
    reporter: Reporter | None = None,
) -> CCNet5DObservedTrainingResult:
    """Fit pseudo-targets with Adam or AdamW, optionally returning final EMA parameters.

    EMA starts at the first post-update parameters and advances once per update.
    Raw parameters are used throughout fitting; only the returned model is averaged.
    A fixed trace budget stacks patches in one forward/backward, without gradient
    accumulation. The loss is the mean over selected complete trace samples.
    """
    if not isinstance(model, CCNet5D):
        raise TypeError("model must be a CCNet5D")
    if not isinstance(source, CCNet5DObservedPatchSource):
        raise TypeError("source must be a CCNet5DObservedPatchSource")
    update_count = _positive_integer(optimizer_updates, "optimizer_updates")
    report_interval = _positive_integer(report_every_steps, "report_every_steps")
    rate = _positive_finite_float(learning_rate, "learning_rate")
    decay = validate_ccnet5d_ema_decay(ema_decay)
    averaged = None
    device_value = torch.device(device)
    model.to(device_value)
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer_name must be adam or adamw")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, Real)
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be nonnegative and finite")
    if supervised_traces_per_update is not None:
        supervised_traces_per_update = _positive_integer(
            supervised_traces_per_update, "supervised_traces_per_update"
        )
    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=rate, weight_decay=weight_decay)
    parameter = next(model.parameters())
    history: list[dict[str, object]] = []
    interval_loss = 0.0
    interval_count = 0
    final_loss = float("nan")
    supervised_trace_presentations = 0

    for step in range(1, update_count + 1):
        if supervised_traces_per_update is None:
            batch = source.sample()
            input_values = batch.model_input[None]
            target_values = batch.pseudo_target[None]
            mask_values = batch.pseudo_target_mask[None]
        else:
            input_values, target_values, mask_values = source.sample_batch(
                supervised_traces_per_update
            )
        inputs = torch.as_tensor(
            input_values,
            dtype=parameter.dtype,
            device=device_value,
        )[:, None]
        targets = torch.as_tensor(
            target_values,
            dtype=parameter.dtype,
            device=device_value,
        )
        target_mask = torch.as_tensor(
            mask_values,
            dtype=torch.bool,
            device=device_value,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        if prediction.shape != inputs.shape:
            raise ValueError("CCNet-5D output shape must match the pseudo-mask input patch")
        prediction_traces = prediction[:, 0].movedim(1, -1)[target_mask]
        target_traces = targets.movedim(1, -1)[target_mask]
        loss = masked_trace_loss(prediction_traces, target_traces, loss_name=loss_name)
        final_loss = float(loss.detach().cpu().item())
        if not math.isfinite(final_loss):
            raise RuntimeError(f"non-finite CCNet-5D training loss at step {step}")
        loss.backward()
        optimizer.step()
        if decay is not None:
            with torch.no_grad():
                if averaged is None:
                    averaged = [p.detach().clone() for p in model.parameters()]
                else:
                    for average, parameter_value in zip(averaged, model.parameters(), strict=True):
                        average.lerp_(parameter_value, 1 - decay)
        supervised_trace_presentations += int(mask_values.sum())
        interval_loss += final_loss
        interval_count += 1

        if step % report_interval == 0 or step == update_count:
            mean_loss = interval_loss / interval_count
            history.append(
                {
                    "step": step,
                    "loss": mean_loss,
                    "learning_rate": rate,
                }
            )
            if supervised_traces_per_update is not None:
                history[-1]["supervised_trace_presentations"] = supervised_trace_presentations
            if reporter is not None:
                reporter(f"ccnet5d observed-only step {step}/{update_count}: loss={mean_loss:.8g}")
            interval_loss = 0.0
            interval_count = 0

    if averaged is not None:
        with torch.no_grad():
            for parameter_value, average in zip(model.parameters(), averaged, strict=True):
                parameter_value.copy_(average)
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
