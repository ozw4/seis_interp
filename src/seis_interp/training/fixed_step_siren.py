"""Fixed-step SIREN training on sampled points, without target validation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.models.siren import Siren
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import (
    RandomPointSampler,
    build_trace_coordinate_points,
    validated_normalized_time_offsets,
)
from seis_interp.training.trace_envelope_loss import (
    gaussian_envelope_kernels,
    trace_envelope_mse,
    trace_envelope_weight,
    validate_trace_envelope_loss_options,
)

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


def train_siren_complete_trace_steps(
    model: Siren,
    normalized_time: np.ndarray,
    normalized_spatial: np.ndarray,
    normalized_amplitudes: np.ndarray,
    *,
    device: torch.device | str,
    learning_rate: float,
    traces_per_step: int | None,
    microbatch_size: int,
    max_steps: int,
    report_interval: int,
    random_seed: int,
    learning_rate_schedule: str = "constant",
    minimum_learning_rate: float | None = None,
    reporter: Reporter | None = None,
    normalized_time_offsets: np.ndarray | None = None,
    envelope_loss: Mapping | None = None,
) -> FixedStepSirenResult:
    """Fit compact observed traces with bounded point-wise gradient accumulation.

    The caller supplies already normalized, aligned observed-only spatial and
    amplitude rows. Each selected trace contributes every time sample once per
    update. A smaller ``traces_per_step`` samples without replacement using a
    local NumPy generator; ``None`` or the full pool size uses fixed row order.
    ``microbatch_size`` bounds points on both host and device. Each microbatch
    mean loss is weighted by its fraction of the step's points before backward,
    including the short last microbatch. Adam steps only after the whole batch.
    Optional ``normalized_time_offsets`` align with the compact observed rows
    and shift only their coordinate times, without moving amplitude samples.
    The optional envelope loss uses complete traces per microbatch while its
    weight is positive. At zero weight the original point chunks are restored.

    No normalization, evaluation, checkpoint selection, or global reseeding is
    performed. The optional cosine schedule spans the declared ``max_steps``;
    reported learning rates are those after the corresponding scheduler step.
    """
    if not isinstance(model, Siren):
        raise TypeError("model must be a Siren")
    time, spatial, amplitudes = _complete_trace_arrays(
        normalized_time, normalized_spatial, normalized_amplitudes, model.input_features
    )
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    count = (
        len(spatial)
        if traces_per_step is None
        else _positive_integer(traces_per_step, "traces_per_step")
    )
    if count > len(spatial):
        raise ValueError("traces_per_step must not exceed the available observed trace count")
    microbatch = _positive_integer(microbatch_size, "microbatch_size")
    steps = _positive_integer(max_steps, "max_steps")
    interval = _positive_integer(report_interval, "report_interval")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    minimum = _complete_trace_minimum_rate(
        learning_rate_schedule, minimum_learning_rate, learning_rate_value
    )
    offsets = validated_normalized_time_offsets(normalized_time_offsets, len(spatial))
    envelope = validate_trace_envelope_loss_options(
        envelope_loss, time_count=len(time), microbatch_size=microbatch
    )
    kernels = (
        gaussian_envelope_kernels(envelope["sigma_samples"], device=device, dtype=torch.float32)
        if envelope is not None
        else None
    )
    rng = np.random.default_rng(int(random_seed))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=minimum)
        if minimum is not None
        else None
    )
    history: list[dict[str, int | float]] = []
    interval_losses: list[float] = []
    interval_waveform_losses: list[float] = []
    interval_envelope_losses: list[float] = []
    for step in range(1, steps + 1):
        rows = (
            np.arange(len(spatial), dtype=np.int64)
            if count == len(spatial)
            else rng.choice(len(spatial), size=count, replace=False)
        )
        envelope_weight = (
            trace_envelope_weight(
                step, weight=envelope["weight"], decay_steps=envelope["decay_steps"]
            )
            if envelope is not None
            else 0.0
        )
        if envelope_weight > 0.0:
            final_batch_loss, waveform_loss, weighted_envelope = _complete_trace_envelope_update(
                model,
                optimizer,
                time,
                spatial,
                amplitudes,
                rows,
                microbatch_size=microbatch,
                device=device,
                step=step,
                normalized_time_offsets=offsets,
                kernels=kernels,
                envelope_weight=envelope_weight,
            )
        else:
            final_batch_loss = _complete_trace_update(
                model,
                optimizer,
                time,
                spatial,
                amplitudes,
                rows,
                microbatch_size=microbatch,
                device=device,
                step=step,
                normalized_time_offsets=offsets,
            )
            waveform_loss, weighted_envelope = final_batch_loss, 0.0
        if scheduler is not None:
            scheduler.step()
        interval_losses.append(final_batch_loss)
        if envelope is not None:
            interval_waveform_losses.append(waveform_loss)
            interval_envelope_losses.append(weighted_envelope)
        if step % interval == 0 or step == steps:
            train_loss = float(np.mean(interval_losses, dtype=np.float64))
            record: dict[str, int | float] = {"step": step, "train_loss": train_loss}
            if scheduler is not None:
                record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if envelope is not None:
                record.update(
                    waveform_mse=float(np.mean(interval_waveform_losses, dtype=np.float64)),
                    weighted_envelope_mse=float(
                        np.mean(interval_envelope_losses, dtype=np.float64)
                    ),
                    envelope_weight=envelope_weight,
                )
                interval_waveform_losses.clear()
                interval_envelope_losses.clear()
            history.append(record)
            interval_losses.clear()
            if reporter is not None:
                reporter(f"siren_volume step {step}/{steps}: train_loss={train_loss:.8g}")
    return FixedStepSirenResult(
        steps_completed=steps, final_batch_loss=final_batch_loss, history=tuple(history)
    )


def _complete_trace_envelope_update(
    model,
    optimizer,
    time,
    spatial,
    amplitudes,
    rows,
    *,
    microbatch_size,
    device,
    step,
    normalized_time_offsets,
    kernels,
    envelope_weight,
):
    traces_per_microbatch = microbatch_size // len(time)
    optimizer.zero_grad(set_to_none=True)
    totals = np.zeros(3, dtype=np.float64)
    for start in range(0, len(rows), traces_per_microbatch):
        selected_rows = rows[start : start + traces_per_microbatch]
        coordinates = build_trace_coordinate_points(
            time, spatial, selected_rows, normalized_time_offsets=normalized_time_offsets
        )
        coordinate_tensor, target_tensor = to_model_tensors(
            coordinates, amplitudes[selected_rows].reshape(-1), device=device
        )
        prediction = model(coordinate_tensor)
        if prediction.shape != target_tensor.shape:
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("model output shape must match the target shape")
        waveform = torch.nn.functional.mse_loss(prediction, target_tensor)
        envelope = trace_envelope_mse(
            prediction.reshape(len(selected_rows), len(time)),
            target_tensor.reshape(len(selected_rows), len(time)),
            kernels=kernels,
        )
        weighted_envelope = envelope * envelope_weight
        loss = waveform + weighted_envelope
        values = torch.stack((loss, waveform, weighted_envelope)).detach().cpu().numpy()
        if not np.isfinite(values).all():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"non-finite training loss at step {step}")
        fraction = len(selected_rows) / len(rows)
        (loss * fraction).backward()
        totals += values.astype(np.float64) * fraction
    if not np.isfinite(totals).all():
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite training loss at step {step}")
    optimizer.step()
    return tuple(float(value) for value in totals)


def _complete_trace_arrays(time, spatial, amplitudes, input_features):
    time = np.asarray(time)
    spatial = np.asarray(spatial)
    amplitudes = np.asarray(amplitudes)
    if time.ndim != 1 or not time.size or time.dtype != np.float64 or not np.isfinite(time).all():
        raise ValueError("normalized_time must be a nonempty finite float64 vector")
    if (
        spatial.ndim != 2
        or not len(spatial)
        or spatial.shape[1] != input_features - 1
        or spatial.dtype != np.float64
        or not np.isfinite(spatial).all()
    ):
        raise ValueError(
            "normalized_spatial must be a nonempty finite float64 matrix matching model features"
        )
    if amplitudes.shape != (len(spatial), len(time)) or amplitudes.dtype not in (
        np.float32,
        np.float64,
    ):
        raise ValueError(
            "normalized_amplitudes must be float32 or float64 with observed trace/time shape"
        )
    return time, spatial, amplitudes


def _complete_trace_minimum_rate(schedule, minimum, initial):
    if schedule == "constant":
        if minimum is not None:
            raise ValueError("minimum_learning_rate requires learning_rate_schedule='cosine'")
        return None
    if schedule != "cosine":
        raise ValueError("learning_rate_schedule must be 'constant' or 'cosine'")
    value = _positive_finite_float(minimum, "minimum_learning_rate")
    if value >= initial:
        raise ValueError("minimum_learning_rate must be strictly less than learning_rate")
    return value


def _complete_trace_update(
    model,
    optimizer,
    time,
    spatial,
    amplitudes,
    rows,
    *,
    microbatch_size,
    device,
    step,
    normalized_time_offsets,
):
    point_count = len(rows) * len(time)
    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0
    for start in range(0, point_count, microbatch_size):
        stop = min(start + microbatch_size, point_count)
        trace_positions, time_indices = np.divmod(np.arange(start, stop), len(time))
        selected_rows = rows[trace_positions]
        coordinates = np.empty((stop - start, spatial.shape[1] + 1), dtype=np.float64)
        coordinates[:, 0] = time[time_indices]
        if normalized_time_offsets is not None:
            coordinates[:, 0] += normalized_time_offsets[selected_rows]
        coordinates[:, 1:] = spatial[selected_rows]
        coordinate_tensor, target_tensor = to_model_tensors(
            coordinates, amplitudes[selected_rows, time_indices], device=device
        )
        prediction = model(coordinate_tensor)
        if prediction.shape != target_tensor.shape:
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("model output shape must match the target shape")
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"non-finite training loss at step {step}")
        fraction = (stop - start) / point_count
        (loss * fraction).backward()
        batch_loss += loss_value * fraction
    if not math.isfinite(batch_loss):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite training loss at step {step}")
    optimizer.step()
    return batch_loss


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
