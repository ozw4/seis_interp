"""Fixed-step observed-only training for a profile-wise NeRSI model."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.models.nersi import Nersi
from seis_interp.training.c3_volume_nersi_data import C3VolumeNersiData
from seis_interp.training.nersi_coordinate_jitter import jitter_profile_coordinates
from seis_interp.training.nersi_optimization import (
    nersi_step_learning_rate,
    validate_nersi_optimization,
)
from seis_interp.training.nersi_profile_mixup import observed_neighbor_profile_mixup
from seis_interp.training.trace_relative_loss import masked_trace_loss

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class FixedStepNersiResult:
    """Completed updates, final batch loss, and interval-mean loss history."""

    steps_completed: int
    supervised_trace_presentations: int
    final_batch_loss: float
    history: tuple[dict[str, int | float], ...]


def observed_profile_trace_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_trace_mask: torch.Tensor,
    *,
    loss_name: str = "masked_trace_relative_mse",
) -> torch.Tensor:
    """Select observed complete traces and apply the shared neural loss."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction must have shape (batch, 1, time, receiver_y)")
    if target.shape != prediction.shape:
        raise ValueError("target shape must match prediction shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must contain floating-point values")
    if not isinstance(observed_trace_mask, torch.Tensor):
        raise TypeError("observed_trace_mask must be a tensor")
    if observed_trace_mask.dtype != torch.bool:
        raise ValueError("observed_trace_mask must be boolean")
    if observed_trace_mask.device != prediction.device or target.device != prediction.device:
        raise ValueError("prediction, target, and observed_trace_mask must share a device")
    expected_mask_shape = (prediction.shape[0], prediction.shape[3])
    if tuple(observed_trace_mask.shape) != expected_mask_shape:
        raise ValueError(
            "observed_trace_mask must have shape (batch, receiver_y) matching prediction"
        )
    if not bool(torch.any(observed_trace_mask)):
        raise ValueError("observed_profile_trace_loss requires at least one observed trace")

    prediction_traces = prediction[:, 0].transpose(1, 2)[observed_trace_mask]
    target_traces = target[:, 0].transpose(1, 2)[observed_trace_mask]
    return masked_trace_loss(prediction_traces, target_traces, loss_name=loss_name)


def train_nersi_fixed_steps(
    model: Nersi,
    data: C3VolumeNersiData,
    *,
    device: torch.device | str,
    learning_rate: float,
    profiles_per_step: int,
    max_steps: int,
    report_interval: int,
    random_seed: int,
    loss_name: str = "masked_trace_relative_mse",
    reporter: Reporter | None = None,
    coordinate_jitter_cells: float = 0.0,
    profile_mixup_max_fraction: float = 0.0,
    augmentation_seed: int = 0,
    optimization: dict | None = None,
    gradient_accumulation_steps: int = 1,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> FixedStepNersiResult:
    """Perform exactly ``max_steps`` Adam/AdamW updates on observed profile samples.

    Profiles are sampled without replacement from the data adapter's stable
    training candidates. A full-candidate update retains that stable order and
    consumes no random draw. Model initialization is owned by the caller; this
    function uses a local NumPy generator for profile sampling. Optional augmentation
    has an independent generator. Mixup appends virtual profiles at the two parents'
    common observed receivers without replacing the original supervised batch.
    Accumulation samples one effective batch without replacement, then splits it
    into micro-batches. Adam, the schedule and EMA advance once per effective batch.
    """
    if not isinstance(model, Nersi):
        raise TypeError("model must be a Nersi")
    _validate_training_data(data)
    rate = _positive_finite_float(learning_rate, "learning_rate")
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")
    decay_rate = _nonnegative_finite_float(weight_decay, "weight_decay")
    if optimizer_name != "adamw" and decay_rate:
        raise ValueError("weight_decay requires optimizer_name='adamw'")
    options = None if optimization is None else validate_nersi_optimization(optimization, rate)
    batch_count = _positive_integer(profiles_per_step, "profiles_per_step")
    accumulation = _positive_integer(gradient_accumulation_steps, "gradient_accumulation_steps")
    effective_batch_count = batch_count * accumulation
    steps = _positive_integer(max_steps, "max_steps")
    interval = _positive_integer(report_interval, "report_interval")
    seed = _nonnegative_integer(random_seed, "random_seed")
    if (
        isinstance(coordinate_jitter_cells, bool)
        or not isinstance(coordinate_jitter_cells, Real)
        or not math.isfinite(coordinate_jitter_cells)
        or not 0 <= coordinate_jitter_cells <= 0.5
    ):
        raise ValueError("coordinate_jitter_cells must be finite and in [0, 0.5]")
    augmentation_random_seed = _nonnegative_integer(augmentation_seed, "augmentation_seed")
    if (
        isinstance(profile_mixup_max_fraction, bool)
        or not isinstance(profile_mixup_max_fraction, Real)
        or not math.isfinite(profile_mixup_max_fraction)
        or not 0 <= profile_mixup_max_fraction <= 0.5
    ):
        raise ValueError("profile_mixup_max_fraction must be finite and in [0, 0.5]")
    if coordinate_jitter_cells and profile_mixup_max_fraction:
        raise ValueError("coordinate jitter and profile mixup must not be combined")
    jitter_rng = (
        np.random.default_rng(augmentation_random_seed) if coordinate_jitter_cells else None
    )
    mixup_rng = (
        np.random.default_rng(augmentation_random_seed) if profile_mixup_max_fraction else None
    )
    candidates = data.training_profile_indices
    if batch_count > len(candidates):
        raise ValueError("profiles_per_step must not exceed the available training profile count")
    if effective_batch_count > len(candidates):
        raise ValueError(
            "profiles_per_step * gradient_accumulation_steps must not exceed "
            "the available training profile count"
        )

    rng = np.random.default_rng(seed)
    model.to(device)
    model.train()
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=rate)
        if optimizer_name == "adam"
        else torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay_rate)
    )
    decay = None if options is None else options["ema_decay"]
    averaged = None
    history: list[dict[str, int | float]] = []
    interval_losses: list[float] = []
    supervised_trace_presentations = 0

    for step in range(1, steps + 1):
        if options is not None:
            optimizer.param_groups[0]["lr"] = nersi_step_learning_rate(step, steps, rate, options)
        selected = (
            candidates
            if effective_batch_count == len(candidates)
            else rng.choice(candidates, size=effective_batch_count, replace=False)
        )
        batch_coordinates = np.ascontiguousarray(data.normalized_coordinates[selected])
        batch_targets = np.ascontiguousarray(data.normalized_profiles[selected])
        batch_mask = np.ascontiguousarray(data.observed_trace_mask[selected])
        if jitter_rng is not None:
            batch_coordinates = jitter_profile_coordinates(
                batch_coordinates,
                data.spatial_shape,
                jitter_cells=coordinate_jitter_cells,
                rng=jitter_rng,
            )
        if mixup_rng is not None:
            extra_coordinates, extra_targets, extra_mask = observed_neighbor_profile_mixup(
                data, selected, max_fraction=profile_mixup_max_fraction, rng=mixup_rng
            )
            batch_coordinates = np.concatenate((batch_coordinates, extra_coordinates))
            batch_targets = np.concatenate((batch_targets, extra_targets))
            batch_mask = np.concatenate((batch_mask, extra_mask))
        optimizer.zero_grad(set_to_none=True)
        trace_count = int(np.count_nonzero(batch_mask))
        chunk_size = len(batch_coordinates) if accumulation == 1 else batch_count
        final_batch_loss = 0.0
        for start in range(0, len(batch_coordinates), chunk_size):
            stop = start + chunk_size
            coordinates = torch.as_tensor(
                batch_coordinates[start:stop], dtype=torch.float32, device=device
            )
            targets = torch.as_tensor(batch_targets[start:stop], dtype=torch.float32, device=device)
            mask = torch.as_tensor(batch_mask[start:stop], dtype=torch.bool, device=device)
            prediction = model(coordinates)
            loss = observed_profile_trace_loss(prediction, targets, mask, loss_name=loss_name)
            if accumulation != 1:
                # Micro-batches have unequal O counts: equal micro-loss weights are biased.
                loss = loss * (int(np.count_nonzero(batch_mask[start:stop])) / trace_count)
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite training loss at step {step}")
            final_batch_loss += value
            loss.backward()
        optimizer.step()
        if decay is not None:
            with torch.no_grad():
                if averaged is None:
                    averaged = [parameter.detach().clone() for parameter in model.parameters()]
                else:
                    for average, parameter in zip(averaged, model.parameters(), strict=True):
                        average.lerp_(parameter, 1 - decay)
        supervised_trace_presentations += trace_count
        interval_losses.append(final_batch_loss)

        if step % interval == 0 or step == steps:
            train_loss = float(np.mean(interval_losses, dtype=np.float64))
            history.append({"step": step, "train_loss": train_loss})
            if options is not None:
                history[-1]["learning_rate"] = optimizer.param_groups[0]["lr"]
            interval_losses.clear()
            if reporter is not None:
                reporter(f"nersi_volume step {step}/{steps}: train_loss={train_loss:.8g}")

    if averaged is not None:
        with torch.no_grad():
            for parameter, average in zip(model.parameters(), averaged, strict=True):
                parameter.copy_(average)
    return FixedStepNersiResult(
        steps_completed=steps,
        supervised_trace_presentations=supervised_trace_presentations,
        final_batch_loss=final_batch_loss,
        history=tuple(history),
    )


def _validate_training_data(data: C3VolumeNersiData) -> None:
    if not isinstance(data, C3VolumeNersiData):
        raise TypeError("data must be C3VolumeNersiData")
    coordinates = data.normalized_coordinates
    profiles = data.normalized_profiles
    mask = data.observed_trace_mask
    candidates = data.training_profile_indices
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or not len(coordinates)
        or coordinates.dtype.kind != "f"
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("normalized_coordinates must be a nonempty finite (N, 3) float array")
    if (
        not isinstance(profiles, np.ndarray)
        or profiles.shape != (len(coordinates), 1, *tuple(data.profile_shape))
        or profiles.dtype.kind != "f"
    ):
        raise ValueError("normalized_profiles must match profile count and profile_shape")
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != (len(coordinates), data.profile_shape[1])
    ):
        raise ValueError("observed_trace_mask must be boolean with shape (N, receiver_y)")
    expected_candidates = np.flatnonzero(np.any(mask, axis=1)).astype(np.int64, copy=False)
    if (
        not isinstance(candidates, np.ndarray)
        or candidates.dtype != np.int64
        or candidates.ndim != 1
        or not np.array_equal(candidates, expected_candidates)
    ):
        raise ValueError(
            "training_profile_indices must contain exactly the observed profiles in stable order"
        )
    if not len(candidates):
        raise ValueError("training_profile_indices must contain at least one profile")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _nonnegative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be nonnegative and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return converted
