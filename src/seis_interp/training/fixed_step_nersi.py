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
) -> FixedStepNersiResult:
    """Perform exactly ``max_steps`` Adam updates on observed profile samples.

    Profiles are sampled without replacement from the data adapter's stable
    training candidates. A full-candidate update retains that stable order and
    consumes no random draw. Model initialization is owned by the caller; this
    function uses only a local NumPy generator for profile sampling.
    """
    if not isinstance(model, Nersi):
        raise TypeError("model must be a Nersi")
    _validate_training_data(data)
    rate = _positive_finite_float(learning_rate, "learning_rate")
    batch_count = _positive_integer(profiles_per_step, "profiles_per_step")
    steps = _positive_integer(max_steps, "max_steps")
    interval = _positive_integer(report_interval, "report_interval")
    seed = _nonnegative_integer(random_seed, "random_seed")
    candidates = data.training_profile_indices
    if batch_count > len(candidates):
        raise ValueError("profiles_per_step must not exceed the available training profile count")

    rng = np.random.default_rng(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    history: list[dict[str, int | float]] = []
    interval_losses: list[float] = []
    supervised_trace_presentations = 0

    for step in range(1, steps + 1):
        selected = (
            candidates
            if batch_count == len(candidates)
            else rng.choice(candidates, size=batch_count, replace=False)
        )
        coordinates = torch.as_tensor(
            np.ascontiguousarray(data.normalized_coordinates[selected]),
            dtype=torch.float32,
            device=device,
        )
        targets = torch.as_tensor(
            np.ascontiguousarray(data.normalized_profiles[selected]),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.as_tensor(
            np.ascontiguousarray(data.observed_trace_mask[selected]),
            dtype=torch.bool,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinates)
        loss = observed_profile_trace_loss(
            prediction,
            targets,
            mask,
            loss_name=loss_name,
        )
        final_batch_loss = float(loss.detach().cpu())
        if not math.isfinite(final_batch_loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        optimizer.step()
        supervised_trace_presentations += int(np.count_nonzero(data.observed_trace_mask[selected]))
        interval_losses.append(final_batch_loss)

        if step % interval == 0 or step == steps:
            train_loss = float(np.mean(interval_losses, dtype=np.float64))
            history.append({"step": step, "train_loss": train_loss})
            interval_losses.clear()
            if reporter is not None:
                reporter(f"nersi_volume step {step}/{steps}: train_loss={train_loss:.8g}")

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
