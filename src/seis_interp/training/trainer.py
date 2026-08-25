"""Minimal Adam training loop for SIREN with a configurable loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from seis_interp.evaluation.metrics import (
    median_trace_signal_to_noise_ratio_db,
    signal_to_noise_ratio_db,
)
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.training.checkpoints import save_siren_checkpoint
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import RandomPointSampler
from seis_interp.training.prediction import predict_points


@dataclass(frozen=True)
class TrainingResult:
    """Best validation result and the complete epoch history."""

    best_epoch: int
    best_validation_median_trace_snr_db: float
    best_validation_global_snr_db: float
    epochs_completed: int
    global_steps: int
    stopped_early: bool
    history: tuple[dict[str, int | float], ...]


def build_loss(name: str) -> torch.nn.Module:
    """Return the training loss selected by a configuration name."""
    if name == "l1":
        return torch.nn.L1Loss()
    if name == "l2":
        return torch.nn.MSELoss()
    raise ValueError(f"unsupported loss: {name}")


def train_siren(
    model: Siren,
    sampler: RandomPointSampler,
    validation_coordinates: np.ndarray,
    validation_targets: np.ndarray,
    normalization: NormalizationParameters,
    *,
    device: torch.device | str,
    loss: str,
    learning_rate: float,
    batch_size: int,
    steps_per_epoch: int,
    max_epochs: int,
    early_stopping_patience: int,
    validation_batch_size: int,
    validation_samples_per_trace: int,
    checkpoint_path: Path,
) -> TrainingResult:
    """Train a SIREN and save only improvements in median per-trace validation S/N."""
    loss_function = build_loss(loss)
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    batch_size_value = _positive_integer(batch_size, "batch_size")
    steps_value = _positive_integer(steps_per_epoch, "steps_per_epoch")
    max_epochs_value = _positive_integer(max_epochs, "max_epochs")
    patience_value = _positive_integer(early_stopping_patience, "early_stopping_patience")
    validation_batch_value = _positive_integer(validation_batch_size, "validation_batch_size")
    samples_per_trace = _positive_integer(
        validation_samples_per_trace, "validation_samples_per_trace"
    )
    coordinates, targets = _validated_validation_data(
        validation_coordinates, validation_targets, model.input_features
    )
    trace_targets = _trace_shaped(targets, samples_per_trace)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    history: list[dict[str, int | float]] = []
    global_steps = 0
    best_epoch = 0
    best_validation_median_trace_snr_db = -math.inf
    best_validation_global_snr_db = -math.inf
    non_improving_epochs = 0
    stopped_early = False

    for epoch in range(1, max_epochs_value + 1):
        model.train()
        training_losses: list[float] = []
        for _ in range(steps_value):
            batch_coordinates, batch_targets = sampler.sample(batch_size_value)
            coordinate_tensor, target_tensor = to_model_tensors(
                batch_coordinates, batch_targets, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinate_tensor)
            loss = loss_function(prediction, target_tensor)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu().item()))
            global_steps += 1

        validation_prediction = predict_points(
            model,
            coordinates,
            batch_size=validation_batch_value,
            device=device,
        )
        validation_global_snr_db = signal_to_noise_ratio_db(targets, validation_prediction)
        validation_median_trace_snr_db = median_trace_signal_to_noise_ratio_db(
            trace_targets, _trace_shaped(validation_prediction, samples_per_trace)
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_steps,
                "train_loss": float(np.mean(training_losses, dtype=np.float64)),
                "validation_global_snr_db": validation_global_snr_db,
                "validation_median_trace_snr_db": validation_median_trace_snr_db,
            }
        )

        if epoch == 1 or validation_median_trace_snr_db > best_validation_median_trace_snr_db:
            best_epoch = epoch
            best_validation_median_trace_snr_db = validation_median_trace_snr_db
            best_validation_global_snr_db = validation_global_snr_db
            non_improving_epochs = 0
            save_siren_checkpoint(
                checkpoint_path,
                model,
                normalization,
                epoch=epoch,
                global_step=global_steps,
                validation_median_trace_snr_db=validation_median_trace_snr_db,
                validation_global_snr_db=validation_global_snr_db,
            )
        else:
            non_improving_epochs += 1
            if non_improving_epochs >= patience_value:
                stopped_early = True
                break

    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_median_trace_snr_db=best_validation_median_trace_snr_db,
        best_validation_global_snr_db=best_validation_global_snr_db,
        epochs_completed=len(history),
        global_steps=global_steps,
        stopped_early=stopped_early,
        history=tuple(history),
    )


def _validated_validation_data(
    coordinates: np.ndarray,
    targets: np.ndarray,
    input_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_array = np.asarray(coordinates)
    target_array = np.asarray(targets)
    if coordinate_array.ndim != 2 or coordinate_array.shape[0] == 0:
        raise ValueError("validation_coordinates must be a non-empty two-dimensional array")
    if coordinate_array.shape[1] != input_features:
        raise ValueError(
            f"model expects {input_features} input features but validation coordinates "
            f"have {coordinate_array.shape[1]}"
        )
    if target_array.ndim != 1 or target_array.shape[0] != coordinate_array.shape[0]:
        raise ValueError(
            "validation_targets must have shape (n_validation_points,) matching coordinates"
        )
    return coordinate_array, target_array


def _trace_shaped(values: np.ndarray, samples_per_trace: int) -> np.ndarray:
    """Reshape trace-major validation points into one row per validation trace."""
    if values.shape[0] % samples_per_trace != 0:
        raise ValueError(
            f"validation point count {values.shape[0]} must be divisible by "
            f"validation_samples_per_trace {samples_per_trace}"
        )
    return values.reshape(-1, samples_per_trace)


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
