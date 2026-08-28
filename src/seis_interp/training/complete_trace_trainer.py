"""Train SIREN from random complete-trace batches and streamed global S/N."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import ModelCoordinateParameters
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
)
from seis_interp.training.checkpoints import save_siren_checkpoint
from seis_interp.training.ffid_batches import RandomCompleteTraceBatchSampler
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.trainer import build_loss

Reporter = Callable[[str], None]
GlobalSnrEvaluator = Callable[[torch.nn.Module], float]
COSINE_LEARNING_RATE_SCHEDULE = "cosine"


@dataclass(frozen=True)
class RandomCompleteTraceTrainingResult:
    """Best global-validation result and the complete epoch history."""

    best_epoch: int
    best_validation_global_snr_db: float
    epochs_completed: int
    global_steps: int
    stopped_early: bool
    training_trace_count: int
    validation_ffid_count: int
    history: tuple[dict[str, int | float], ...]


def train_siren_by_random_complete_traces(
    model: Siren,
    sampler: RandomCompleteTraceBatchSampler,
    validation_evaluator: GlobalSnrEvaluator,
    normalization: NormalizationParameters,
    *,
    device: torch.device | str,
    loss: str,
    optimizer: str,
    learning_rate: float,
    traces_per_update: int,
    steps_per_epoch: int,
    max_epochs: int,
    early_stopping_patience: int,
    validation_ffid_count: int,
    checkpoint_path: Path,
    model_coordinates: ModelCoordinateParameters | None = None,
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    learning_rate_schedule: str | None = None,
    minimum_learning_rate: float | None = None,
    training_evaluator: GlobalSnrEvaluator | None = None,
    reporter: Reporter | None = None,
) -> RandomCompleteTraceTrainingResult:
    """Train on random complete traces and select checkpoints by global validation S/N."""
    loss_function = build_loss(loss)
    if optimizer != "adam":
        raise ValueError(f"random_complete_traces supports only adam optimizer, got {optimizer!r}")
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    traces_value = _positive_integer(traces_per_update, "traces_per_update")
    steps_value = _positive_integer(steps_per_epoch, "steps_per_epoch")
    max_epochs_value = _positive_integer(max_epochs, "max_epochs")
    patience_value = _positive_integer(early_stopping_patience, "early_stopping_patience")
    validation_count = _positive_integer(validation_ffid_count, "validation_ffid_count")
    training_trace_count = _positive_integer(
        sampler.training_trace_count,
        "training_trace_count",
    )
    minimum_learning_rate_value = _validated_minimum_learning_rate(
        learning_rate_schedule,
        minimum_learning_rate,
        initial_learning_rate=learning_rate_value,
    )
    if traces_value > training_trace_count:
        raise ValueError(
            "traces_per_update must not exceed the number of available training traces "
            f"({training_trace_count})"
        )
    target_amplitude_scaling = validated_amplitude_scaling(amplitude_scaling)
    if sampler.amplitude_scaling != target_amplitude_scaling:
        raise ValueError(
            "amplitude_scaling must match the RandomCompleteTraceBatchSampler target scaling: "
            f"{target_amplitude_scaling!r} != {sampler.amplitude_scaling!r}"
        )
    report = reporter or _print_progress
    metric_prefix = (
        ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
        if target_amplitude_scaling == PER_TRACE_RMS_SCALING
        else None
    )
    validation_label = (
        f"{metric_prefix}_global_snr_db" if metric_prefix else "validation_global_snr_db"
    )
    training_label = (
        f"{metric_prefix}_training_global_snr_db" if metric_prefix else "training_global_snr_db"
    )
    progress_domain = f" validation_metric_domain={metric_prefix}" if metric_prefix else ""

    model.to(device)
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    learning_rate_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            torch_optimizer,
            T_max=max_epochs_value * steps_value,
            eta_min=minimum_learning_rate_value,
        )
        if minimum_learning_rate_value is not None
        else None
    )
    history: list[dict[str, int | float]] = []
    global_steps = 0
    best_epoch = 0
    best_validation_global_snr_db = -math.inf
    non_improving_epochs = 0
    stopped_early = False

    for epoch in range(1, max_epochs_value + 1):
        report(
            f"random_complete_traces {epoch}/{max_epochs_value} start: "
            f"steps_per_epoch={steps_value} traces_per_update={traces_value} "
            f"amplitude_scaling={target_amplitude_scaling}{progress_domain}"
        )
        model.train()
        batch_losses: list[float] = []
        for _ in range(steps_value):
            loss_value = _train_complete_trace_update(
                model,
                sampler,
                loss_function,
                torch_optimizer,
                traces_per_update=traces_value,
                device=device,
                epoch=epoch,
                global_step=global_steps + 1,
            )
            batch_losses.append(loss_value)
            global_steps += 1
            if learning_rate_scheduler is not None:
                learning_rate_scheduler.step()

        history_row: dict[str, int | float] = {
            "epoch": epoch,
            "global_step": global_steps,
            "mean_trace_batch_loss": float(np.mean(batch_losses, dtype=np.float64)),
        }
        if learning_rate_scheduler is not None:
            history_row["learning_rate"] = _optimizer_learning_rate(torch_optimizer)
        if training_evaluator is not None:
            history_row["training_global_snr_db"] = _validated_global_snr(
                training_evaluator(model),
                "training",
            )
        validation_global_snr_db = _validated_global_snr(
            validation_evaluator(model),
            "validation",
        )
        history_row["validation_global_snr_db"] = validation_global_snr_db
        history.append(history_row)

        training_progress = (
            f" {training_label}={history_row['training_global_snr_db']:.8g}"
            if "training_global_snr_db" in history_row
            else ""
        )
        learning_rate_progress = (
            f" learning_rate={history_row['learning_rate']:.8g}"
            if "learning_rate" in history_row
            else ""
        )
        report(
            f"random_complete_traces {epoch}/{max_epochs_value} end: "
            f"mean_trace_batch_loss={history_row['mean_trace_batch_loss']:.8g}"
            f"{learning_rate_progress}{training_progress} "
            f"{validation_label}={validation_global_snr_db:.8g}"
        )

        if epoch == 1 or validation_global_snr_db > best_validation_global_snr_db:
            best_epoch = epoch
            best_validation_global_snr_db = validation_global_snr_db
            non_improving_epochs = 0
            save_siren_checkpoint(
                checkpoint_path,
                model,
                normalization,
                model_coordinates=model_coordinates,
                amplitude_scaling=target_amplitude_scaling,
                epoch=epoch,
                global_step=global_steps,
                validation_median_trace_snr_db=None,
                validation_global_snr_db=validation_global_snr_db,
            )
        else:
            non_improving_epochs += 1
            if non_improving_epochs >= patience_value:
                stopped_early = True
                break

    return RandomCompleteTraceTrainingResult(
        best_epoch=best_epoch,
        best_validation_global_snr_db=best_validation_global_snr_db,
        epochs_completed=len(history),
        global_steps=global_steps,
        stopped_early=stopped_early,
        training_trace_count=training_trace_count,
        validation_ffid_count=validation_count,
        history=tuple(history),
    )


def _train_complete_trace_update(
    model: Siren,
    sampler: RandomCompleteTraceBatchSampler,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    traces_per_update: int,
    device: torch.device | str,
    epoch: int,
    global_step: int,
) -> float:
    """Run one optimizer update without retaining its batch during validation."""
    batch_coordinates, batch_targets = sampler.sample(traces_per_update)
    coordinate_tensor, target_tensor = to_model_tensors(
        batch_coordinates,
        batch_targets,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(coordinate_tensor)
    batch_loss = loss_function(prediction, target_tensor)
    loss_value = float(batch_loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise RuntimeError(
            "non-finite training loss for random_complete_traces: "
            f"epoch={epoch}, global_step={global_step}, loss={loss_value}"
        )
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_value


def _validated_global_snr(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} global S/N must be finite or positive infinity")
    converted = float(value)
    if math.isnan(converted) or converted == -math.inf:
        raise ValueError(f"{name} global S/N must be finite or positive infinity")
    return converted


def _validated_minimum_learning_rate(
    schedule: str | None,
    minimum_learning_rate: float | None,
    *,
    initial_learning_rate: float,
) -> float | None:
    if schedule is None:
        if minimum_learning_rate is not None:
            raise ValueError("minimum_learning_rate requires learning_rate_schedule")
        return None
    if schedule != COSINE_LEARNING_RATE_SCHEDULE:
        raise ValueError(f"learning_rate_schedule must be 'cosine' or None, got {schedule!r}")
    if minimum_learning_rate is None:
        raise ValueError("minimum_learning_rate is required when learning_rate_schedule='cosine'")
    minimum = _positive_finite_float(minimum_learning_rate, "minimum_learning_rate")
    if minimum >= initial_learning_rate:
        raise ValueError("minimum_learning_rate must be strictly less than learning_rate")
    return minimum


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("random_complete_traces optimizer must have one parameter group")
    learning_rate = float(optimizer.param_groups[0]["lr"])
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise RuntimeError("optimizer learning rate must remain positive and finite")
    return learning_rate


def _print_progress(message: str) -> None:
    print(message, flush=True)


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
