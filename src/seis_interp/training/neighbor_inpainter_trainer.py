"""Data-source-agnostic training loop for the neighbor-trace inpainter."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import torch

from seis_interp.models.neighbor_trace_inpainter import NeighborTraceInpainter
from seis_interp.training.amplitude_scaling import ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
from seis_interp.training.neighbor_inpainter_checkpoints import (
    save_neighbor_inpainter_checkpoint,
)

DEFAULT_TOTAL_STEPS = 2500
DEFAULT_BATCH_SIZE = 96
DEFAULT_NEIGHBOR_DROPOUT = 0.05
DEFAULT_DERIVATIVE_WEIGHT = 0.1
DEFAULT_LEARNING_RATE = 5.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_VALIDATION_INTERVAL = 100
MINIMUM_LEARNING_RATE_FACTOR = 0.03
MAX_GRADIENT_NORM = 1.0

NeighborInpainterBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
Reporter = Callable[[str], None]


class NeighborInpainterBatchProvider(Protocol):
    """Supply one random training batch using caller-owned randomness."""

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> NeighborInpainterBatch: ...


class NeighborInpainterValidationEvaluator(Protocol):
    """Return raw global validation S/N for the supplied model."""

    def __call__(self, model: NeighborTraceInpainter) -> float: ...


@dataclass(frozen=True)
class NeighborInpainterTrainingResult:
    """Best raw validation result and history at every validation step."""

    best_step: int
    best_validation_global_snr_db: float
    steps_completed: int
    training_trace_count: int | None
    history: tuple[dict[str, int | float], ...]


def train_neighbor_trace_inpainter(
    model: NeighborTraceInpainter,
    batch_provider: NeighborInpainterBatchProvider,
    validation_evaluator: NeighborInpainterValidationEvaluator,
    *,
    device: torch.device | str,
    generator: torch.Generator,
    checkpoint_path: Path,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    neighbor_dropout: float = DEFAULT_NEIGHBOR_DROPOUT,
    derivative_weight: float = DEFAULT_DERIVATIVE_WEIGHT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL,
    use_bfloat16: bool = True,
    training_trace_count: int | None = None,
    reporter: Reporter | None = None,
) -> NeighborInpainterTrainingResult:
    """Train with the successful proxy recipe and select by raw validation S/N.

    The caller owns all seed setup and passes the seeded ``generator`` used by
    ``batch_provider`` for both target sampling and neighbor dropout. The
    trainer does not inspect or seed any global random-number generator.
    """
    steps = _positive_integer(total_steps, "total_steps")
    batch_size_value = _positive_integer(batch_size, "batch_size")
    validation_interval_value = _positive_integer(
        validation_interval,
        "validation_interval",
    )
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    weight_decay_value = _nonnegative_finite_float(weight_decay, "weight_decay")
    neighbor_dropout_value = _probability_below_one(neighbor_dropout, "neighbor_dropout")
    derivative_weight_value = _nonnegative_finite_float(
        derivative_weight,
        "derivative_weight",
    )
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator seeded by the caller")
    if not isinstance(use_bfloat16, bool):
        raise ValueError("use_bfloat16 must be a boolean")
    training_count = (
        None
        if training_trace_count is None
        else _positive_integer(training_trace_count, "training_trace_count")
    )
    device_value = torch.device(device)
    report = reporter or _print_progress

    model.to(device_value)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_value,
        weight_decay=weight_decay_value,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=learning_rate_value * MINIMUM_LEARNING_RATE_FACTOR,
    )
    use_cuda_bfloat16 = use_bfloat16 and device_value.type == "cuda"
    history: list[dict[str, int | float]] = []
    best_step = 0
    best_validation_global_snr_db = -math.inf

    for step in range(1, steps + 1):
        model.train()
        raw_batch = batch_provider(
            batch_size_value,
            generator=generator,
            neighbor_dropout=neighbor_dropout_value,
        )
        neighbors, availability, target_coordinates, targets = _validated_batch(
            raw_batch,
            model=model,
            batch_size=batch_size_value,
        )
        neighbors = neighbors.to(device_value)
        availability = availability.to(device_value)
        target_coordinates = target_coordinates.to(device_value)
        targets = targets.to(device_value)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_cuda_bfloat16
            else nullcontext()
        )
        with autocast_context:
            prediction = model(neighbors, availability, target_coordinates).float()
            mse = torch.square(prediction - targets).mean()
            derivative_mse = torch.square(
                torch.diff(prediction, dim=1) - torch.diff(targets, dim=1)
            ).mean()
            loss = mse + derivative_weight_value * derivative_mse
        _require_finite_tensor(loss, "training loss", step=step)
        _require_finite_tensor(mse, "training MSE", step=step)
        _require_finite_tensor(derivative_mse, "training derivative MSE", step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRADIENT_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()

        should_validate = step == 1 or step % validation_interval_value == 0 or step == steps
        if should_validate:
            learning_rate_at_step = _optimizer_learning_rate(optimizer)
            model.eval()
            with torch.inference_mode():
                validation_global_snr_db = _finite_float(
                    validation_evaluator(model),
                    "validation global S/N",
                )
            history_row: dict[str, int | float] = {
                "step": step,
                "loss": _tensor_float(loss),
                "mse": _tensor_float(mse),
                "derivative_mse": _tensor_float(derivative_mse),
                "learning_rate": learning_rate_at_step,
                "validation_global_snr_db": validation_global_snr_db,
            }
            history.append(history_row)
            if validation_global_snr_db > best_validation_global_snr_db:
                best_step = step
                best_validation_global_snr_db = validation_global_snr_db
                save_neighbor_inpainter_checkpoint(
                    checkpoint_path,
                    model,
                    best_step=best_step,
                    best_validation_global_snr_db=best_validation_global_snr_db,
                )
            report(
                f"neighbor_trace_inpainter {step}/{steps}: "
                f"loss={history_row['loss']:.8g} mse={history_row['mse']:.8g} "
                f"derivative_mse={history_row['derivative_mse']:.8g} "
                f"learning_rate={learning_rate_at_step:.8g} "
                f"{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}_global_snr_db="
                f"{validation_global_snr_db:.8g}"
            )

        del raw_batch, neighbors, availability, target_coordinates, targets

    return NeighborInpainterTrainingResult(
        best_step=best_step,
        best_validation_global_snr_db=best_validation_global_snr_db,
        steps_completed=steps,
        training_trace_count=training_count,
        history=tuple(history),
    )


def _validated_batch(
    batch: object,
    *,
    model: NeighborTraceInpainter,
    batch_size: int,
) -> NeighborInpainterBatch:
    if not isinstance(batch, tuple) or len(batch) != 4:
        raise TypeError(
            "batch_provider must return a four-tensor tuple: "
            "(neighbors, availability, target_coordinates, targets)"
        )
    names = ("neighbors", "availability", "target_coordinates", "targets")
    if not all(isinstance(value, torch.Tensor) for value in batch):
        invalid_name = next(
            name
            for name, value in zip(names, batch, strict=True)
            if not isinstance(value, torch.Tensor)
        )
        raise TypeError(f"batch {invalid_name} must be a torch.Tensor")
    neighbors, availability, target_coordinates, targets = batch
    expected_neighbors = (batch_size, model.neighbor_count)
    if neighbors.ndim != 3 or neighbors.shape[:2] != expected_neighbors:
        raise ValueError(
            "batch neighbors must have shape "
            f"({batch_size}, {model.neighbor_count}, time), got {tuple(neighbors.shape)}"
        )
    time_count = neighbors.shape[2]
    if time_count < 2:
        raise ValueError("training traces must contain at least two time samples")
    if availability.shape != expected_neighbors:
        raise ValueError(
            f"batch availability must have shape {expected_neighbors}, "
            f"got {tuple(availability.shape)}"
        )
    if target_coordinates.shape != (batch_size, 3):
        raise ValueError(
            f"batch target_coordinates must have shape ({batch_size}, 3), "
            f"got {tuple(target_coordinates.shape)}"
        )
    if targets.shape != (batch_size, time_count):
        raise ValueError(
            f"batch targets must have shape ({batch_size}, {time_count}), "
            f"got {tuple(targets.shape)}"
        )
    if not neighbors.is_floating_point():
        raise TypeError("batch neighbors must have a floating-point dtype")
    if availability.dtype != torch.bool and not availability.is_floating_point():
        raise TypeError("batch availability must have a boolean or floating-point dtype")
    if not target_coordinates.is_floating_point():
        raise TypeError("batch target_coordinates must have a floating-point dtype")
    if not targets.is_floating_point():
        raise TypeError("batch targets must have a floating-point dtype")
    return neighbors, availability, target_coordinates, targets


def _require_finite_tensor(value: torch.Tensor, name: str, *, step: int) -> None:
    if not bool(torch.isfinite(value.detach()).item()):
        raise RuntimeError(f"non-finite {name} at step {step}")


def _tensor_float(value: torch.Tensor) -> float:
    converted = float(value.detach().cpu().item())
    if not math.isfinite(converted):
        raise RuntimeError("history values must remain finite")
    return converted


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("neighbor inpainter optimizer must have one parameter group")
    return _positive_finite_float(
        optimizer.param_groups[0]["lr"],
        "optimizer learning rate",
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _nonnegative_finite_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return converted


def _probability_below_one(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0 or converted >= 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return converted


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _print_progress(message: str) -> None:
    print(message, flush=True)
