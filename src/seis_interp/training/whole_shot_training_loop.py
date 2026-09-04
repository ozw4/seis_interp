"""Shared loop control for whole-shot gather trainers.

Owns argument validation, optimizer/scheduler setup, the step loop, validation
cadence, strict best-checkpoint selection, history, and progress reporting.
Model-specific forward/loss code and its autocast scope stay in the trainers
and enter through the ``training_step`` callback.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.training.amplitude_scaling import ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
from seis_interp.training.whole_shot_batches import (
    WholeShotBatch,
    WholeShotBatchProvider,
    validated_whole_shot_batch,
)

MINIMUM_LEARNING_RATE_FACTOR = 0.03
MAX_GRADIENT_NORM = 1.0


@dataclass(frozen=True)
class WholeShotStepResult:
    """One training step's loss plus ordered history and finite-check tensors."""

    loss: torch.Tensor
    history_metrics: tuple[tuple[str, torch.Tensor], ...]
    finite_checks: tuple[tuple[str, torch.Tensor], ...] = ()


@dataclass(frozen=True)
class WholeShotLoopResult:
    """Best raw validation result and history at every validation step."""

    best_step: int
    best_validation_global_snr_db: float
    steps_completed: int
    training_ffid_count: int
    training_trace_count: int
    history: tuple[dict[str, int | float], ...]


def run_whole_shot_training_loop(
    model: torch.nn.Module,
    batch_provider: WholeShotBatchProvider,
    validation_evaluator: Callable[[torch.nn.Module], float],
    *,
    training_step: Callable[
        [torch.nn.Module, WholeShotBatch, bool],
        WholeShotStepResult,
    ],
    checkpoint_saver: Callable[..., None],
    progress_name: str,
    progress_metric_labels: tuple[tuple[str, str], ...],
    device: torch.device | str,
    generator: torch.Generator,
    checkpoint_path: Path,
    total_steps: int,
    batch_size: int,
    neighbor_dropout: float,
    learning_rate: float,
    weight_decay: float,
    validation_interval: int,
    use_bfloat16: bool,
    training_ffid_count: int,
    training_trace_count: int,
    reporter: Callable[[str], None] | None = None,
) -> WholeShotLoopResult:
    """Train whole-shot batches and select by strict best raw validation S/N."""
    steps = _positive_integer(total_steps, "total_steps")
    batch_size_value = _positive_integer(batch_size, "batch_size")
    validation_interval_value = _positive_integer(
        validation_interval,
        "validation_interval",
    )
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    weight_decay_value = _nonnegative_finite_float(weight_decay, "weight_decay")
    neighbor_dropout_value = _probability_below_one(neighbor_dropout, "neighbor_dropout")
    training_ffids = _positive_integer(training_ffid_count, "training_ffid_count")
    training_traces = _positive_integer(training_trace_count, "training_trace_count")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator seeded by the caller")
    if not isinstance(use_bfloat16, bool):
        raise ValueError("use_bfloat16 must be a boolean")

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
    best_validation_global_snr_db = _evaluate_validation(model, validation_evaluator)
    history.append(
        {
            "step": 0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_global_snr_db": best_validation_global_snr_db,
        }
    )
    checkpoint_saver(
        checkpoint_path,
        model,
        best_step=best_step,
        best_validation_global_snr_db=best_validation_global_snr_db,
    )
    report(
        f"{progress_name} 0/{steps}: "
        f"{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}_global_snr_db="
        f"{best_validation_global_snr_db:.8g}"
    )

    for step in range(1, steps + 1):
        model.train()
        raw_batch = batch_provider(
            batch_size_value,
            generator=generator,
            neighbor_dropout=neighbor_dropout_value,
        )
        batch = validated_whole_shot_batch(raw_batch, batch_size=batch_size_value)
        device_batch = tuple(value.to(device_value) for value in batch)

        optimizer.zero_grad(set_to_none=True)
        step_result = training_step(model, device_batch, use_cuda_bfloat16)
        _require_finite_tensor(step_result.loss, "training loss", step=step)
        for name, value in step_result.finite_checks:
            _require_finite_tensor(value, name, step=step)
        step_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRADIENT_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()

        should_validate = step % validation_interval_value == 0 or step == steps
        if should_validate:
            learning_rate_at_step = float(optimizer.param_groups[0]["lr"])
            validation_global_snr_db = _evaluate_validation(model, validation_evaluator)
            history_row: dict[str, int | float] = {
                "step": step,
                "loss": float(step_result.loss.detach().cpu()),
            }
            for key, value in step_result.history_metrics:
                history_row[key] = float(value.detach().cpu())
            history_row["learning_rate"] = learning_rate_at_step
            history_row["validation_global_snr_db"] = validation_global_snr_db
            history.append(history_row)
            if validation_global_snr_db > best_validation_global_snr_db:
                best_step = step
                best_validation_global_snr_db = validation_global_snr_db
                checkpoint_saver(
                    checkpoint_path,
                    model,
                    best_step=best_step,
                    best_validation_global_snr_db=best_validation_global_snr_db,
                )
            progress_fields = " ".join(
                f"{label}={history_row[key]:.8g}" for key, label in progress_metric_labels
            )
            report(
                f"{progress_name} {step}/{steps}: "
                f"{progress_fields} "
                f"learning_rate={learning_rate_at_step:.8g} "
                f"{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}_global_snr_db="
                f"{validation_global_snr_db:.8g}"
            )

        del raw_batch, batch, device_batch, step_result

    return WholeShotLoopResult(
        best_step=best_step,
        best_validation_global_snr_db=best_validation_global_snr_db,
        steps_completed=steps,
        training_ffid_count=training_ffids,
        training_trace_count=training_traces,
        history=tuple(history),
    )


def _evaluate_validation(
    model: torch.nn.Module,
    validation_evaluator: Callable[[torch.nn.Module], float],
) -> float:
    """Evaluate one full validation checkpoint candidate."""
    model.eval()
    with torch.inference_mode():
        return _finite_float(
            validation_evaluator(model),
            "validation global S/N",
        )


def _require_finite_tensor(value: torch.Tensor, name: str, *, step: int) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} is non-finite at step {step}")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _nonnegative_finite_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
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
