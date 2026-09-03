"""Data-source-agnostic training loop for trace-graph gather interpolation."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import torch

from seis_interp.models.trace_graph_interpolator import TraceGraphInterpolator
from seis_interp.training.amplitude_scaling import ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
from seis_interp.training.trace_graph_checkpoints import save_trace_graph_checkpoint
from seis_interp.training.trace_graph_losses import (
    amplitude_envelope_loss,
    masked_mean_square,
    slope_consistency_loss,
    spectrum_loss,
)
from seis_interp.training.whole_shot_batches import (
    WholeShotBatchProvider,
    validated_whole_shot_batch,
)

MINIMUM_LEARNING_RATE_FACTOR = 0.03
MAX_GRADIENT_NORM = 1.0

Reporter = Callable[[str], None]


class TraceGraphValidationEvaluator(Protocol):
    """Return raw global validation S/N for the supplied model."""

    def __call__(self, model: TraceGraphInterpolator) -> float: ...


@dataclass(frozen=True)
class TraceGraphTrainingResult:
    """Best raw validation result and history at every validation step."""

    best_step: int
    best_validation_global_snr_db: float
    steps_completed: int
    training_ffid_count: int
    training_trace_count: int
    history: tuple[dict[str, int | float], ...]


def train_trace_graph_interpolator(
    model: TraceGraphInterpolator,
    batch_provider: WholeShotBatchProvider,
    validation_evaluator: TraceGraphValidationEvaluator,
    *,
    device: torch.device | str,
    generator: torch.Generator,
    checkpoint_path: Path,
    total_steps: int,
    batch_size: int,
    neighbor_dropout: float,
    spectrum_weight: float,
    slope_weight: float,
    amplitude_weight: float,
    learning_rate: float,
    weight_decay: float,
    validation_interval: int,
    use_bfloat16: bool,
    training_ffid_count: int,
    training_trace_count: int,
    reporter: Reporter | None = None,
) -> TraceGraphTrainingResult:
    """Train masked whole-shot batches and select by raw validation global S/N."""
    if not isinstance(model, TraceGraphInterpolator):
        raise TypeError("model must be a TraceGraphInterpolator")
    steps = _positive_integer(total_steps, "total_steps")
    batch_size_value = _positive_integer(batch_size, "batch_size")
    validation_interval_value = _positive_integer(
        validation_interval,
        "validation_interval",
    )
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    weight_decay_value = _nonnegative_finite_float(weight_decay, "weight_decay")
    neighbor_dropout_value = _probability_below_one(neighbor_dropout, "neighbor_dropout")
    spectrum_weight_value = _nonnegative_finite_float(spectrum_weight, "spectrum_weight")
    slope_weight_value = _nonnegative_finite_float(slope_weight, "slope_weight")
    amplitude_weight_value = _nonnegative_finite_float(amplitude_weight, "amplitude_weight")
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
    save_trace_graph_checkpoint(
        checkpoint_path,
        model,
        best_step=best_step,
        best_validation_global_snr_db=best_validation_global_snr_db,
    )
    report(
        f"trace_graph_interpolator 0/{steps}: "
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
        (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            targets,
            target_availability,
        ) = (value.to(device_value) for value in batch)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_cuda_bfloat16
            else nullcontext()
        )
        with autocast_context:
            prediction = model(
                neighbors,
                availability,
                source_deltas,
                target_coordinates,
            ).float()
        mask_mse = masked_mean_square(prediction, targets, target_availability)
        loss = mask_mse
        zero = torch.zeros((), dtype=mask_mse.dtype, device=mask_mse.device)
        spectrum_value = zero
        slope_value = zero
        amplitude_value = zero
        if spectrum_weight_value > 0.0:
            spectrum_value = spectrum_loss(prediction, targets, target_availability)
            loss = loss + spectrum_weight_value * spectrum_value
        if slope_weight_value > 0.0:
            slope_value = slope_consistency_loss(prediction, targets, target_availability)
            loss = loss + slope_weight_value * slope_value
        if amplitude_weight_value > 0.0:
            amplitude_value = amplitude_envelope_loss(
                prediction,
                targets,
                target_availability,
            )
            loss = loss + amplitude_weight_value * amplitude_value
        _require_finite_tensor(loss, "training loss", step=step)
        _require_finite_tensor(mask_mse, "training mask MSE", step=step)
        loss.backward()
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
                "loss": float(loss.detach().cpu()),
                "mask_mse": float(mask_mse.detach().cpu()),
                "spectrum_loss": float(spectrum_value.detach().cpu()),
                "slope_loss": float(slope_value.detach().cpu()),
                "amplitude_loss": float(amplitude_value.detach().cpu()),
                "learning_rate": learning_rate_at_step,
                "validation_global_snr_db": validation_global_snr_db,
            }
            history.append(history_row)
            if validation_global_snr_db > best_validation_global_snr_db:
                best_step = step
                best_validation_global_snr_db = validation_global_snr_db
                save_trace_graph_checkpoint(
                    checkpoint_path,
                    model,
                    best_step=best_step,
                    best_validation_global_snr_db=best_validation_global_snr_db,
                )
            report(
                f"trace_graph_interpolator {step}/{steps}: "
                f"loss={history_row['loss']:.8g} mask_mse={history_row['mask_mse']:.8g} "
                f"spectrum={history_row['spectrum_loss']:.8g} "
                f"slope={history_row['slope_loss']:.8g} "
                f"amplitude={history_row['amplitude_loss']:.8g} "
                f"learning_rate={learning_rate_at_step:.8g} "
                f"{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}_global_snr_db="
                f"{validation_global_snr_db:.8g}"
            )

        del raw_batch, batch, neighbors, availability, source_deltas
        del target_coordinates, targets, target_availability

    return TraceGraphTrainingResult(
        best_step=best_step,
        best_validation_global_snr_db=best_validation_global_snr_db,
        steps_completed=steps,
        training_ffid_count=training_ffids,
        training_trace_count=training_traces,
        history=tuple(history),
    )


def _evaluate_validation(
    model: TraceGraphInterpolator,
    validation_evaluator: TraceGraphValidationEvaluator,
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
