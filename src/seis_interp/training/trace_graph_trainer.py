"""Trace-graph loss step and thin wrapper over the shared whole-shot loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

import torch

from seis_interp.models.trace_graph_interpolator import TraceGraphInterpolator
from seis_interp.training.trace_graph_checkpoints import save_trace_graph_checkpoint
from seis_interp.training.trace_graph_losses import (
    amplitude_envelope_loss,
    masked_mean_square,
    slope_consistency_loss,
    spectrum_loss,
)
from seis_interp.training.whole_shot_batches import WholeShotBatch, WholeShotBatchProvider
from seis_interp.training.whole_shot_training_loop import (
    WholeShotStepResult,
    run_whole_shot_training_loop,
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
    spectrum_weight_value = _nonnegative_finite_loss_weight(spectrum_weight, "spectrum_weight")
    slope_weight_value = _nonnegative_finite_loss_weight(slope_weight, "slope_weight")
    amplitude_weight_value = _nonnegative_finite_loss_weight(
        amplitude_weight,
        "amplitude_weight",
    )

    def training_step(
        step_model: torch.nn.Module,
        batch: WholeShotBatch,
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        return _trace_graph_training_step(
            step_model,
            batch,
            spectrum_weight=spectrum_weight_value,
            slope_weight=slope_weight_value,
            amplitude_weight=amplitude_weight_value,
            use_cuda_bfloat16=use_cuda_bfloat16,
        )

    result = run_whole_shot_training_loop(
        model,
        batch_provider,
        validation_evaluator,
        training_step=training_step,
        checkpoint_saver=save_trace_graph_checkpoint,
        progress_name="trace_graph_interpolator",
        progress_metric_labels=(
            ("loss", "loss"),
            ("mask_mse", "mask_mse"),
            ("spectrum_loss", "spectrum"),
            ("slope_loss", "slope"),
            ("amplitude_loss", "amplitude"),
        ),
        device=device,
        generator=generator,
        checkpoint_path=checkpoint_path,
        total_steps=total_steps,
        batch_size=batch_size,
        neighbor_dropout=neighbor_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_interval=validation_interval,
        use_bfloat16=use_bfloat16,
        training_ffid_count=training_ffid_count,
        training_trace_count=training_trace_count,
        reporter=reporter,
    )
    return TraceGraphTrainingResult(
        best_step=result.best_step,
        best_validation_global_snr_db=result.best_validation_global_snr_db,
        steps_completed=result.steps_completed,
        training_ffid_count=result.training_ffid_count,
        training_trace_count=result.training_trace_count,
        history=result.history,
    )


def _trace_graph_training_step(
    model: TraceGraphInterpolator,
    batch: WholeShotBatch,
    *,
    spectrum_weight: float,
    slope_weight: float,
    amplitude_weight: float,
    use_cuda_bfloat16: bool,
) -> WholeShotStepResult:
    """Run only the model forward in autocast; compute every loss outside it."""
    (
        neighbors,
        availability,
        source_deltas,
        target_coordinates,
        targets,
        target_availability,
    ) = batch
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
    if spectrum_weight > 0.0:
        spectrum_value = spectrum_loss(prediction, targets, target_availability)
        loss = loss + spectrum_weight * spectrum_value
    if slope_weight > 0.0:
        slope_value = slope_consistency_loss(prediction, targets, target_availability)
        loss = loss + slope_weight * slope_value
    if amplitude_weight > 0.0:
        amplitude_value = amplitude_envelope_loss(
            prediction,
            targets,
            target_availability,
        )
        loss = loss + amplitude_weight * amplitude_value
    return WholeShotStepResult(
        loss=loss,
        history_metrics=(
            ("mask_mse", mask_mse),
            ("spectrum_loss", spectrum_value),
            ("slope_loss", slope_value),
            ("amplitude_loss", amplitude_value),
        ),
        finite_checks=(("training mask MSE", mask_mse),),
    )


def _nonnegative_finite_loss_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted
