"""Shot-gather loss step and thin wrapper over the shared whole-shot loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

import torch

from seis_interp.models.shot_gather_inpainter import ShotGatherInpainter
from seis_interp.training.shot_gather_inpainter_checkpoints import (
    save_shot_gather_inpainter_checkpoint,
)
from seis_interp.training.whole_shot_batches import WholeShotBatch, WholeShotBatchProvider
from seis_interp.training.whole_shot_training_loop import (
    WholeShotStepResult,
    run_whole_shot_training_loop,
)

Reporter = Callable[[str], None]


class ShotGatherValidationEvaluator(Protocol):
    """Return raw global validation S/N for the supplied model."""

    def __call__(self, model: ShotGatherInpainter) -> float: ...


@dataclass(frozen=True)
class ShotGatherTrainingResult:
    """Best raw validation result and history at every validation step."""

    best_step: int
    best_validation_global_snr_db: float
    steps_completed: int
    training_ffid_count: int
    training_trace_count: int
    history: tuple[dict[str, int | float], ...]


def train_shot_gather_inpainter(
    model: ShotGatherInpainter,
    batch_provider: WholeShotBatchProvider,
    validation_evaluator: ShotGatherValidationEvaluator,
    *,
    device: torch.device | str,
    generator: torch.Generator,
    checkpoint_path: Path,
    total_steps: int,
    batch_size: int,
    neighbor_dropout: float,
    derivative_weight: float,
    learning_rate: float,
    weight_decay: float,
    validation_interval: int,
    use_bfloat16: bool,
    training_ffid_count: int,
    training_trace_count: int,
    reporter: Reporter | None = None,
) -> ShotGatherTrainingResult:
    """Train masked whole-shot batches and select by raw validation global S/N."""
    if not isinstance(model, ShotGatherInpainter):
        raise TypeError("model must be a ShotGatherInpainter")
    derivative_weight_value = _nonnegative_finite_loss_weight(
        derivative_weight,
        "derivative_weight",
    )

    def training_step(
        step_model: torch.nn.Module,
        batch: WholeShotBatch,
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        return _shot_gather_training_step(
            step_model,
            batch,
            derivative_weight=derivative_weight_value,
            use_cuda_bfloat16=use_cuda_bfloat16,
        )

    result = run_whole_shot_training_loop(
        model,
        batch_provider,
        validation_evaluator,
        training_step=training_step,
        checkpoint_saver=save_shot_gather_inpainter_checkpoint,
        progress_name="shot_gather_inpainter",
        progress_metric_labels=(
            ("loss", "loss"),
            ("mse", "mse"),
            ("derivative_mse", "derivative_mse"),
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
    return ShotGatherTrainingResult(
        best_step=result.best_step,
        best_validation_global_snr_db=result.best_validation_global_snr_db,
        steps_completed=result.steps_completed,
        training_ffid_count=result.training_ffid_count,
        training_trace_count=result.training_trace_count,
        history=result.history,
    )


def _shot_gather_training_step(
    model: ShotGatherInpainter,
    batch: WholeShotBatch,
    *,
    derivative_weight: float,
    use_cuda_bfloat16: bool,
) -> WholeShotStepResult:
    """Run forward, MSE, derivative MSE, and loss inside one autocast scope."""
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
        mse = _masked_mean_square(prediction - targets, target_availability)
        derivative_mse = _masked_mean_square(
            torch.diff(prediction, dim=-1) - torch.diff(targets, dim=-1),
            target_availability,
        )
        loss = mse + derivative_weight * derivative_mse
    return WholeShotStepResult(
        loss=loss,
        history_metrics=(
            ("mse", mse),
            ("derivative_mse", derivative_mse),
        ),
        finite_checks=(
            ("training MSE", mse),
            ("training derivative MSE", derivative_mse),
        ),
    )


def _masked_mean_square(values: torch.Tensor, trace_mask: torch.Tensor) -> torch.Tensor:
    """Average squared samples over the available receiver traces only."""
    if values.ndim != 4 or trace_mask.shape != values.shape[:3]:
        raise ValueError("values and trace_mask shapes do not describe a gather batch")
    if trace_mask.dtype != torch.bool:
        raise TypeError("trace_mask must have dtype torch.bool")
    trace_count = torch.count_nonzero(trace_mask)
    if int(trace_count.item()) == 0:
        raise ValueError("a training batch must contain at least one target trace")
    point_count = trace_count * values.shape[-1]
    mask = trace_mask[..., None].to(dtype=values.dtype)
    return torch.sum(torch.square(values) * mask) / point_count


def _nonnegative_finite_loss_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted
