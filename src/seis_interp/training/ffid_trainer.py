"""Train one shared SIREN with one complete training FFID per update."""

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
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
)
from seis_interp.training.checkpoints import save_siren_checkpoint
from seis_interp.training.correlation_loss import (
    DEFAULT_TRACE_CORRELATION_EPS,
    trace_correlation_loss,
)
from seis_interp.training.ffid_batches import FullFfidBatch, FullFfidBatchSampler
from seis_interp.training.model_inputs import to_model_tensors

Reporter = Callable[[str], None]
ValidationEvaluator = Callable[[torch.nn.Module], float]


@dataclass(frozen=True)
class FullFfidTrainingResult:
    """Best streamed validation result and the complete epoch history."""

    best_epoch: int
    best_validation_global_snr_db: float
    epochs_completed: int
    global_steps: int
    stopped_early: bool
    training_ffid_count: int
    validation_ffid_count: int
    history: tuple[dict[str, int | float], ...]


@dataclass(frozen=True)
class _FfidBatchLoss:
    """Detached loss components for one optimizer update."""

    total: float
    mse: float
    correlation: float | None


def train_siren_by_ffid(
    model: Siren,
    sampler: FullFfidBatchSampler,
    validation_evaluator: ValidationEvaluator,
    normalization: NormalizationParameters,
    *,
    device: torch.device | str,
    loss: str,
    optimizer: str,
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
    validation_ffid_count: int,
    checkpoint_path: Path,
    reporter: Reporter | None = None,
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    correlation_weight: float = 0.0,
    correlation_eps: float = DEFAULT_TRACE_CORRELATION_EPS,
) -> FullFfidTrainingResult:
    """Train with every training FFID once per epoch and select by global S/N."""
    if loss != "l2":
        raise ValueError(f"full_ffid_epoch supports only l2 loss, got {loss!r}")
    if optimizer != "adam":
        raise ValueError(f"full_ffid_epoch supports only adam optimizer, got {optimizer!r}")
    learning_rate_value = _positive_finite_float(learning_rate, "learning_rate")
    max_epochs_value = _positive_integer(max_epochs, "max_epochs")
    patience_value = _positive_integer(early_stopping_patience, "early_stopping_patience")
    validation_count = _positive_integer(validation_ffid_count, "validation_ffid_count")
    training_count = _positive_integer(sampler.ffid_count, "training_ffid_count")
    correlation_weight_value = _nonnegative_finite_float(
        correlation_weight,
        "correlation_weight",
    )
    correlation_eps_value = _positive_finite_float(correlation_eps, "correlation_eps")
    uses_correlation = correlation_weight_value > 0.0
    target_amplitude_scaling = validated_amplitude_scaling(amplitude_scaling)
    if sampler.amplitude_scaling != target_amplitude_scaling:
        raise ValueError(
            "amplitude_scaling must match the FullFfidBatchSampler target scaling: "
            f"{target_amplitude_scaling!r} != {sampler.amplitude_scaling!r}"
        )
    report = reporter or _print_progress

    model.to(device)
    loss_function = torch.nn.MSELoss()
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    history: list[dict[str, int | float]] = []
    global_steps = 0
    best_epoch = 0
    best_validation_global_snr_db = -math.inf
    non_improving_epochs = 0
    stopped_early = False

    for epoch in range(1, max_epochs_value + 1):
        correlation_label = (
            f" correlation_weight={correlation_weight_value:g} "
            f"correlation_eps={correlation_eps_value:g}"
            if uses_correlation
            else ""
        )
        if target_amplitude_scaling == PER_TRACE_RMS_SCALING:
            report(
                f"full_ffid_epoch {epoch}/{max_epochs_value} start: "
                f"amplitude_scaling={target_amplitude_scaling} "
                f"validation_metric_domain={ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}"
                f"{correlation_label}"
            )
        else:
            report(f"full_ffid_epoch {epoch}/{max_epochs_value} start{correlation_label}")
        model.train()
        ffid_batch_losses: list[float] = []
        ffid_batch_mse_losses: list[float] = []
        ffid_batch_correlation_losses: list[float] = []
        visited_ffids: set[int] = set()
        for batch in sampler.iter_epoch():
            if batch.ffid in visited_ffids:
                raise RuntimeError(f"FFID {batch.ffid} was yielded more than once in one epoch")
            visited_ffids.add(batch.ffid)
            batch_loss = _train_ffid_batch(
                model,
                batch,
                loss_function,
                torch_optimizer,
                device=device,
                epoch=epoch,
                global_step=global_steps + 1,
                correlation_weight=correlation_weight_value,
                correlation_eps=correlation_eps_value,
            )
            ffid_batch_losses.append(batch_loss.total)
            ffid_batch_mse_losses.append(batch_loss.mse)
            if batch_loss.correlation is not None:
                ffid_batch_correlation_losses.append(batch_loss.correlation)
            global_steps += 1
            # Let the sampler build the next FFID without this batch coexisting.
            del batch

        if len(visited_ffids) != training_count:
            raise RuntimeError(
                "full FFID sampler did not yield the configured epoch coverage: "
                f"expected {training_count}, got {len(visited_ffids)}"
            )

        validation_global_snr_db = float(validation_evaluator(model))
        if math.isnan(validation_global_snr_db) or validation_global_snr_db == -math.inf:
            raise ValueError("validation global S/N must be finite or positive infinity")
        mean_ffid_batch_loss = float(np.mean(ffid_batch_losses, dtype=np.float64))
        history_row: dict[str, int | float] = {
            "epoch": epoch,
            "global_step": global_steps,
            "mean_ffid_batch_loss": mean_ffid_batch_loss,
            "validation_global_snr_db": validation_global_snr_db,
        }
        if uses_correlation:
            if len(ffid_batch_correlation_losses) != training_count:
                raise RuntimeError("correlation loss was not recorded for every training FFID")
            history_row.update(
                {
                    "mean_ffid_batch_mse_loss": float(
                        np.mean(ffid_batch_mse_losses, dtype=np.float64)
                    ),
                    "mean_ffid_batch_correlation_loss": float(
                        np.mean(ffid_batch_correlation_losses, dtype=np.float64)
                    ),
                }
            )
        history.append(history_row)
        validation_label = (
            "oracle_per_trace_unit_rms_global_snr_db"
            if target_amplitude_scaling == PER_TRACE_RMS_SCALING
            else "validation_global_snr_db"
        )
        if uses_correlation:
            report(
                f"full_ffid_epoch {epoch}/{max_epochs_value} end: "
                f"mean_ffid_batch_loss={mean_ffid_batch_loss:.8g} "
                f"mean_ffid_batch_mse_loss={history_row['mean_ffid_batch_mse_loss']:.8g} "
                "mean_ffid_batch_correlation_loss="
                f"{history_row['mean_ffid_batch_correlation_loss']:.8g} "
                f"{validation_label}={validation_global_snr_db:.8g}"
            )
        else:
            report(
                f"full_ffid_epoch {epoch}/{max_epochs_value} end: "
                f"mean_ffid_batch_loss={mean_ffid_batch_loss:.8g} "
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

    return FullFfidTrainingResult(
        best_epoch=best_epoch,
        best_validation_global_snr_db=best_validation_global_snr_db,
        epochs_completed=len(history),
        global_steps=global_steps,
        stopped_early=stopped_early,
        training_ffid_count=training_count,
        validation_ffid_count=validation_count,
        history=tuple(history),
    )


def _train_ffid_batch(
    model: Siren,
    batch: FullFfidBatch,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    epoch: int,
    global_step: int,
    correlation_weight: float,
    correlation_eps: float,
) -> _FfidBatchLoss:
    """Run one update in a frame whose batch tensors are released on return."""
    coordinate_tensor, target_tensor = to_model_tensors(
        batch.coordinates,
        batch.targets,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(coordinate_tensor)
    mse_loss = loss_function(prediction, target_tensor)
    correlation_loss: torch.Tensor | None = None
    if correlation_weight > 0.0:
        prediction_traces, target_traces = _trace_matrices(
            prediction,
            target_tensor,
            batch,
        )
        correlation_loss = trace_correlation_loss(
            prediction_traces,
            target_traces,
            eps=correlation_eps,
        )
        batch_loss = mse_loss + correlation_weight * correlation_loss
    else:
        batch_loss = mse_loss
    mse_value = float(mse_loss.detach().cpu().item())
    correlation_value = (
        float(correlation_loss.detach().cpu().item()) if correlation_loss is not None else None
    )
    if correlation_value is not None and not math.isfinite(mse_value):
        raise RuntimeError(
            f"non-finite MSE loss for FFID {batch.ffid}: "
            f"epoch={epoch}, global_step={global_step}, mse_loss={mse_value}"
        )
    if correlation_value is not None and not math.isfinite(correlation_value):
        raise RuntimeError(
            f"non-finite correlation loss for FFID {batch.ffid}: "
            f"epoch={epoch}, global_step={global_step}, correlation_loss={correlation_value}"
        )
    loss_value = float(batch_loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise RuntimeError(
            f"non-finite training loss for FFID {batch.ffid}: "
            f"epoch={epoch}, global_step={global_step}, loss={loss_value}"
        )
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return _FfidBatchLoss(
        total=loss_value,
        mse=mse_value,
        correlation=correlation_value,
    )


def _trace_matrices(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: FullFfidBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore trace-major matrices for the trace-wise auxiliary objective."""
    if batch.trace_count <= 0 or batch.point_count <= 0:
        raise ValueError("full FFID batch trace_count and point_count must be positive")
    if batch.point_count % batch.trace_count != 0:
        raise ValueError("full FFID batch point_count must be divisible by trace_count")
    if prediction.numel() != batch.point_count or target.numel() != batch.point_count:
        raise ValueError(
            "full FFID batch tensor sizes must match point_count: "
            f"prediction={prediction.numel()}, target={target.numel()}, "
            f"point_count={batch.point_count}"
        )
    samples_per_trace = batch.point_count // batch.trace_count
    return (
        prediction.reshape(batch.trace_count, samples_per_trace),
        target.reshape(batch.trace_count, samples_per_trace),
    )


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


def _nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return converted
