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
from seis_interp.training.checkpoints import save_siren_checkpoint
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
        report(f"full_ffid_epoch {epoch}/{max_epochs_value} start")
        model.train()
        ffid_batch_losses: list[float] = []
        visited_ffids: set[int] = set()
        for batch in sampler.iter_epoch():
            if batch.ffid in visited_ffids:
                raise RuntimeError(f"FFID {batch.ffid} was yielded more than once in one epoch")
            visited_ffids.add(batch.ffid)
            ffid_batch_losses.append(
                _train_ffid_batch(
                    model,
                    batch,
                    loss_function,
                    torch_optimizer,
                    device=device,
                )
            )
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
        history.append(
            {
                "epoch": epoch,
                "global_step": global_steps,
                "mean_ffid_batch_loss": mean_ffid_batch_loss,
                "validation_global_snr_db": validation_global_snr_db,
            }
        )
        report(
            f"full_ffid_epoch {epoch}/{max_epochs_value} end: "
            f"mean_ffid_batch_loss={mean_ffid_batch_loss:.8g} "
            f"validation_global_snr_db={validation_global_snr_db:.8g}"
        )

        if epoch == 1 or validation_global_snr_db > best_validation_global_snr_db:
            best_epoch = epoch
            best_validation_global_snr_db = validation_global_snr_db
            non_improving_epochs = 0
            save_siren_checkpoint(
                checkpoint_path,
                model,
                normalization,
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
) -> float:
    """Run one update in a frame whose batch tensors are released on return."""
    coordinate_tensor, target_tensor = to_model_tensors(
        batch.coordinates,
        batch.targets,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(coordinate_tensor)
    batch_loss = loss_function(prediction, target_tensor)
    batch_loss.backward()
    optimizer.step()
    loss_value = float(batch_loss.detach().cpu().item())
    optimizer.zero_grad(set_to_none=True)
    return loss_value


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
