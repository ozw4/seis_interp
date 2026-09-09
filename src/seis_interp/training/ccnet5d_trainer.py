"""Fixed-plan supervised training for CCNet-5D."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.data.c3_supervised_source import C3SupervisedSource
from seis_interp.evaluation.ccnet5d_selection import evaluate_ccnet5d_selection
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training.ccnet5d_patches import CCNetPatchPlan, load_ccnet_patch

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class CCNet5DTrainingResult:
    """Completed training, fixed-cadence selection, and the best CPU snapshot."""

    epochs_completed: int
    steps_completed: int
    best_epoch: int
    best_step: int
    best_selection_metrics: dict[str, object]
    final_selection_metrics: dict[str, object]
    training_history: tuple[dict[str, object], ...]
    selection_history: tuple[dict[str, object], ...]
    best_state_dict: dict[str, torch.Tensor]


def train_ccnet5d(
    model: CCNet5D,
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
    *,
    device: torch.device | str,
    random_seed: int,
    max_epochs: int,
    learning_rate: float,
    decay_after_epochs: int,
    decay_factor: float,
    validate_every_steps: int,
    report_every_steps: int,
    reporter: Reporter | None = None,
    batch_size: int = 1,
    cudnn_benchmark: bool = True,
) -> CCNet5DTrainingResult:
    """Run exact epochs of Adam/MSE and select by missing-trace RSE.

    Each epoch partitions its seeded descriptor permutation into batches and
    retains the final partial batch. With batch_size > 1, reported loss is
    weighted by the number of samples in each batch; all patches have the same
    shape. Batch one preserves the original arithmetic and history fields.
    """
    if not isinstance(model, CCNet5D):
        raise TypeError("model must be a CCNet5D")
    batch_count = _positive_integer(batch_size, "batch_size")
    if not isinstance(cudnn_benchmark, bool):
        raise ValueError("cudnn_benchmark must be a boolean")
    seed = _nonnegative_integer(random_seed, "random_seed")
    epoch_count = _positive_integer(max_epochs, "max_epochs")
    initial_learning_rate = _positive_finite_float(learning_rate, "learning_rate")
    decay_epoch = _positive_integer(decay_after_epochs, "decay_after_epochs")
    learning_rate_factor = _positive_finite_float(decay_factor, "decay_factor")
    if learning_rate_factor > 1.0:
        raise ValueError("decay_factor must be at most 1")
    validation_interval = _positive_integer(validate_every_steps, "validate_every_steps")
    report_interval = _positive_integer(report_every_steps, "report_every_steps")
    fit_count = len(plan.fit)
    if fit_count == 0:
        raise ValueError("fit patch plan must not be empty")

    if not cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    device_value = torch.device(device)
    model.to(device_value)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_function = torch.nn.MSELoss(reduction="mean")
    shuffle_rng = np.random.default_rng(seed)
    total_steps = epoch_count * math.ceil(fit_count / batch_count)
    global_step = 0
    interval_losses: list[float] = []
    interval_patch_counts: list[int] = []
    training_history: list[dict[str, object]] = []
    selection_history: list[dict[str, object]] = []
    best_selection_metrics: dict[str, object] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_step = 0

    for epoch in range(1, epoch_count + 1):
        current_learning_rate = _learning_rate_for_epoch(
            epoch,
            initial=initial_learning_rate,
            decay_after_epochs=decay_epoch,
            decay_factor=learning_rate_factor,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        descriptor_order = shuffle_rng.permutation(fit_count)

        for start in range(0, fit_count, batch_count):
            batch_indices = descriptor_order[start : start + batch_count]
            global_step += 1
            if batch_count == 1:
                inputs, labels, _ = load_ccnet_patch(
                    source,
                    plan,
                    region="fit",
                    index=int(batch_indices[0]),
                )
                inputs, labels = _validated_training_patch(
                    inputs,
                    labels,
                    patch_shape=plan.patch_shape,
                )
                input_tensor = torch.from_numpy(inputs[None, None]).to(device_value)
                target_tensor = torch.from_numpy(labels[None, None]).to(device_value)
            else:
                inputs, labels = _load_training_batch(source, plan, batch_indices)
                input_tensor = torch.from_numpy(inputs[:, None]).to(device_value)
                target_tensor = torch.from_numpy(labels[:, None]).to(device_value)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_tensor)
            if prediction.shape != target_tensor.shape:
                raise ValueError("CCNet-5D training output shape must match complete labels")
            loss = loss_function(prediction, target_tensor)
            batch_loss = float(loss.detach().cpu().item())
            if not math.isfinite(batch_loss):
                raise RuntimeError(f"non-finite training loss at step {global_step}")
            loss.backward()
            optimizer.step()
            interval_losses.append(batch_loss)
            if batch_count > 1:
                interval_patch_counts.append(len(batch_indices))

            if global_step % report_interval == 0 or global_step == total_steps:
                if batch_count == 1:
                    interval_mean = float(np.mean(interval_losses, dtype=np.float64))
                else:
                    interval_mean = float(
                        np.average(interval_losses, weights=interval_patch_counts)
                    )
                history_row: dict[str, object] = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": interval_mean,
                    "learning_rate": current_learning_rate,
                }
                if batch_count > 1:
                    history_row["sample_count"] = sum(interval_patch_counts) * math.prod(
                        plan.patch_shape
                    )
                    history_row["loss_aggregation"] = "sample_weighted_mean"
                training_history.append(history_row)
                interval_losses.clear()
                interval_patch_counts.clear()
                if reporter is not None:
                    reporter(
                        f"ccnet5d epoch {epoch}/{epoch_count} "
                        f"step {global_step}/{total_steps}: "
                        f"train_loss={interval_mean:.8g} "
                        f"learning_rate={current_learning_rate:.8g}"
                    )

            if global_step % validation_interval == 0 or global_step == total_steps:
                selection_metrics = evaluate_ccnet5d_selection(
                    model,
                    source,
                    plan,
                    device=device_value,
                )
                relative_error = _selection_relative_squared_error(selection_metrics)
                stored_metrics = dict(selection_metrics)
                selection_history.append(
                    {"epoch": epoch, "step": global_step, "metrics": stored_metrics}
                )
                if best_selection_metrics is None or relative_error < float(
                    best_selection_metrics["relative_squared_error"]
                ):
                    best_epoch = epoch
                    best_step = global_step
                    best_selection_metrics = stored_metrics
                    best_state_dict = _cpu_state_dict(model.state_dict())

    if best_selection_metrics is None or best_state_dict is None:
        raise RuntimeError("CCNet-5D training completed without selection")
    final_selection_metrics = dict(selection_history[-1]["metrics"])
    return CCNet5DTrainingResult(
        epochs_completed=epoch_count,
        steps_completed=global_step,
        best_epoch=best_epoch,
        best_step=best_step,
        best_selection_metrics=dict(best_selection_metrics),
        final_selection_metrics=final_selection_metrics,
        training_history=tuple(training_history),
        selection_history=tuple(selection_history),
        best_state_dict=best_state_dict,
    )


def _load_training_batch(
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    patches = []
    for index in indices:
        inputs, labels, _ = load_ccnet_patch(source, plan, region="fit", index=int(index))
        patches.append(_validated_training_patch(inputs, labels, patch_shape=plan.patch_shape))
    return np.stack([patch[0] for patch in patches]), np.stack([patch[1] for patch in patches])


def _validated_training_patch(
    inputs: object,
    labels: object,
    *,
    patch_shape: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(inputs, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("CCNet-5D patch inputs and labels must be NumPy arrays")
    if inputs.shape != patch_shape or labels.shape != patch_shape:
        raise ValueError("CCNet-5D patch inputs and labels must match plan.patch_shape")
    if inputs.dtype != np.float32 or labels.dtype != np.float32:
        raise TypeError("CCNet-5D patch inputs and labels must have dtype float32")
    return inputs, labels


def _learning_rate_for_epoch(
    epoch: int,
    *,
    initial: float,
    decay_after_epochs: int,
    decay_factor: float,
) -> float:
    if epoch <= decay_after_epochs:
        return initial
    return initial * decay_factor


def _selection_relative_squared_error(metrics: Mapping[str, object]) -> float:
    try:
        value = metrics["relative_squared_error"]
    except KeyError as error:
        raise ValueError("selection metrics are missing relative_squared_error") from error
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("selection relative_squared_error must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("selection relative_squared_error must be a non-negative finite number")
    return converted


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
