"""Fixed-step CCNet-5D fitting on hidden traces from the observed set."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from numbers import Integral, Real

import torch

from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training.ccnet5d_ema import validate_ccnet5d_ema_decay
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_patch_gradients import precompute_ccnet5d_patch_gradients
from seis_interp.training.ccnet5d_query_training import backward_ccnet5d_trace_queries
from seis_interp.training.cuda_conv_kernels import summarize_cuda_conv_kernels
from seis_interp.training.exact_resume import object_state, restore_objects
from seis_interp.training.mixed_precision import mixed_precision_dtype
from seis_interp.training.neural_diagnostics import NeuralTrainingDiagnostics
from seis_interp.training.patch_prefetch import PatchPrefetch
from seis_interp.training.trace_relative_loss import masked_trace_loss

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class CCNet5DObservedTrainingResult:
    """Final fixed-step training counters and bounded loss history."""

    steps_completed: int
    supervised_trace_presentations: int
    final_loss: float
    history: tuple[dict[str, object], ...]
    shadow_parameters: dict[float, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    shadow_snapshots: dict[int, dict[float, tuple[torch.Tensor, ...]]] = field(default_factory=dict)
    conv_kernels_first_update: tuple[str, ...] = ()
    time_conv_tensor_cores: bool | None = None
    diagnostics: dict = field(default_factory=dict)


def train_ccnet5d_observed_steps(
    model: CCNet5D,
    source: CCNet5DObservedPatchSource,
    *,
    device: torch.device | str,
    optimizer_updates: int,
    learning_rate: float,
    report_every_steps: int,
    loss_name: str = "masked_trace_relative_mse",
    ema_decay: float | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    supervised_traces_per_update: int | None = None,
    ema_eval_decays: Sequence[float] = (),
    ema_snapshot_updates: Sequence[int] = (),
    profile_first_update: bool = False,
    mixed_precision: str = "off",
    reuse_patch: bool = False,
    spatial_query_chunk_size: int | None = None,
    cache_patch_gradients: bool = False,
    reporter: Reporter | None = None,
    before_update: Callable[[int], None] | None = None,
    after_update: Callable[[int], None] | None = None,
    recovery=None,
    prefetch: bool = False,
) -> CCNet5DObservedTrainingResult:
    """Fit pseudo-targets with Adam or AdamW, optionally returning final EMA parameters.

    EMA starts at the first post-update parameters and advances once per update.
    Raw parameters are used throughout fitting; only the returned model is averaged.
    A fixed trace budget stacks patches in one forward/backward, without gradient
    accumulation. The loss is the mean over selected complete trace samples.
    ``ema_eval_decays`` keeps extra parameter averages, advanced like the primary
    one and returned as ``shadow_parameters``; the primary decay reuses its own.
    ``ema_snapshot_updates`` keeps CPU copies of every average (the primary and
    each shadow) after the listed updates, returned as ``shadow_snapshots``.
    ``profile_first_update`` records convolution CUDA kernels from the first
    update; it has no effect on CPU. ``mixed_precision="bf16"`` autocasts only
    the CUDA forward, keeping the loss accumulation and parameter averages
    at their usual precision.
    ``reuse_patch`` consumes shuffled, disjoint trace groups from each fixed
    pseudo-mask, with a fresh forward and optimizer/EMA step for each group.
    ``spatial_query_chunk_size`` restricts computation to the selected traces'
    dependencies and checkpoints spatial convolution chunks. Time stays whole.
    ``cache_patch_gradients`` forwards each patch once, saves every group's
    gradients before any weight change, then applies one optimizer/EMA update
    per saved group. Those gradients intentionally share the patch's old weights.
    Optional update callbacks bracket sampling through optimizer/EMA updates,
    excluding initialization, reporting and final EMA copying, for probe timing.
    """
    if not isinstance(model, CCNet5D):
        raise TypeError("model must be a CCNet5D")
    if not isinstance(source, CCNet5DObservedPatchSource):
        raise TypeError("source must be a CCNet5DObservedPatchSource")
    update_count = _positive_integer(optimizer_updates, "optimizer_updates")
    report_interval = _positive_integer(report_every_steps, "report_every_steps")
    rate = _positive_finite_float(learning_rate, "learning_rate")
    decay = validate_ccnet5d_ema_decay(ema_decay)
    averaged = None
    shadow_decays = _shadow_decays(ema_eval_decays, decay)
    shadows = {value: None for value in shadow_decays if value != decay}
    snapshot_updates = frozenset(ema_snapshot_updates)
    if snapshot_updates and decay is None:
        raise ValueError("ema_snapshot_updates require an enabled ema_decay")
    snapshots: dict[int, dict[float, tuple[torch.Tensor, ...]]] = {}
    device_value = torch.device(device)
    autocast_dtype = mixed_precision_dtype(mixed_precision, device_value)
    if mixed_precision == "fp16":
        raise ValueError("CCNet-5D training.mixed_precision must be off or bf16")
    model.to(device_value)
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer_name must be adam or adamw")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, Real)
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be nonnegative and finite")
    if supervised_traces_per_update is not None:
        supervised_traces_per_update = _positive_integer(
            supervised_traces_per_update, "supervised_traces_per_update"
        )
    if not isinstance(reuse_patch, bool):
        raise ValueError("reuse_patch must be Boolean")
    if reuse_patch and supervised_traces_per_update is None:
        raise ValueError("reuse_patch requires supervised_traces_per_update")
    if spatial_query_chunk_size is not None:
        spatial_query_chunk_size = _positive_integer(
            spatial_query_chunk_size, "spatial_query_chunk_size"
        )
    if not isinstance(cache_patch_gradients, bool):
        raise ValueError("cache_patch_gradients must be Boolean")
    if cache_patch_gradients and (not reuse_patch or spatial_query_chunk_size is not None):
        raise ValueError("cache_patch_gradients requires reuse_patch and the full-patch forward")
    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=rate, weight_decay=weight_decay)
    parameter = next(model.parameters())
    history: list[dict[str, object]] = []
    interval_loss = 0.0
    interval_count = 0
    final_loss = float("nan")
    supervised_trace_presentations = 0
    conv_kernels_first_update: tuple[str, ...] = ()
    time_conv_tensor_cores = None
    reused_batches = deque()
    cached_gradients = deque()
    cached_groups_remaining = 0
    patch_forwards_completed = 0
    diagnostics = NeuralTrainingDiagnostics(update_count, loss_name)
    patches_per_update = []

    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        source.load_state_dict(state["source"])
        restore_objects({"diagnostics": diagnostics}, state["objects"])
        averaged, shadows, snapshots = state["averaged"], state["shadows"], state["snapshots"]
        history = state["history"]
        interval_loss, interval_count = state["interval_loss"], state["interval_count"]
        final_loss = state["final_loss"]
        supervised_trace_presentations = state["supervised_trace_presentations"]
        patches_per_update = state["patches_per_update"]
        reused_batches, cached_gradients = state["reused_batches"], state["cached_gradients"]
        cached_groups_remaining = state["cached_groups_remaining"]
        patch_forwards_completed = state["patch_forwards_completed"]
        conv_kernels_first_update = state["conv_kernels_first_update"]
        time_conv_tensor_cores = state["time_conv_tensor_cores"]

    def draw():
        previous = source.patches_drawn
        if supervised_traces_per_update is None:
            patch = source.sample()
            batch = (
                patch.model_input[None],
                patch.pseudo_target[None],
                patch.pseudo_target_mask[None],
            )
        else:
            batch = source.sample_batch(supervised_traces_per_update)
        return batch, source.patches_drawn - previous

    # Reused/cached patches already have their own group queue.
    ahead = PatchPrefetch(
        draw, enabled=prefetch and not reuse_patch, pending=state.get("prefetched")
    )
    try:
        for step in range(state.get("step", 0) + 1, update_count + 1):
            if before_update is not None:
                before_update(step)
            previous_patches = source.patches_drawn
            update_profile = None
            if cache_patch_gradients:
                saved = cached_gradients.popleft() if cached_gradients else None
                if saved is None:
                    batches = list(
                        islice(
                            source.sample_reused_batches(supervised_traces_per_update),
                            update_count - step + 1,
                        )
                    )
                    model.train()
                    update_profile = (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        )
                        if profile_first_update and device_value.type == "cuda" and step == 1
                        else None
                    )
                    with update_profile if update_profile is not None else nullcontext():
                        groups = precompute_ccnet5d_patch_gradients(
                            model,
                            batches,
                            device=device_value,
                            autocast_dtype=autocast_dtype,
                            loss_name=loss_name,
                            reporter=reporter if step == 1 else None,
                        )
                    cached_gradients = deque(groups)
                    cached_groups_remaining = len(groups)
                    patch_forwards_completed += batches[0][0].shape[0]
                    saved = cached_gradients.popleft()
                    del batches, groups
                optimizer.zero_grad(set_to_none=True)
                for value, gradient in zip(model.parameters(), saved.gradients, strict=True):
                    value.grad = gradient
                diagnostics.gradients(model)
                optimizer.step()
                cached_groups_remaining -= 1
                final_loss = saved.loss
                supervised_count = saved.supervised_traces
                del saved
            else:
                if reuse_patch:
                    if not reused_batches:
                        reused_batches = deque(
                            source.sample_reused_batches(supervised_traces_per_update)
                        )
                    input_values, target_values, mask_values = reused_batches.popleft()
                    patch_count = source.patches_drawn - previous_patches
                else:
                    (input_values, target_values, mask_values), patch_count = ahead.take(
                        prefetch_next=step < update_count
                    )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                update_profile = None
                if spatial_query_chunk_size is not None:
                    final_loss = backward_ccnet5d_trace_queries(
                        model,
                        input_values,
                        target_values,
                        mask_values,
                        device=device_value,
                        spatial_chunk_size=spatial_query_chunk_size,
                        autocast_dtype=autocast_dtype,
                        loss_name=loss_name,
                    )
                    optimizer.step()
                else:
                    inputs = torch.as_tensor(
                        input_values, dtype=parameter.dtype, device=device_value
                    )[:, None]
                    targets = torch.as_tensor(
                        target_values, dtype=parameter.dtype, device=device_value
                    )
                    target_mask = torch.as_tensor(
                        mask_values, dtype=torch.bool, device=device_value
                    )
                    update_profile = (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        )
                        if profile_first_update and device_value.type == "cuda" and step == 1
                        else None
                    )
                    with update_profile if update_profile is not None else nullcontext():
                        with (
                            torch.autocast(
                                device_type=torch.device(device).type, dtype=autocast_dtype
                            )
                            if autocast_dtype is not None
                            else nullcontext()
                        ):
                            prediction = model(inputs)
                        if autocast_dtype is not None:
                            prediction = prediction.float()
                        if prediction.shape != inputs.shape:
                            raise ValueError(
                                "CCNet-5D output shape must match the pseudo-mask input patch"
                            )
                        prediction_traces = prediction[:, 0].movedim(1, -1)[target_mask]
                        target_traces = targets.movedim(1, -1)[target_mask]
                        loss = masked_trace_loss(
                            prediction_traces, target_traces, loss_name=loss_name
                        )
                        final_loss = float(loss.detach().cpu().item())
                        if not math.isfinite(final_loss):
                            raise RuntimeError(f"non-finite CCNet-5D training loss at step {step}")
                        diagnostics.observe(step, prediction_traces, target_traces, loss)
                        loss.backward()
                        diagnostics.gradients(model)
                        optimizer.step()
                supervised_count = int(mask_values.sum())
            if update_profile is not None:
                conv_kernels_first_update, time_conv_tensor_cores = summarize_cuda_conv_kernels(
                    update_profile.events(),
                    time_kernel_shape=(
                        model.time_kernel_size,
                        model.kernel_size,
                        model.kernel_size,
                    ),
                )
                update_profile = None
            if decay is not None:
                averaged = _advance_average(averaged, model, decay)
            for value, shadow in shadows.items():
                shadows[value] = _advance_average(shadow, model, value)
            if step in snapshot_updates:
                snapshots[step] = {
                    value: tuple(
                        t.detach().cpu().clone()
                        for t in (averaged if value == decay else shadows[value])
                    )
                    for value in shadow_decays or (decay,)
                }
            supervised_trace_presentations += supervised_count
            patches_per_update.append(
                source.patches_drawn - previous_patches if cache_patch_gradients else patch_count
            )
            interval_loss += final_loss
            interval_count += 1
            if after_update is not None:
                after_update(step)

            if (
                cache_patch_gradients
                and cached_groups_remaining == 0
                and reporter is not None
                and step % report_interval != 0
                and step != update_count
            ):
                reporter(f"ccnet5d cached patch complete: step {step}/{update_count}")

            report_message = None
            if (
                step % report_interval == 0
                or step == update_count
                or (step == 1 and (reuse_patch or spatial_query_chunk_size is not None))
            ):
                mean_loss = interval_loss / interval_count
                history.append(
                    {
                        "step": step,
                        "loss": final_loss
                        if recovery is not None and recovery.record_all_steps
                        else mean_loss,
                        "learning_rate": rate,
                    }
                )
                if supervised_traces_per_update is not None:
                    history[-1]["supervised_trace_presentations"] = supervised_trace_presentations
                if cache_patch_gradients:
                    history[-1]["patch_forwards_completed"] = patch_forwards_completed
                report_message = (
                    f"ccnet5d observed-only step {step}/{update_count}: loss={mean_loss:.8g}"
                )
                interval_loss = 0.0
                interval_count = 0

            if recovery is not None and recovery.record_all_steps and report_message is None:
                history.append({"step": step, "loss": final_loss, "learning_rate": rate})

            if recovery is not None and (recovery.due(step) or step == update_count):
                prefetched = ahead.checkpoint()
                recovery.save(
                    step,
                    model,
                    optimizer,
                    {
                        "source": source.state_dict(),
                        "objects": object_state({"diagnostics": diagnostics}),
                        "averaged": averaged,
                        "shadows": shadows,
                        "snapshots": snapshots,
                        "history": history,
                        "interval_loss": interval_loss,
                        "interval_count": interval_count,
                        "final_loss": final_loss,
                        "supervised_trace_presentations": supervised_trace_presentations,
                        "patches_per_update": patches_per_update,
                        "reused_batches": reused_batches,
                        "cached_gradients": cached_gradients,
                        "cached_groups_remaining": cached_groups_remaining,
                        "patch_forwards_completed": patch_forwards_completed,
                        "prefetched": prefetched,
                        "conv_kernels_first_update": conv_kernels_first_update,
                        "time_conv_tensor_cores": time_conv_tensor_cores,
                    },
                    force=True,
                    phase="before_prediction" if step == update_count else "training",
                )
            if reporter is not None and report_message is not None:
                reporter(report_message)
    finally:
        ahead.close()

    if averaged is not None:
        with torch.no_grad():
            for parameter_value, average in zip(model.parameters(), averaged, strict=True):
                parameter_value.copy_(average)
    shadow_parameters = {
        value: tuple(averaged if value == decay else shadows[value]) for value in shadow_decays
    }
    return CCNet5DObservedTrainingResult(
        steps_completed=update_count,
        supervised_trace_presentations=supervised_trace_presentations,
        final_loss=final_loss,
        history=tuple(history),
        shadow_parameters=shadow_parameters,
        shadow_snapshots=snapshots,
        conv_kernels_first_update=conv_kernels_first_update,
        time_conv_tensor_cores=time_conv_tensor_cores,
        diagnostics=diagnostics.records() | {"patches_per_update": patches_per_update},
    )


@torch.no_grad()
def _advance_average(
    average: list[torch.Tensor] | None, model: CCNet5D, decay: float
) -> list[torch.Tensor]:
    """Start at the first post-update parameters, then lerp toward each update."""
    if average is None:
        return [p.detach().clone() for p in model.parameters()]
    for value, parameter_value in zip(average, model.parameters(), strict=True):
        value.lerp_(parameter_value, 1 - decay)
    return average


def _shadow_decays(values: Sequence[float], primary: float | None) -> tuple[float, ...]:
    """Distinct enabled decays; shadows require an enabled primary EMA."""
    decays = tuple(validate_ccnet5d_ema_decay(value) for value in values)
    if decays and (primary is None or None in decays or len(set(decays)) != len(decays)):
        raise ValueError("ema_eval_decays must be distinct enabled decays beside ema_decay")
    return decays


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted
