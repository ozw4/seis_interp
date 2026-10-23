"""Real CCNet updates and one full-time, largest-halo prediction forward."""

import gc
import math
import statistics
import time
from contextlib import nullcontext


def measure_ccnet5d_memory(inputs, config, device, *, optimizer_updates=1, warmup_updates=0):
    """Return measurements only; never load or score held-out target amplitudes."""
    if (
        isinstance(warmup_updates, bool)
        or not isinstance(warmup_updates, int)
        or warmup_updates < 0
    ):
        raise ValueError("warmup_updates must be a nonnegative integer")
    if (
        isinstance(optimizer_updates, bool)
        or not isinstance(optimizer_updates, int)
        or optimizer_updates < 1
    ):
        raise ValueError("optimizer_updates must be a positive integer")
    import numpy as np
    import torch

    from seis_interp.data.forge_aligned_inputs import align_model_inputs
    from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles
    from seis_interp.training.amplitude_scaling import normalize_by_global_rms
    from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
    from seis_interp.training.ccnet5d_observed_training import train_ccnet5d_observed_steps
    from seis_interp.training.forge_c3_methods import pad_ccnet_source_lines
    from seis_interp.training.forge_mvp_methods import initialize_model
    from seis_interp.training.mixed_precision import mixed_precision_dtype

    started = time.monotonic()
    inputs = pad_ccnet_source_lines(inputs, config)
    aligned, _ = align_model_inputs(inputs.model_space(), config["alignment"])
    volume = aligned.volume()
    model = initialize_model(config)
    placement, mask, supervision = np.random.SeedSequence(config["model_seed"]).spawn(3)
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=aligned.global_rms,
        patch_shape=(len(aligned.time_s), *config["patches"]["shape"][1:]),
        inner_mask_fraction=config["patches"]["inner_mask_fraction"],
        placement_seed=placement,
        inner_mask_seed=mask,
        supervision_seed=supervision,
    )
    tr = config["training"]
    cuda = torch.device(device).type == "cuda"
    benchmark = config.get("execution", {}).get("cudnn_benchmark", False)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = not benchmark

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    def reset_peak():
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)

    def peak():
        return (
            (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
            if cuda
            else (0, 0)
        )

    reset_peak()
    training_started = time.monotonic()
    total_updates = warmup_updates + optimizer_updates
    update_seconds = []
    update_started = None

    def before_update(step):
        nonlocal update_started
        if step > warmup_updates:
            synchronize()
            update_started = time.monotonic()

    def after_update(step):
        if step > warmup_updates:
            synchronize()
            update_seconds.append(time.monotonic() - update_started)

    trained = train_ccnet5d_observed_steps(
        model,
        source,
        device=device,
        optimizer_updates=total_updates,
        learning_rate=tr["learning_rate"],
        report_every_steps=min(optimizer_updates, tr["report_interval"]),
        loss_name=config["loss"],
        ema_decay=tr["ema_decay"],
        ema_eval_decays=tr.get("ema_eval_decays", ()),
        optimizer_name=tr["optimizer"],
        weight_decay=tr["weight_decay"],
        supervised_traces_per_update=tr["supervised_traces_per_update"],
        mixed_precision=tr["mixed_precision"],
        reuse_patch=config.get("execution", {}).get("reuse_patch", False),
        cache_patch_gradients=config.get("execution", {}).get("cache_patch_gradients", False),
        profile_first_update=optimizer_updates == 1,
        reporter=print,
        before_update=before_update,
        after_update=after_update,
    )
    synchronize()
    training_seconds = time.monotonic() - training_started
    allocated, reserved = peak()
    record = {
        "training_loss": trained.final_loss,
        "time_conv_tensor_cores": trained.time_conv_tensor_cores,
        "training_peak_allocated_bytes": allocated,
        "training_peak_reserved_bytes": reserved,
        "training_seconds": training_seconds,
        "seconds_per_update": statistics.median(update_seconds),
        "per_update_seconds": update_seconds,
        "warmup_optimizer_updates": warmup_updates,
        "timed_optimizer_updates": optimizer_updates,
        "timing_includes_startup": False,
        "timing_statistic": "per_update_median",
        "cudnn_benchmark": benchmark,
        "diagnostics": {
            **trained.diagnostics,
            "mean_patches_per_update": source.patches_drawn / total_updates,
            "supervised_traces_per_update": trained.supervised_trace_presentations / total_updates,
            "effective_inner_fraction_per_patch": source.effective_inner_fractions,
        },
    }
    del trained, source
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    core = (len(aligned.time_s), *config["prediction"]["core_shape"][1:])
    # This is the full-time prediction path's spatial halo rule, including padding
    # for the larger source-line training patches.
    tile = max(
        iter_ccnet5d_tiles(
            volume.values.shape, core, halo_radius=len(model.network) * (model.kernel_size // 2)
        ),
        key=lambda item: math.prod(s.stop - s.start for s in item.input_slices),
    )
    region = tile.input_slices
    block = np.ascontiguousarray(
        normalize_by_global_rms(
            np.where(volume.observed_trace_mask[region[1:]][None], volume.values[region], 0),
            aligned.global_rms,
        )
    )
    reset_peak()
    model.eval()
    dtype = mixed_precision_dtype(config["prediction"].get("mixed_precision", "off"), device)
    autocast = torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
    with torch.no_grad(), autocast:
        prediction = model(torch.from_numpy(block).to(device)[None, None])[0, 0][
            tile.local_core_slices
        ]
    if prediction.shape[0] != len(aligned.time_s):
        raise RuntimeError("probe must predict the full aligned time canvas")
    if dtype is not None and prediction.dtype != dtype:
        raise RuntimeError("probe prediction dtype differs from the configured mixed precision")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("non-finite probe prediction")
    forward_dtype = str(prediction.dtype)
    prediction = prediction.float().cpu().numpy()
    synchronize()
    allocated, reserved = peak()
    record.update(
        status="passed",
        aligned_time_samples=len(aligned.time_s),
        maximum_inference_input_shape=list(block.shape),
        spatial_core=list(core[1:]),
        inference_peak_allocated_bytes=allocated,
        inference_peak_reserved_bytes=reserved,
        inference_forward_dtype=forward_dtype,
        output_dtype=str(prediction.dtype),
        parameter_count=sum(p.numel() for p in model.parameters()),
        training_probe_optimizer_updates=total_updates,
        prediction_calls=1,
        time_split=False,
        runtime_seconds=time.monotonic() - started,
    )
    record["measured_peak_reserved_gib_ceil"] = math.ceil(
        max(record["training_peak_reserved_bytes"], record["inference_peak_reserved_bytes"]) / 2**30
    )
    return record


def benchmark_comparisons(records):
    """Report both trial orders and compare medians of all steady-state updates."""
    candidates = {}
    for row in records:
        if (
            row.get("status") != "passed"
            or "benchmark_order" not in row
            or not row.get("warmup_optimizer_updates")
            or not row.get("per_update_seconds")
        ):
            continue
        candidates.setdefault(row["candidate_id"], []).append(row)
    comparisons = {}
    for candidate, trials in candidates.items():
        orders = []
        all_updates = {"off": [], "on": []}
        for order in ("off,on", "on,off"):
            pair = [row for row in trials if row["benchmark_order"] == order]
            if len(pair) != 2 or {row["cudnn_benchmark"] for row in pair} != {False, True}:
                continue
            times = {}
            for row in pair:
                flag = "on" if row["cudnn_benchmark"] else "off"
                times[flag] = statistics.median(row["per_update_seconds"])
                all_updates[flag].extend(row["per_update_seconds"])
            orders.append(
                {"order": order.split(","), **times, "speedup": times["off"] / times["on"]}
            )
        if len(orders) == 2:
            times = {flag: statistics.median(values) for flag, values in all_updates.items()}
            comparisons[candidate] = {
                **times,
                "speedup": times["off"] / times["on"],
                "timing_statistic": "per_update_median",
                "orders": orders,
            }
    return comparisons
