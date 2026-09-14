"""Independent source-line NeRSI models with disjoint prediction ownership."""

import gc
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.file_checksums import file_sha256
from seis_interp.models.nersi import Nersi
from seis_interp.processing.trace_time_alignment import validate_time_alignment
from seis_interp.training.c3_volume_nersi_data import build_c3_volume_nersi_data
from seis_interp.training.c3_volume_nersi_prediction import predict_c3_volume_nersi
from seis_interp.training.fixed_step_nersi import train_nersi_fixed_steps
from seis_interp.training.nersi_checkpoints import (
    load_fixed_step_nersi_checkpoint,
    nersi_checkpoint_input_binding,
    save_fixed_step_nersi_checkpoint,
    validate_fixed_step_nersi_checkpoint_input_binding,
)
from seis_interp.training.randomness import seed_global_model_initialization


def source_line_ranges(count: int, width: int) -> list[tuple[int, int]]:
    """Partition local source-line indices exactly once, including a short final block."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (count, width)):
        raise ValueError("source-line count and width must be positive integers")
    if width > count:
        raise ValueError("source-line width must not exceed the volume")
    return [(start, min(start + width, count)) for start in range(0, count, width)]


def local_nersi_inputs(observed, start, stop, *, amplitude_scale, trace_scale, time_alignment):
    """Slice geometry and O storage, discarding T before any local preprocessing."""
    if not 0 <= start < stop <= observed.values.shape[1]:
        raise ValueError("invalid local source-line range")
    mask = observed.observed_trace_mask[start:stop].copy()
    values = np.zeros_like(observed.values[:, start:stop])
    values[:, mask] = observed.values[:, start:stop][:, mask]
    local = ObservedC3Volume(
        values=values,
        time_s=observed.time_s.copy(),
        array_rows=observed.array_rows[start:stop].copy(),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=observed.evaluation_target_trace_mask[start:stop].copy(),
    )
    data = build_c3_volume_nersi_data(
        local,
        amplitude_scale=amplitude_scale,
        trace_amplitude_scale=None if trace_scale is None else trace_scale[start:stop],
        time_alignment=time_alignment,
    )
    return local, data


def fit_local_nersi_block(
    observed,
    settings,
    *,
    start,
    stop,
    amplitude_scale,
    trace_scale,
    inputs_lock,
    checkpoint_path: Path,
    device,
    reporter=None,
):
    """Fit one fresh model and return its immutable checkpoint record and prediction."""
    device = torch.device(device)
    local, data = local_nersi_inputs(
        observed,
        start,
        stop,
        amplitude_scale=amplitude_scale,
        trace_scale=trace_scale,
        time_alignment=settings.time_alignment,
    )
    seed_global_model_initialization(settings.training.model_initialization_seed, device=device)
    model = Nersi(**settings.model_constructor_config(data.profile_shape))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    trained = train_nersi_fixed_steps(
        model,
        data,
        device=device,
        learning_rate=settings.training.learning_rate,
        profiles_per_step=settings.training.profiles_per_step,
        max_steps=settings.training.max_steps,
        report_interval=settings.training.report_interval,
        random_seed=settings.training.sampling_seed,
        gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        loss_name=settings.training.loss,
        optimization=settings.optimization,
        reporter=reporter,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = predict_c3_volume_nersi(
        model,
        data,
        local,
        batch_size=settings.prediction_batch_size,
        device=device,
    ).values
    prediction_seconds = time.perf_counter() - started
    save_fixed_step_nersi_checkpoint(
        checkpoint_path,
        model,
        data,
        global_step=trained.steps_completed,
        final_batch_loss=trained.final_batch_loss,
        input_binding=nersi_checkpoint_input_binding(inputs_lock),
        model_initialization_seed=settings.training.model_initialization_seed,
        sampling_seed=settings.training.sampling_seed,
        optimization=settings.optimization,
    )
    record = {
        "source_line_range": [start, stop],
        "checkpoint": {"path": checkpoint_path.name, "sha256": file_sha256(checkpoint_path)},
        "loss": settings.training.loss,
        "time_alignment": data.time_alignment,
        "model_initialization_seed": settings.training.model_initialization_seed,
        "sampling_seed": settings.training.sampling_seed,
        "micro_batch_profiles": settings.training.profiles_per_step,
        "gradient_accumulation_steps": settings.training.gradient_accumulation_steps,
        "effective_batch_profiles": (
            settings.training.profiles_per_step * settings.training.gradient_accumulation_steps
        ),
        "optimizer_updates": trained.steps_completed,
        "supervised_trace_presentations": trained.supervised_trace_presentations,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "history": list(trained.history),
    }
    del model
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return record, prediction


def restore_local_nersi_prediction(
    observed,
    settings,
    *,
    records,
    width,
    amplitude_scale,
    trace_scale,
    inputs_lock,
    checkpoint_directory: Path,
    device,
    time_alignments=None,
):
    """Strict-load every bound local checkpoint and reconstruct the complete volume."""
    expected = source_line_ranges(observed.values.shape[1], width)
    if [r["source_line_range"] for r in records] != [list(r) for r in expected]:
        raise ValueError("local checkpoint ranges must partition the full volume in order")
    if time_alignments is None:
        time_alignments = [settings.time_alignment] * len(expected)
    if len(time_alignments) != len(expected):
        raise ValueError("one time alignment is required per local model")
    prediction = np.empty_like(observed.values, dtype=np.float32)
    for record, (start, stop), alignment in zip(records, expected, time_alignments, strict=True):
        local_settings = replace(
            settings,
            time_alignment=None if alignment is None else validate_time_alignment(alignment),
        )
        name = record["checkpoint"]["path"]
        if Path(name).name != name:
            raise ValueError("checkpoint path must be a filename")
        path = checkpoint_directory / name
        if file_sha256(path) != record["checkpoint"]["sha256"]:
            raise ValueError("local checkpoint hash mismatch")
        local, data = local_nersi_inputs(
            observed,
            start,
            stop,
            amplitude_scale=amplitude_scale,
            trace_scale=trace_scale,
            time_alignment=local_settings.time_alignment,
        )
        checkpoint = load_fixed_step_nersi_checkpoint(path)
        validate_fixed_step_nersi_checkpoint_input_binding(checkpoint, inputs_lock, data)
        if (
            checkpoint.model.constructor_config()
            != settings.model_constructor_config(data.profile_shape)
            or record["loss"] != settings.training.loss
            or checkpoint.global_step != settings.training.max_steps
            or checkpoint.model_initialization_seed != settings.training.model_initialization_seed
            or checkpoint.sampling_seed != settings.training.sampling_seed
            or checkpoint.optimization != settings.optimization
        ):
            raise ValueError("local checkpoint model or recorded loss differs from config")
        prediction[:, start:stop] = predict_c3_volume_nersi(
            checkpoint.model,
            data,
            local,
            batch_size=settings.prediction_batch_size,
            device=device,
        ).values
        del checkpoint
    return prediction
