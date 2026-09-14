"""Fit independent source-line NeRSI functions and evaluate their full-volume union."""

import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from seis_interp import run_records
from seis_interp.c3_poc_run_records import poc_coverage_metadata, validate_poc_prediction
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.evaluation.normalized_trace_reference import evaluate_normalized_trace_reference
from seis_interp.nersi_config import validate_nersi_poc_config
from seis_interp.processing.nersi_local_time_alignment import fit_observed_receiver_y_shear
from seis_interp.processing.trace_rms_idw import interpolate_trace_rms_idw
from seis_interp.processing.trace_time_alignment import validate_time_alignment
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.devices import resolve_device
from seis_interp.training.local_nersi import (
    fit_local_nersi_block,
    restore_local_nersi_prediction,
    source_line_ranges,
)


def interpolate_local_nersi_run(
    *, config_path, output_dir, input_paths, expected_inputs_lock=None, reporter=print
):
    """Keep local artifacts separate from single-model runs; never select on partial T."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    settings = validate_nersi_poc_config(config)
    local_options = config.get("local_models")
    if not isinstance(local_options, dict) or set(local_options) not in (
        {"source_line_width"},
        {"source_line_width", "alignment_search"},
    ):
        raise ValueError("local_models requires source_line_width and optional alignment_search")
    search = local_options.get("alignment_search")
    if "alignment_search" in local_options and (
        not isinstance(search, dict)
        or set(search) != {"candidates", "distances"}
        or settings.time_alignment is None
        or settings.time_alignment["boundary"] != "fourier_periodic"
    ):
        raise ValueError(
            "alignment_search requires candidates/distances and Fourier-periodic alignment"
        )
    if (
        settings.cartesian_profile_coordinates
        or settings.nyquist_fractions
        or settings.augmentation
    ):
        raise ValueError(
            "local runs currently require index coordinates without augmentation/bandlimit"
        )
    if settings.trace_rms_idw is None:
        raise ValueError("local normalized reference requires observed_trace_rms_idw")
    device = resolve_device(settings.training.device)
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "method": "nersi_local_source_lines",
        "started_at_utc": run_records.utc_timestamp(),
        **run_records.current_git_metadata(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "loss_or_native_objective": settings.training.loss,
        "independent_evaluation": False,
        "blocks": [],
        "coordinate_normalization": "per_block_local_index_bounds",
        "source_line_range_units": "zero_based_volume_local_half_open",
        "amplitude_scale_fit": "shared_full_volume_O_only",
    }
    inputs = load_c3_random80_poc_inputs(config=config, **input_paths)
    if expected_inputs_lock is not None and inputs.inputs_lock != expected_inputs_lock:
        raise ValueError("local NeRSI inputs differ from the frozen full input lock")
    metadata["implementation_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[3])): file_sha256(path)
        for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    }
    observed = inputs.observed_volume
    width = local_options["source_line_width"]
    ranges = source_line_ranges(observed.values.shape[1], width)
    amplitude_scale = compute_observed_global_rms(observed.values, observed.observed_trace_mask)
    trace_scale = interpolate_trace_rms_idw(
        observed.values, observed.observed_trace_mask, **settings.trace_rms_idw
    )
    time_alignments = [settings.time_alignment] * len(ranges)
    if search is not None:
        fits = [
            fit_observed_receiver_y_shear(
                observed.values[:, start:stop], observed.observed_trace_mask[start:stop], **search
            )
            for start, stop in ranges
        ]
        metadata["time_alignment_fits"] = fits
        time_alignments = [
            validate_time_alignment(
                {
                    **settings.time_alignment,
                    "receiver_y_shift_samples_per_cell": fit["receiver_y_shift_samples_per_cell"],
                }
            )
            for fit in fits
        ]
        reporter(f"O-only local shears: {time_alignments}")
    output.mkdir(parents=True, exist_ok=False)
    metadata["observed_global_rms"] = amplitude_scale
    np.save(output / "trace_rms.npy", trace_scale, allow_pickle=False)
    metadata["trace_rms"] = {
        "path": "trace_rms.npy",
        "sha256": file_sha256(output / "trace_rms.npy"),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prediction = np.empty_like(observed.values, dtype=np.float32)
    try:
        for (start, stop), alignment in zip(ranges, time_alignments, strict=True):
            reporter(f"Local source lines [{start}, {stop}) of {observed.values.shape[1]}")
            record, block_prediction = fit_local_nersi_block(
                observed,
                replace(settings, time_alignment=alignment),
                start=start,
                stop=stop,
                amplitude_scale=amplitude_scale,
                trace_scale=trace_scale,
                inputs_lock=inputs.inputs_lock,
                checkpoint_path=output / f"lines_{start:02d}_{stop:02d}.pt",
                device=device,
                reporter=reporter,
            )
            prediction[:, start:stop] = block_prediction
            metadata["blocks"].append(record)
            run_records.write_run_outputs(
                output, config, inputs.inputs_lock, {}, metadata, metadata_file_name="metadata.json"
            )
        validate_poc_prediction(prediction, observed)
        np.save(output / "prediction.npy", prediction, allow_pickle=False)
        metadata["prediction"] = {
            "path": "prediction.npy",
            "sha256": file_sha256(output / "prediction.npy"),
        }
        reporter("Strict-loading every local checkpoint and reproducing full-volume prediction.")
        restored = restore_local_nersi_prediction(
            observed,
            settings,
            records=metadata["blocks"],
            width=width,
            amplitude_scale=amplitude_scale,
            trace_scale=trace_scale,
            inputs_lock=inputs.inputs_lock,
            checkpoint_directory=output,
            device=device,
            time_alignments=time_alignments,
        )
        np.testing.assert_allclose(restored, prediction, rtol=1e-6, atol=1e-6)
        metadata["checkpoint_restoration_max_abs_error"] = float(
            np.max(np.abs(restored - prediction))
        )
        del restored
        evaluation_started = time.perf_counter()
        physical = evaluate_c3_volume_prediction(
            prediction,
            observed,
            interim_dir=input_paths["interim_dir"],
            volume_metadata=inputs.volume_metadata,
            target_coverage_mask=observed.evaluation_target_trace_mask,
        )
        normalized = evaluate_normalized_trace_reference(
            prediction,
            observed,
            trace_scale,
            interim_dir=input_paths["interim_dir"],
            volume_metadata=inputs.volume_metadata,
        )
        metadata.update(
            status="success",
            finished_at_utc=run_records.utc_timestamp(),
            coverage=poc_coverage_metadata(
                observed.evaluation_target_trace_mask,
                observed.evaluation_target_trace_mask,
                time_sample_count=observed.values.shape[0],
            ),
            compute={
                key: sum(r[key] for r in metadata["blocks"])
                for key in (
                    "parameter_count",
                    "optimizer_updates",
                    "supervised_trace_presentations",
                )
            },
            evaluation_seconds=time.perf_counter() - evaluation_started,
            end_to_end_seconds=time.perf_counter() - started,
            resources=run_records.runtime_resource_metadata(device),
        )
        metrics = {"physical": physical, "normalized_reference": normalized}
        run_records.write_run_outputs(
            output,
            config,
            inputs.inputs_lock,
            metrics,
            metadata,
            metadata_file_name="metadata.json",
        )
        reporter(json.dumps(metrics, allow_nan=False))
        return metrics
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        run_records.write_run_outputs(
            output, config, inputs.inputs_lock, {}, metadata, metadata_file_name="metadata.json"
        )
        raise
