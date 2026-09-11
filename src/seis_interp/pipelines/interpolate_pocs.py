"""Run Fourier POCS-5D on one verified C3 benchmark volume."""

from __future__ import annotations

import platform
import resource
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_interp import config_values, run_records
from seis_interp.configuration import ConfigurationError, load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation.c3_volume_metrics import (
    evaluate_c3_volume_prediction,
    validate_c3_volume_evaluation_config,
)
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.processing.pocs_windows import WindowedPocsResult, interpolate_pocs_volume

METHOD = "pocs"
PREDICTION_RELATIVE_PATH = Path("artifacts") / "prediction.npy"

_POCS_KEYS = frozenset(
    (
        "n_iterations",
        "threshold_start",
        "threshold_end",
        "window_shape",
        "overlap",
    )
)

ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class _PocsSettings:
    n_iterations: int
    threshold_start: float
    threshold_end: float
    window_shape: tuple[int, int, int, int, int] | None
    overlap: tuple[int, int, int, int, int] | None


def interpolate_pocs_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    output_dir: Path,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Reconstruct, evaluate, and record one immutable Fourier POCS-5D run."""
    output_directory = Path(output_dir)
    run_records.check_new_output_directory(output_directory)
    run_started = time.perf_counter()
    config = load_resolved_config(Path(config_path))
    settings = _pocs_settings(config)
    validate_c3_volume_evaluation_config(config)

    started_at_utc = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    interim_directory = Path(interim_dir)
    processed_directory = Path(processed_dir)
    mask_directory = Path(mask_dir)
    case_directory = Path(case_dir)
    volume_directory = Path(volume_dir)

    _report(progress_reporter, "Loading and verifying C3 inputs.")
    load_started = time.perf_counter()
    inputs = load_c3_random80_poc_inputs(
        config=config,
        interim_dir=interim_directory,
        processed_dir=processed_directory,
        mask_dir=mask_directory,
        case_dir=case_directory,
        volume_dir=volume_directory,
    )
    observed_volume = inputs.observed_volume
    case = inputs.case
    volume_metadata = inputs.volume_metadata
    inputs_lock = inputs.inputs_lock
    load_and_verification_seconds = time.perf_counter() - load_started

    _report(progress_reporter, "Reconstructing the selected volume with Fourier POCS-5D.")
    reconstruction_started = time.perf_counter()
    reconstructed = interpolate_pocs_volume(
        observed_volume.values,
        observed_volume.observed_trace_mask,
        n_iterations=settings.n_iterations,
        threshold_start=settings.threshold_start,
        threshold_end=settings.threshold_end,
        window_shape=settings.window_shape,
        overlap=settings.overlap,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    target_coverage_mask = _validate_complete_prediction(reconstructed, observed_volume)

    _report(progress_reporter, "Evaluating reconstruction on evaluation-target traces.")
    evaluation_started = time.perf_counter()
    evaluation = evaluate_c3_volume_prediction(
        reconstructed.values,
        observed_volume,
        interim_dir=interim_directory,
        volume_metadata=volume_metadata,
        target_coverage_mask=target_coverage_mask,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    warnings: list[str] = []
    metrics = dict(evaluation)
    metrics.update(
        {
            "method": METHOD,
            "case_id": case["case_id"],
            "volume_id": volume_metadata["volume_id"],
            "uncovered_sample_count": reconstructed.uncovered_sample_count,
            "warnings": warnings,
        }
    )

    _report(progress_reporter, "Writing prediction and immutable run records.")
    output_directory.mkdir(parents=True, exist_ok=False)
    prediction_path = output_directory / PREDICTION_RELATIVE_PATH
    prediction_path.parent.mkdir(parents=True, exist_ok=False)
    np.save(prediction_path, reconstructed.values, allow_pickle=False)
    end_to_end_seconds = time.perf_counter() - run_started
    finished_at_utc = run_records.utc_timestamp()
    run_metadata = _run_metadata(
        settings=settings,
        case=case,
        volume_metadata=volume_metadata,
        values=reconstructed.values,
        block_count=reconstructed.block_count,
        empty_block_count=reconstructed.empty_block_count,
        uncovered_sample_count=reconstructed.uncovered_sample_count,
        warnings=warnings,
        git_metadata=git_metadata,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        load_and_verification_seconds=load_and_verification_seconds,
        reconstruction_seconds=reconstruction_seconds,
        evaluation_seconds=evaluation_seconds,
        end_to_end_seconds=end_to_end_seconds,
    )

    run_records.write_run_outputs(
        output_directory,
        deepcopy(config),
        inputs_lock,
        metrics,
        run_metadata,
    )
    return metrics


def _validate_complete_prediction(
    reconstructed: WindowedPocsResult,
    observed_volume: ObservedC3Volume,
) -> np.ndarray:
    values = reconstructed.values
    if not isinstance(values, np.ndarray) or values.shape != observed_volume.values.shape:
        shape = getattr(values, "shape", None)
        raise ValueError(
            "POCS prediction shape does not match the prepared C3 volume: "
            f"prediction={shape}, expected={observed_volume.values.shape}"
        )
    if values.dtype.kind not in "fiu" or values.dtype.kind == "b":
        raise ValueError("POCS prediction must contain real numeric physical amplitudes")
    if reconstructed.uncovered_sample_count != 0:
        raise ValueError(
            "POCS reconstruction must cover every target sample; "
            f"uncovered_sample_count={reconstructed.uncovered_sample_count}"
        )
    if not np.isfinite(values).all():
        raise ValueError("POCS prediction contains non-finite values")
    observed_mask = observed_volume.observed_trace_mask
    if not np.array_equal(
        values[:, observed_mask],
        observed_volume.values[:, observed_mask],
    ):
        raise ValueError("POCS prediction must preserve observed samples exactly")
    return np.ones_like(observed_volume.evaluation_target_trace_mask, dtype=np.bool_)


def _pocs_settings(config: Mapping[str, object]) -> _PocsSettings:
    value = config.get("pocs")
    if not isinstance(value, Mapping):
        raise ConfigurationError("pocs configuration must be a mapping")
    actual_keys = set(value)
    if actual_keys != _POCS_KEYS:
        missing = sorted(_POCS_KEYS - actual_keys)
        unexpected = sorted(actual_keys - _POCS_KEYS, key=repr)
        raise ConfigurationError(
            "pocs configuration must contain exactly "
            f"{sorted(_POCS_KEYS)}; missing={missing}, unexpected={unexpected}"
        )

    n_iterations = config_values.positive_integer(value["n_iterations"], "pocs.n_iterations")
    if n_iterations < 2:
        raise ConfigurationError("pocs.n_iterations must be at least 2")
    threshold_start = config_values.positive_float(value["threshold_start"], "pocs.threshold_start")
    threshold_end = config_values.positive_float(value["threshold_end"], "pocs.threshold_end")
    if threshold_start > 1.0:
        raise ConfigurationError("pocs.threshold_start must be at most 1")
    if threshold_end > threshold_start:
        raise ConfigurationError(
            "pocs.threshold_end must be less than or equal to pocs.threshold_start"
        )

    raw_window = value["window_shape"]
    raw_overlap = value["overlap"]
    if (raw_window is None) != (raw_overlap is None):
        raise ConfigurationError("pocs.window_shape and pocs.overlap must both be null or lists")
    if raw_window is None:
        window_shape = None
        overlap = None
    else:
        window_values = config_values.validated_positive_integer_list(
            raw_window, "pocs.window_shape"
        )
        if len(window_values) != 5:
            raise ConfigurationError("pocs.window_shape must contain exactly five integers")
        overlap_values = _nonnegative_integer_list(raw_overlap, "pocs.overlap")
        if len(overlap_values) != 5:
            raise ConfigurationError("pocs.overlap must contain exactly five integers")
        if any(item >= length for item, length in zip(overlap_values, window_values, strict=True)):
            raise ConfigurationError(
                "each pocs.overlap value must be less than its pocs.window_shape value"
            )
        window_shape = window_values  # type: ignore[assignment]
        overlap = overlap_values  # type: ignore[assignment]

    return _PocsSettings(
        n_iterations=n_iterations,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        window_shape=window_shape,
        overlap=overlap,
    )


def _nonnegative_integer_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be a list")
    return tuple(
        config_values.nonnegative_integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _run_metadata(
    *,
    settings: _PocsSettings,
    case: Mapping[str, object],
    volume_metadata: Mapping[str, object],
    values: np.ndarray,
    block_count: int,
    empty_block_count: int,
    uncovered_sample_count: int,
    warnings: list[str],
    git_metadata: Mapping[str, str | bool],
    started_at_utc: str,
    finished_at_utc: str,
    load_and_verification_seconds: float,
    reconstruction_seconds: float,
    evaluation_seconds: float,
    end_to_end_seconds: float,
) -> dict[str, object]:
    mask = case["mask"]
    role_counts = volume_metadata["role_counts"]
    assert isinstance(mask, Mapping)
    assert isinstance(role_counts, Mapping)
    trace_count = int(volume_metadata["trace_count"])
    target_count = int(role_counts["evaluation_target"])
    maximum_block_shape = [
        int(length if settings.window_shape is None else min(length, requested))
        for length, requested in zip(
            values.shape,
            settings.window_shape or values.shape,
            strict=True,
        )
    ]
    return {
        "method": METHOD,
        "input_amplitude_domain": "physical",
        "output_amplitude_domain": "physical",
        "benchmark_global_rms_normalization": False,
        "observed_data_constraint": True,
        "native_iterative_updates": True,
        "case_id": case["case_id"],
        "volume_id": volume_metadata["volume_id"],
        **git_metadata,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": "success",
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "random_seed": mask["random_seed"],
        "input": {
            "dataset_id": case["dataset_id"],
            "partition": case["partition"],
            "mask": {
                "kind": mask["kind"],
                "random_seed": mask["random_seed"],
                "requested_missing_fraction": mask["missing_fraction"],
            },
            "selected_volume": {
                "selection": deepcopy(volume_metadata["selection"]),
                "shape": list(values.shape),
                "dtype": values.dtype.name,
                "observed_trace_count": int(role_counts["observed"]),
                "evaluation_target_trace_count": target_count,
                "actual_missing_trace_count": target_count,
                "actual_missing_fraction": target_count / trace_count,
            },
        },
        "pocs": {
            "n_iterations": settings.n_iterations,
            "threshold_start": settings.threshold_start,
            "threshold_end": settings.threshold_end,
            "threshold_schedule": "geometric",
            "threshold_mode": "hard",
            "threshold_reference": "initial_observed_spectrum_max_per_frequency_and_block",
            "fft_norm": "ortho",
            "frequency_bins": "all_rfft_bins",
            "amplitude_normalization": "none",
            "fft_internal_dtype": "float64_and_complex128",
        },
        "window": {
            "requested_shape": (
                None if settings.window_shape is None else list(settings.window_shape)
            ),
            "requested_overlap": None if settings.overlap is None else list(settings.overlap),
            "maximum_block_shape": maximum_block_shape,
            "blend": "positive_hann_interior_synthesis_only",
            "padding": "none",
            "block_count": block_count,
            "empty_block_count": empty_block_count,
            "uncovered_sample_count": uncovered_sample_count,
        },
        "coverage": {
            "target_trace_count": target_count,
            "covered_target_trace_count": target_count,
            "target_coverage_fraction": 1.0,
            "uncovered_sample_count": uncovered_sample_count,
        },
        "resources": {
            "load_and_verification_seconds": load_and_verification_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "evaluation_seconds": evaluation_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "process_max_rss_scope": "whole_process",
        },
        "prediction": {
            "artifact": PREDICTION_RELATIVE_PATH.as_posix(),
            "axis_order": list(VOLUME_AXIS_ORDER),
            "shape": list(values.shape),
            "dtype": values.dtype.name,
        },
        "warnings": list(warnings),
    }


def _report(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)
