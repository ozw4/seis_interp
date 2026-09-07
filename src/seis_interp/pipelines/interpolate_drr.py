"""Run damped rank-reduction 5D interpolation on a verified C3 volume."""

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
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs, load_c3_volume_run_inputs
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.processing.drr import DrrFrequencySelection, select_drr_frequencies
from seis_interp.processing.drr_windows import WindowedDrrResult, interpolate_drr_volume
from seis_interp.processing.level_four_hankel import level_four_hankel_matrix_shape

METHOD = "damped_rank_reduction_5d"
PREDICTION_RELATIVE_PATH = Path("artifacts") / "prediction.npy"

_DRR_KEYS = frozenset(
    (
        "rank",
        "damping_power",
        "n_iterations",
        "frequency_min_hz",
        "frequency_max_hz",
        "spatial_window_shape",
        "spatial_overlap",
    )
)
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class _DrrSettings:
    rank: int
    damping_power: int
    n_iterations: int
    frequency_min_hz: float
    frequency_max_hz: float | None
    spatial_window_shape: tuple[int, int, int, int] | None
    spatial_overlap: tuple[int, int, int, int] | None


def interpolate_drr_run(
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
    """Reconstruct, evaluate, and record one immutable CPU/NumPy DRR run."""
    output_directory = Path(output_dir)
    run_records.check_new_output_directory(output_directory)
    config = load_resolved_config(Path(config_path))
    settings = _drr_settings(config)
    _validate_evaluation_contract(config)
    started_at_utc = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()

    _report(progress_reporter, "Loading and verifying C3 inputs.")
    load_started = time.perf_counter()
    inputs = load_c3_volume_run_inputs(
        config=config,
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        mask_dir=Path(mask_dir),
        case_dir=Path(case_dir),
        volume_dir=Path(volume_dir),
    )
    load_seconds = time.perf_counter() - load_started
    observed = inputs.observed_volume

    _report(progress_reporter, "Reconstructing the selected volume with damped rank-reduction 5D.")
    reconstruction_started = time.perf_counter()
    frequencies = select_drr_frequencies(
        observed.time_s,
        frequency_min_hz=settings.frequency_min_hz,
        frequency_max_hz=settings.frequency_max_hz,
    )
    reconstructed = interpolate_drr_volume(
        observed.values,
        observed.observed_trace_mask,
        observed.time_s,
        rank=settings.rank,
        damping_power=settings.damping_power,
        n_iterations=settings.n_iterations,
        frequency_min_hz=settings.frequency_min_hz,
        frequency_max_hz=settings.frequency_max_hz,
        spatial_window_shape=settings.spatial_window_shape,
        spatial_overlap=settings.spatial_overlap,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started

    _report(progress_reporter, "Evaluating reconstruction on evaluation-target traces.")
    evaluation_started = time.perf_counter()
    evaluation = evaluate_c3_volume_prediction(
        reconstructed.values,
        observed,
        interim_dir=Path(interim_dir),
        volume_metadata=inputs.volume_metadata,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    uncovered_samples = reconstructed.uncovered_trace_count * observed.values.shape[0]
    warnings = _uncovered_warnings(reconstructed.uncovered_trace_count, uncovered_samples)
    metrics = dict(evaluation)
    metrics.update(
        {
            "method": METHOD,
            "case_id": inputs.case["case_id"],
            "volume_id": inputs.volume_metadata["volume_id"],
            "uncovered_trace_count": reconstructed.uncovered_trace_count,
            "uncovered_sample_count": uncovered_samples,
            "warnings": warnings,
        }
    )
    metadata = _run_metadata(
        settings=settings,
        inputs=inputs,
        reconstructed=reconstructed,
        frequencies=frequencies,
        git_metadata=git_metadata,
        started_at_utc=started_at_utc,
        load_seconds=load_seconds,
        reconstruction_seconds=reconstruction_seconds,
        evaluation_seconds=evaluation_seconds,
        warnings=warnings,
    )

    _report(progress_reporter, "Writing prediction and immutable run records.")
    output_directory.mkdir(parents=True, exist_ok=False)
    prediction_path = output_directory / PREDICTION_RELATIVE_PATH
    prediction_path.parent.mkdir(parents=True, exist_ok=False)
    np.save(prediction_path, reconstructed.values, allow_pickle=False)
    run_records.write_run_outputs(
        output_directory, deepcopy(config), inputs.inputs_lock, metrics, metadata
    )
    return metrics


def _validate_evaluation_contract(config: Mapping[str, object]) -> None:
    config_values.require_exact(
        config, "evaluation.primary_metric", "physical_amplitude_global_snr_db"
    )
    config_values.require_exact(config, "evaluation.domain", "evaluation_target")


def _drr_settings(config: Mapping[str, object]) -> _DrrSettings:
    value = config.get("drr")
    if not isinstance(value, Mapping):
        raise ConfigurationError("drr configuration must be a mapping")
    actual_keys = set(value)
    if actual_keys != _DRR_KEYS:
        missing = sorted(_DRR_KEYS - actual_keys)
        unexpected = sorted(actual_keys - _DRR_KEYS, key=repr)
        raise ConfigurationError(
            f"drr configuration must contain exactly {sorted(_DRR_KEYS)}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    rank = config_values.positive_integer(value["rank"], "drr.rank")
    power = config_values.positive_integer(value["damping_power"], "drr.damping_power")
    iterations = config_values.positive_integer(value["n_iterations"], "drr.n_iterations")
    minimum = config_values.nonnegative_float(value["frequency_min_hz"], "drr.frequency_min_hz")
    maximum = None
    if value["frequency_max_hz"] is not None:
        maximum = config_values.positive_float(value["frequency_max_hz"], "drr.frequency_max_hz")
        if maximum <= minimum:
            raise ConfigurationError(
                "drr.frequency_max_hz must be greater than drr.frequency_min_hz"
            )
    shape, overlap = _spatial_window_settings(value)
    return _DrrSettings(rank, power, iterations, minimum, maximum, shape, overlap)


def _spatial_window_settings(
    value: Mapping[str, object],
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    raw_shape, raw_overlap = value["spatial_window_shape"], value["spatial_overlap"]
    if (raw_shape is None) != (raw_overlap is None):
        raise ConfigurationError(
            "drr.spatial_window_shape and drr.spatial_overlap must both be null or lists"
        )
    if raw_shape is None:
        return None, None
    shape = config_values.validated_positive_integer_list(raw_shape, "drr.spatial_window_shape")
    if len(shape) != 4:
        raise ConfigurationError("drr.spatial_window_shape must contain exactly four integers")
    if not isinstance(raw_overlap, list) or len(raw_overlap) != 4:
        raise ConfigurationError("drr.spatial_overlap must be a list of exactly four integers")
    overlap = tuple(
        config_values.nonnegative_integer(item, f"drr.spatial_overlap[{index}]")
        for index, item in enumerate(raw_overlap)
    )
    if any(item >= length for item, length in zip(overlap, shape, strict=True)):
        raise ConfigurationError(
            "each drr.spatial_overlap value must be less than its spatial_window_shape value"
        )
    return shape, overlap  # type: ignore[return-value]


def _run_metadata(
    *,
    settings: _DrrSettings,
    inputs: C3VolumeRunInputs,
    reconstructed: WindowedDrrResult,
    frequencies: DrrFrequencySelection,
    git_metadata: Mapping[str, str | bool],
    started_at_utc: str,
    load_seconds: float,
    reconstruction_seconds: float,
    evaluation_seconds: float,
    warnings: list[str],
) -> dict[str, object]:
    values = reconstructed.values
    mask = inputs.case["mask"]
    assert isinstance(mask, Mapping)
    return {
        "method": METHOD,
        "case_id": inputs.case["case_id"],
        "volume_id": inputs.volume_metadata["volume_id"],
        **git_metadata,
        "started_at_utc": started_at_utc,
        "finished_at_utc": run_records.utc_timestamp(),
        "status": "success",
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "random_seed": mask["random_seed"],
        "input": _input_metadata(inputs),
        "drr": {
            "rank": settings.rank,
            "damping_power": settings.damping_power,
            "n_iterations": settings.n_iterations,
            "level_count": 4,
            "embedding_rule": "floor_axis_length_over_2_plus_1",
            "damping_reference": "rank_plus_one_singular_value",
            "svd_backend": "numpy.linalg.svd",
            "full_matrices": False,
            "observed_data_consistency": "hard_reinsertion_each_iteration_and_time_domain",
            "amplitude_normalization": "none",
        },
        "fft": _fft_metadata(settings, frequencies),
        "window": _window_metadata(settings, reconstructed),
        "resources": {
            "load_and_verification_seconds": load_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "evaluation_seconds": evaluation_seconds,
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


def _input_metadata(inputs: C3VolumeRunInputs) -> dict[str, object]:
    case, volume = inputs.case, inputs.volume_metadata
    mask, counts = case["mask"], volume["role_counts"]
    assert isinstance(mask, Mapping) and isinstance(counts, Mapping)
    target_count = int(counts["evaluation_target"])
    values = inputs.observed_volume.values
    return {
        "dataset_id": case["dataset_id"],
        "partition": case["partition"],
        "mask": {
            "kind": mask["kind"],
            "random_seed": mask["random_seed"],
            "requested_missing_fraction": mask["missing_fraction"],
        },
        "selected_volume": {
            "selection": deepcopy(volume["selection"]),
            "shape": list(values.shape),
            "dtype": values.dtype.name,
            "observed_trace_count": int(counts["observed"]),
            "evaluation_target_trace_count": target_count,
            "actual_missing_trace_count": target_count,
            "actual_missing_fraction": target_count / int(volume["trace_count"]),
        },
    }


def _fft_metadata(settings: _DrrSettings, selection: DrrFrequencySelection) -> dict[str, object]:
    first, last = int(selection.bin_indices[0]), int(selection.bin_indices[-1])
    return {
        "norm": "ortho",
        "padding": "next_power_of_two",
        "internal_dtype": "float64_and_complex128",
        "frequency_min_hz": settings.frequency_min_hz,
        "frequency_max_hz": settings.frequency_max_hz,
        "sample_interval_s": selection.sample_interval_s,
        "length": selection.fft_length,
        "nyquist_hz": 0.5 / selection.sample_interval_s,
        "processed_bin_count": int(len(selection.bin_indices)),
        "first_bin_index": first,
        "last_bin_index": last,
        "first_frequency_hz": float(selection.frequencies_hz[first]),
        "last_frequency_hz": float(selection.frequencies_hz[last]),
        "outside_band_missing_prediction": "zero_before_time_truncation",
    }


def _window_metadata(settings: _DrrSettings, result: WindowedDrrResult) -> dict[str, object]:
    spatial = result.values.shape[1:]
    maximum_spatial = tuple(
        min(length, requested)
        for length, requested in zip(spatial, settings.spatial_window_shape or spatial, strict=True)
    )
    return {
        "requested_spatial_shape": (
            None if settings.spatial_window_shape is None else list(settings.spatial_window_shape)
        ),
        "requested_spatial_overlap": (
            None if settings.spatial_overlap is None else list(settings.spatial_overlap)
        ),
        "maximum_block_shape": [result.values.shape[0], *maximum_spatial],
        "maximum_hankel_matrix_shape": list(level_four_hankel_matrix_shape(maximum_spatial)),
        "blend": "positive_hann_interior_synthesis_only",
        "spatial_padding": "none",
        "block_count": result.block_count,
        "empty_block_count": result.empty_block_count,
        "uncovered_trace_count": result.uncovered_trace_count,
        "uncovered_sample_count": result.uncovered_trace_count * result.values.shape[0],
    }


def _uncovered_warnings(traces: int, samples: int) -> list[str]:
    if traces == 0:
        return []
    return [
        f"{traces} traces ({samples} samples) were not covered by any non-empty DRR block "
        "and remain zero for evaluation"
    ]


def _report(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)
