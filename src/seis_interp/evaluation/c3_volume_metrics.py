"""Evaluate dense C3 volume predictions on held-out physical amplitudes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from seis_interp import config_values
from seis_interp.data.c3_volume_adapter import (
    ObservedC3Volume,
    volume_to_trace_predictions,
)
from seis_interp.data.trace_store import AMPLITUDES_FILE_NAME
from seis_interp.evaluation.physical_amplitude_metrics import (
    physical_amplitude_energies,
    physical_amplitude_mean_trace_relative_mse,
    physical_amplitude_snr,
    physical_amplitude_target_metrics,
)
from seis_interp.processing.c3_volume_index import validated_index_range

_TARGET_TRACE_CHUNK_SIZE = 1024


def validate_c3_volume_evaluation_config(config: Mapping[str, object]) -> None:
    """Require target-only, physical-amplitude global SNR evaluation settings."""
    config_values.require_exact(
        config, "evaluation.primary_metric", "physical_amplitude_global_snr_db"
    )
    config_values.require_exact(config, "evaluation.domain", "evaluation_target")


def evaluate_c3_volume_prediction(
    predicted_values: np.ndarray,
    observed_volume: ObservedC3Volume,
    *,
    interim_dir: Path,
    volume_metadata: Mapping[str, object],
    target_coverage_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one prediction on the volume's evaluation-target traces.

    Reference amplitudes are read only for evaluation-target rows and the selected
    time interval. Energies are accumulated across all target samples, so the SNR
    is a physical-amplitude global SNR rather than an average of per-trace values.
    """
    time_start, time_stop = _validated_evaluation_inputs(
        predicted_values,
        observed_volume,
        volume_metadata,
    )
    target_trace_count, covered_target_trace_count = _validated_target_coverage(
        target_coverage_mask,
        target_mask=observed_volume.evaluation_target_trace_mask,
    )
    array_rows, trace_predictions = volume_to_trace_predictions(
        predicted_values,
        observed_volume.array_rows,
    )
    target_positions = np.flatnonzero(observed_volume.evaluation_target_trace_mask.reshape(-1))
    if target_positions.size == 0:
        raise ValueError("evaluation target trace mask must select at least one trace")

    amplitudes = np.load(
        Path(interim_dir) / AMPLITUDES_FILE_NAME,
        mmap_mode="r",
        allow_pickle=False,
    )
    _validate_amplitude_source(
        amplitudes,
        array_rows=array_rows,
        time_stop=time_stop,
    )

    reference_energy = 0.0
    error_energy = 0.0
    trace_relative_mse_sum = 0.0
    for start in range(0, len(target_positions), _TARGET_TRACE_CHUNK_SIZE):
        positions = target_positions[start : start + _TARGET_TRACE_CHUNK_SIZE]
        rows = array_rows[positions]
        reference = _finite_float64(
            amplitudes[rows, time_start:time_stop],
            name="evaluation target reference amplitudes",
        )
        prediction = _finite_float64(
            trace_predictions[positions],
            name="evaluation target predictions",
        )
        chunk_reference_energy, chunk_error_energy = physical_amplitude_energies(
            reference, prediction
        )
        reference_energy += chunk_reference_energy
        error_energy += chunk_error_energy
        trace_relative_mse_sum += len(positions) * physical_amplitude_mean_trace_relative_mse(
            reference,
            prediction,
        )
        if not all(
            math.isfinite(value)
            for value in (reference_energy, error_energy, trace_relative_mse_sum)
        ):
            raise ValueError("evaluation energies must be finite")

    trace_count = int(len(target_positions))
    sample_count = trace_count * (time_stop - time_start)
    target_metrics = physical_amplitude_target_metrics(
        trace_count=trace_count,
        sample_count=sample_count,
        reference_energy=reference_energy,
        error_energy=error_energy,
    )
    target_metrics.update(
        mean_trace_relative_mse=trace_relative_mse_sum / trace_count,
        covered_target_trace_count=covered_target_trace_count,
        target_trace_count=target_trace_count,
    )
    zero_fill_metrics = _zero_fill_metrics(
        sample_count=sample_count,
        reference_energy=reference_energy,
    )
    observed_max_abs_error = _observed_max_abs_error(predicted_values, observed_volume)

    return {
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "physical",
        "evaluation_target": target_metrics,
        "zero_fill": zero_fill_metrics,
        "observed_max_abs_error": observed_max_abs_error,
    }


def _validated_evaluation_inputs(
    predicted_values: np.ndarray,
    observed_volume: ObservedC3Volume,
    volume_metadata: Mapping[str, object],
) -> tuple[int, int]:
    if not isinstance(observed_volume, ObservedC3Volume):
        raise TypeError("observed_volume must be an ObservedC3Volume")
    if not isinstance(predicted_values, np.ndarray):
        raise ValueError("predicted_values must be a NumPy array")
    if not isinstance(observed_volume.values, np.ndarray) or observed_volume.values.ndim != 5:
        raise ValueError("observed volume values must be a five-dimensional NumPy array")
    if predicted_values.shape != observed_volume.values.shape:
        raise ValueError(
            "predicted_values shape must match observed volume values, got "
            f"{predicted_values.shape} and {observed_volume.values.shape}"
        )

    spatial_shape = predicted_values.shape[1:]
    _validate_trace_mask(
        observed_volume.observed_trace_mask,
        name="observed trace mask",
        spatial_shape=spatial_shape,
    )
    _validate_trace_mask(
        observed_volume.evaluation_target_trace_mask,
        name="evaluation target trace mask",
        spatial_shape=spatial_shape,
    )
    observed_mask = observed_volume.observed_trace_mask
    target_mask = observed_volume.evaluation_target_trace_mask
    if np.any(observed_mask & target_mask) or not np.all(observed_mask | target_mask):
        raise ValueError(
            "observed and evaluation target trace masks must disjointly cover the volume"
        )
    if not np.any(observed_mask):
        raise ValueError("observed trace mask must select at least one trace")

    if not isinstance(observed_volume.time_s, np.ndarray) or observed_volume.time_s.ndim != 1:
        raise ValueError("observed volume time_s must be a one-dimensional NumPy array")
    if len(observed_volume.time_s) != predicted_values.shape[0]:
        raise ValueError("observed volume time_s length must match the prediction time axis")
    if not isinstance(volume_metadata, Mapping):
        raise TypeError("volume_metadata must be a mapping")
    selection = volume_metadata.get("selection")
    if not isinstance(selection, Mapping) or "time" not in selection:
        raise ValueError("volume_metadata.selection.time is required")
    time_start, time_stop = validated_index_range(
        selection["time"],
        name="volume_metadata.selection.time",
    )
    if time_stop - time_start != predicted_values.shape[0]:
        raise ValueError(
            "volume metadata time selection length must match the prediction time axis"
        )
    return time_start, time_stop


def _validate_trace_mask(
    mask: np.ndarray,
    *,
    name: str,
    spatial_shape: tuple[int, ...],
) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 4 or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a four-dimensional boolean NumPy array")
    if mask.shape != spatial_shape:
        raise ValueError(f"{name} shape must match the prediction spatial shape")


def _validated_target_coverage(
    coverage_mask: np.ndarray | None,
    *,
    target_mask: np.ndarray,
) -> tuple[int, int]:
    target_trace_count = int(np.count_nonzero(target_mask))
    if coverage_mask is None:
        return target_trace_count, target_trace_count
    _validate_trace_mask(
        coverage_mask,
        name="target coverage mask",
        spatial_shape=target_mask.shape,
    )
    covered_target_trace_count = int(np.count_nonzero(coverage_mask & target_mask))
    if covered_target_trace_count != target_trace_count:
        raise ValueError("target coverage mask must cover every evaluation target trace")
    return target_trace_count, covered_target_trace_count


def _validate_amplitude_source(
    amplitudes: np.ndarray,
    *,
    array_rows: np.ndarray,
    time_stop: int,
) -> None:
    if not isinstance(amplitudes, np.ndarray) or amplitudes.ndim != 2:
        raise ValueError("amplitudes must be a two-dimensional NumPy array")
    if amplitudes.dtype.kind not in "fiu" or amplitudes.dtype.kind == "b":
        raise ValueError("amplitudes must contain real numeric values")
    if time_stop > amplitudes.shape[1]:
        raise ValueError("volume time selection is outside the source amplitudes")
    if len(array_rows) and int(array_rows.max()) >= amplitudes.shape[0]:
        raise ValueError("volume array_row values are outside the source amplitudes")


def _finite_float64(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        converted = np.asarray(values).astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} contain non-finite values")
    return converted


def _zero_fill_metrics(
    *,
    sample_count: int,
    reference_energy: float,
) -> dict[str, object]:
    snr_db, snr_status = physical_amplitude_snr(reference_energy, reference_energy)
    rmse = float(math.sqrt(reference_energy / sample_count))
    _require_finite_metric(rmse, "zero-fill RMSE")
    return {
        "snr_db": snr_db,
        "snr_status": snr_status,
        "rmse": rmse,
    }


def _observed_max_abs_error(
    predicted_values: np.ndarray,
    observed_volume: ObservedC3Volume,
) -> float:
    observed_mask = observed_volume.observed_trace_mask
    prediction = _finite_float64(
        predicted_values[:, observed_mask],
        name="observed predictions",
    )
    observed = _finite_float64(
        observed_volume.values[:, observed_mask],
        name="observed amplitudes",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        max_abs_error = float(np.max(np.abs(prediction - observed)))
    _require_finite_metric(max_abs_error, "observed maximum absolute error")
    return max_abs_error


def _require_finite_metric(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
