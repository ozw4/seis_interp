"""Evaluate global validation S/N one FFID at a time."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
)
from seis_interp.training.ffid_batches import (
    build_global_rms_trace_points,
    build_per_trace_rms_trace_points,
)
from seis_interp.training.prediction import predict_points


def evaluate_model_global_snr_by_ffid(
    model: torch.nn.Module,
    *,
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    rows_by_ffid: Mapping[int, np.ndarray],
    amplitude_rms: float,
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    prediction_batch_size: int,
    device: torch.device | str,
) -> float:
    """Return point-weighted target-domain S/N while retaining one FFID's points.

    Each FFID is built and predicted independently in sorted FFID order. Energies
    are accumulated as float64 scalars, matching ``signal_to_noise_ratio_db``
    without concatenating survey-wide coordinates, targets, or predictions.

    ``per_trace_rms`` divides each validation trace by its own target RMS. That
    branch is an oracle-normalized waveform diagnostic, not physical-amplitude
    interpolation, because the scale is unavailable for an unseen query trace.
    """
    sorted_groups = _validated_sorted_rows_by_ffid(rows_by_ffid)
    _positive_finite_float(amplitude_rms, "amplitude_rms")
    scaling = validated_amplitude_scaling(amplitude_scaling)
    batch_size = _positive_integer(prediction_batch_size, "prediction_batch_size")

    reference_energy = 0.0
    error_energy = 0.0
    for _, rows in sorted_groups:
        ffid_reference_energy, ffid_error_energy = _evaluate_ffid_energies(
            model,
            normalized_time=normalized_time,
            normalized_spatial_by_array_row=normalized_spatial_by_array_row,
            amplitudes=amplitudes,
            rows=rows,
            amplitude_rms=amplitude_rms,
            amplitude_scaling=scaling,
            prediction_batch_size=batch_size,
            device=device,
        )
        reference_energy += ffid_reference_energy
        error_energy += ffid_error_energy

    if not math.isfinite(reference_energy) or reference_energy == 0.0:
        if reference_energy == 0.0:
            raise ValueError("reference energy must be greater than zero")
        raise ValueError("reference energy must be finite")
    if not math.isfinite(error_energy):
        raise ValueError("error energy must be finite")
    if error_energy == 0.0:
        return float("inf")
    return float(10.0 * np.log10(reference_energy / error_energy))


def _evaluate_ffid_energies(
    model: torch.nn.Module,
    *,
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    rows: np.ndarray,
    amplitude_rms: float,
    amplitude_scaling: str,
    prediction_batch_size: int,
    device: torch.device | str,
) -> tuple[float, float]:
    """Build one FFID and release all of its arrays when the scalar energies return."""
    if amplitude_scaling == TRAIN_GLOBAL_RMS_SCALING:
        coordinates, targets = build_global_rms_trace_points(
            normalized_time,
            normalized_spatial_by_array_row,
            amplitudes,
            rows,
            amplitude_rms=amplitude_rms,
        )
    elif amplitude_scaling == PER_TRACE_RMS_SCALING:
        coordinates, targets = build_per_trace_rms_trace_points(
            normalized_time,
            normalized_spatial_by_array_row,
            amplitudes,
            rows,
        )
    else:  # pragma: no cover - public entry point validates the name.
        raise RuntimeError(f"unsupported amplitude scaling: {amplitude_scaling!r}")
    predictions = predict_points(
        model,
        coordinates,
        batch_size=prediction_batch_size,
        device=device,
    )
    prediction_array = np.asarray(predictions)
    if prediction_array.shape != targets.shape:
        raise ValueError(
            "prediction shape must match the FFID targets, got "
            f"{prediction_array.shape} and {targets.shape}"
        )

    reference_energy = 0.0
    error_energy = 0.0
    for start in range(0, len(targets), prediction_batch_size):
        stop = start + prediction_batch_size
        target_float = _float64_energy_slice(targets[start:stop], name="targets")
        prediction_float = _float64_energy_slice(
            prediction_array[start:stop],
            name="predictions",
        )
        reference_energy += float(np.sum(np.square(target_float), dtype=np.float64))
        difference = target_float - prediction_float
        error_energy += float(np.sum(np.square(difference), dtype=np.float64))
    return reference_energy, error_energy


def _float64_energy_slice(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return one bounded float64 energy slice after validating numeric finiteness."""
    try:
        converted = np.asarray(values).astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("targets and predictions must contain real numeric values") from error
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} contain non-finite values")
    return converted


def _validated_sorted_rows_by_ffid(
    rows_by_ffid: Mapping[int, np.ndarray],
) -> tuple[tuple[int, np.ndarray], ...]:
    if not isinstance(rows_by_ffid, Mapping):
        raise TypeError("rows_by_ffid must be a mapping")
    if not rows_by_ffid:
        raise ValueError("rows_by_ffid must not be empty")

    groups: list[tuple[int, np.ndarray]] = []
    used_rows: set[int] = set()
    for raw_ffid, raw_rows in rows_by_ffid.items():
        if isinstance(raw_ffid, bool) or not isinstance(raw_ffid, Integral):
            raise ValueError("FFID mapping keys must be non-negative integers")
        ffid = int(raw_ffid)
        if ffid < 0:
            raise ValueError("FFID mapping keys must be non-negative integers")
        rows = np.asarray(raw_rows)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError(f"rows for FFID {ffid} must be a non-empty one-dimensional array")
        if rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
            raise ValueError(f"rows for FFID {ffid} must have an integer dtype")
        if np.any(rows < 0):
            raise ValueError(f"rows for FFID {ffid} must be non-negative")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"rows for FFID {ffid} must contain unique values")
        integer_rows = np.sort(rows.astype(np.int64, copy=False))
        overlap = sorted(used_rows.intersection(int(row) for row in integer_rows))
        if overlap:
            raise ValueError(f"array_row values occur in more than one FFID group: {overlap}")
        used_rows.update(int(row) for row in integer_rows)
        groups.append((ffid, integer_rows))
    return tuple(sorted(groups, key=lambda item: item[0]))


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
