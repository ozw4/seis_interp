"""Classify and exclude traces from bounded-chunk amplitude scans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

_CONFIG_KEYS = frozenset(("exclude_all_zero", "max_abs_amplitude"))
_ROW_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class TraceAmplitudeFilterConfig:
    """Amplitude conditions that determine whether a trace is eligible."""

    exclude_all_zero: bool
    max_abs_amplitude: float

    def __post_init__(self) -> None:
        """Validate values and canonicalize the threshold to a Python float."""
        if not isinstance(self.exclude_all_zero, bool):
            raise ValueError("exclude_all_zero must be a boolean")
        threshold = _positive_finite_float(self.max_abs_amplitude, "max_abs_amplitude")
        object.__setattr__(self, "max_abs_amplitude", threshold)

    @classmethod
    def from_mapping(
        cls,
        payload: object,
        *,
        name: str = "trace_amplitude_filter",
    ) -> TraceAmplitudeFilterConfig:
        """Build a config from a mapping containing exactly the canonical keys."""
        if not isinstance(name, str) or not name:
            raise ValueError("configuration name must be a non-empty string")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name} must be a mapping")
        keys = frozenset(payload)
        if keys != _CONFIG_KEYS:
            missing = sorted(_CONFIG_KEYS - keys)
            unexpected = sorted(repr(key) for key in keys - _CONFIG_KEYS)
            raise ValueError(f"{name} has invalid keys: missing={missing}, unexpected={unexpected}")
        try:
            return cls(
                exclude_all_zero=payload["exclude_all_zero"],
                max_abs_amplitude=payload["max_abs_amplitude"],
            )
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error

    def to_dict(self) -> dict[str, object]:
        """Return the canonical serializable configuration mapping."""
        return {
            "exclude_all_zero": self.exclude_all_zero,
            "max_abs_amplitude": self.max_abs_amplitude,
        }


@dataclass(frozen=True)
class TraceAmplitudeFilterResult:
    """Sorted array-row labels partitioned into eligible and exclusion reasons."""

    eligible_array_rows: np.ndarray
    all_zero_array_rows: np.ndarray
    excess_amplitude_array_rows: np.ndarray


def validated_trace_amplitude_filter_config(
    value: object,
    *,
    name: str = "trace_amplitude_filter",
) -> TraceAmplitudeFilterConfig:
    """Return a validated trace amplitude filter configuration."""
    if isinstance(value, TraceAmplitudeFilterConfig):
        return value
    return TraceAmplitudeFilterConfig.from_mapping(value, name=name)


def filter_trace_amplitudes(
    amplitudes: np.ndarray,
    config: TraceAmplitudeFilterConfig | Mapping[str, object],
    *,
    array_rows: np.ndarray | None = None,
) -> TraceAmplitudeFilterResult:
    """Classify trace rows without materializing a full amplitude-array copy.

    Every amplitude is required to be finite. An amplitude whose absolute value
    is strictly greater than ``max_abs_amplitude`` excludes its entire trace.
    Exact all-zero traces are excluded only when ``exclude_all_zero`` is true.
    """
    amplitude_array = _validated_amplitude_array(amplitudes)
    filter_config = validated_trace_amplitude_filter_config(config)
    row_labels = _validated_array_rows(array_rows, trace_count=amplitude_array.shape[0])

    all_zero = np.zeros(amplitude_array.shape[0], dtype=bool)
    excess_amplitude = np.zeros(amplitude_array.shape[0], dtype=bool)
    threshold = filter_config.max_abs_amplitude

    for start in range(0, amplitude_array.shape[0], _ROW_CHUNK_SIZE):
        stop = min(start + _ROW_CHUNK_SIZE, amplitude_array.shape[0])
        chunk = amplitude_array[start:stop]
        finite_rows = np.all(np.isfinite(chunk), axis=1)
        if not np.all(finite_rows):
            first_invalid = start + int(np.flatnonzero(~finite_rows)[0])
            raise ValueError(
                "amplitudes contain non-finite values for "
                f"array_row {int(row_labels[first_invalid])}"
            )
        all_zero[start:stop] = np.all(chunk == 0, axis=1)
        # Comparing both signed bounds avoids signed-integer minimum overflow
        # that can occur when applying np.abs directly to an integer array.
        excess_amplitude[start:stop] = np.any(
            (chunk > threshold) | (chunk < -threshold),
            axis=1,
        )

    excluded_all_zero = all_zero if filter_config.exclude_all_zero else np.zeros_like(all_zero)
    eligible = ~(excluded_all_zero | excess_amplitude)
    return TraceAmplitudeFilterResult(
        eligible_array_rows=_immutable_sorted_rows(row_labels[eligible]),
        all_zero_array_rows=_immutable_sorted_rows(row_labels[excluded_all_zero]),
        excess_amplitude_array_rows=_immutable_sorted_rows(row_labels[excess_amplitude]),
    )


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite real number") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number, got {value!r}")
    return number


def _validated_amplitude_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"amplitudes must be a non-empty two-dimensional array, got shape {array.shape}"
        )
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError("amplitudes must contain real numeric values")
    return array


def _validated_array_rows(values: np.ndarray | None, *, trace_count: int) -> np.ndarray:
    if values is None:
        return np.arange(trace_count, dtype=np.int64)
    rows = np.asarray(values)
    if rows.shape != (trace_count,) or rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError(
            "array_rows must be a one-dimensional integer array matching the amplitude rows"
        )
    if (
        len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.int64).max)
    ):
        raise ValueError("array_rows must contain unique non-negative int64-compatible values")
    return rows.astype(np.int64, copy=False)


def _immutable_sorted_rows(rows: np.ndarray) -> np.ndarray:
    result = np.sort(np.asarray(rows, dtype=np.int64))
    result.setflags(write=False)
    return result
