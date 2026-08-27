"""Fit and apply training-only coordinate and amplitude normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.model_coordinates import build_spatial_model_coordinates
from seis_interp.processing.trace_splits import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

_VALID_SPLITS = frozenset((TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT))
_PARAMETER_KEYS = frozenset(
    ("coordinate_order", "coordinate_min", "coordinate_max", "amplitude_rms")
)
_LINEAR_SPATIAL_FEATURE_COUNT = 3
_AZIMUTH_FEATURE_MIN = (-1.0, -1.0)
_AZIMUTH_FEATURE_MAX = (1.0, 1.0)
_AMPLITUDE_ROW_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class NormalizationParameters:
    """Linear coordinate ranges, fixed azimuth bounds, and training amplitude scale."""

    coordinate_order: tuple[str, ...]
    coordinate_min: tuple[float, ...]
    coordinate_max: tuple[float, ...]
    amplitude_rms: float

    def __post_init__(self) -> None:
        """Validate and canonicalize the small serializable parameter payload."""
        if isinstance(self.coordinate_order, (str, bytes)):
            raise ValueError("coordinate_order must be a sequence of strings")
        try:
            coordinate_order = tuple(self.coordinate_order)
        except TypeError as error:
            raise ValueError("coordinate_order must be a sequence of strings") from error
        if coordinate_order != MODEL_COORDINATE_ORDER:
            raise ValueError(
                f"coordinate_order must match the model coordinate order: {MODEL_COORDINATE_ORDER}"
            )

        coordinate_min = _finite_float_tuple(
            self.coordinate_min, "coordinate_min", len(MODEL_COORDINATE_ORDER)
        )
        coordinate_max = _finite_float_tuple(
            self.coordinate_max, "coordinate_max", len(MODEL_COORDINATE_ORDER)
        )
        if any(
            minimum > maximum
            for minimum, maximum in zip(coordinate_min, coordinate_max, strict=True)
        ):
            raise ValueError("coordinate_min must not exceed coordinate_max")
        if coordinate_min[-2:] != _AZIMUTH_FEATURE_MIN:
            raise ValueError("azimuth sine/cosine coordinate_min must be (-1.0, -1.0)")
        if coordinate_max[-2:] != _AZIMUTH_FEATURE_MAX:
            raise ValueError("azimuth sine/cosine coordinate_max must be (1.0, 1.0)")
        amplitude_rms = _positive_finite_float(self.amplitude_rms, "amplitude_rms")

        object.__setattr__(self, "coordinate_order", coordinate_order)
        object.__setattr__(self, "coordinate_min", coordinate_min)
        object.__setattr__(self, "coordinate_max", coordinate_max)
        object.__setattr__(self, "amplitude_rms", amplitude_rms)

    def to_dict(self) -> dict[str, object]:
        """Return the plain JSON-compatible normalization schema."""
        return {
            "coordinate_order": list(self.coordinate_order),
            "coordinate_min": list(self.coordinate_min),
            "coordinate_max": list(self.coordinate_max),
            "amplitude_rms": self.amplitude_rms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NormalizationParameters:
        """Build validated parameters from the plain JSON-compatible schema."""
        if not isinstance(payload, Mapping):
            raise ValueError("normalization parameters must be a mapping")
        keys = frozenset(payload)
        if keys != _PARAMETER_KEYS:
            missing = sorted(_PARAMETER_KEYS - keys)
            unexpected = sorted(keys - _PARAMETER_KEYS)
            raise ValueError(
                "normalization parameters have invalid keys: "
                f"missing={missing}, unexpected={unexpected}"
            )

        coordinate_order = payload["coordinate_order"]
        if isinstance(coordinate_order, (str, bytes)) or not isinstance(coordinate_order, Sequence):
            raise ValueError("coordinate_order must be a sequence of strings")
        if not all(isinstance(value, str) for value in coordinate_order):
            raise ValueError("coordinate_order must contain only strings")

        return cls(
            coordinate_order=tuple(coordinate_order),
            coordinate_min=_finite_float_tuple(
                payload["coordinate_min"], "coordinate_min", len(MODEL_COORDINATE_ORDER)
            ),
            coordinate_max=_finite_float_tuple(
                payload["coordinate_max"], "coordinate_max", len(MODEL_COORDINATE_ORDER)
            ),
            amplitude_rms=_positive_finite_float(payload["amplitude_rms"], "amplitude_rms"),
        )


def fit_normalization_parameters(
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
    *,
    split_column: str = "split",
    amplitudes_are_finite: bool = False,
) -> NormalizationParameters:
    """Fit coordinate ranges and global RMS without held-out trace leakage.

    ``amplitudes_are_finite`` may be set only when an upstream loader has
    validated every amplitude. Training chunks are still checked while their
    RMS energy is accumulated.
    """
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    if not isinstance(split_column, str) or not split_column:
        raise ValueError("split_column must be a non-empty string")
    if not isinstance(amplitudes_are_finite, bool):
        raise ValueError("amplitudes_are_finite must be a boolean")
    required_columns = ("array_row", split_column)
    missing = [column for column in required_columns if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")

    array_rows = validated_array_rows(trace_table)
    amplitude_array = _validated_amplitude_array(
        amplitudes,
        check_finite=not amplitudes_are_finite,
    )
    time_array = _validated_numeric_array(time_s, "time_s", dimensions=1).astype(
        np.float64, copy=False
    )
    if amplitude_array.shape[1] != len(time_array):
        raise ValueError(
            f"amplitudes has {amplitude_array.shape[1]} samples but time_s has "
            f"{len(time_array)} values"
        )
    if np.any(array_rows < 0) or np.any(array_rows >= amplitude_array.shape[0]):
        raise ValueError(
            f"array_row values must be within amplitudes row range [0, {amplitude_array.shape[0]})"
        )

    split_values = trace_table[split_column]
    valid_split_mask = split_values.isin(_VALID_SPLITS)
    if not bool(valid_split_mask.all()):
        invalid = split_values.loc[~valid_split_mask].unique().tolist()
        raise ValueError(f"trace table contains invalid split values: {invalid}")
    train_mask = split_values.eq(TRAIN_SPLIT).to_numpy(dtype=bool)
    if not np.any(train_mask):
        raise ValueError("trace table contains no training rows")

    spatial_coordinates = build_spatial_model_coordinates(trace_table)

    train_linear_spatial = spatial_coordinates[train_mask, :_LINEAR_SPATIAL_FEATURE_COUNT]
    coordinate_min = (
        float(np.min(time_array)),
        *(float(value) for value in np.min(train_linear_spatial, axis=0)),
        *_AZIMUTH_FEATURE_MIN,
    )
    coordinate_max = (
        float(np.max(time_array)),
        *(float(value) for value in np.max(train_linear_spatial, axis=0)),
        *_AZIMUTH_FEATURE_MAX,
    )

    train_array_rows = np.sort(array_rows[train_mask])
    # Avoid a second pass after this function's own all-amplitude scan. When
    # that scan is skipped in favor of an upstream contract, retain a finite
    # check inside the training chunks that are read for RMS accumulation.
    amplitude_rms = _training_amplitude_rms(
        amplitude_array,
        train_array_rows,
        check_finite=amplitudes_are_finite,
    )
    if not np.isfinite(amplitude_rms) or amplitude_rms <= 0.0:
        raise ValueError(f"training amplitude RMS must be positive and finite, got {amplitude_rms}")

    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=coordinate_min,
        coordinate_max=coordinate_max,
        amplitude_rms=amplitude_rms,
    )


def normalize_time(
    time_s: np.ndarray,
    parameters: NormalizationParameters,
) -> np.ndarray:
    """Normalize the common time axis to the fitted min-max range."""
    time_array = _validated_numeric_array(time_s, "time_s", dimensions=1)
    minimum = parameters.coordinate_min[0]
    maximum = parameters.coordinate_max[0]
    if maximum == minimum:
        return np.zeros(time_array.shape, dtype=np.float64)
    return 2.0 * (time_array.astype(np.float64, copy=False) - minimum) / (maximum - minimum) - 1.0


def normalize_spatial_coordinates(
    trace_table: pd.DataFrame,
    parameters: NormalizationParameters,
) -> np.ndarray:
    """Normalize linear spatial coordinates and preserve azimuth sine/cosine."""
    coordinates = build_spatial_model_coordinates(trace_table)

    minimum = np.asarray(
        parameters.coordinate_min[1 : 1 + _LINEAR_SPATIAL_FEATURE_COUNT], dtype=np.float64
    )
    maximum = np.asarray(
        parameters.coordinate_max[1 : 1 + _LINEAR_SPATIAL_FEATURE_COUNT], dtype=np.float64
    )
    coordinate_range = maximum - minimum
    normalized = np.zeros(coordinates.shape, dtype=np.float64)
    varying = coordinate_range != 0.0
    normalized_linear = normalized[:, :_LINEAR_SPATIAL_FEATURE_COUNT]
    normalized_linear[:, varying] = (
        2.0
        * (coordinates[:, :_LINEAR_SPATIAL_FEATURE_COUNT][:, varying] - minimum[varying])
        / coordinate_range[varying]
        - 1.0
    )
    normalized[:, _LINEAR_SPATIAL_FEATURE_COUNT:] = coordinates[:, _LINEAR_SPATIAL_FEATURE_COUNT:]
    return normalized


def normalize_amplitudes(
    amplitudes: np.ndarray,
    parameters: NormalizationParameters,
) -> np.ndarray:
    """Scale amplitudes by the positive global training RMS."""
    amplitude_array = _validated_numeric_array(amplitudes, "amplitudes")
    return amplitude_array / parameters.amplitude_rms


def denormalize_amplitudes(
    normalized_amplitudes: np.ndarray,
    parameters: NormalizationParameters,
) -> np.ndarray:
    """Restore amplitudes from the global training RMS scale."""
    amplitude_array = _validated_numeric_array(normalized_amplitudes, "amplitudes")
    return amplitude_array * parameters.amplitude_rms


def write_normalization_parameters(
    path: Path,
    parameters: NormalizationParameters,
) -> None:
    """Write parameters as indented JSON with a terminating newline."""
    Path(path).write_text(
        json.dumps(parameters.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def read_normalization_parameters(path: Path) -> NormalizationParameters:
    """Read and validate normalization parameters from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationParameters.from_dict(payload)


def _validated_numeric_array(
    values: np.ndarray,
    name: str,
    *,
    dimensions: int | None = None,
) -> np.ndarray:
    """Return a non-empty, finite, real numeric array without copying it."""
    array = np.asarray(values)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validated_amplitude_array(
    values: np.ndarray,
    *,
    check_finite: bool,
) -> np.ndarray:
    """Validate a 2-D amplitude array, optionally checking finiteness by row chunk."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"amplitudes must be 2-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("amplitudes must not be empty")
    if array.dtype.kind not in "iuf":
        raise ValueError("amplitudes must contain real numeric values")

    if check_finite:
        for start in range(0, array.shape[0], _AMPLITUDE_ROW_CHUNK_SIZE):
            chunk = array[start : start + _AMPLITUDE_ROW_CHUNK_SIZE]
            if not np.all(np.isfinite(chunk)):
                raise ValueError("amplitudes contains non-finite values")
    return array


def _training_amplitude_rms(
    amplitudes: np.ndarray,
    training_array_rows: np.ndarray,
    *,
    check_finite: bool,
) -> float:
    """Accumulate float64 training energy, optionally validating the same chunks."""
    sum_squares = 0.0
    sample_count = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(training_array_rows), _AMPLITUDE_ROW_CHUNK_SIZE):
            rows = training_array_rows[start : start + _AMPLITUDE_ROW_CHUNK_SIZE]
            chunk = amplitudes[rows].astype(np.float64, copy=False)
            if check_finite and not np.all(np.isfinite(chunk)):
                raise ValueError("training amplitudes contain non-finite values")
            sum_squares += float(np.sum(np.square(chunk), dtype=np.float64))
            sample_count += int(chunk.size)
        return float(np.sqrt(sum_squares / sample_count))


def _finite_float_tuple(values: object, name: str, length: int) -> tuple[float, ...]:
    """Validate a fixed-length sequence of finite real values."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of {length} finite numbers")
    try:
        sequence = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of {length} finite numbers") from error
    if len(sequence) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(sequence)}")
    converted: list[float] = []
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain only finite real numbers")
        converted_value = float(value)
        if not np.isfinite(converted_value):
            raise ValueError(f"{name} must contain only finite real numbers")
        converted.append(converted_value)
    return tuple(converted)


def _positive_finite_float(value: object, name: str) -> float:
    """Validate a scalar that must be finite and greater than zero."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return converted
