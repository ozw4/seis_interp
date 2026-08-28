"""Build training-time model coordinates from prepared geometry scales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.processing.normalization import (
    NormalizationParameters,
    normalize_spatial_coordinates,
)

CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES = "cmp_offset_azimuth"
CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES = "cmp_cartesian_half_offset"
CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES = "cmp_cartesian_half_offset_radius"
DEFAULT_COORDINATE_FEATURES = CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES

CMP_CARTESIAN_HALF_OFFSET_COORDINATE_ORDER = (
    "time_s",
    "cmp_x_m",
    "cmp_y_m",
    "half_offset_x_m",
    "half_offset_y_m",
)
CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_ORDER = (
    *CMP_CARTESIAN_HALF_OFFSET_COORDINATE_ORDER,
    "offset_m",
)

_SUPPORTED_COORDINATE_FEATURES = frozenset(
    (
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    )
)
_CARTESIAN_COORDINATE_FEATURES = frozenset(
    (
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    )
)
_CARTESIAN_REQUIRED_COLUMNS = (
    "cmp_x_m",
    "cmp_y_m",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_PARAMETER_KEYS = frozenset(
    (
        "coordinate_features",
        "coordinate_order",
        "coordinate_scale_min",
        "coordinate_scale_max",
        "half_offset_scale_m",
    )
)


@dataclass(frozen=True)
class ModelCoordinateParameters:
    """Serializable feature order and physical scales for model coordinates."""

    coordinate_features: str
    coordinate_order: tuple[str, ...]
    coordinate_scale_min: tuple[float, ...]
    coordinate_scale_max: tuple[float, ...]

    def __post_init__(self) -> None:
        mode = validated_coordinate_features(self.coordinate_features)
        expected_order = coordinate_order_for_features(mode)
        order = _string_tuple(self.coordinate_order, "coordinate_order")
        if order != expected_order:
            raise ValueError(
                f"coordinate_order must match {mode!r}: expected {expected_order}, got {order}"
            )
        minimum = _finite_float_tuple(
            self.coordinate_scale_min,
            "coordinate_scale_min",
            len(expected_order),
        )
        maximum = _finite_float_tuple(
            self.coordinate_scale_max,
            "coordinate_scale_max",
            len(expected_order),
        )
        if any(left > right for left, right in zip(minimum, maximum, strict=True)):
            raise ValueError("coordinate_scale_min must not exceed coordinate_scale_max")
        if mode in _CARTESIAN_COORDINATE_FEATURES:
            half_offset_min = minimum[3:5]
            half_offset_max = maximum[3:5]
            if (
                half_offset_min[0] != half_offset_min[1]
                or half_offset_max[0] != half_offset_max[1]
                or half_offset_min[0] != -half_offset_max[0]
            ):
                raise ValueError(
                    "Cartesian half-offset axes must share one symmetric coordinate scale"
                )

        object.__setattr__(self, "coordinate_features", mode)
        object.__setattr__(self, "coordinate_order", order)
        object.__setattr__(self, "coordinate_scale_min", minimum)
        object.__setattr__(self, "coordinate_scale_max", maximum)

    @property
    def input_features(self) -> int:
        """Return the model input width implied by this coordinate representation."""
        return len(self.coordinate_order)

    @property
    def half_offset_scale_m(self) -> float | None:
        """Return the shared Cartesian half-offset scale when the mode has one."""
        if self.coordinate_features not in _CARTESIAN_COORDINATE_FEATURES:
            return None
        return self.coordinate_scale_max[3]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible coordinate transform contract."""
        return {
            "coordinate_features": self.coordinate_features,
            "coordinate_order": list(self.coordinate_order),
            "coordinate_scale_min": list(self.coordinate_scale_min),
            "coordinate_scale_max": list(self.coordinate_scale_max),
            "half_offset_scale_m": self.half_offset_scale_m,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelCoordinateParameters:
        """Restore and validate a serialized coordinate transform contract."""
        if not isinstance(payload, Mapping):
            raise ValueError("model coordinates must be a mapping")
        keys = frozenset(payload)
        if keys != _PARAMETER_KEYS:
            missing = sorted(_PARAMETER_KEYS - keys)
            unexpected = sorted(keys - _PARAMETER_KEYS)
            raise ValueError(
                f"model coordinates have invalid keys: missing={missing}, unexpected={unexpected}"
            )
        parameters = cls(
            coordinate_features=validated_coordinate_features(payload["coordinate_features"]),
            coordinate_order=_string_tuple(payload["coordinate_order"], "coordinate_order"),
            coordinate_scale_min=_finite_float_tuple(
                payload["coordinate_scale_min"],
                "coordinate_scale_min",
            ),
            coordinate_scale_max=_finite_float_tuple(
                payload["coordinate_scale_max"],
                "coordinate_scale_max",
            ),
        )
        if payload["half_offset_scale_m"] != parameters.half_offset_scale_m:
            raise ValueError(
                "half_offset_scale_m does not match the shared symmetric coordinate scale"
            )
        return parameters


def validated_coordinate_features(value: object, *, name: str = "coordinate_features") -> str:
    """Return a supported coordinate feature mode or reject it."""
    if not isinstance(value, str) or value not in _SUPPORTED_COORDINATE_FEATURES:
        raise ValueError(
            f"{name} must be one of {sorted(_SUPPORTED_COORDINATE_FEATURES)}, got {value!r}"
        )
    return str(value)


def coordinate_order_for_features(coordinate_features: str) -> tuple[str, ...]:
    """Return the authoritative model input order for one feature mode."""
    mode = validated_coordinate_features(coordinate_features)
    if mode == CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES:
        return MODEL_COORDINATE_ORDER
    if mode == CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES:
        return CMP_CARTESIAN_HALF_OFFSET_COORDINATE_ORDER
    return CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_ORDER


def model_coordinate_parameters(
    coordinate_features: str,
    normalization: NormalizationParameters,
) -> ModelCoordinateParameters:
    """Derive training-time feature scales from prepared training parameters."""
    mode = validated_coordinate_features(coordinate_features)
    if mode == CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES:
        return ModelCoordinateParameters(
            coordinate_features=mode,
            coordinate_order=MODEL_COORDINATE_ORDER,
            coordinate_scale_min=normalization.coordinate_min,
            coordinate_scale_max=normalization.coordinate_max,
        )

    prepared_max_offset_m = normalization.coordinate_max[3]
    if prepared_max_offset_m < 0.0:
        raise ValueError(
            f"prepared training maximum offset must be non-negative, got {prepared_max_offset_m}"
        )
    half_offset_scale_m = 0.5 * prepared_max_offset_m
    coordinate_order = coordinate_order_for_features(mode)
    coordinate_scale_min = (
        *normalization.coordinate_min[:3],
        -half_offset_scale_m,
        -half_offset_scale_m,
    )
    coordinate_scale_max = (
        *normalization.coordinate_max[:3],
        half_offset_scale_m,
        half_offset_scale_m,
    )
    if mode == CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES:
        coordinate_scale_min = (*coordinate_scale_min, normalization.coordinate_min[3])
        coordinate_scale_max = (*coordinate_scale_max, normalization.coordinate_max[3])
    return ModelCoordinateParameters(
        coordinate_features=mode,
        coordinate_order=coordinate_order,
        coordinate_scale_min=coordinate_scale_min,
        coordinate_scale_max=coordinate_scale_max,
    )


def normalize_training_spatial_coordinates(
    trace_table: pd.DataFrame,
    normalization: NormalizationParameters,
    parameters: ModelCoordinateParameters,
) -> np.ndarray:
    """Return normalized spatial features for the selected training representation."""
    if parameters.coordinate_features == CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES:
        return normalize_spatial_coordinates(trace_table, normalization)

    include_offset = (
        parameters.coordinate_features == CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES
    )
    physical = _cartesian_physical_coordinates(
        trace_table,
        include_offset=include_offset,
    )
    normalized = np.zeros(physical.shape, dtype=np.float64)
    minimum = np.asarray(parameters.coordinate_scale_min[1:3], dtype=np.float64)
    maximum = np.asarray(parameters.coordinate_scale_max[1:3], dtype=np.float64)
    coordinate_range = maximum - minimum
    varying = coordinate_range != 0.0
    normalized_cmp = normalized[:, :2]
    normalized_cmp[:, varying] = (
        2.0 * (physical[:, :2][:, varying] - minimum[varying]) / coordinate_range[varying] - 1.0
    )

    half_offset_scale_m = parameters.half_offset_scale_m
    assert half_offset_scale_m is not None
    if half_offset_scale_m != 0.0:
        normalized[:, 2:4] = physical[:, 2:4] / half_offset_scale_m
    if include_offset:
        offset_minimum = parameters.coordinate_scale_min[-1]
        offset_maximum = parameters.coordinate_scale_max[-1]
        if offset_maximum != offset_minimum:
            normalized[:, -1] = (
                2.0 * (physical[:, -1] - offset_minimum) / (offset_maximum - offset_minimum) - 1.0
            )
    return normalized


def _cartesian_physical_coordinates(
    trace_table: pd.DataFrame,
    *,
    include_offset: bool,
) -> np.ndarray:
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    required_columns = (
        (*_CARTESIAN_REQUIRED_COLUMNS, "offset_m")
        if include_offset
        else _CARTESIAN_REQUIRED_COLUMNS
    )
    missing = [column for column in required_columns if column not in trace_table]
    if missing:
        raise ValueError(f"trace table is missing required Cartesian coordinate columns: {missing}")
    non_numeric = [
        column
        for column in required_columns
        if not (
            is_integer_dtype(trace_table[column].dtype) or is_float_dtype(trace_table[column].dtype)
        )
    ]
    if non_numeric:
        raise ValueError(f"Cartesian coordinate columns must be numeric: {non_numeric}")
    physical_columns = trace_table[list(required_columns)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(physical_columns)):
        raise ValueError("Cartesian coordinate columns contain non-finite values")

    cmp_coordinates = physical_columns[:, :2]
    half_offset = 0.5 * (physical_columns[:, 2:4] - physical_columns[:, 4:6])
    coordinates = np.column_stack((cmp_coordinates, half_offset))
    if include_offset:
        coordinates = np.column_stack((coordinates, physical_columns[:, -1]))
    return coordinates


def _string_tuple(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    converted = tuple(values)
    if not converted or not all(isinstance(value, str) for value in converted):
        raise ValueError(f"{name} must be a non-empty sequence of strings")
    return converted


def _finite_float_tuple(
    values: object,
    name: str,
    length: int | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a sequence of finite numbers") from error
    if length is not None and len(converted) != length:
        raise ValueError(f"{name} must contain {length} values")
    if not converted or not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be a non-empty sequence of finite numbers")
    return converted
