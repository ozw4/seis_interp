"""Validate shared configuration values for training pipelines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

from seis_interp.configuration import ConfigurationError, get_required_config_value
from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return int(value)


def odd_positive_integer(value: object, name: str) -> int:
    converted = positive_integer(value, name)
    if converted % 2 == 0:
        raise ConfigurationError(f"{name} must be odd")
    return converted


def finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigurationError(f"{name} must be a finite number")
    return converted


def positive_float(value: object, name: str) -> float:
    converted = finite_float(value, name)
    if converted <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return converted


def nonnegative_float(value: object, name: str) -> float:
    converted = finite_float(value, name)
    if converted < 0.0:
        raise ConfigurationError(f"{name} must be non-negative")
    return converted


def probability(value: object, name: str) -> float:
    converted = finite_float(value, name)
    if converted < 0.0 or converted >= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1)")
    return converted


def require_exact(
    config: Mapping[str, object],
    dotted_path: str,
    expected: object,
) -> None:
    actual = get_required_config_value(config, dotted_path)
    if actual != expected:
        raise ConfigurationError(f"{dotted_path} must be {expected!r}, got {actual!r}")


def optional_ffid_range(config: Mapping[str, object]) -> tuple[int, int] | None:
    training = config.get("training")
    if not isinstance(training, Mapping) or "ffid_range" not in training:
        return None
    value = training["ffid_range"]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in value)
        or int(value[0]) < 0
        or int(value[0]) > int(value[1])
    ):
        raise ConfigurationError("training.ffid_range must be [minimum, maximum] integers")
    return int(value[0]), int(value[1])


def validated_effective_split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_EFFECTIVE_SPLITS):
        raise ConfigurationError(
            "evaluation.required_effective_split_counts must contain exactly "
            f"{list(_EFFECTIVE_SPLITS)}"
        )
    return {
        split: positive_integer(
            value[split],
            f"evaluation.required_effective_split_counts.{split}",
        )
        for split in _EFFECTIVE_SPLITS
    }


def validated_ffid_split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_EFFECTIVE_SPLITS):
        raise ConfigurationError(
            f"evaluation.required_ffid_split_counts must contain exactly {list(_EFFECTIVE_SPLITS)}"
        )
    return {
        split: positive_integer(
            value[split],
            f"evaluation.required_ffid_split_counts.{split}",
        )
        for split in _EFFECTIVE_SPLITS
    }


def validated_sorted_ffids(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(ffid, bool) or not isinstance(ffid, Integral) or int(ffid) < 0 for ffid in value
    ):
        raise ConfigurationError(f"{name} must be a sorted unique list of non-negative integers")
    converted = [int(ffid) for ffid in value]
    if converted != sorted(set(converted)):
        raise ConfigurationError(f"{name} must be a sorted unique list of non-negative integers")
    return tuple(converted)


def validated_positive_integer_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty list")
    return tuple(positive_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def validated_target_coordinate_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("model.target_coordinates must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in value):
        raise ConfigurationError("model.target_coordinates must contain non-empty strings")
    converted = tuple(value)
    if len(set(converted)) != len(converted):
        raise ConfigurationError("model.target_coordinates must not contain duplicates")
    return converted


# The SIREN pipeline reports the rejected value in its messages; keep that
# format separate from the plain validators above instead of merging them.
def validated_positive_config_integer(value: object, dotted_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ConfigurationError(f"{dotted_path} must be a positive integer, got {value!r}")
    return int(value)


def validated_positive_config_float(value: object, dotted_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{dotted_path} must be a positive finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(f"{dotted_path} must be a positive finite number, got {value!r}")
    return converted


def validated_nonnegative_config_float(value: object, dotted_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(
            f"{dotted_path} must be a non-negative finite number, got {value!r}"
        )
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ConfigurationError(
            f"{dotted_path} must be a non-negative finite number, got {value!r}"
        )
    return converted
