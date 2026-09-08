"""Validate the fixed main C3 benchmark crop before data preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from seis_interp.configuration import ConfigurationError, get_required_config_value
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER, validated_index_range

MAIN_C3_BENCHMARK_SHAPE = (384, 16, 32, 8, 32)
MAIN_C3_TIME_RANGE = (0, 384)
MAIN_C3_SOURCE_LINE_RANGE = (25, 41)


@dataclass(frozen=True)
class C3BenchmarkDimensions:
    """Explicit expected dimensions for preparation and small synthetic fixtures."""

    time_range: tuple[int, int]
    sail_line_numbers: tuple[int, int]
    shape: tuple[int, int, int, int, int]

    def validate(self, config: Mapping[str, object]) -> None:
        """Require the configured contract to equal the caller's fixed expectations."""
        if config.get("data", {}).get("dataset_id") == "seg_c3_na" and self != MAIN_C3_DIMENSIONS:
            raise ConfigurationError("SEG C3 NA must use the fixed main benchmark dimensions")
        if self == MAIN_C3_DIMENSIONS:
            validate_c3_benchmark_contract(config)
        _require_value(config, "c3_benchmark.shape", list(self.shape))
        _require_value(
            config, "c3_benchmark.sail_lines.requested_inclusive", list(self.sail_line_numbers)
        )
        _require_value(config, "c3_benchmark.axis_order", list(VOLUME_AXIS_ORDER))
        _require_range(config, "benchmark_volume.selection.time", self.time_range)
        _require_range(config, "c3_benchmark.training.time_samples", self.time_range)
        _require_value(config, "c3_benchmark.start_rule", "centered_contiguous")
        if (
            self.shape[0] != self.time_range[1] - self.time_range[0]
            or self.shape[1] != self.sail_line_numbers[1] - self.sail_line_numbers[0] + 1
        ):
            raise ConfigurationError(
                "expected dimensions must agree with fixed time and sail lines"
            )


MAIN_C3_DIMENSIONS = C3BenchmarkDimensions((0, 384), (25, 40), MAIN_C3_BENCHMARK_SHAPE)


def validate_c3_benchmark_contract(
    config: Mapping[str, object], *, require_resolved: bool = False
) -> None:
    """Check the main crop contract, allowing unresolved spatial starts for planning.

    Original-number mappings must be explicitly recorded by geometry QC. Without
    original numbers the existing full-survey global source-line ranks are used.
    This function never derives physical time from a sample index.
    """
    _require_value(config, "c3_benchmark.axis_order", list(VOLUME_AXIS_ORDER))
    shape = get_required_config_value(config, "c3_benchmark.shape")
    if (
        not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in shape)
        or tuple(shape) != MAIN_C3_BENCHMARK_SHAPE
    ):
        raise ConfigurationError(f"c3_benchmark.shape must be {list(MAIN_C3_BENCHMARK_SHAPE)}")
    _require_range(config, "c3_benchmark.sail_lines.requested_inclusive", (25, 40))
    source_range = benchmark_source_line_range(config)
    _require_value(config, "c3_benchmark.start_rule", "centered_contiguous")
    _require_range(config, "c3_benchmark.training.time_samples", MAIN_C3_TIME_RANGE)
    _require_range(config, "benchmark_volume.selection.time", MAIN_C3_TIME_RANGE)
    _require_range(config, "benchmark_volume.selection.source_line", source_range)
    for axis, length in zip(VOLUME_AXIS_ORDER[2:], MAIN_C3_BENCHMARK_SHAPE[2:], strict=True):
        path = f"benchmark_volume.selection.{axis}"
        value = get_required_config_value(config, path)
        if value is None:
            if require_resolved:
                raise ConfigurationError(
                    f"{path} is unresolved; resolve centered_contiguous starts with geometry "
                    "QC and save integer ranges before preparing the main benchmark volume"
                )
            continue
        start, stop = validated_index_range(value, name=path)
        if stop - start != length:
            raise ConfigurationError(f"{path} must select exactly {length} indices")


def benchmark_source_line_range(config: Mapping[str, object]) -> tuple[int, int]:
    """Check the recorded conversion without silently shifting selected sail lines."""
    path = "c3_benchmark.sail_lines"
    numbering = get_required_config_value(config, f"{path}.numbering")
    bounds = validated_index_range(
        get_required_config_value(config, f"{path}.index_range"), name=f"{path}.index_range"
    )
    if numbering == "global_source_line_index":
        if bounds != MAIN_C3_SOURCE_LINE_RANGE:
            raise ConfigurationError(f"{path}.index_range must be [25, 41]")
    elif numbering == "original_sail_line_number":
        lines = get_required_config_value(config, f"{path}.lines")
        if not isinstance(lines, list) or len(lines) != 16 or bounds[1] - bounds[0] != 16:
            raise ConfigurationError("sail-line mapping must contain exactly 16 lines")
        try:
            numbers = [row["requested_sail_line_number"] for row in lines]
            indices = [row["source_line_index"] for row in lines]
        except (KeyError, TypeError) as error:
            raise ConfigurationError(
                "sail-line mapping must contain numbers and indices"
            ) from error
        if (
            any(type(v) is not int for v in numbers + indices)
            or sorted(numbers) != list(range(25, 41))
            or sorted(indices) != list(range(*bounds))
        ):
            raise ConfigurationError("sail-line mapping must preserve the requested 25..40 set")
    else:
        raise ConfigurationError(f"{path}.numbering is unsupported")
    return bounds


def _require_value(config: Mapping[str, object], path: str, expected: object) -> None:
    if get_required_config_value(config, path) != expected:
        raise ConfigurationError(f"{path} must be {expected!r}")


def _require_range(config: Mapping[str, object], path: str, expected: tuple[int, int]) -> None:
    value = validated_index_range(get_required_config_value(config, path), name=path)
    if value != expected:
        raise ConfigurationError(f"{path} must be {list(expected)}")
