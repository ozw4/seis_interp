from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from seis_interp.processing import trace_amplitude_filter as filter_module
from seis_interp.processing.trace_amplitude_filter import (
    TraceAmplitudeFilterConfig,
    filter_trace_amplitudes,
    validated_trace_amplitude_filter_config,
)


def _config(*, exclude_all_zero: bool = True) -> TraceAmplitudeFilterConfig:
    return TraceAmplitudeFilterConfig(
        exclude_all_zero=exclude_all_zero,
        max_abs_amplitude=10.0,
    )


def test_config_validates_strict_mapping_and_serializes_canonical_values() -> None:
    config = validated_trace_amplitude_filter_config(
        {"exclude_all_zero": True, "max_abs_amplitude": np.float32(12.5)}
    )

    assert config == TraceAmplitudeFilterConfig(True, 12.5)
    assert config.to_dict() == {
        "exclude_all_zero": True,
        "max_abs_amplitude": 12.5,
    }
    assert type(config.max_abs_amplitude) is float
    with pytest.raises(FrozenInstanceError):
        config.max_abs_amplitude = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {"exclude_all_zero": True},
        {
            "exclude_all_zero": True,
            "max_abs_amplitude": 1.0,
            "unexpected": False,
        },
    ],
)
def test_config_rejects_missing_or_unexpected_mapping_keys(payload: object) -> None:
    with pytest.raises(ValueError, match="invalid keys"):
        validated_trace_amplitude_filter_config(payload)


def test_config_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        validated_trace_amplitude_filter_config([True, 10.0])


def test_config_mapping_errors_include_the_requested_configuration_path() -> None:
    with pytest.raises(
        ValueError,
        match=r"sampling\.trace_amplitude_filter: max_abs_amplitude",
    ):
        TraceAmplitudeFilterConfig.from_mapping(
            {"exclude_all_zero": True, "max_abs_amplitude": 0.0},
            name="sampling.trace_amplitude_filter",
        )


@pytest.mark.parametrize("value", [1, "true", None, np.bool_(True)])
def test_config_requires_a_boolean_zero_trace_switch(value: object) -> None:
    with pytest.raises(ValueError, match="exclude_all_zero must be a boolean"):
        TraceAmplitudeFilterConfig(value, 10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [True, "10", 0.0, -1.0, np.nan, np.inf, -np.inf, 10**400],
)
def test_config_requires_a_positive_finite_real_threshold(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite real"):
        TraceAmplitudeFilterConfig(True, value)  # type: ignore[arg-type]


def test_filter_classifies_rows_by_exact_zero_and_strict_amplitude_limit() -> None:
    amplitudes = np.asarray(
        [
            [0.0, 1.0, -2.0],
            [0.0, -0.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.01, 0.0, 0.0],
            [0.0, -11.0, 0.0],
        ],
        dtype=np.float32,
    )
    original = amplitudes.copy()
    labels = np.asarray([50, 20, 40, 10, 30], dtype=np.int64)

    result = filter_trace_amplitudes(amplitudes, _config(), array_rows=labels)

    np.testing.assert_array_equal(result.eligible_array_rows, [40, 50])
    np.testing.assert_array_equal(result.all_zero_array_rows, [20])
    np.testing.assert_array_equal(result.excess_amplitude_array_rows, [10, 30])
    np.testing.assert_array_equal(amplitudes, original)
    for rows in (
        result.eligible_array_rows,
        result.all_zero_array_rows,
        result.excess_amplitude_array_rows,
    ):
        assert rows.dtype == np.int64
        assert not rows.flags.writeable


def test_filter_keeps_zero_traces_when_zero_exclusion_is_disabled() -> None:
    amplitudes = np.asarray([[0, 0], [1, 2], [11, 0]], dtype=np.int16)

    result = filter_trace_amplitudes(amplitudes, _config(exclude_all_zero=False))

    np.testing.assert_array_equal(result.eligible_array_rows, [0, 1])
    assert result.all_zero_array_rows.size == 0
    np.testing.assert_array_equal(result.excess_amplitude_array_rows, [2])


def test_filter_handles_minimum_signed_integer_without_absolute_value_overflow() -> None:
    amplitudes = np.asarray([[np.iinfo(np.int64).min], [10]], dtype=np.int64)

    result = filter_trace_amplitudes(amplitudes, _config())

    np.testing.assert_array_equal(result.eligible_array_rows, [1])
    np.testing.assert_array_equal(result.excess_amplitude_array_rows, [0])


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_filter_rejects_nonfinite_amplitudes_with_array_row_label(nonfinite: float) -> None:
    amplitudes = np.asarray([[1.0, 2.0], [nonfinite, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="non-finite.*array_row 77"):
        filter_trace_amplitudes(
            amplitudes,
            _config(),
            array_rows=np.asarray([99, 77]),
        )


def test_filter_reads_memmap_in_bounded_row_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "amplitudes.npy"
    np.save(path, np.arange(1, 25, dtype=np.float32).reshape(6, 4))
    amplitudes = np.load(path, mmap_mode="r", allow_pickle=False)
    config = _config()
    inspected_shapes: list[tuple[int, ...]] = []
    actual_isfinite = filter_module.np.isfinite

    def recording_isfinite(values: object, *args: object, **kwargs: object) -> np.ndarray:
        inspected_shapes.append(np.asarray(values).shape)
        return actual_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(filter_module, "_ROW_CHUNK_SIZE", 2)
    monkeypatch.setattr(filter_module.np, "isfinite", recording_isfinite)

    result = filter_trace_amplitudes(amplitudes, config)

    assert isinstance(amplitudes, np.memmap)
    assert inspected_shapes == [(2, 4), (2, 4), (2, 4)]
    assert amplitudes.shape not in inspected_shapes
    np.testing.assert_array_equal(result.eligible_array_rows, [0, 1])
    np.testing.assert_array_equal(result.excess_amplitude_array_rows, [2, 3, 4, 5])


@pytest.mark.parametrize(
    "amplitudes",
    [
        np.asarray([1.0, 2.0]),
        np.empty((0, 3), dtype=np.float32),
        np.empty((2, 0), dtype=np.float32),
        np.asarray([[True, False]]),
        np.asarray([[1.0 + 2.0j]]),
        np.asarray([["1", "2"]]),
    ],
)
def test_filter_rejects_invalid_amplitude_arrays(amplitudes: np.ndarray) -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        filter_trace_amplitudes(amplitudes, _config())


@pytest.mark.parametrize(
    "array_rows",
    [
        np.asarray([1]),
        np.asarray([[1, 2]]),
        np.asarray([1.0, 2.0]),
        np.asarray([True, False]),
        np.asarray([1, 1]),
        np.asarray([-1, 2]),
        np.asarray([0, np.iinfo(np.int64).max + 1], dtype=np.uint64),
    ],
)
def test_filter_rejects_invalid_array_row_labels(array_rows: np.ndarray) -> None:
    with pytest.raises(ValueError, match="array_rows"):
        filter_trace_amplitudes(
            np.asarray([[1.0], [2.0]]),
            _config(),
            array_rows=array_rows,
        )
