"""Numerical checks distinguish valid zeros from lost or overflowing signals."""

import json

import numpy as np
import pytest

from seis_interp.processing.normalization_qc import summarize_amplitude_normalization


@pytest.mark.parametrize("chunk_rows", [1, 4])
def test_extreme_scale_loses_squared_signal_and_clean_scale_preserves_it(chunk_rows):
    values = np.array([[1, -2, 3], [0, 0, 0], [10, -20, 30]], dtype=np.float32)
    original = values.copy()
    settings = dict(array_rows=np.arange(3), time_range=(0, 3), chunk_rows=chunk_rows)
    failed = summarize_amplitude_normalization(values, amplitude_scale=4.1921e34, **settings)
    passed = summarize_amplitude_normalization(values, amplitude_scale=10.0, **settings)
    assert failed["status"] == "failed"
    assert failed["normalized_nonfinite_sample_count"] == 0
    assert failed["nonzero_traces_with_zero_float32_squared_energy"] == 2
    assert passed["status"] == "passed"
    assert passed["physical_zero_trace_count"] == 1
    assert passed["nonzero_traces_with_zero_float32_squared_energy"] == 0
    assert passed["normalized_max_abs_amplitude"] == 3.0
    np.testing.assert_array_equal(values, original)
    json.dumps(failed, allow_nan=False)


def test_rows_and_time_outside_the_authorized_selection_are_not_read():
    values = np.full((3, 4), np.nan, dtype=np.float32)
    values[2, 1:3] = [3, 4]
    result = summarize_amplitude_normalization(
        values, np.array([2]), time_range=(1, 3), amplitude_scale=1
    )
    assert result["status"] == "passed"
    assert result["physical_rms"] == np.sqrt(12.5)
    assert result["sample_count"] == 2
    assert result["time_samples"] == [1, 3]


def test_overflow_of_squared_finite_values_is_reported():
    values = np.array([[1e30]], dtype=np.float32)
    result = summarize_amplitude_normalization(
        values, np.array([0]), time_range=(0, 1), amplitude_scale=1
    )
    assert result["status"] == "failed"
    assert result["normalized_nonfinite_sample_count"] == 0
    assert result["traces_with_nonfinite_float32_squared_energy"] == 1
    json.dumps(result, allow_nan=False)


def test_nonfinite_normalized_cast_is_reported_without_nonfinite_json():
    values = np.array([[1e30]], dtype=np.float32)
    result = summarize_amplitude_normalization(
        values, np.array([0]), time_range=(0, 1), amplitude_scale=1e-30
    )
    assert result["normalized_nonfinite_sample_count"] == 1
    assert result["normalized_max_abs_amplitude"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scale", [0, -1, True, np.inf, np.nan])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="amplitude_scale"):
        summarize_amplitude_normalization(
            np.ones((1, 1), dtype=np.float32),
            np.array([0]),
            time_range=(0, 1),
            amplitude_scale=scale,
        )
