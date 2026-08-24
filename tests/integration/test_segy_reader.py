from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.segy_reader import build_time_axis, read_trace_amplitudes
from tests.fixtures.tiny_segy import TinySegyFile, write_tiny_segy


@pytest.fixture
def tiny_segy(tmp_path: Path) -> TinySegyFile:
    return write_tiny_segy(tmp_path / "tiny.sgy")


def test_reads_only_the_requested_traces(tiny_segy: TinySegyFile) -> None:
    amplitudes = read_trace_amplitudes(tiny_segy.path, [1, 4])

    assert amplitudes.shape == (2, tiny_segy.sample_count)
    np.testing.assert_array_equal(amplitudes[0], tiny_segy.amplitudes[1])
    np.testing.assert_array_equal(amplitudes[1], tiny_segy.amplitudes[4])


def test_preserves_the_requested_order(tiny_segy: TinySegyFile) -> None:
    amplitudes = read_trace_amplitudes(tiny_segy.path, [5, 0, 3])

    np.testing.assert_array_equal(amplitudes, tiny_segy.amplitudes[[5, 0, 3]])


def test_returns_c_contiguous_float32(tiny_segy: TinySegyFile) -> None:
    amplitudes = read_trace_amplitudes(tiny_segy.path, [0])

    assert amplitudes.dtype == np.float32
    assert amplitudes.flags["C_CONTIGUOUS"]


def test_empty_indices_are_an_error(tiny_segy: TinySegyFile) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        read_trace_amplitudes(tiny_segy.path, [])


def test_duplicate_indices_are_an_error(tiny_segy: TinySegyFile) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        read_trace_amplitudes(tiny_segy.path, [1, 1])


def test_negative_indices_are_an_error(tiny_segy: TinySegyFile) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        read_trace_amplitudes(tiny_segy.path, [-1])


def test_out_of_range_indices_are_an_error(tiny_segy: TinySegyFile) -> None:
    with pytest.raises(ValueError, match="out of range"):
        read_trace_amplitudes(tiny_segy.path, [len(tiny_segy.headers)])


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_trace_amplitudes(tmp_path / "absent.sgy", [0])


def test_time_axis_matches_known_values() -> None:
    time_s = build_time_axis(4, 0.004)

    np.testing.assert_allclose(time_s, [0.0, 0.004, 0.008, 0.012])
    assert time_s.dtype == np.float64


@pytest.mark.parametrize("sample_count", [0, -1])
def test_invalid_sample_count_is_an_error(sample_count: int) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        build_time_axis(sample_count, 0.004)


@pytest.mark.parametrize("interval", [0.0, -0.004, float("nan")])
def test_invalid_sample_interval_is_an_error(interval: float) -> None:
    with pytest.raises(ValueError, match="sample_interval_s"):
        build_time_axis(4, interval)
