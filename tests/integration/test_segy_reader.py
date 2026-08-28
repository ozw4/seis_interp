from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.data import segy_reader as segy_reader_module
from seis_interp.data.segy_reader import (
    build_time_axis,
    iter_trace_amplitude_chunks,
    read_trace_amplitudes,
)
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


def test_chunk_reader_opens_once_and_preserves_order(
    tiny_segy: TinySegyFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_count = 0
    original_open = segy_reader_module.segyio.open

    def recording_open(*args: object, **kwargs: object):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(segy_reader_module.segyio, "open", recording_open)

    chunks = list(
        iter_trace_amplitude_chunks(
            tiny_segy.path,
            [5, 0, 3, 2, 1],
            chunk_size=2,
        )
    )

    assert open_count == 1
    assert [chunk.shape for chunk in chunks] == [(2, 8), (2, 8), (1, 8)]
    assert all(chunk.dtype == np.float32 for chunk in chunks)
    assert all(chunk.flags["C_CONTIGUOUS"] for chunk in chunks)
    np.testing.assert_array_equal(
        np.vstack(chunks),
        tiny_segy.amplitudes[[5, 0, 3, 2, 1]],
    )


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_chunk_reader_rejects_invalid_chunk_size(
    tiny_segy: TinySegyFile, chunk_size: object
) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        list(
            iter_trace_amplitude_chunks(
                tiny_segy.path,
                [0],
                chunk_size=chunk_size,
            )
        )


def test_chunk_reader_rejects_non_finite_values_in_a_later_chunk(
    tiny_segy: TinySegyFile,
) -> None:
    with segy_reader_module.segyio.open(
        str(tiny_segy.path),
        mode="r+",
        strict=False,
        ignore_geometry=True,
    ) as handle:
        corrupted = np.asarray(handle.trace[3], dtype=np.float32).copy()
        corrupted[0] = np.nan
        handle.trace[3] = corrupted

    with pytest.raises(ValueError, match="non-finite"):
        list(
            iter_trace_amplitude_chunks(
                tiny_segy.path,
                [0, 1, 2, 3],
                chunk_size=2,
            )
        )


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
