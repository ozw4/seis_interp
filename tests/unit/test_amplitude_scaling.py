from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.training import amplitude_scaling as scaling_module
from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    extract_per_trace_rms_scaled_rows,
    per_trace_rms_scaled_amplitudes,
    per_trace_rms_scaled_rows,
    validated_amplitude_scaling,
    validation_metric_domain_for_scaling,
)


def test_per_trace_rms_scaling_makes_each_trace_unit_rms_without_modifying_input() -> None:
    amplitudes = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [-2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    original = amplitudes.copy()

    scaled = per_trace_rms_scaled_amplitudes(amplitudes)

    trace_rms = np.sqrt(np.mean(np.square(scaled.astype(np.float64)), axis=1, dtype=np.float64))
    np.testing.assert_allclose(trace_rms, np.ones(3), rtol=1.0e-7)
    np.testing.assert_array_equal(amplitudes, original)
    assert scaled.dtype == np.float32


def test_per_trace_rms_scaling_reads_memmap_in_bounded_row_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amplitudes = np.arange(1, 25, dtype=np.float32).reshape(6, 4)
    path = tmp_path / "amplitudes.npy"
    np.save(path, amplitudes)
    memory_mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    square_shapes: list[tuple[int, ...]] = []
    actual_square = scaling_module.np.square

    def recording_square(values: object, *args: object, **kwargs: object) -> np.ndarray:
        square_shapes.append(np.asarray(values).shape)
        return actual_square(values, *args, **kwargs)

    monkeypatch.setattr(scaling_module, "_ROW_CHUNK_SIZE", 2)
    monkeypatch.setattr(scaling_module.np, "square", recording_square)

    scaled = per_trace_rms_scaled_amplitudes(memory_mapped)

    assert isinstance(memory_mapped, np.memmap)
    assert not memory_mapped.flags.writeable
    assert square_shapes == [(2, 4), (2, 4), (2, 4)]
    assert amplitudes.shape not in square_shapes
    np.testing.assert_allclose(
        np.sqrt(np.mean(scaled.astype(np.float64) ** 2, axis=1)),
        np.ones(6),
    )


def test_selected_row_scaling_does_not_read_or_require_unselected_trace_rms() -> None:
    amplitudes = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 12.0],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    scaled = per_trace_rms_scaled_rows(amplitudes, np.asarray([2, 0]))

    selected = scaled[[0, 2]].astype(np.float64)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(selected), axis=1)),
        np.ones(2),
    )
    np.testing.assert_array_equal(scaled[[1, 3]], np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "array_row 0"),
        (np.asarray([[1.0, 2.0], [np.nan, 3.0]]), "non-finite.*array_row 1"),
        (np.asarray([[1.0, 2.0], [np.inf, 3.0]]), "non-finite.*array_row 1"),
        (np.asarray([[1.0, 2.0], [-np.inf, 3.0]]), "non-finite.*array_row 1"),
    ],
)
def test_per_trace_rms_scaling_rejects_undefined_scales(
    bad_value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        per_trace_rms_scaled_amplitudes(bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [np.asarray([1.0, 2.0]), np.empty((0, 3)), np.asarray([[True, False]])],
)
def test_per_trace_rms_scaling_rejects_invalid_arrays(bad_value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        per_trace_rms_scaled_amplitudes(bad_value)


def test_extract_scaled_rows_returns_unit_rms_float32_in_requested_order() -> None:
    amplitudes = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [-2.0, 1.0, 2.0],
            [6.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    original = amplitudes.copy()
    array_rows = np.asarray([3, 0, 2], dtype=np.int64)

    extracted = extract_per_trace_rms_scaled_rows(amplitudes, array_rows)

    assert extracted.shape == (3, 3)
    assert extracted.dtype == np.float32
    trace_rms = np.sqrt(np.mean(np.square(extracted.astype(np.float64)), axis=1))
    np.testing.assert_allclose(trace_rms, np.ones(3), rtol=1.0e-7)
    expected = per_trace_rms_scaled_amplitudes(amplitudes)[array_rows]
    np.testing.assert_allclose(extracted, expected, rtol=1.0e-7)
    np.testing.assert_array_equal(amplitudes, original)


def test_extract_scaled_rows_is_compact_unlike_row_aligned_scaling() -> None:
    amplitudes = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [-2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    array_rows = np.asarray([2, 0], dtype=np.int64)

    extracted = extract_per_trace_rms_scaled_rows(amplitudes, array_rows)
    row_aligned = per_trace_rms_scaled_rows(amplitudes, array_rows)

    assert extracted.shape == (2, 3)
    assert row_aligned.shape == amplitudes.shape
    np.testing.assert_allclose(extracted, row_aligned[array_rows], rtol=1.0e-7)
    np.testing.assert_array_equal(row_aligned[1], np.zeros(3, dtype=np.float32))


def test_extract_scaled_rows_matches_across_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amplitudes = np.arange(1, 21, dtype=np.float32).reshape(5, 4)
    array_rows = np.asarray([4, 1, 3, 0, 2], dtype=np.int64)
    single_chunk = extract_per_trace_rms_scaled_rows(amplitudes, array_rows)

    monkeypatch.setattr(scaling_module, "_ROW_CHUNK_SIZE", 2)
    multi_chunk = extract_per_trace_rms_scaled_rows(amplitudes, array_rows)

    np.testing.assert_array_equal(single_chunk, multi_chunk)


def test_amplitude_scaling_names_are_strict() -> None:
    assert validated_amplitude_scaling(TRAIN_GLOBAL_RMS_SCALING) == TRAIN_GLOBAL_RMS_SCALING
    assert validated_amplitude_scaling(PER_TRACE_RMS_SCALING) == PER_TRACE_RMS_SCALING
    assert validation_metric_domain_for_scaling(TRAIN_GLOBAL_RMS_SCALING) == "train_global_rms"
    assert (
        validation_metric_domain_for_scaling(PER_TRACE_RMS_SCALING) == "oracle_per_trace_unit_rms"
    )

    with pytest.raises(ValueError, match="must be one of"):
        validated_amplitude_scaling("global_rms", name="training.amplitude_scaling")
    with pytest.raises(ValueError, match="training.amplitude_scaling"):
        validated_amplitude_scaling(None, name="training.amplitude_scaling")
