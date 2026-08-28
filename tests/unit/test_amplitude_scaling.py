from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.training import amplitude_scaling as scaling_module
from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
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
