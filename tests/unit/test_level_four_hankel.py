from __future__ import annotations

from itertools import product
from math import prod

import numpy as np
import pytest

from seis_interp.processing.level_four_hankel import (
    average_level_four_hankel,
    hankelize_level_four,
    level_four_hankel_matrix_shape,
)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1, 1), (1, 1)),
        ((2, 4, 6, 8), (120, 24)),
        ((3, 5, 7, 9), (120, 120)),
        ((1, 3, 4, 5), (18, 12)),
    ],
)
def test_matrix_shape_matches_four_axis_embedding(
    shape: tuple[int, int, int, int], expected: tuple[int, int]
) -> None:
    assert level_four_hankel_matrix_shape(shape) == expected


def test_hankel_entries_map_every_offset_and_start_to_the_grid() -> None:
    shape = (3, 4, 2, 5)
    values = np.arange(prod(shape)).reshape(shape) * (2.0 - 3.0j) + 1.0j
    original = values.copy()

    matrix = hankelize_level_four(values)

    offsets = list(product(range(2), range(3), range(2), range(3)))
    starts = list(product(range(2), range(2), range(1), range(3)))
    for row, offset in enumerate(offsets):
        for column, start in enumerate(starts):
            index = tuple(first + delta for first, delta in zip(start, offset, strict=True))
            assert matrix[row, column] == values[index]
    assert matrix.dtype == np.complex128
    assert matrix.flags.c_contiguous
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 4, 5), (3, 1, 5, 2)])
def test_complex_grid_round_trip_preserves_values_and_inputs(
    shape: tuple[int, int, int, int], dtype: type[np.complexfloating]
) -> None:
    rng = np.random.default_rng(21)
    values = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    original = values.copy()
    matrix = hankelize_level_four(values)
    original_matrix = matrix.copy()

    restored = average_level_four_hankel(matrix, shape)

    np.testing.assert_allclose(restored, values, rtol=1.0e-14, atol=1.0e-14)
    assert restored.dtype == np.complex128
    assert np.any(restored.imag != 0.0)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(matrix, original_matrix)


def test_non_hankel_matrix_averaging_matches_explicit_entry_reference() -> None:
    shape = (3, 4, 3, 2)
    embedding_shape = (2, 3, 2, 2)
    start_shape = (2, 2, 2, 1)
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(24, 8)) + 1j * rng.normal(size=(24, 8))
    original = matrix.copy()
    expected = np.zeros(shape, dtype=np.complex128)
    counts = np.zeros(shape, dtype=np.int64)
    for row, offset in enumerate(product(*(range(length) for length in embedding_shape))):
        for column, start in enumerate(product(*(range(length) for length in start_shape))):
            index = tuple(first + delta for first, delta in zip(start, offset, strict=True))
            expected[index] += matrix[row, column]
            counts[index] += 1
    expected /= counts

    result = average_level_four_hankel(matrix, shape)

    np.testing.assert_allclose(result, expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_array_equal(matrix, original)


def test_noncontiguous_inputs_preserve_the_same_matrix_layout() -> None:
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5).astype(np.complex128)
    values = (values + 1.0j)[..., ::-1]

    matrix = hankelize_level_four(values)
    restored = average_level_four_hankel(np.asfortranarray(matrix), values.shape)

    assert matrix.flags.c_contiguous
    np.testing.assert_array_equal(restored, values)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 0, 4), (1, 2, -1, 4), (True, 2, 3, 4), (1.0, 2, 3, 4)],
)
def test_invalid_spatial_shape_is_rejected(shape: object) -> None:
    with pytest.raises(ValueError, match="four positive integers"):
        level_four_hankel_matrix_shape(shape)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="four positive integers"):
        average_level_four_hankel(np.ones((2, 2), dtype=np.complex128), shape)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((2, 2, 2), dtype=np.complex128), "four-dimensional"),
        (np.ones((2, 2, 2, 2), dtype=np.float64), "complex64 or complex128"),
        (np.ones((0, 2, 2, 2), dtype=np.complex128), "positive integers"),
        ([[1.0j]], "four-dimensional"),
    ],
)
def test_hankelize_rejects_invalid_array_contracts(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hankelize_level_four(values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 2, 2), dtype=np.complex128), "two-dimensional"),
        (np.ones((16, 1), dtype=np.float64), "complex dtype"),
        (np.ones((2, 2), dtype=np.complex128), "matrix shape"),
        ([[1.0j]], "two-dimensional"),
    ],
)
def test_averaging_rejects_invalid_array_contracts(matrix: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        average_level_four_hankel(matrix, (2, 2, 2, 2))  # type: ignore[arg-type]
