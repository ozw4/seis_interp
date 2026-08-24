from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_interp.training.model_inputs import to_model_tensors


def make_coordinates(point_count: int = 3) -> np.ndarray:
    """Return float64 coordinates shaped like the interim trace table."""
    return np.arange(point_count * 2, dtype=np.float64).reshape(point_count, 2)


def test_float64_coordinates_become_float32() -> None:
    coordinates, _ = to_model_tensors(make_coordinates(), np.zeros(3, dtype=np.float32))

    assert coordinates.dtype == torch.float32
    assert coordinates.shape == (3, 2)
    torch.testing.assert_close(coordinates, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))


def test_float32_targets_stay_float32() -> None:
    targets_array = np.array([0.5, -0.25, 1.0], dtype=np.float32)

    _, targets = to_model_tensors(make_coordinates(), targets_array)

    assert targets.dtype == torch.float32
    torch.testing.assert_close(targets, torch.tensor([[0.5], [-0.25], [1.0]]))


def test_one_dimensional_targets_become_a_column() -> None:
    _, targets = to_model_tensors(make_coordinates(), np.zeros(3, dtype=np.float32))

    assert targets.shape == (3, 1)


def test_column_targets_keep_their_shape() -> None:
    _, targets = to_model_tensors(make_coordinates(), np.zeros((3, 1), dtype=np.float32))

    assert targets.shape == (3, 1)


def test_returns_contiguous_cpu_tensors() -> None:
    coordinates, targets = to_model_tensors(
        np.asfortranarray(make_coordinates()), np.zeros(3, dtype=np.float64)
    )

    assert coordinates.is_contiguous()
    assert targets.is_contiguous()
    assert coordinates.device == torch.device("cpu")
    assert targets.device == torch.device("cpu")


def test_rejects_coordinates_that_are_not_two_dimensional() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        to_model_tensors(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("shape", [(3, 2), (3, 1, 1)])
def test_rejects_targets_with_an_unusable_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"\(n_points, 1\)"):
        to_model_tensors(make_coordinates(), np.zeros(shape, dtype=np.float32))


def test_rejects_mismatched_point_counts() -> None:
    with pytest.raises(ValueError, match="same number of points"):
        to_model_tensors(make_coordinates(3), np.zeros(4, dtype=np.float32))


def test_does_not_modify_the_input_arrays() -> None:
    coordinate_array = make_coordinates()
    target_array = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    coordinate_copy = coordinate_array.copy()
    target_copy = target_array.copy()

    to_model_tensors(coordinate_array, target_array)

    assert np.array_equal(coordinate_array, coordinate_copy)
    assert np.array_equal(target_array, target_copy)
