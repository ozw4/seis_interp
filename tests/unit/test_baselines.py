from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from seis_interp.evaluation.baselines import (
    inverse_distance_weighted_predict,
    nearest_neighbor_predict,
)

Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def test_nearest_neighbor_returns_known_nearest_traces() -> None:
    train_coordinates = np.array([[0.0], [2.0], [5.0]])
    train_amplitudes = np.array([[0.0, 1.0], [2.0, 3.0], [5.0, 6.0]])
    query_coordinates = np.array([[0.25], [3.0], [4.75]])

    prediction = nearest_neighbor_predict(train_coordinates, train_amplitudes, query_coordinates)

    np.testing.assert_array_equal(prediction, [[0.0, 1.0], [2.0, 3.0], [5.0, 6.0]])


def test_nearest_neighbor_tie_uses_first_training_trace() -> None:
    prediction = nearest_neighbor_predict(
        np.array([[0.0], [2.0]]),
        np.array([[10.0], [20.0]]),
        np.array([[1.0]]),
    )

    np.testing.assert_array_equal(prediction, [[10.0]])


def test_idw_midpoint_between_two_points_returns_their_mean() -> None:
    prediction = inverse_distance_weighted_predict(
        np.array([[0.0], [2.0]]),
        np.array([[2.0, 4.0], [6.0, 8.0]]),
        np.array([[1.0]]),
        neighbors=2,
    )

    np.testing.assert_allclose(prediction, [[4.0, 6.0]])


def test_idw_exact_match_returns_matching_trace() -> None:
    prediction = inverse_distance_weighted_predict(
        np.array([[0.0], [2.0]]),
        np.array([[2.0, 4.0], [6.0, 8.0]]),
        np.array([[2.0]]),
        neighbors=1,
    )

    np.testing.assert_array_equal(prediction, [[6.0, 8.0]])


def test_idw_duplicate_exact_coordinates_return_amplitude_mean() -> None:
    prediction = inverse_distance_weighted_predict(
        np.array([[0.0], [0.0], [2.0]]),
        np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 12.0]]),
        np.array([[0.0]]),
        neighbors=1,
    )

    np.testing.assert_array_equal(prediction, [[1.0, 3.0]])


def test_idw_neighbors_above_training_count_uses_all_traces() -> None:
    prediction = inverse_distance_weighted_predict(
        np.array([[0.0], [2.0]]),
        np.array([[2.0], [6.0]]),
        np.array([[1.0]]),
        neighbors=20,
    )

    np.testing.assert_allclose(prediction, [[4.0]])


@pytest.mark.parametrize(
    "predictor",
    [nearest_neighbor_predict, inverse_distance_weighted_predict],
)
def test_float32_amplitudes_produce_float32_output(predictor: Predictor) -> None:
    prediction = predictor(
        np.array([[0.0], [2.0]]),
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
        np.array([[1.0]]),
    )

    assert prediction.dtype == np.float32


@pytest.mark.parametrize(
    "predictor",
    [nearest_neighbor_predict, inverse_distance_weighted_predict],
)
def test_output_has_expected_shape_and_is_c_contiguous(predictor: Predictor) -> None:
    amplitudes = np.arange(24.0).reshape(3, 8)[:, ::2]

    prediction = predictor(
        np.array([[0.0], [1.0], [2.0]]),
        amplitudes,
        np.array([[0.25], [1.75]]),
    )

    assert prediction.shape == (2, 4)
    assert prediction.flags.c_contiguous


@pytest.mark.parametrize(
    ("train_coordinates", "train_amplitudes", "query_coordinates"),
    [
        (np.zeros(2), np.zeros((2, 1)), np.zeros((1, 1))),
        (np.zeros((2, 1)), np.zeros(2), np.zeros((1, 1))),
        (np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(1)),
        (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((1, 1))),
        (np.zeros((1, 0)), np.zeros((1, 1)), np.zeros((1, 0))),
        (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((0, 1))),
        (np.zeros((1, 1)), np.zeros((1, 0)), np.zeros((1, 1))),
        (np.zeros((2, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
        (np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 1))),
    ],
)
@pytest.mark.parametrize(
    "predictor",
    [nearest_neighbor_predict, inverse_distance_weighted_predict],
)
def test_rejects_invalid_shapes_and_empty_arrays(
    predictor: Predictor,
    train_coordinates: np.ndarray,
    train_amplitudes: np.ndarray,
    query_coordinates: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        predictor(train_coordinates, train_amplitudes, query_coordinates)


@pytest.mark.parametrize(
    "input_name", ["train_coordinates", "train_amplitudes", "query_coordinates"]
)
@pytest.mark.parametrize(
    "predictor",
    [nearest_neighbor_predict, inverse_distance_weighted_predict],
)
def test_rejects_non_finite_inputs(predictor: Predictor, input_name: str) -> None:
    arrays = {
        "train_coordinates": np.array([[0.0], [1.0]]),
        "train_amplitudes": np.array([[0.0], [1.0]]),
        "query_coordinates": np.array([[0.5]]),
    }
    arrays[input_name].flat[0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        predictor(
            arrays["train_coordinates"],
            arrays["train_amplitudes"],
            arrays["query_coordinates"],
        )


@pytest.mark.parametrize("neighbors", [0, -1, 1.5, True])
def test_idw_rejects_invalid_neighbors(neighbors: object) -> None:
    with pytest.raises(ValueError, match="neighbors"):
        inverse_distance_weighted_predict(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            neighbors=neighbors,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("parameter", [0.0, -1.0, np.inf, np.nan, True])
@pytest.mark.parametrize("parameter_name", ["power", "epsilon"])
def test_idw_rejects_invalid_positive_finite_parameters(
    parameter_name: str,
    parameter: object,
) -> None:
    keyword_arguments = {parameter_name: parameter}

    with pytest.raises(ValueError, match=parameter_name):
        inverse_distance_weighted_predict(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            **keyword_arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "predictor",
    [nearest_neighbor_predict, inverse_distance_weighted_predict],
)
def test_does_not_modify_inputs(predictor: Predictor) -> None:
    train_coordinates = np.array([[0.0, 1.0], [2.0, 3.0]])
    train_amplitudes = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    query_coordinates = np.array([[1.0, 2.0]])
    originals = tuple(
        array.copy() for array in (train_coordinates, train_amplitudes, query_coordinates)
    )

    predictor(train_coordinates, train_amplitudes, query_coordinates)

    for array, original in zip(
        (train_coordinates, train_amplitudes, query_coordinates), originals, strict=True
    ):
        np.testing.assert_array_equal(array, original)
