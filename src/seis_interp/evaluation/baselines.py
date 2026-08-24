"""Simple spatial trace interpolation baselines implemented with NumPy."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def nearest_neighbor_predict(
    train_coordinates: np.ndarray,
    train_amplitudes: np.ndarray,
    query_coordinates: np.ndarray,
) -> np.ndarray:
    """Predict each query trace from its nearest training coordinate.

    Distances are squared Euclidean distances over all coordinate features.
    NumPy's ``argmin`` therefore resolves equal-distance ties in training input
    order. The returned array is C-contiguous and the inputs are not modified.
    """
    train_coordinate_array, amplitude_array, query_coordinate_array = _validate_inputs(
        train_coordinates,
        train_amplitudes,
        query_coordinates,
    )
    squared_distances = _pairwise_squared_distances(
        train_coordinate_array,
        query_coordinate_array,
    )
    nearest_indices = np.argmin(squared_distances, axis=1)
    return np.ascontiguousarray(amplitude_array[nearest_indices])


def inverse_distance_weighted_predict(
    train_coordinates: np.ndarray,
    train_amplitudes: np.ndarray,
    query_coordinates: np.ndarray,
    *,
    neighbors: int = 8,
    power: float = 2.0,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Predict traces with inverse-distance weighting over nearby coordinates.

    Coordinate matches within ``epsilon`` bypass weighting. If multiple
    training traces match a query coordinate, their amplitudes are averaged.
    Otherwise, the closest ``min(neighbors, n_train)`` traces are combined
    using one normalized weight per complete trace.
    """
    neighbor_count, distance_power, match_epsilon = _validate_idw_parameters(
        neighbors,
        power,
        epsilon,
    )
    train_coordinate_array, amplitude_array, query_coordinate_array = _validate_inputs(
        train_coordinates,
        train_amplitudes,
        query_coordinates,
    )

    distances = np.sqrt(_pairwise_squared_distances(train_coordinate_array, query_coordinate_array))
    neighbor_count = min(neighbor_count, train_coordinate_array.shape[0])
    predictions = np.empty(
        (query_coordinate_array.shape[0], amplitude_array.shape[1]),
        dtype=_prediction_dtype(amplitude_array),
        order="C",
    )

    for query_index, query_distances in enumerate(distances):
        matching_rows = query_distances <= match_epsilon
        if np.any(matching_rows):
            predictions[query_index] = np.mean(amplitude_array[matching_rows], axis=0)
            continue

        nearest_indices = np.argsort(query_distances, kind="stable")[:neighbor_count]
        nearest_distances = query_distances[nearest_indices]
        weights = 1.0 / nearest_distances**distance_power
        normalized_weights = weights / np.sum(weights)
        predictions[query_index] = normalized_weights @ amplitude_array[nearest_indices]

    return predictions


def _validate_inputs(
    train_coordinates: np.ndarray,
    train_amplitudes: np.ndarray,
    query_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_coordinate_array = np.asarray(train_coordinates)
    amplitude_array = np.asarray(train_amplitudes)
    query_coordinate_array = np.asarray(query_coordinates)

    arrays = (
        ("train_coordinates", train_coordinate_array),
        ("train_amplitudes", amplitude_array),
        ("query_coordinates", query_coordinate_array),
    )
    for name, array in arrays:
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")

    if train_coordinate_array.shape[0] < 1:
        raise ValueError("train_coordinates must contain at least one row")
    if train_coordinate_array.shape[1] < 1:
        raise ValueError("train_coordinates must contain at least one feature")
    if query_coordinate_array.shape[0] < 1:
        raise ValueError("query_coordinates must contain at least one row")
    if amplitude_array.shape[1] < 1:
        raise ValueError("train_amplitudes must contain at least one sample")
    if train_coordinate_array.shape[0] != amplitude_array.shape[0]:
        raise ValueError(
            "train_coordinates and train_amplitudes must contain the same number of rows, "
            f"got {train_coordinate_array.shape[0]} and {amplitude_array.shape[0]}"
        )
    if train_coordinate_array.shape[1] != query_coordinate_array.shape[1]:
        raise ValueError(
            "train_coordinates and query_coordinates must contain the same number of features, "
            f"got {train_coordinate_array.shape[1]} and {query_coordinate_array.shape[1]}"
        )

    for name, array in arrays:
        try:
            finite = np.isfinite(array)
        except TypeError as error:
            raise ValueError(f"{name} must contain numeric values") from error
        if not np.all(finite):
            raise ValueError(f"{name} contains non-finite values")

    if not _is_real_numeric(train_coordinate_array) or not _is_real_numeric(query_coordinate_array):
        raise ValueError("coordinates must contain real numeric values")

    return train_coordinate_array, amplitude_array, query_coordinate_array


def _validate_idw_parameters(
    neighbors: int,
    power: float,
    epsilon: float,
) -> tuple[int, float, float]:
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors < 1:
        raise ValueError("neighbors must be an integer greater than or equal to 1")
    if (
        isinstance(power, bool)
        or not isinstance(power, Real)
        or not np.isfinite(power)
        or power <= 0.0
    ):
        raise ValueError("power must be finite and greater than 0")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError("epsilon must be finite and greater than 0")
    return int(neighbors), float(power), float(epsilon)


def _pairwise_squared_distances(
    train_coordinates: np.ndarray,
    query_coordinates: np.ndarray,
) -> np.ndarray:
    train_for_distance = train_coordinates.astype(np.float64, copy=False)
    query_for_distance = query_coordinates.astype(np.float64, copy=False)
    differences = query_for_distance[:, np.newaxis, :] - train_for_distance[np.newaxis, :, :]
    return np.sum(differences * differences, axis=2)


def _prediction_dtype(train_amplitudes: np.ndarray) -> np.dtype:
    if np.issubdtype(train_amplitudes.dtype, np.inexact):
        return train_amplitudes.dtype
    return np.dtype(np.float64)


def _is_real_numeric(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.complexfloating
    )
