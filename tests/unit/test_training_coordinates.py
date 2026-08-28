from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.processing.normalization import (
    NormalizationParameters,
    normalize_spatial_coordinates,
    normalize_time,
)
from seis_interp.processing.training_coordinates import (
    CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
    CMP_CARTESIAN_HALF_OFFSET_COORDINATE_ORDER,
    CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_ORDER,
    CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
    ModelCoordinateParameters,
    model_coordinate_parameters,
    normalize_training_spatial_coordinates,
    normalize_training_time_coordinate,
)


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(0.0, 100.0, 200.0, 100.0, -1.0, -1.0),
        coordinate_max=(2.0, 300.0, 600.0, 1000.0, 1.0, 1.0),
        amplitude_rms=3.0,
    )


def _trace_table() -> pd.DataFrame:
    cmp_x = np.asarray([100.0, 200.0, 300.0])
    cmp_y = np.asarray([200.0, 400.0, 600.0])
    half_offset_x = np.asarray([50.0, 0.0, 300.0])
    half_offset_y = np.asarray([0.0, -275.0, 400.0])
    return pd.DataFrame(
        {
            "cmp_x_m": cmp_x,
            "cmp_y_m": cmp_y,
            "source_x_m": cmp_x + half_offset_x,
            "source_y_m": cmp_y + half_offset_y,
            "receiver_x_m": cmp_x - half_offset_x,
            "receiver_y_m": cmp_y - half_offset_y,
            "offset_m": [100.0, 550.0, 1000.0],
            "azimuth_deg": [90.0, 180.0, np.degrees(np.arctan2(600.0, 800.0))],
        }
    )


def test_cartesian_half_offset_uses_prepared_cmp_bounds_and_one_symmetric_scale() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
    )

    coordinates = normalize_training_spatial_coordinates(
        _trace_table(),
        _normalization(),
        parameters,
    )

    assert parameters.coordinate_order == CMP_CARTESIAN_HALF_OFFSET_COORDINATE_ORDER
    assert parameters.input_features == 5
    assert parameters.half_offset_scale_m == 500.0
    assert parameters.coordinate_scale_min == (0.0, 100.0, 200.0, -500.0, -500.0)
    assert parameters.coordinate_scale_max == (2.0, 300.0, 600.0, 500.0, 500.0)
    np.testing.assert_allclose(
        coordinates,
        [
            [-1.0, -1.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, -0.55],
            [1.0, 1.0, 0.6, 0.8],
        ],
    )


def test_cartesian_radius_appends_the_exact_legacy_normalized_offset() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
        _normalization(),
    )

    coordinates = normalize_training_spatial_coordinates(
        _trace_table(),
        _normalization(),
        parameters,
    )
    legacy_offset = normalize_spatial_coordinates(_trace_table(), _normalization())[:, 2]

    assert parameters.coordinate_order == CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_ORDER
    assert parameters.input_features == 6
    assert parameters.half_offset_scale_m == 500.0
    assert parameters.coordinate_scale_min == (
        0.0,
        100.0,
        200.0,
        -500.0,
        -500.0,
        100.0,
    )
    assert parameters.coordinate_scale_max == (
        2.0,
        300.0,
        600.0,
        500.0,
        500.0,
        1000.0,
    )
    np.testing.assert_array_equal(coordinates[:, -1], legacy_offset)
    np.testing.assert_allclose(
        coordinates,
        [
            [-1.0, -1.0, 0.1, 0.0, -1.0],
            [0.0, 0.0, 0.0, -0.55, 0.0],
            [1.0, 1.0, 0.6, 0.8, 1.0],
        ],
    )


def test_default_coordinate_mode_preserves_existing_normalized_features() -> None:
    parameters = model_coordinate_parameters(
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        _normalization(),
    )

    actual = normalize_training_spatial_coordinates(
        _trace_table(),
        _normalization(),
        parameters,
    )

    np.testing.assert_array_equal(
        actual,
        normalize_spatial_coordinates(_trace_table(), _normalization()),
    )
    assert parameters.coordinate_order == MODEL_COORDINATE_ORDER
    assert parameters.input_features == 6
    assert parameters.half_offset_scale_m is None
    assert parameters.time_coordinate_scale == 1.0
    assert "time_coordinate_scale" not in parameters.to_dict()


def test_time_coordinate_scale_is_applied_after_existing_minmax_normalization() -> None:
    time_s = np.asarray([0.0, 1.0, 2.0])
    parameters = model_coordinate_parameters(
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        _normalization(),
        time_coordinate_scale=4.0,
    )

    scaled = normalize_training_time_coordinate(
        time_s,
        _normalization(),
        parameters,
    )

    np.testing.assert_array_equal(scaled, [-4.0, 0.0, 4.0])
    assert parameters.to_dict()["time_coordinate_scale"] == 4.0
    assert ModelCoordinateParameters.from_dict(parameters.to_dict()) == parameters


def test_default_time_coordinate_scale_preserves_the_existing_array_exactly() -> None:
    time_s = np.asarray([0.0, 0.25, 1.5, 2.0])
    parameters = model_coordinate_parameters(
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        _normalization(),
    )

    actual = normalize_training_time_coordinate(
        time_s,
        _normalization(),
        parameters,
    )

    np.testing.assert_array_equal(actual, normalize_time(time_s, _normalization()))


@pytest.mark.parametrize(
    "coordinate_features",
    [
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    ],
)
def test_coordinate_parameter_payload_round_trips(coordinate_features: str) -> None:
    expected = model_coordinate_parameters(
        coordinate_features,
        _normalization(),
    )

    assert ModelCoordinateParameters.from_dict(expected.to_dict()) == expected


def test_cartesian_half_offset_requires_source_and_receiver_coordinates() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
    )

    with pytest.raises(ValueError, match="source_x_m"):
        normalize_training_spatial_coordinates(
            _trace_table().drop(columns="source_x_m"),
            _normalization(),
            parameters,
        )


def test_cartesian_radius_requires_the_stored_offset() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
        _normalization(),
    )

    with pytest.raises(ValueError, match="offset_m"):
        normalize_training_spatial_coordinates(
            _trace_table().drop(columns="offset_m"),
            _normalization(),
            parameters,
        )
