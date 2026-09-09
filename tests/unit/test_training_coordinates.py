from __future__ import annotations

from dataclasses import replace

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
    normalize_training_trace_time_offsets,
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


@pytest.mark.parametrize("shear", [0.0006, -0.0006])
@pytest.mark.parametrize("time_scale", [1.0, 4.0])
def test_time_shear_matches_physical_receiver_y_and_keeps_coordinates_unchanged(
    shear: float, time_scale: float
) -> None:
    normalization = _normalization()
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        normalization,
        time_coordinate_scale=time_scale,
        relative_receiver_y_time_shear_s_per_m=shear,
    )
    table = _trace_table()
    spatial = normalize_training_spatial_coordinates(table, normalization, parameters)
    before = spatial.copy()
    time = np.asarray([0.25, 1.0, 1.5])
    base_time = normalize_training_time_coordinate(time, normalization, parameters)

    offsets = normalize_training_trace_time_offsets(spatial, parameters)

    assert offsets is not None
    assert offsets.shape == (3,)
    assert offsets.dtype == np.float64
    relative_y = (table["receiver_y_m"] - table["source_y_m"]).to_numpy()
    physical_sheared_time = time[None, :] + shear * relative_y[:, None]
    expected = (physical_sheared_time - 1.0) * time_scale
    np.testing.assert_allclose(base_time[None, :] + offsets[:, None], expected, atol=1e-14)
    np.testing.assert_array_equal(spatial, before)
    np.testing.assert_array_equal(base_time, (time - 1.0) * time_scale)
    if shear > 0:
        assert offsets[1] > 0.0  # receiver_y - source_y is +550 m.
        assert offsets[2] < 0.0


def test_time_shear_offsets_are_float64_even_for_float32_inputs() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
        time_coordinate_scale=4.0,
        relative_receiver_y_time_shear_s_per_m=0.0006,
    )
    offsets = normalize_training_trace_time_offsets(
        np.asarray([[0.0, 0.0, 0.0, -0.5]], dtype=np.float32), parameters
    )

    assert offsets is not None
    assert offsets.dtype == np.float64
    np.testing.assert_allclose(offsets, [1.2])


@pytest.mark.parametrize(
    "mode",
    [
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    ],
)
def test_zero_shear_preserves_legacy_payload_and_bypasses_offset_work(
    mode: str,
) -> None:
    legacy = model_coordinate_parameters(mode, _normalization())
    explicit_zero = model_coordinate_parameters(
        mode, _normalization(), relative_receiver_y_time_shear_s_per_m=0.0
    )

    assert explicit_zero == legacy
    assert explicit_zero.relative_receiver_y_time_shear_s_per_m == 0.0
    assert "relative_receiver_y_time_shear_s_per_m" not in legacy.to_dict()
    assert explicit_zero.to_dict() == legacy.to_dict()
    assert ModelCoordinateParameters.from_dict(legacy.to_dict()) == legacy
    assert normalize_training_trace_time_offsets(np.asarray([np.nan]), explicit_zero) is None


@pytest.mark.parametrize("shear", [0.0006, -0.0006])
def test_nonzero_shear_serialization_round_trips(shear: float) -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
        time_coordinate_scale=4.0,
        relative_receiver_y_time_shear_s_per_m=shear,
    )

    payload = parameters.to_dict()

    assert payload["relative_receiver_y_time_shear_s_per_m"] == shear
    assert ModelCoordinateParameters.from_dict(payload) == parameters
    payload["unexpected_shear_key"] = 0.0
    with pytest.raises(ValueError, match="unexpected_shear_key"):
        ModelCoordinateParameters.from_dict(payload)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.0006", None, 1j, np.nan, np.inf, -np.inf, 10**1000],
)
def test_time_shear_rejects_nonfinite_or_nonreal_values_in_factory_and_payload(
    invalid,
) -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES, _normalization()
    )
    with pytest.raises(ValueError, match="relative_receiver_y_time_shear_s_per_m"):
        model_coordinate_parameters(
            CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
            _normalization(),
            relative_receiver_y_time_shear_s_per_m=invalid,
        )
    payload = parameters.to_dict()
    payload["relative_receiver_y_time_shear_s_per_m"] = invalid
    with pytest.raises(ValueError, match="relative_receiver_y_time_shear_s_per_m"):
        ModelCoordinateParameters.from_dict(payload)


@pytest.mark.parametrize(
    "mode",
    [
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    ],
)
def test_nonzero_shear_requires_five_dimensional_cartesian_coordinates(
    mode: str,
) -> None:
    with pytest.raises(ValueError, match="requires cmp_cartesian_half_offset"):
        model_coordinate_parameters(
            mode, _normalization(), relative_receiver_y_time_shear_s_per_m=0.0006
        )


def test_nonzero_shear_requires_positive_finite_time_span_but_zero_does_not() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES, _normalization()
    )
    for minimum, maximum in ((0.0, 0.0), (-1e308, 1e308)):
        zero = replace(
            parameters,
            coordinate_scale_min=(minimum, *parameters.coordinate_scale_min[1:]),
            coordinate_scale_max=(maximum, *parameters.coordinate_scale_max[1:]),
        )
        assert normalize_training_trace_time_offsets(np.empty((0, 4)), zero) is None
        with pytest.raises(ValueError, match="positive finite time span"):
            replace(zero, relative_receiver_y_time_shear_s_per_m=0.0006)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(4),
        np.zeros((2, 3)),
        np.zeros((2, 5)),
        np.zeros((2, 4), dtype=bool),
        np.zeros((2, 4), dtype=complex),
        np.full((2, 4), "0"),
        np.asarray([[np.nan, 0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0, np.inf]]),
    ],
)
def test_nonzero_shear_rejects_invalid_normalized_spatial_coordinates(
    invalid: np.ndarray,
) -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
        relative_receiver_y_time_shear_s_per_m=0.0006,
    )
    with pytest.raises(ValueError, match="normalized_spatial"):
        normalize_training_trace_time_offsets(invalid, parameters)


def test_nonzero_shear_empty_rows_and_nonfinite_result() -> None:
    parameters = model_coordinate_parameters(
        CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
        _normalization(),
        relative_receiver_y_time_shear_s_per_m=0.0006,
    )
    empty = normalize_training_trace_time_offsets(np.empty((0, 4)), parameters)
    assert empty is not None
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    with pytest.raises(ValueError, match="trace time offsets must be finite"):
        normalize_training_trace_time_offsets(
            np.ones((1, 4)),
            replace(parameters, relative_receiver_y_time_shear_s_per_m=1e308),
        )


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
