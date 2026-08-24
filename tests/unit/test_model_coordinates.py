from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_schema import MODEL_SPATIAL_FEATURE_ORDER
from seis_interp.processing.model_coordinates import build_spatial_model_coordinates


def make_trace_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cmp_x_m": [10.0, 20.0, 30.0, 40.0],
            "cmp_y_m": [100.0, 200.0, 300.0, 400.0],
            "offset_m": [500.0, 600.0, 700.0, 800.0],
            "azimuth_deg": [0.0, 90.0, 180.0, 270.0],
            "unrelated": [1, 2, 3, 4],
        },
        index=[40, 10, 70, 20],
    )


def test_builds_model_features_in_the_authoritative_order() -> None:
    coordinates = build_spatial_model_coordinates(make_trace_table())

    assert MODEL_SPATIAL_FEATURE_ORDER == (
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_sin",
        "azimuth_cos",
    )
    np.testing.assert_allclose(
        coordinates,
        [
            [10.0, 100.0, 500.0, 0.0, 1.0],
            [20.0, 200.0, 600.0, 1.0, 0.0],
            [30.0, 300.0, 700.0, 0.0, -1.0],
            [40.0, 400.0, 800.0, -1.0, 0.0],
        ],
        atol=1.0e-15,
    )
    assert coordinates.dtype == np.float64
    assert coordinates.shape == (4, 5)


def test_preserves_dataframe_row_order_and_does_not_modify_input() -> None:
    trace_table = make_trace_table().iloc[[2, 0, 3, 1]]
    original = trace_table.copy(deep=True)

    coordinates = build_spatial_model_coordinates(trace_table)

    np.testing.assert_array_equal(coordinates[:, 0], [30.0, 10.0, 40.0, 20.0])
    pd.testing.assert_frame_equal(trace_table, original)


def test_unit_circle_encoding_is_continuous_across_azimuth_wrap() -> None:
    trace_table = pd.DataFrame(
        {
            "cmp_x_m": [0.0, 0.0],
            "cmp_y_m": [0.0, 0.0],
            "offset_m": [1.0, 1.0],
            "azimuth_deg": [359.0, 1.0],
        }
    )

    coordinates = build_spatial_model_coordinates(trace_table)
    angular_distance = np.linalg.norm(coordinates[0, -2:] - coordinates[1, -2:])

    assert angular_distance == pytest.approx(2.0 * np.sin(np.deg2rad(1.0)))
    np.testing.assert_allclose(np.linalg.norm(coordinates[:, -2:], axis=1), 1.0)


@pytest.mark.parametrize("missing", ["cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"])
def test_rejects_missing_physical_coordinate_columns(missing: str) -> None:
    with pytest.raises(ValueError, match="missing"):
        build_spatial_model_coordinates(make_trace_table().drop(columns=missing))


def test_rejects_a_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        build_spatial_model_coordinates([])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_physical_coordinates(bad_value: float) -> None:
    trace_table = make_trace_table()
    trace_table.loc[trace_table.index[0], "azimuth_deg"] = bad_value

    with pytest.raises(ValueError, match="non-finite"):
        build_spatial_model_coordinates(trace_table)


def test_rejects_non_numeric_physical_coordinates() -> None:
    trace_table = make_trace_table()
    trace_table["cmp_x_m"] = trace_table["cmp_x_m"].astype(object)
    trace_table.loc[trace_table.index[0], "cmp_x_m"] = "not-a-number"

    with pytest.raises(ValueError, match="numeric"):
        build_spatial_model_coordinates(trace_table)
