from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.processing.normalization import (
    NormalizationParameters,
    denormalize_amplitudes,
    fit_normalization_parameters,
    normalize_amplitudes,
    normalize_spatial_coordinates,
    normalize_time,
    read_normalization_parameters,
    write_normalization_parameters,
)


def make_trace_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": [2, 0, 3, 1],
            "split": ["train", "train", "validation", "test"],
            "cmp_x_m": [20.0, 10.0, 1000.0, -1000.0],
            "cmp_y_m": [5.0, 5.0, 2000.0, -2000.0],
            "offset_m": [200.0, 100.0, 3000.0, -3000.0],
            "azimuth_deg": [20.0, 10.0, 300.0, 1.0],
        },
        index=[40, 10, 70, 20],
    )


def make_amplitudes() -> np.ndarray:
    # array rows 0 and 2 are training rows; held-out magnitudes are extreme.
    return np.asarray(
        [
            [3.0, 4.0, 0.0],
            [1.0e6, 1.0e6, 1.0e6],
            [0.0, 0.0, 12.0],
            [-1.0e6, -1.0e6, -1.0e6],
        ],
        dtype=np.float32,
    )


def make_time_axis() -> np.ndarray:
    return np.asarray([0.0, 1.0, 2.0], dtype=np.float64)


def fit_default() -> NormalizationParameters:
    return fit_normalization_parameters(make_trace_table(), make_amplitudes(), make_time_axis())


def test_fit_uses_only_training_rows_for_spatial_ranges() -> None:
    parameters = fit_default()

    assert parameters.coordinate_order == MODEL_COORDINATE_ORDER
    assert parameters.coordinate_order == (
        "time_s",
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_sin",
        "azimuth_cos",
    )
    assert parameters.coordinate_min == (0.0, 10.0, 5.0, 100.0, -1.0, -1.0)
    assert parameters.coordinate_max == (2.0, 20.0, 5.0, 200.0, 1.0, 1.0)


def test_fit_uses_only_training_amplitudes_for_global_rms() -> None:
    parameters = fit_default()

    expected = np.sqrt((3.0**2 + 4.0**2 + 12.0**2) / 6.0)
    assert parameters.amplitude_rms == pytest.approx(expected)


def test_array_row_not_dataframe_position_aligns_amplitudes() -> None:
    trace_table = make_trace_table().iloc[[1, 0, 2, 3]]
    amplitudes = make_amplitudes()

    parameters = fit_normalization_parameters(trace_table, amplitudes, make_time_axis())

    expected = np.sqrt((3.0**2 + 4.0**2 + 12.0**2) / 6.0)
    assert parameters.amplitude_rms == pytest.approx(expected)


def test_fit_allows_trace_table_to_reference_subset_of_amplitude_rows() -> None:
    trace_table = make_trace_table().iloc[:2].copy()
    amplitudes = np.vstack([make_amplitudes(), np.full((2, 3), 7.0, dtype=np.float32)])

    parameters = fit_normalization_parameters(trace_table, amplitudes, make_time_axis())

    assert parameters.coordinate_min[1:4] == (10.0, 5.0, 100.0)
    assert parameters.coordinate_min[4:] == (-1.0, -1.0)
    assert parameters.coordinate_max[4:] == (1.0, 1.0)


def test_training_extrema_map_to_minus_and_plus_one() -> None:
    trace_table = make_trace_table()
    normalized = normalize_spatial_coordinates(trace_table, fit_default())

    np.testing.assert_array_equal(normalized[0, [0, 2]], [1.0, 1.0])
    np.testing.assert_array_equal(normalized[1, [0, 2]], [-1.0, -1.0])


def test_constant_axis_maps_every_query_to_zero() -> None:
    normalized = normalize_spatial_coordinates(make_trace_table(), fit_default())

    np.testing.assert_array_equal(normalized[:, 1], np.zeros(4))


def test_queries_outside_training_range_are_not_clipped() -> None:
    normalized = normalize_spatial_coordinates(make_trace_table(), fit_default())

    assert normalized[2, 0] > 1.0
    assert normalized[3, 0] < -1.0


def test_azimuth_is_encoded_as_sine_and_cosine_in_fixed_output_columns() -> None:
    trace_table = make_trace_table()

    normalized = normalize_spatial_coordinates(trace_table, fit_default())

    radians = np.deg2rad(trace_table["azimuth_deg"].to_numpy(dtype=np.float64))
    assert normalized.shape == (len(trace_table), 5)
    np.testing.assert_allclose(normalized[:, 3], np.sin(radians), atol=1.0e-15)
    np.testing.assert_allclose(normalized[:, 4], np.cos(radians), atol=1.0e-15)
    assert np.all(normalized[:, 3:] >= -1.0)
    assert np.all(normalized[:, 3:] <= 1.0)
    np.testing.assert_allclose(np.sum(normalized[:, 3:] ** 2, axis=1), 1.0)


def test_azimuth_encoding_does_not_depend_on_training_azimuth_range() -> None:
    first_table = make_trace_table()
    second_table = make_trace_table()
    second_table.loc[second_table["split"] == "train", "azimuth_deg"] = [120.0, 240.0]
    query = pd.DataFrame(
        {
            "cmp_x_m": [15.0],
            "cmp_y_m": [5.0],
            "offset_m": [150.0],
            "azimuth_deg": [350.0],
        }
    )

    first_parameters = fit_normalization_parameters(
        first_table, make_amplitudes(), make_time_axis()
    )
    second_parameters = fit_normalization_parameters(
        second_table, make_amplitudes(), make_time_axis()
    )
    first_normalized = normalize_spatial_coordinates(query, first_parameters)
    second_normalized = normalize_spatial_coordinates(query, second_parameters)

    assert first_parameters.coordinate_min[4:] == (-1.0, -1.0)
    assert first_parameters.coordinate_max[4:] == (1.0, 1.0)
    assert second_parameters.coordinate_min[4:] == (-1.0, -1.0)
    assert second_parameters.coordinate_max[4:] == (1.0, 1.0)
    np.testing.assert_allclose(first_normalized, second_normalized)
    np.testing.assert_allclose(
        first_normalized[0, 3:],
        [np.sin(np.deg2rad(350.0)), np.cos(np.deg2rad(350.0))],
    )


def test_azimuth_encoding_is_periodic_across_degree_wraps() -> None:
    query = pd.DataFrame(
        {
            "cmp_x_m": [15.0, 15.0, 15.0],
            "cmp_y_m": [5.0, 5.0, 5.0],
            "offset_m": [150.0, 150.0, 150.0],
            "azimuth_deg": [-10.0, 350.0, 710.0],
        }
    )

    normalized = normalize_spatial_coordinates(query, fit_default())

    np.testing.assert_allclose(normalized[0, 3:], normalized[1, 3:], atol=1.0e-15)
    np.testing.assert_allclose(normalized[1, 3:], normalized[2, 3:], atol=1.0e-15)


def test_held_out_coordinates_do_not_change_fitted_parameters() -> None:
    trace_table = make_trace_table()
    changed_held_out = trace_table.copy()
    held_out = changed_held_out["split"] != "train"
    changed_held_out.loc[held_out, ["cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"]] = [
        [1.0e9, -1.0e9, 1.0e9, -720.0],
        [-1.0e9, 1.0e9, 2.0e9, 1080.0],
    ]

    original = fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())
    changed = fit_normalization_parameters(
        changed_held_out,
        make_amplitudes(),
        make_time_axis(),
    )

    assert changed == original


def test_time_normalization_returns_float64_and_handles_constant_axis() -> None:
    parameters = fit_default()
    normalized = normalize_time(np.asarray([0.0, 1.0, 2.0], dtype=np.float32), parameters)
    constant_parameters = NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(1.0, *parameters.coordinate_min[1:]),
        coordinate_max=(1.0, *parameters.coordinate_max[1:]),
        amplitude_rms=parameters.amplitude_rms,
    )

    np.testing.assert_array_equal(normalized, [-1.0, 0.0, 1.0])
    assert normalized.dtype == np.float64
    np.testing.assert_array_equal(
        normalize_time(np.asarray([1.0, 2.0]), constant_parameters),
        np.zeros(2),
    )


def test_amplitude_round_trip_restores_values_and_float32_dtype() -> None:
    amplitudes = make_amplitudes()
    parameters = fit_default()

    normalized = normalize_amplitudes(amplitudes, parameters)
    restored = denormalize_amplitudes(normalized, parameters)

    assert normalized.dtype == np.float32
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, amplitudes)


def test_json_write_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "normalization.json"
    parameters = fit_default()

    write_normalization_parameters(path, parameters)

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw) == parameters.to_dict()
    assert read_normalization_parameters(path) == parameters


def test_from_dict_rejects_wrong_coordinate_order() -> None:
    payload = fit_default().to_dict()
    payload["coordinate_order"] = list(reversed(MODEL_COORDINATE_ORDER))

    with pytest.raises(ValueError, match="coordinate_order"):
        NormalizationParameters.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "index", "value"), [("coordinate_min", 4, -0.5), ("coordinate_max", 5, 0.5)]
)
def test_from_dict_rejects_non_fixed_azimuth_feature_bounds(
    field: str,
    index: int,
    value: float,
) -> None:
    payload = fit_default().to_dict()
    bounds = payload[field]
    assert isinstance(bounds, list)
    bounds[index] = value

    with pytest.raises(ValueError, match="azimuth"):
        NormalizationParameters.from_dict(payload)


@pytest.mark.parametrize(
    "missing_column",
    ["array_row", "split", "cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"],
)
def test_fit_rejects_missing_columns(missing_column: str) -> None:
    with pytest.raises(ValueError, match=r"missing required (?:coordinate )?columns"):
        fit_normalization_parameters(
            make_trace_table().drop(columns=missing_column),
            make_amplitudes(),
            make_time_axis(),
        )


@pytest.mark.parametrize("bad_split", ["unknown", None])
def test_fit_rejects_invalid_split_values(bad_split: object) -> None:
    trace_table = make_trace_table()
    trace_table.loc[trace_table.index[0], "split"] = bad_split

    with pytest.raises(ValueError, match="invalid split"):
        fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())


def test_fit_rejects_table_without_training_rows() -> None:
    trace_table = make_trace_table()
    trace_table["split"] = "test"

    with pytest.raises(ValueError, match="no training"):
        fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())


@pytest.mark.parametrize("array_row", [-1, 4, 100])
def test_fit_rejects_out_of_range_array_row(array_row: int) -> None:
    trace_table = make_trace_table()
    trace_table.loc[trace_table.index[0], "array_row"] = array_row

    with pytest.raises(ValueError, match="amplitudes row range"):
        fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())


def test_fit_rejects_duplicate_array_rows() -> None:
    trace_table = make_trace_table()
    trace_table.loc[trace_table.index[1], "array_row"] = 2

    with pytest.raises(ValueError, match="duplicate array_row"):
        fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())


def test_fit_rejects_non_finite_spatial_coordinate() -> None:
    trace_table = make_trace_table()
    trace_table.loc[trace_table.index[-1], "cmp_x_m"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        fit_normalization_parameters(trace_table, make_amplitudes(), make_time_axis())


def test_fit_rejects_non_finite_held_out_amplitude() -> None:
    amplitudes = make_amplitudes()
    amplitudes[1, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        fit_normalization_parameters(make_trace_table(), amplitudes, make_time_axis())


def test_fit_rejects_non_finite_time() -> None:
    time_s = make_time_axis()
    time_s[1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        fit_normalization_parameters(make_trace_table(), make_amplitudes(), time_s)


def test_fit_rejects_zero_training_rms() -> None:
    amplitudes = make_amplitudes()
    amplitudes[[0, 2]] = 0.0

    with pytest.raises(ValueError, match="RMS"):
        fit_normalization_parameters(make_trace_table(), amplitudes, make_time_axis())


def test_fit_and_transforms_do_not_modify_inputs() -> None:
    trace_table = make_trace_table()
    amplitudes = make_amplitudes()
    time_s = make_time_axis()
    original_table = trace_table.copy(deep=True)
    original_amplitudes = amplitudes.copy()
    original_time = time_s.copy()

    parameters = fit_normalization_parameters(trace_table, amplitudes, time_s)
    normalize_spatial_coordinates(trace_table, parameters)
    normalize_time(time_s, parameters)
    normalize_amplitudes(amplitudes, parameters)
    denormalize_amplitudes(amplitudes, parameters)

    pd.testing.assert_frame_equal(trace_table, original_table)
    np.testing.assert_array_equal(amplitudes, original_amplitudes)
    np.testing.assert_array_equal(time_s, original_time)
