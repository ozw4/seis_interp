from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.c3_volume_siren_data import (
    VOLUME_AMPLITUDE_SCALE_SOURCE,
    VOLUME_COORDINATE_SCOPE,
    build_c3_volume_siren_data,
    build_c3_volume_siren_sampler,
    validate_c3_volume_siren_scaling,
)
from seis_interp.training.point_sampler import RandomPointSampler


def _volume_and_index(dtype=np.float32) -> tuple[ObservedC3Volume, pd.DataFrame]:
    spatial_shape = (1, 2, 1, 2)
    amplitudes = np.array(
        [[3.0, 4.0, 0.0], [1000.0, 2000.0, 3000.0], [0.0, 0.0, 12.0], [-1000.0, -2000.0, -3000.0]],
        dtype=dtype,
    )
    mask = np.array([True, False, True, False]).reshape(spatial_shape)
    rows = np.array([10, 2, 30, 7], dtype=np.int64)
    volume = ObservedC3Volume(
        values=np.ascontiguousarray(amplitudes.T.reshape(3, *spatial_shape)),
        time_s=np.array([0.25, 0.5, 0.75], dtype=np.float64),
        array_rows=rows.reshape(spatial_shape),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    index_table = pd.DataFrame(
        {
            "array_row": rows,
            "source_x_m": [0.0, 10.0, 0.0, 10.0],
            "source_y_m": [0.0, 0.0, 10.0, 10.0],
            "relative_receiver_x_m": [0.0, 4.0, -6.0, 0.0],
            "relative_receiver_y_m": [2.0, 0.0, 0.0, -8.0],
        },
        index=[5, 9, 2, 7],
    )
    return volume, index_table


def test_geometry_and_normalization_match_hand_calculated_four_direction_traces() -> None:
    volume, index_table = _volume_and_index()

    data = build_c3_volume_siren_data(volume, index_table)

    assert data.normalization.coordinate_min == (0.25, -3.0, 0.0, 2.0, -1.0, -1.0)
    assert data.normalization.coordinate_max == (0.75, 12.0, 10.0, 8.0, 1.0, 1.0)
    np.testing.assert_array_equal(data.normalized_time, [-1.0, 0.0, 1.0])
    # CMPs are (0,1), (12,0), (-3,10), (10,6); offsets are 2,4,6,8.
    # Source-minus-receiver azimuths are 180,270,90,0 degrees respectively.
    np.testing.assert_allclose(
        data.normalized_spatial,
        [
            [-0.6, -0.8, -1.0, 0.0, -1.0],
            [1.0, -1.0, -1.0 / 3.0, -1.0, 0.0],
            [-1.0, 1.0, 1.0 / 3.0, 1.0, 0.0],
            [11.0 / 15.0, 0.2, 1.0, 0.0, 1.0],
        ],
        atol=1e-14,
    )
    assert data.model_coordinates.coordinate_features == "cmp_offset_azimuth"
    assert data.model_coordinates.time_coordinate_scale == 1.0
    assert data.model_coordinates.coordinate_order == (
        "time_s",
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_sin",
        "azimuth_cos",
    )
    assert VOLUME_COORDINATE_SCOPE == "selected_volume_geometry"
    assert VOLUME_AMPLITUDE_SCALE_SOURCE == "observed_trace_samples_only"


@pytest.mark.parametrize(
    "coordinate_features", ["cmp_cartesian_half_offset", "cmp_cartesian_half_offset_radius"]
)
@pytest.mark.parametrize("time_scale", [1.0, 4.0])
def test_cartesian_coordinates_and_time_scale_match_physical_hand_calculation(
    coordinate_features: str, time_scale: float
) -> None:
    volume, index_table = _volume_and_index()
    default = build_c3_volume_siren_data(volume, index_table)
    data = build_c3_volume_siren_data(
        volume,
        index_table,
        coordinate_features=coordinate_features,
        time_coordinate_scale=time_scale,
    )

    # Half offsets are source-minus-receiver / 2, divided by max(offset) / 2 = 4.
    expected = np.array(
        [
            [-0.6, -0.8, 0.0, -0.25],
            [1.0, -1.0, -0.5, 0.0],
            [-1.0, 1.0, 0.75, 0.0],
            [11.0 / 15.0, 0.2, 0.0, 1.0],
        ]
    )
    if coordinate_features == "cmp_cartesian_half_offset_radius":
        expected = np.column_stack((expected, [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]))
    np.testing.assert_allclose(data.normalized_spatial, expected, atol=1e-14)
    np.testing.assert_array_equal(data.normalized_time, [-time_scale, 0.0, time_scale])
    assert data.model_coordinates.input_features == expected.shape[1] + 1
    assert data.model_coordinates.coordinate_features == coordinate_features
    assert data.model_coordinates.time_coordinate_scale == time_scale
    assert data.model_coordinates.half_offset_scale_m == 4.0
    assert data.normalization == default.normalization
    np.testing.assert_array_equal(
        data.normalized_observed_amplitudes, default.normalized_observed_amplitudes
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"coordinate_features": "unknown"}, "coordinate_features"),
        ({"time_coordinate_scale": 0.0}, "time_coordinate_scale"),
        ({"time_coordinate_scale": True}, "time_coordinate_scale"),
        ({"time_coordinate_scale": np.inf}, "time_coordinate_scale"),
    ],
)
def test_invalid_coordinate_options_use_existing_validation_contract(changes, message) -> None:
    volume, index_table = _volume_and_index()
    with pytest.raises(ValueError, match=message):
        build_c3_volume_siren_data(volume, index_table, **changes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compact_observed_arrays_have_expected_rms_shapes_dtypes_and_order(dtype) -> None:
    volume, index_table = _volume_and_index(dtype)

    data = build_c3_volume_siren_data(volume, index_table)

    expected_rms = np.sqrt((3.0**2 + 4.0**2 + 12.0**2) / 6.0)
    assert data.normalization.amplitude_rms == pytest.approx(expected_rms)
    assert data.normalized_time.shape == (3,)
    assert data.normalized_spatial.shape == (4, 5)
    assert data.normalized_observed_amplitudes.shape == (2, 3)
    assert data.normalized_time.dtype == np.float64
    assert data.normalized_spatial.dtype == np.float64
    assert data.normalized_observed_amplitudes.dtype == dtype
    assert data.observed_flat_indices.dtype == np.int64
    np.testing.assert_array_equal(data.observed_flat_indices, [0, 2])
    np.testing.assert_allclose(
        data.normalized_observed_amplitudes,
        np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]], dtype=dtype) / expected_rms,
    )
    for array in (
        data.normalized_time,
        data.normalized_spatial,
        data.observed_flat_indices,
        data.normalized_observed_amplitudes,
    ):
        assert array.flags.c_contiguous
        assert not np.shares_memory(array, volume.values)


def test_coordinate_ranges_use_all_geometry_and_do_not_depend_on_visibility() -> None:
    volume, index_table = _volume_and_index()
    changed = replace(
        volume,
        observed_trace_mask=~volume.observed_trace_mask,
        evaluation_target_trace_mask=volume.observed_trace_mask.copy(),
    )

    first = build_c3_volume_siren_data(volume, index_table)
    second = build_c3_volume_siren_data(changed, index_table)

    assert first.normalization.coordinate_min == second.normalization.coordinate_min
    assert first.normalization.coordinate_max == second.normalization.coordinate_max
    assert first.model_coordinates == second.model_coordinates
    np.testing.assert_array_equal(first.normalized_spatial, second.normalized_spatial)
    assert first.normalization.amplitude_rms != second.normalization.amplitude_rms


def test_target_values_do_not_affect_data_or_seeded_sampler_output() -> None:
    volume, index_table = _volume_and_index(np.float64)
    data_variants = []
    batches = []
    for target_value in (0.0, 1.0e300, np.nan):
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = target_value
        data = build_c3_volume_siren_data(replace(volume, values=values), index_table)
        data_variants.append(data)
        batches.append(build_c3_volume_siren_sampler(data, random_seed=19).sample(64))

    for data, batch in zip(data_variants[1:], batches[1:], strict=True):
        assert data.normalization == data_variants[0].normalization
        assert data.model_coordinates == data_variants[0].model_coordinates
        np.testing.assert_array_equal(data.normalized_time, data_variants[0].normalized_time)
        np.testing.assert_array_equal(data.normalized_spatial, data_variants[0].normalized_spatial)
        np.testing.assert_array_equal(
            data.observed_flat_indices, data_variants[0].observed_flat_indices
        )
        np.testing.assert_array_equal(
            data.normalized_observed_amplitudes, data_variants[0].normalized_observed_amplitudes
        )
        np.testing.assert_array_equal(batch[0], batches[0][0])
        np.testing.assert_array_equal(batch[1], batches[0][1])


def test_changing_observed_values_changes_the_rms_and_training_targets() -> None:
    volume, index_table = _volume_and_index()
    changed_values = volume.values.copy()
    changed_values[0, 0, 0, 0, 0] = 30.0

    first = build_c3_volume_siren_data(volume, index_table)
    changed = build_c3_volume_siren_data(replace(volume, values=changed_values), index_table)

    assert first.normalization.amplitude_rms != changed.normalization.amplitude_rms
    assert not np.array_equal(
        first.normalized_observed_amplitudes, changed.normalized_observed_amplitudes
    )
    np.testing.assert_array_equal(first.normalized_spatial, changed.normalized_spatial)


def test_sampler_is_reproducible_and_only_returns_compact_observed_points() -> None:
    volume, index_table = _volume_and_index(np.float64)
    data = build_c3_volume_siren_data(volume, index_table)
    first = build_c3_volume_siren_sampler(data, random_seed=17)
    second = build_c3_volume_siren_sampler(data, random_seed=17)

    coordinates, targets = first.sample(200)
    repeated_coordinates, repeated_targets = second.sample(200)

    assert isinstance(first, RandomPointSampler)
    assert first.amplitude_scaling == "train_global_rms"
    np.testing.assert_array_equal(coordinates, repeated_coordinates)
    np.testing.assert_array_equal(targets, repeated_targets)
    for point, target in zip(coordinates, targets, strict=True):
        time_index = int(np.flatnonzero(data.normalized_time == point[0])[0])
        flat_index = int(np.flatnonzero(np.all(data.normalized_spatial == point[1:], axis=1))[0])
        assert flat_index in (0, 2)
        physical_target = volume.values.reshape(3, 4)[time_index, flat_index]
        assert target == pytest.approx(physical_target / np.sqrt(169.0 / 6.0))


def test_building_data_and_sampling_do_not_modify_source_arrays_or_index() -> None:
    volume, index_table = _volume_and_index()
    originals = {
        field: getattr(volume, field).copy()
        for field in (
            "values",
            "time_s",
            "array_rows",
            "observed_trace_mask",
            "evaluation_target_trace_mask",
        )
    }
    original_table = index_table.copy(deep=True)

    data = build_c3_volume_siren_data(volume, index_table)
    build_c3_volume_siren_sampler(data, random_seed=17).sample(8)

    for field, original in originals.items():
        np.testing.assert_array_equal(getattr(volume, field), original)
    pd.testing.assert_frame_equal(index_table, original_table)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("row_count", "row count"),
        ("row_order", "array_row order"),
        ("missing_geometry", "geometry columns"),
        ("nonfinite_geometry", "finite real values"),
        ("missing_array_row", "array_row"),
    ],
)
def test_rejects_misaligned_or_invalid_index_geometry(invalid: str, message: str) -> None:
    volume, index_table = _volume_and_index()
    if invalid == "row_count":
        index_table = index_table.iloc[:-1]
    elif invalid == "row_order":
        index_table = index_table.iloc[::-1]
    elif invalid == "missing_geometry":
        index_table = index_table.drop(columns="relative_receiver_x_m")
    elif invalid == "nonfinite_geometry":
        index_table.loc[index_table.index[-1], "source_x_m"] = np.nan
    else:
        index_table = index_table.drop(columns="array_row")

    with pytest.raises(ValueError, match=message):
        build_c3_volume_siren_data(volume, index_table)


@pytest.mark.parametrize(
    "invalid", ["zero_rms", "no_observed", "observed_nan", "time_nan", "time_shape"]
)
def test_rejects_invalid_observed_data(invalid: str) -> None:
    volume, index_table = _volume_and_index()
    if invalid == "zero_rms":
        volume.values[:, volume.observed_trace_mask] = 0.0
        message = "RMS must be positive"
    elif invalid == "no_observed":
        volume.observed_trace_mask[:] = False
        message = "at least one observed"
    elif invalid == "observed_nan":
        volume.values[0, 0, 0, 0, 0] = np.nan
        message = "observed amplitudes.*finite"
    elif invalid == "time_nan":
        volume.time_s[-1] = np.nan
        message = "time_s.*finite"
    else:
        volume = replace(volume, time_s=volume.time_s[:-1])
        message = "time_s.*matching"

    with pytest.raises(ValueError, match=message):
        build_c3_volume_siren_data(volume, index_table)


def test_requires_an_observed_c3_volume() -> None:
    _, index_table = _volume_and_index()

    with pytest.raises(ValueError, match="ObservedC3Volume"):
        build_c3_volume_siren_data(object(), index_table)  # type: ignore[arg-type]


_INTERPOLATION = {"neighbors": 8, "power": 2.0, "distance_scales_m": [10.0, 1.0, 1.0, 1.0]}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_per_trace_scaling_fits_unit_rms_and_interpolates_in_declared_physical_metric(dtype):
    volume, index_table = _volume_and_index(dtype)
    default = build_c3_volume_siren_data(volume, index_table)
    data = build_c3_volume_siren_data(
        volume,
        index_table,
        amplitude_scaling="per_trace_rms",
        scale_interpolation=_INTERPOLATION,
    )
    rms = np.array([5.0 / np.sqrt(3), 12.0 / np.sqrt(3)])
    assert data.normalization == default.normalization
    assert data.amplitude_scaling == "per_trace_rms"
    assert data.scale_interpolation == _INTERPOLATION
    assert data.scale_interpolation is not _INTERPOLATION
    np.testing.assert_allclose(data.trace_amplitude_scales[[0, 2]], rms)
    # Axis divisors [10,1,1,1] give these squared physical metric distances.
    expected_targets = [
        (rms[0] / 21 + rms[1] / 201) / (1 / 21 + 1 / 201),
        (rms[0] / 201 + rms[1] / 101) / (1 / 201 + 1 / 101),
    ]
    np.testing.assert_allclose(data.trace_amplitude_scales[[1, 3]], expected_targets)
    np.testing.assert_allclose(
        np.mean(data.normalized_observed_amplitudes.astype(np.float64) ** 2, axis=1), 1.0, rtol=1e-7
    )
    np.testing.assert_allclose(
        data.normalized_observed_amplitudes * rms[:, None],
        volume.values[:, volume.observed_trace_mask].T,
        rtol=1e-7,
    )
    assert data.normalized_observed_amplitudes.dtype == dtype
    assert data.trace_amplitude_scales.dtype == np.float64
    assert data.trace_amplitude_scales.flags.c_contiguous
    np.testing.assert_array_equal(data.trace_array_rows, volume.array_rows.reshape(-1))
    assert data.trace_array_rows.dtype == np.int64
    assert not np.shares_memory(data.trace_array_rows, volume.array_rows)


def test_per_trace_scaling_keeps_mixed_zero_observed_traces_with_physical_scale_zero():
    volume, index_table = _volume_and_index()
    volume.values[:, 0, 0, 0, 0] = 0
    data = build_c3_volume_siren_data(
        volume,
        index_table,
        amplitude_scaling="per_trace_rms",
        scale_interpolation=_INTERPOLATION,
    )
    assert data.trace_amplitude_scales[0] == 0
    np.testing.assert_array_equal(data.normalized_observed_amplitudes[0], 0)
    assert np.isfinite(data.normalized_observed_amplitudes).all()
    assert (data.trace_amplitude_scales[[1, 3]] > 0).all()
    assert data.observed_flat_indices.tolist() == [0, 2]
    volume.values[:, volume.observed_trace_mask] = 0
    with pytest.raises(ValueError, match="observed amplitude RMS must be positive and finite"):
        build_c3_volume_siren_data(
            volume,
            index_table,
            amplitude_scaling="per_trace_rms",
            scale_interpolation=_INTERPOLATION,
        )


@pytest.mark.parametrize(
    "coordinate_features,time_scale",
    [
        ("cmp_offset_azimuth", 1.0),
        ("cmp_cartesian_half_offset", 4.0),
        ("cmp_cartesian_half_offset_radius", 2.0),
    ],
)
def test_per_trace_scales_and_seeded_coordinates_never_depend_on_target_truth(
    coordinate_features, time_scale
):
    volume, index_table = _volume_and_index(np.float64)
    baseline = build_c3_volume_siren_data(
        volume,
        index_table,
        coordinate_features=coordinate_features,
        time_coordinate_scale=time_scale,
    )
    baseline_sampler = build_c3_volume_siren_sampler(baseline, random_seed=17)
    expected = None
    for target_value in [0.0, 1e300, np.nan]:
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = target_value
        data = build_c3_volume_siren_data(
            replace(volume, values=values),
            index_table,
            amplitude_scaling="per_trace_rms",
            scale_interpolation=_INTERPOLATION,
            coordinate_features=coordinate_features,
            time_coordinate_scale=time_scale,
        )
        sampler = build_c3_volume_siren_sampler(data, random_seed=17)
        assert sampler.amplitude_scaling == "per_trace_rms"
        batches = [sampler.sample(50) for _ in range(3)]
        if expected is None:
            expected = (data, batches)
            for coordinates, _ in batches:
                np.testing.assert_array_equal(coordinates, baseline_sampler.sample(50)[0])
        else:
            np.testing.assert_array_equal(
                data.trace_amplitude_scales, expected[0].trace_amplitude_scales
            )
            np.testing.assert_array_equal(
                data.normalized_observed_amplitudes, expected[0].normalized_observed_amplitudes
            )
            for batch, first in zip(batches, expected[1], strict=True):
                np.testing.assert_array_equal(batch[0], first[0])
                np.testing.assert_array_equal(batch[1], first[1])


def test_coordinate_representation_and_time_scale_do_not_change_observed_rms_or_idw_gains():
    volume, index_table = _volume_and_index()
    volume.values[:, 0, 0, 0, 0] = 0.0
    baseline = build_c3_volume_siren_data(
        volume,
        index_table,
        amplitude_scaling="per_trace_rms",
        scale_interpolation=_INTERPOLATION,
    )
    for features in ("cmp_cartesian_half_offset", "cmp_cartesian_half_offset_radius"):
        data = build_c3_volume_siren_data(
            volume,
            index_table,
            amplitude_scaling="per_trace_rms",
            scale_interpolation=_INTERPOLATION,
            coordinate_features=features,
            time_coordinate_scale=4.0,
        )
        np.testing.assert_array_equal(data.trace_amplitude_scales, baseline.trace_amplitude_scales)
        np.testing.assert_array_equal(data.trace_array_rows, baseline.trace_array_rows)
        np.testing.assert_array_equal(
            data.normalized_observed_amplitudes, baseline.normalized_observed_amplitudes
        )
        assert data.normalization == baseline.normalization
        assert data.scale_interpolation == baseline.scale_interpolation
        assert data.trace_amplitude_scales[0] == 0.0


@pytest.mark.parametrize("shear", [0.125, -0.125])
def test_shear_uses_signed_relative_receiver_y_without_changing_amplitudes_or_sampler_rng(shear):
    volume, index_table = _volume_and_index()
    settings = {
        "coordinate_features": "cmp_cartesian_half_offset",
        "time_coordinate_scale": 4.0,
        "amplitude_scaling": "per_trace_rms",
        "scale_interpolation": _INTERPOLATION,
    }
    baseline = build_c3_volume_siren_data(volume, index_table, **settings)
    data = build_c3_volume_siren_data(
        volume, index_table, **settings, relative_receiver_y_time_shear_s_per_m=shear
    )
    # tau=t+s*relative_receiver_y; dt_normalized/dt=2*4/(0.75-0.25)=16.
    expected_offsets = 16.0 * shear * np.array([2.0, 0.0, 0.0, -8.0])
    np.testing.assert_array_equal(data.normalized_time_offsets, expected_offsets)
    np.testing.assert_array_equal(data.normalized_time, [-4.0, 0.0, 4.0])
    np.testing.assert_array_equal(data.normalized_spatial, baseline.normalized_spatial)
    np.testing.assert_array_equal(data.trace_amplitude_scales, baseline.trace_amplitude_scales)
    np.testing.assert_array_equal(
        data.normalized_observed_amplitudes, baseline.normalized_observed_amplitudes
    )
    assert baseline.normalized_time_offsets is None
    assert data.normalized_time_offsets.dtype == np.float64
    assert data.normalized_time_offsets.flags.c_contiguous
    points, targets = build_c3_volume_siren_sampler(data, random_seed=19).sample(100)
    baseline_points, baseline_targets = build_c3_volume_siren_sampler(
        baseline, random_seed=19
    ).sample(100)
    np.testing.assert_array_equal(points[:, 1:], baseline_points[:, 1:])
    np.testing.assert_array_equal(targets, baseline_targets)
    for point, original in zip(points, baseline_points, strict=True):
        flat_row = int(np.flatnonzero(np.all(data.normalized_spatial == point[1:], axis=1))[0])
        assert flat_row in data.observed_flat_indices
        assert point[0] == original[0] + expected_offsets[flat_row]


def test_sheared_training_coordinates_and_per_trace_scales_ignore_poisoned_target_waveforms():
    volume, index_table = _volume_and_index(np.float64)
    baseline = None
    for poison in (0.0, 1e300, np.nan):
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = poison
        data = build_c3_volume_siren_data(
            replace(volume, values=values),
            index_table,
            coordinate_features="cmp_cartesian_half_offset",
            time_coordinate_scale=4.0,
            relative_receiver_y_time_shear_s_per_m=0.125,
            amplitude_scaling="per_trace_rms",
            scale_interpolation=_INTERPOLATION,
        )
        batch = build_c3_volume_siren_sampler(data, random_seed=3).sample(64)
        if baseline is None:
            baseline = data, batch
        else:
            np.testing.assert_array_equal(
                data.normalized_time_offsets, baseline[0].normalized_time_offsets
            )
            np.testing.assert_array_equal(
                data.trace_amplitude_scales, baseline[0].trace_amplitude_scales
            )
            np.testing.assert_array_equal(
                data.normalized_observed_amplitudes, baseline[0].normalized_observed_amplitudes
            )
            np.testing.assert_array_equal(batch[0], baseline[1][0])
            np.testing.assert_array_equal(batch[1], baseline[1][1])


def test_explicit_global_default_preserves_arrays_and_has_no_trace_scale_metadata():
    volume, index_table = _volume_and_index()
    implicit = build_c3_volume_siren_data(volume, index_table)
    explicit = build_c3_volume_siren_data(
        volume,
        index_table,
        amplitude_scaling="train_global_rms",
        coordinate_features="cmp_offset_azimuth",
        time_coordinate_scale=1.0,
        relative_receiver_y_time_shear_s_per_m=0.0,
    )
    assert explicit.trace_amplitude_scales is None and explicit.scale_interpolation is None
    assert explicit.trace_array_rows is None
    assert implicit.normalized_time_offsets is explicit.normalized_time_offsets is None
    assert implicit.model_coordinates.to_dict() == explicit.model_coordinates.to_dict()
    for name in [
        "normalized_time",
        "normalized_spatial",
        "normalized_observed_amplitudes",
        "observed_flat_indices",
    ]:
        np.testing.assert_array_equal(getattr(explicit, name), getattr(implicit, name))
    for actual, expected in zip(
        build_c3_volume_siren_sampler(explicit, random_seed=19).sample(100),
        build_c3_volume_siren_sampler(implicit, random_seed=19).sample(100),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mode,changes",
    [
        ("train_global_rms", {}),
        ("unknown", {}),
        ("per_trace_rms", {"neighbors": True}),
        ("per_trace_rms", {"neighbors": 0}),
        ("per_trace_rms", {"power": np.inf}),
        ("per_trace_rms", {"power": True}),
        ("per_trace_rms", {"unknown": 1}),
        ("per_trace_rms", {"power": 10**400}),
        ("per_trace_rms", {"distance_scales_m": [1, 2, 3]}),
        ("per_trace_rms", {"distance_scales_m": [1, 2, 3, 0]}),
        ("per_trace_rms", {"distance_scales_m": [1, 2, 3, np.nan]}),
        ("per_trace_rms", {"distance_scales_m": [1, 2, 3, True]}),
        ("per_trace_rms", {"distance_scales_m": [1, 2, 3, 10**400]}),
    ],
)
def test_scaling_contract_rejects_invalid_interpolation_settings(mode, changes):
    with pytest.raises(ValueError):
        validate_c3_volume_siren_scaling(mode, {**_INTERPOLATION, **changes})


def test_per_trace_scaling_requires_explicit_interpolation():
    with pytest.raises(ValueError, match="scale_interpolation"):
        validate_c3_volume_siren_scaling("per_trace_rms", None)
