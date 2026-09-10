from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.c3_volume_nersi_data import (
    PROFILE_AXIS_ORDER,
    PROFILE_COORDINATE_ORDER,
    build_c3_volume_nersi_data,
    nersi_profiles_to_volume,
    volume_to_nersi_profiles,
)


def _observed_volume(
    *,
    shape: tuple[int, int, int, int, int] = (8, 2, 3, 2, 4),
    dtype: np.dtype | type = np.float32,
) -> ObservedC3Volume:
    values = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    spatial_shape = shape[1:]
    observed_mask = np.ones(spatial_shape, dtype=np.bool_)
    observed_mask[0, 0, 0] = False
    values[:, ~observed_mask] = 0
    return ObservedC3Volume(
        values=np.ascontiguousarray(values),
        time_s=np.arange(shape[0], dtype=np.float64) * 0.008,
        array_rows=np.arange(np.prod(spatial_shape), dtype=np.int64).reshape(spatial_shape),
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=~observed_mask,
    )


def test_profile_order_matches_c_order_profile_keys() -> None:
    shape = (3, 2, 3, 2, 4)
    values = np.empty(shape, dtype=np.float32)
    for source_line in range(shape[1]):
        for shot in range(shape[2]):
            for receiver_x in range(shape[3]):
                for time in range(shape[0]):
                    for receiver_y in range(shape[4]):
                        values[time, source_line, shot, receiver_x, receiver_y] = (
                            10_000 * source_line
                            + 1_000 * shot
                            + 100 * receiver_x
                            + 10 * time
                            + receiver_y
                        )

    profiles = volume_to_nersi_profiles(values)

    assert profiles.shape == (12, 1, 3, 4)
    for source_line in range(shape[1]):
        for shot in range(shape[2]):
            for receiver_x in range(shape[3]):
                index = (source_line * shape[2] + shot) * shape[3] + receiver_x
                np.testing.assert_array_equal(
                    profiles[index, 0], values[:, source_line, shot, receiver_x, :]
                )


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_profile_volume_round_trip_preserves_shape_values_and_dtype(dtype: type) -> None:
    values = np.arange(5 * 2 * 3 * 2 * 4, dtype=dtype).reshape(5, 2, 3, 2, 4)

    restored = nersi_profiles_to_volume(
        volume_to_nersi_profiles(values),
        values.shape[1:],
    )

    assert restored.shape == values.shape
    assert restored.dtype == values.dtype
    assert restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, values)


def test_profile_mask_reshape_matches_profile_value_order() -> None:
    volume = _observed_volume()
    data = build_c3_volume_nersi_data(volume)

    expected = volume.observed_trace_mask.reshape(-1, volume.values.shape[-1])
    np.testing.assert_array_equal(data.observed_trace_mask, expected)
    for profile_index, receiver_y in np.ndindex(expected.shape):
        source_line, remainder = divmod(profile_index, 3 * 2)
        shot, receiver_x = divmod(remainder, 2)
        assert (
            expected[profile_index, receiver_y]
            == volume.observed_trace_mask[source_line, shot, receiver_x, receiver_y]
        )


def test_coordinates_use_independent_unit_intervals_and_singleton_zero() -> None:
    volume = _observed_volume(shape=(8, 2, 1, 3, 4))

    data = build_c3_volume_nersi_data(volume)

    expected = np.array(
        [
            [source_line, 0.0, receiver_x / 2.0]
            for source_line in (0.0, 1.0)
            for receiver_x in range(3)
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(data.normalized_coordinates, expected)
    assert data.normalized_coordinates.dtype == np.float32
    assert data.normalized_coordinates.flags.c_contiguous
    assert PROFILE_COORDINATE_ORDER == (
        "source_line",
        "shot_in_line",
        "relative_receiver_x",
    )
    assert PROFILE_AXIS_ORDER == ("time", "relative_receiver_y")


def test_global_rms_uses_only_observed_samples_with_float64_accumulation() -> None:
    volume = _observed_volume(shape=(2, 1, 2, 1, 2), dtype=np.float32)
    values = np.array(
        [
            [[[[3.0, 1000.0]], [[0.0, 2000.0]]]],
            [[[[4.0, 3000.0]], [[12.0, 4000.0]]]],
        ],
        dtype=np.float32,
    ).reshape(2, 1, 2, 1, 2)
    observed_mask = np.array([True, False, True, False]).reshape(1, 2, 1, 2)
    volume = replace(
        volume,
        values=values,
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=~observed_mask,
    )

    data = build_c3_volume_nersi_data(volume)

    assert data.amplitude_scale == pytest.approx(np.sqrt((3.0**2 + 4.0**2 + 12.0**2) / 4.0))
    selected = (
        data.normalized_profiles[:, 0]
        .transpose(1, 0, 2)
        .reshape(2, -1)[:, observed_mask.reshape(-1)]
    )
    np.testing.assert_allclose(
        selected,
        values[:, observed_mask] / data.amplitude_scale,
    )


def test_target_storage_does_not_change_scale_or_masked_training_values() -> None:
    volume = _observed_volume(shape=(8, 2, 1, 1, 2), dtype=np.float64)
    target_mask = volume.evaluation_target_trace_mask
    variants = []
    for target_value in (0.0, 1.0e100):
        values = volume.values.copy()
        values[:, target_mask] = target_value
        variants.append(build_c3_volume_nersi_data(replace(volume, values=values)))

    first, changed = variants
    assert changed.amplitude_scale == first.amplitude_scale
    np.testing.assert_array_equal(changed.observed_trace_mask, first.observed_trace_mask)
    observed_samples = np.broadcast_to(
        first.observed_trace_mask[:, None, None, :], first.normalized_profiles.shape
    )
    np.testing.assert_array_equal(
        changed.normalized_profiles[observed_samples],
        first.normalized_profiles[observed_samples],
    )


def test_fully_missing_profile_is_excluded_only_from_training_candidates() -> None:
    volume = _observed_volume(shape=(8, 2, 1, 1, 4))
    data = build_c3_volume_nersi_data(volume)

    assert data.normalized_coordinates.shape[0] == 2
    assert data.normalized_profiles.shape == (2, 1, 8, 4)
    np.testing.assert_array_equal(data.training_profile_indices, [1])


@pytest.mark.parametrize("invalid", ["zero_rms", "mask_shape", "overlap", "nonfinite"])
def test_invalid_observed_volume_is_rejected(invalid: str) -> None:
    volume = _observed_volume(shape=(8, 2, 1, 1, 4))
    if invalid == "zero_rms":
        values = volume.values.copy()
        values[:, volume.observed_trace_mask] = 0
        volume = replace(volume, values=values)
        match = "RMS must be positive"
    elif invalid == "mask_shape":
        volume = replace(volume, observed_trace_mask=volume.observed_trace_mask[..., :-1])
        match = "observed_trace_mask"
    elif invalid == "overlap":
        target = volume.evaluation_target_trace_mask.copy()
        target[volume.observed_trace_mask] = True
        volume = replace(volume, evaluation_target_trace_mask=target)
        match = "disjointly cover"
    else:
        values = volume.values.copy()
        values[0, 1, 0, 0, 0] = np.nan
        volume = replace(volume, values=values)
        match = "finite"

    with pytest.raises(ValueError, match=match):
        build_c3_volume_nersi_data(volume)


@pytest.mark.parametrize(
    ("profiles", "spatial_shape", "match"),
    [
        (np.ones((2, 1, 8, 4)), (3, 1, 1, 4), "profile count"),
        (np.ones((2, 2, 8, 4)), (2, 1, 1, 4), "one amplitude channel"),
        (np.ones((2, 1, 8, 3)), (2, 1, 1, 4), "receiver-y"),
        (np.ones((2, 1, 8, 4)), (2, 1, 4), "four positive integers"),
    ],
)
def test_inverse_mapping_rejects_inconsistent_shapes(profiles, spatial_shape, match) -> None:
    with pytest.raises(ValueError, match=match):
        nersi_profiles_to_volume(profiles, spatial_shape)
