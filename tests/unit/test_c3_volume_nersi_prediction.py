from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.c3_volume_nersi_data import C3VolumeNersiData
from seis_interp.training.c3_volume_nersi_prediction import predict_c3_volume_nersi


class _CoordinateProfileModel(torch.nn.Module):
    profile_shape = (8, 8)

    def __init__(self, *, invalid: str | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.invalid = invalid
        self.batch_sizes: list[int] = []

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        assert not self.training
        self.batch_sizes.append(len(coordinates))
        values = coordinates[:, 2] * 4.0 + coordinates[:, 1] * 2.0 + coordinates[:, 0]
        output = values[:, None, None, None].expand(-1, 1, 8, 8) + self.anchor
        if self.invalid == "shape":
            return output[..., :-1]
        if self.invalid == "nonfinite":
            output = output.clone()
            output[0, 0, 0, 0] = torch.nan
        return output


def _inputs(dtype=np.float32) -> tuple[C3VolumeNersiData, ObservedC3Volume]:
    spatial_shape = (1, 2, 2, 8)
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    profile_mask = np.array(
        [
            [True, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
        ],
        dtype=np.bool_,
    )
    mask = profile_mask.reshape(spatial_shape)
    values = np.zeros((8, *spatial_shape), dtype=dtype)
    values[:, mask] = np.array([11.0, -7.0], dtype=dtype)
    observed = ObservedC3Volume(
        values=values,
        time_s=np.arange(8, dtype=np.float64),
        array_rows=np.arange(np.prod(spatial_shape), dtype=np.int64).reshape(spatial_shape),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    data = C3VolumeNersiData(
        normalized_coordinates=coordinates,
        normalized_profiles=np.zeros((4, 1, 8, 8), dtype=np.float32),
        observed_trace_mask=profile_mask,
        training_profile_indices=np.array([0, 2], dtype=np.int64),
        amplitude_scale=3.0,
        spatial_shape=spatial_shape,
        profile_shape=(8, 8),
    )
    return data, observed


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_predicts_every_profile_restores_scale_once_and_reinserts_observed(
    batch_size: int,
) -> None:
    data, observed = _inputs()
    model = _CoordinateProfileModel()

    result = predict_c3_volume_nersi(model, data, observed, batch_size=batch_size, device="cpu")

    # C-order profile constants are [0, 4, 2, 6], followed by physical scale 3.
    before = np.repeat(np.array([0.0, 12.0, 6.0, 18.0])[:, None], 64, axis=1)
    before = before.reshape(4, 1, 8, 8).reshape(1, 2, 2, 8, 8).transpose(3, 0, 1, 2, 4)
    expected = before.astype(np.float32)
    observed_errors = expected[:, observed.observed_trace_mask].astype(np.float64) - (
        observed.values[:, observed.observed_trace_mask].astype(np.float64)
    )
    expected[:, observed.observed_trace_mask] = observed.values[:, observed.observed_trace_mask]

    np.testing.assert_array_equal(result.values, expected)
    assert result.predicted_profile_count == 4
    assert result.observed_model_rmse_before_reinsertion == pytest.approx(
        np.sqrt(np.mean(observed_errors**2, dtype=np.float64))
    )
    assert result.observed_model_max_abs_error_before_reinsertion == pytest.approx(
        np.max(np.abs(observed_errors))
    )
    assert result.values.dtype == np.float32
    assert result.values.flags.c_contiguous
    assert result.values.shape == observed.values.shape
    assert np.isfinite(result.values).all()
    assert max(model.batch_sizes) <= batch_size
    assert sum(model.batch_sizes) == 4
    assert model.training


def test_batch_size_and_target_buffer_values_do_not_change_prediction() -> None:
    data, observed = _inputs()
    baseline = predict_c3_volume_nersi(
        _CoordinateProfileModel(), data, observed, batch_size=1, device="cpu"
    )
    poisoned_values = observed.values.copy()
    poisoned_values[:, observed.evaluation_target_trace_mask] = np.nan
    poisoned = replace(observed, values=poisoned_values)
    changed = predict_c3_volume_nersi(
        _CoordinateProfileModel(), data, poisoned, batch_size=3, device="cpu"
    )

    np.testing.assert_array_equal(changed.values, baseline.values)
    assert changed.observed_model_rmse_before_reinsertion == (
        baseline.observed_model_rmse_before_reinsertion
    )
    assert changed.observed_model_max_abs_error_before_reinsertion == (
        baseline.observed_model_max_abs_error_before_reinsertion
    )


def test_profiles_without_observed_traces_are_still_predicted() -> None:
    data, observed = _inputs()
    assert not data.observed_trace_mask[1].any()
    assert not data.observed_trace_mask[3].any()

    result = predict_c3_volume_nersi(
        _CoordinateProfileModel(), data, observed, batch_size=2, device="cpu"
    )

    assert result.predicted_profile_count == len(data.normalized_coordinates) == 4
    assert np.all(result.values[:, 0, 0, 1] == 12.0)
    assert np.all(result.values[:, 0, 1, 1] == 18.0)


@pytest.mark.parametrize(
    ("invalid", "message"), [("shape", "model output"), ("nonfinite", "finite")]
)
def test_rejects_invalid_model_output(invalid: str, message: str) -> None:
    data, observed = _inputs()

    with pytest.raises(ValueError, match=message):
        predict_c3_volume_nersi(
            _CoordinateProfileModel(invalid=invalid),
            data,
            observed,
            batch_size=2,
            device="cpu",
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(batch_size) -> None:
    data, observed = _inputs()
    with pytest.raises(ValueError, match="batch_size"):
        predict_c3_volume_nersi(
            _CoordinateProfileModel(), data, observed, batch_size=batch_size, device="cpu"
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("profile_shape", "profile_shape"),
        ("spatial_shape", "spatial_shape"),
        ("coordinates", "normalized_coordinates"),
        ("mask", "observed_trace_mask"),
        ("scale", "amplitude_scale"),
    ],
)
def test_rejects_mismatched_prediction_contract(change: str, message: str) -> None:
    data, observed = _inputs()
    model = _CoordinateProfileModel()
    if change == "profile_shape":
        model.profile_shape = (8, 16)
    elif change == "spatial_shape":
        data = replace(data, spatial_shape=(1, 1, 4, 8))
    elif change == "coordinates":
        data = replace(data, normalized_coordinates=data.normalized_coordinates[:-1])
    elif change == "mask":
        profile_mask = data.observed_trace_mask.copy()
        profile_mask[0, 0] = False
        data = replace(data, observed_trace_mask=profile_mask)
    else:
        data = replace(data, amplitude_scale=0.0)

    with pytest.raises(ValueError, match=message):
        predict_c3_volume_nersi(model, data, observed, batch_size=2, device="cpu")
