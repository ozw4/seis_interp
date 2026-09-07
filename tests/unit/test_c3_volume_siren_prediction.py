from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import ModelCoordinateParameters
from seis_interp.training import c3_volume_siren_prediction as prediction_module
from seis_interp.training.c3_volume_siren_data import C3VolumeSirenData
from seis_interp.training.c3_volume_siren_prediction import predict_c3_volume_siren


class _CoordinateModel(torch.nn.Module):
    input_features = 6

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        assert not self.training
        self.batch_sizes.append(len(coordinates))
        return (
            2 * coordinates[:, 0]
            + 3 * coordinates[:, 1]
            - coordinates[:, 2]
            + 0.5 * coordinates[:, 3]
            + coordinates[:, 4]
            - 2 * coordinates[:, 5]
        )[:, None]


def _prediction_inputs(dtype=np.float32) -> tuple[C3VolumeSirenData, ObservedC3Volume]:
    spatial_shape = (1, 2, 1, 3)
    mask = np.array([True, False, False, False, True, False]).reshape(spatial_shape)
    values = np.full((3, *spatial_shape), np.nan, dtype=dtype)
    values[:, mask] = [2.0, -2.0]
    volume = ObservedC3Volume(
        values=values,
        time_s=np.array([0.0, 0.5, 1.0]),
        array_rows=np.array([10, 2, 30, 7, 9, 1], dtype=np.int64).reshape(spatial_shape),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    normalization = NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(0.0, 0.0, 0.0, 0.0, -1.0, -1.0),
        coordinate_max=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        amplitude_rms=2.0,
    )
    data = C3VolumeSirenData(
        normalized_time=np.array([-1.0, 0.0, 1.0]),
        normalized_spatial=np.array(
            [
                [-1.0, -1.0, -1.0, 0.0, 1.0],
                [-0.5, 0.0, -0.5, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0],
                [0.5, -1.0, 0.5, -1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.25, 0.5, 0.75, 1.0, 0.0],
            ]
        ),
        observed_flat_indices=np.array([0, 4], dtype=np.int64),
        normalized_observed_amplitudes=np.array([[1.0] * 3, [-1.0] * 3], dtype=dtype),
        normalization=normalization,
        model_coordinates=ModelCoordinateParameters(
            coordinate_features="cmp_offset_azimuth",
            coordinate_order=MODEL_COORDINATE_ORDER,
            coordinate_scale_min=normalization.coordinate_min,
            coordinate_scale_max=normalization.coordinate_max,
        ),
    )
    return data, volume


@pytest.mark.parametrize("batch_size,trace_chunk_counts", [(2, [1] * 6), (12, [4, 2])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chunked_volume_prediction_matches_hand_calculated_physical_output(
    batch_size: int,
    trace_chunk_counts: list[int],
    dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, volume = _prediction_inputs(dtype)
    model = _CoordinateModel()
    chunk_counts: list[int] = []
    real_coordinate_builder = prediction_module.build_trace_coordinate_points

    def recording_coordinates(time, spatial, rows):
        chunk_counts.append(len(rows))
        assert len(rows) <= max(1, batch_size // 3)
        return real_coordinate_builder(time, spatial, rows)

    monkeypatch.setattr(prediction_module, "build_trace_coordinate_points", recording_coordinates)

    result = predict_c3_volume_siren(model, data, volume, batch_size=batch_size, device="cpu")

    # f(t,x) * observed RMS 2, before reinserting the two observed traces.
    before_reinsertion = np.array(
        [
            [-13.0, -9.0, -5.0],
            [-5.5, -1.5, 2.5],
            [-2.0, 2.0, 6.0],
            [-0.5, 3.5, 7.5],
            [-1.0, 3.0, 7.0],
            [-0.75, 3.25, 7.25],
        ],
        dtype=dtype,
    )
    expected = before_reinsertion.copy()
    expected[0], expected[4] = 2.0, -2.0
    expected = expected.T.reshape(volume.values.shape)
    np.testing.assert_array_equal(result.values, expected)
    assert result.observed_model_rmse_before_reinsertion == pytest.approx(np.sqrt(502.0 / 6.0))
    assert result.observed_model_max_abs_error_before_reinsertion == 15.0
    np.testing.assert_array_equal(result.values[:, volume.observed_trace_mask], [[2.0, -2.0]] * 3)
    assert chunk_counts == trace_chunk_counts
    assert max(model.batch_sizes) <= batch_size
    assert sum(model.batch_sizes) == result.predicted_point_count == volume.values.size
    assert result.values.shape == (3, 1, 2, 1, 3)
    assert result.values.dtype == dtype
    assert result.values.flags.c_contiguous
    assert np.all(np.isfinite(result.values))
    assert model.training


def test_prediction_ignores_target_truth_and_preserves_inputs_and_eval_mode() -> None:
    data, volume = _prediction_inputs()
    data_arrays = (
        "normalized_time",
        "normalized_spatial",
        "observed_flat_indices",
        "normalized_observed_amplitudes",
    )
    original_data = {name: getattr(data, name).copy() for name in data_arrays}
    original_values = volume.values.copy()
    original_mask = volume.observed_trace_mask.copy()
    model = _CoordinateModel().eval()
    results = []
    for replacement in (np.nan, 0.0, 1.0e30):
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = replacement
        original_variant_values = values.copy()
        variant = replace(volume, values=values)
        result = predict_c3_volume_siren(model, data, variant, batch_size=4, device="cpu")
        results.append(result)
        np.testing.assert_array_equal(variant.values, original_variant_values)
        assert not model.training
    for result in results[1:]:
        np.testing.assert_array_equal(result.values, results[0].values)
        assert (
            result.observed_model_rmse_before_reinsertion
            == results[0].observed_model_rmse_before_reinsertion
        )
        assert (
            result.observed_model_max_abs_error_before_reinsertion
            == results[0].observed_model_max_abs_error_before_reinsertion
        )
    for name, original in original_data.items():
        np.testing.assert_array_equal(getattr(data, name), original)
    np.testing.assert_array_equal(volume.values, original_values)
    np.testing.assert_array_equal(volume.observed_trace_mask, original_mask)


@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_prediction_rejects_invalid_batch_size(batch_size) -> None:
    data, volume = _prediction_inputs()

    with pytest.raises(ValueError, match="batch_size"):
        predict_c3_volume_siren(
            _CoordinateModel(), data, volume, batch_size=batch_size, device="cpu"
        )


@pytest.mark.parametrize(
    "invalid", ["time", "spatial", "observed_indices", "model_width", "coordinates"]
)
def test_prediction_rejects_mismatched_data_and_model_contracts(invalid: str) -> None:
    data, volume = _prediction_inputs()
    model = _CoordinateModel()
    if invalid == "time":
        data = replace(data, normalized_time=data.normalized_time[:-1])
        message = "time count"
    elif invalid == "spatial":
        data = replace(data, normalized_spatial=data.normalized_spatial[:-1])
        message = "spatial shape"
    elif invalid == "observed_indices":
        data = replace(data, observed_flat_indices=np.array([0, 3], dtype=np.int64))
        message = "observed flat indices"
    elif invalid == "model_width":
        model.input_features = 5
        message = "input width"
    else:
        data = replace(
            data, model_coordinates=replace(data.model_coordinates, time_coordinate_scale=2.0)
        )
        message = "normalization and model coordinate"

    with pytest.raises(ValueError, match=message):
        predict_c3_volume_siren(model, data, volume, batch_size=4, device="cpu")
    assert not model.batch_sizes
