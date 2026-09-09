from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import (
    ModelCoordinateParameters,
    model_coordinate_parameters,
    normalize_training_trace_time_offsets,
)
from seis_interp.training import c3_volume_siren_prediction as prediction_module
from seis_interp.training.c3_volume_siren_data import C3VolumeSirenData
from seis_interp.training.c3_volume_siren_prediction import predict_c3_volume_siren
from seis_interp.training.checkpoints import (
    load_fixed_step_siren_checkpoint,
    save_fixed_step_siren_checkpoint,
)


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


class _CartesianCoordinateModel(_CoordinateModel):
    input_features = 5

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        assert not self.training
        self.batch_sizes.append(len(coordinates))
        return (
            2 * coordinates[:, 0]
            + 3 * coordinates[:, 1]
            - coordinates[:, 2]
            + 0.5 * coordinates[:, 3]
            + coordinates[:, 4]
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


def _cartesian_prediction_inputs() -> tuple[C3VolumeSirenData, ObservedC3Volume]:
    data, volume = _prediction_inputs()
    return replace(
        data,
        normalized_time=data.normalized_time * 3.0,
        normalized_spatial=data.normalized_spatial[:, :4].copy(),
        model_coordinates=model_coordinate_parameters(
            "cmp_cartesian_half_offset", data.normalization, time_coordinate_scale=3.0
        ),
    ), volume


def test_cartesian_prediction_uses_five_coordinates_and_scaled_time_before_physical_restore():
    data, volume = _cartesian_prediction_inputs()
    model = _CartesianCoordinateModel()
    result = predict_c3_volume_siren(model, data, volume, batch_size=4, device="cpu")
    constants = np.array([-2.5, -0.75, -1.0, 1.75, 3.5, 1.625])
    expected = 2.0 * (constants[:, None] + np.array([-6.0, 0.0, 6.0]))
    expected[0], expected[4] = 2.0, -2.0
    np.testing.assert_array_equal(result.values, expected.T.reshape(volume.values.shape))
    assert max(model.batch_sizes) <= 4


@pytest.mark.parametrize("invalid", ["width", "bounds", "time_scale"])
def test_cartesian_prediction_rejects_inconsistent_metadata_before_forward(invalid):
    data, volume = _cartesian_prediction_inputs()
    model = _CartesianCoordinateModel()
    message = "normalization and model coordinate"
    if invalid == "width":
        model.input_features = 6
        message = "input width"
    elif invalid == "bounds":
        data = replace(
            data,
            model_coordinates=replace(
                data.model_coordinates,
                coordinate_scale_max=(1.0, 2.0, 1.0, 0.5, 0.5),
            ),
        )
    else:
        data = replace(
            data, model_coordinates=replace(data.model_coordinates, time_coordinate_scale=1.0)
        )
    with pytest.raises(ValueError, match=message):
        predict_c3_volume_siren(model, data, volume, batch_size=4, device="cpu")
    assert not model.batch_sizes


@pytest.mark.parametrize("shear", [0.0, 0.5])
def test_cartesian_scaled_time_checkpoint_restores_identical_volume_prediction(
    tmp_path: Path, shear
):
    data, volume = _cartesian_prediction_inputs()
    coordinates = replace(data.model_coordinates, relative_receiver_y_time_shear_s_per_m=shear)
    data = replace(
        data,
        model_coordinates=coordinates,
        normalized_time_offsets=normalize_training_trace_time_offsets(
            data.normalized_spatial, coordinates
        ),
    )
    model = Siren(input_features=5, hidden_width=4, hidden_layers=1)
    original = predict_c3_volume_siren(model, data, volume, batch_size=4, device="cpu")
    path = tmp_path / "final.pt"
    save_fixed_step_siren_checkpoint(
        path,
        model,
        data.normalization,
        data.model_coordinates,
        global_step=2,
        final_batch_loss=0.5,
    )
    loaded = load_fixed_step_siren_checkpoint(path)
    restored = predict_c3_volume_siren(
        loaded.model,
        replace(
            data, normalization=loaded.normalization, model_coordinates=loaded.model_coordinates
        ),
        volume,
        batch_size=4,
        device="cpu",
    )
    assert loaded.model_coordinates == data.model_coordinates
    np.testing.assert_array_equal(restored.values, original.values)


@pytest.mark.parametrize("shear", [0.5, -0.5])
def test_sheared_prediction_time_sign_and_physical_amplitude_match_hand_calculation(shear):
    data, volume = _cartesian_prediction_inputs()
    baseline = predict_c3_volume_siren(
        _CartesianCoordinateModel(), data, volume, batch_size=4, device="cpu"
    )
    coordinates = replace(data.model_coordinates, relative_receiver_y_time_shear_s_per_m=shear)
    # half-offset scale=.5, time span=1, time factor=3: offset=-6*s*normalized_hy.
    offsets = -6.0 * shear * np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    sheared = replace(data, model_coordinates=coordinates, normalized_time_offsets=offsets)
    expected = baseline.values.reshape(3, -1).copy()
    expected += 4.0 * offsets[None, :]  # Model time coefficient2 and physical RMS2.
    expected[:, volume.observed_trace_mask.ravel()] = volume.values[:, volume.observed_trace_mask]
    for poison in (np.nan, 0.0, 1e30):
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = poison
        result = predict_c3_volume_siren(
            _CartesianCoordinateModel(),
            sheared,
            replace(volume, values=values),
            batch_size=4,
            device="cpu",
        )
        np.testing.assert_array_equal(result.values, expected.reshape(volume.values.shape))


@pytest.mark.parametrize(
    "invalid", ["missing", "shape", "dtype", "nan", "sign", "row_order", "zero_metadata"]
)
def test_sheared_prediction_rejects_inconsistent_offsets_before_forward(invalid):
    data, volume = _cartesian_prediction_inputs()
    coordinates = replace(data.model_coordinates, relative_receiver_y_time_shear_s_per_m=0.5)
    offsets = -3.0 * np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    if invalid == "missing":
        offsets = None
    elif invalid == "shape":
        offsets = offsets[:-1]
    elif invalid == "dtype":
        offsets = offsets.astype(np.float32)
    elif invalid == "nan":
        offsets[1] = np.nan
    elif invalid == "sign":
        offsets = -offsets
    elif invalid == "row_order":
        offsets = offsets[::-1].copy()
    else:
        coordinates = data.model_coordinates
    model = _CartesianCoordinateModel()
    with pytest.raises(ValueError, match="normalized_time_offsets"):
        predict_c3_volume_siren(
            model,
            replace(data, model_coordinates=coordinates, normalized_time_offsets=offsets),
            volume,
            batch_size=4,
            device="cpu",
        )
    assert not model.batch_sizes


_INTERPOLATION = {"neighbors": 8, "power": 2.0, "distance_scales_m": [160.0, 80.0, 40.0, 40.0]}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch_size", [2, 12])
def test_per_trace_prediction_restores_varied_physical_scales_before_observed_reinsertion(
    dtype, batch_size
):
    data, volume = _prediction_inputs(dtype)
    scales = np.array([2.0, 0.0, 3.0, 4.0, 2.0, 10.0])
    data = replace(
        data,
        amplitude_scaling="per_trace_rms",
        trace_amplitude_scales=scales,
        trace_array_rows=volume.array_rows.reshape(-1).copy(),
        scale_interpolation=_INTERPOLATION,
    )
    before_reinsertion = (
        np.array(
            [
                [-6.5, -4.5, -2.5],
                [-2.75, -0.75, 1.25],
                [-1.0, 1.0, 3.0],
                [-0.25, 1.75, 3.75],
                [-0.5, 1.5, 3.5],
                [-0.375, 1.625, 3.625],
            ]
        )
        * scales[:, None]
    )
    expected = before_reinsertion.astype(dtype)
    expected[0], expected[4] = 2.0, -2.0
    baseline = None
    for target_value in [np.nan, 0.0, 1e30]:
        values = volume.values.copy()
        values[:, volume.evaluation_target_trace_mask] = target_value
        result = predict_c3_volume_siren(
            _CoordinateModel(),
            data,
            replace(volume, values=values),
            batch_size=batch_size,
            device="cpu",
        )
        np.testing.assert_array_equal(result.values, expected.T.reshape(values.shape))
        assert result.observed_model_rmse_before_reinsertion == pytest.approx(np.sqrt(502.0 / 6.0))
        assert result.observed_model_max_abs_error_before_reinsertion == 15.0
        if baseline is not None:
            np.testing.assert_array_equal(result.values, baseline)
        baseline = result.values
    np.testing.assert_array_equal(data.trace_amplitude_scales, scales)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"trace_amplitude_scales": None}, "trace_amplitude_scales"),
        ({"trace_amplitude_scales": np.ones(5)}, "trace_amplitude_scales"),
        ({"trace_amplitude_scales": np.ones(6, dtype=np.float32)}, "trace_amplitude_scales"),
        ({"trace_amplitude_scales": np.array([1, 1, 1, 1, 1, -1.0])}, "trace_amplitude_scales"),
        ({"trace_amplitude_scales": np.array([1, 1, 1, 1, 1, np.inf])}, "trace_amplitude_scales"),
        ({"scale_interpolation": None}, "scale_interpolation"),
        ({"trace_array_rows": None}, "trace_array_rows"),
        ({"trace_array_rows": np.array([2, 10, 30, 7, 9, 1])}, "trace_array_rows"),
        ({"trace_array_rows": np.array([10, 2, 30, 7, 9, 9])}, "trace_array_rows"),
        ({"trace_array_rows": np.array([10, 2, 30, 7, 9, 1], dtype=np.int32)}, "trace_array_rows"),
        (
            {"amplitude_scaling": "train_global_rms", "scale_interpolation": None},
            "trace_amplitude_scales",
        ),
    ],
)
def test_prediction_rejects_inconsistent_trace_scale_metadata_before_model_forward(change, match):
    data, volume = _prediction_inputs()
    data = replace(
        data,
        amplitude_scaling="per_trace_rms",
        trace_amplitude_scales=np.ones(6),
        trace_array_rows=volume.array_rows.reshape(-1).copy(),
        scale_interpolation=_INTERPOLATION,
    )
    model = _CoordinateModel()
    with pytest.raises(ValueError, match=match):
        predict_c3_volume_siren(model, replace(data, **change), volume, batch_size=4, device="cpu")
    assert not model.batch_sizes
