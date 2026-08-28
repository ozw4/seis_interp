from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.evaluation import streaming_snr
from seis_interp.evaluation.metrics import signal_to_noise_ratio_db
from seis_interp.evaluation.streaming_snr import evaluate_model_global_snr_by_ffid
from seis_interp.training.ffid_batches import (
    build_global_rms_trace_points,
    build_per_trace_rms_trace_points,
)
from seis_interp.training.prediction import predict_points


class CoordinateModel(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return 0.3 * coordinates[:, 0:1] + 0.1 * coordinates[:, 1:2] - 0.2 * coordinates[:, 2:3]


def _evaluation_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
]:
    time = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    spatial = np.arange(30, dtype=np.float64).reshape(6, 5) / 17.0
    amplitudes = np.asarray(
        [np.sin(time * (row + 1)) + 0.2 * row for row in range(len(spatial))],
        dtype=np.float32,
    )
    rows_by_ffid = {
        30: np.asarray([5]),
        10: np.asarray([2, 0]),
        20: np.asarray([4, 3, 1]),
    }
    return time, spatial, amplitudes, rows_by_ffid


def _materialized_reference(
    model: torch.nn.Module,
    time: np.ndarray,
    spatial: np.ndarray,
    amplitudes: np.ndarray,
    rows_by_ffid: dict[int, np.ndarray],
    *,
    amplitude_rms: float,
) -> float:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for ffid in sorted(rows_by_ffid):
        coordinates, ffid_targets = build_global_rms_trace_points(
            time,
            spatial,
            amplitudes,
            rows_by_ffid[ffid],
            amplitude_rms=amplitude_rms,
        )
        targets.append(ffid_targets)
        predictions.append(predict_points(model, coordinates, batch_size=100, device="cpu"))
    return signal_to_noise_ratio_db(
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def test_streaming_snr_matches_materialized_point_weighted_reference() -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    model = CoordinateModel()
    expected = _materialized_reference(
        model,
        time,
        spatial,
        amplitudes,
        rows_by_ffid,
        amplitude_rms=2.5,
    )

    actual = evaluate_model_global_snr_by_ffid(
        model,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=2.5,
        prediction_batch_size=3,
        device="cpu",
    )

    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


def test_per_trace_streaming_snr_matches_its_oracle_normalized_reference() -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    model = CoordinateModel()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for ffid in sorted(rows_by_ffid):
        coordinates, ffid_targets = build_per_trace_rms_trace_points(
            time,
            spatial,
            amplitudes,
            rows_by_ffid[ffid],
        )
        targets.append(ffid_targets)
        predictions.append(predict_points(model, coordinates, batch_size=100, device="cpu"))
    expected = signal_to_noise_ratio_db(
        np.concatenate(targets),
        np.concatenate(predictions),
    )

    actual = evaluate_model_global_snr_by_ffid(
        model,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=2.5,
        amplitude_scaling="per_trace_rms",
        prediction_batch_size=3,
        device="cpu",
    )

    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


def test_streaming_snr_accepts_read_only_memmap(tmp_path: Path) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    amplitude_path = tmp_path / "amplitudes.npy"
    np.save(amplitude_path, amplitudes)
    memory_mapped = np.load(amplitude_path, mmap_mode="r", allow_pickle=False)
    expected = _materialized_reference(
        CoordinateModel(),
        time,
        spatial,
        amplitudes,
        rows_by_ffid,
        amplitude_rms=1.75,
    )

    actual = evaluate_model_global_snr_by_ffid(
        CoordinateModel(),
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=memory_mapped,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=1.75,
        prediction_batch_size=2,
        device="cpu",
    )

    assert isinstance(memory_mapped, np.memmap)
    assert not memory_mapped.flags.writeable
    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


@pytest.mark.parametrize("starts_training", [True, False])
def test_streaming_snr_restores_model_training_state(starts_training: bool) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    model = CoordinateModel()
    model.train(starts_training)

    evaluate_model_global_snr_by_ffid(
        model,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=2.0,
        prediction_batch_size=3,
        device="cpu",
    )

    assert model.training is starts_training


def test_perfect_streaming_prediction_is_positive_infinity() -> None:
    time = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    spatial = np.zeros((3, 5), dtype=np.float64)
    spatial[:, 0] = [0.5, 1.0, -0.5]
    amplitudes = (time[np.newaxis, :] + spatial[:, :1]).astype(np.float32)

    class ExactSum(torch.nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates[:, 0:1] + coordinates[:, 1:2]

    result = evaluate_model_global_snr_by_ffid(
        ExactSum(),
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid={20: np.asarray([2]), 10: np.asarray([0, 1])},
        amplitude_rms=1.0,
        prediction_batch_size=2,
        device="cpu",
    )

    assert result == float("inf")


def test_streaming_snr_rejects_zero_energy_reference() -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()

    with pytest.raises(ValueError, match="reference energy"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=np.zeros_like(amplitudes),
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


def test_per_trace_streaming_snr_rejects_zero_rms_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, _ = _evaluation_inputs()
    amplitudes[4] = 0.0
    monkeypatch.setattr(
        streaming_snr,
        "predict_points",
        lambda *_args, **_kwargs: pytest.fail("prediction must not run for an invalid target"),
    )

    with pytest.raises(ValueError, match="array_row 4"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid={20: np.asarray([4])},
            amplitude_rms=1.0,
            amplitude_scaling="per_trace_rms",
            prediction_batch_size=3,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("rows_by_ffid", "message"),
    [
        ({}, "must not be empty"),
        ({10: np.asarray([], dtype=np.int64)}, "non-empty"),
        ({10: np.asarray([0, 0])}, "unique"),
        ({10: np.asarray([-1])}, "non-negative"),
        ({10: np.asarray([6])}, r"within \[0, 6\)"),
    ],
)
def test_streaming_snr_rejects_invalid_row_mappings(
    rows_by_ffid: dict[int, np.ndarray],
    message: str,
) -> None:
    time, spatial, amplitudes, _ = _evaluation_inputs()

    with pytest.raises(ValueError, match=message):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


def test_streaming_snr_rejects_rows_shared_across_ffid_groups() -> None:
    time, spatial, amplitudes, _ = _evaluation_inputs()
    with pytest.raises(ValueError, match="more than one FFID"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid={10: np.asarray([0, 1]), 20: np.asarray([1, 2])},
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


def test_streaming_snr_processes_sorted_ffids_without_survey_concatenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    call_sizes: list[int] = []
    first_spatial_values: list[float] = []

    def recording_predict(
        model: torch.nn.Module,
        coordinates: np.ndarray,
        *,
        batch_size: int,
        device: torch.device | str,
    ) -> np.ndarray:
        call_sizes.append(len(coordinates))
        first_spatial_values.append(float(coordinates[0, 1]))
        return np.zeros(len(coordinates), dtype=np.float32)

    monkeypatch.setattr(streaming_snr, "predict_points", recording_predict)

    result = evaluate_model_global_snr_by_ffid(
        CoordinateModel(),
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=1.0,
        prediction_batch_size=2,
        device="cpu",
    )

    # Sorted FFIDs 10, 20, 30 contain 2, 3, and 1 traces respectively.
    assert call_sizes == [8, 12, 4]
    assert first_spatial_values == [spatial[0, 0], spatial[1, 0], spatial[5, 0]]
    assert max(call_sizes) < sum(call_sizes)
    assert result == pytest.approx(0.0)


def test_streaming_snr_releases_one_ffid_before_building_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    original_builder = streaming_snr.build_global_rms_trace_points
    previous_references: list[weakref.ReferenceType[np.ndarray]] = []
    release_checks: list[bool] = []

    def recording_builder(*args, **kwargs):
        if previous_references:
            release_checks.append(all(reference() is None for reference in previous_references))
        coordinates, targets = original_builder(*args, **kwargs)
        previous_references[:] = [weakref.ref(coordinates), weakref.ref(targets)]
        return coordinates, targets

    def recording_predict(
        model: torch.nn.Module,
        coordinates: np.ndarray,
        *,
        batch_size: int,
        device: torch.device | str,
    ) -> np.ndarray:
        predictions = np.zeros(len(coordinates), dtype=np.float32)
        previous_references.append(weakref.ref(predictions))
        return predictions

    monkeypatch.setattr(streaming_snr, "build_global_rms_trace_points", recording_builder)
    monkeypatch.setattr(streaming_snr, "predict_points", recording_predict)

    evaluate_model_global_snr_by_ffid(
        CoordinateModel(),
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=1.0,
        prediction_batch_size=2,
        device="cpu",
    )

    assert release_checks == [True, True]


def test_streaming_snr_float64_energy_work_is_bounded_by_prediction_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    original_converter = streaming_snr._float64_energy_slice
    converted_sizes: list[int] = []

    def recording_converter(values: np.ndarray, *, name: str) -> np.ndarray:
        converted_sizes.append(len(values))
        return original_converter(values, name=name)

    monkeypatch.setattr(streaming_snr, "_float64_energy_slice", recording_converter)

    evaluate_model_global_snr_by_ffid(
        CoordinateModel(),
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        amplitudes=amplitudes,
        rows_by_ffid=rows_by_ffid,
        amplitude_rms=1.0,
        prediction_batch_size=3,
        device="cpu",
    )

    assert converted_sizes
    assert max(converted_sizes) <= 3


def test_streaming_snr_rejects_non_finite_predictions() -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()

    class NanModel(torch.nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return torch.full((len(coordinates), 1), torch.nan, device=coordinates.device)

    with pytest.raises(ValueError, match="predictions contain non-finite"):
        evaluate_model_global_snr_by_ffid(
            NanModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


def test_streaming_snr_rejects_invalid_source_shapes_and_non_finite_coordinates() -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    with pytest.raises(ValueError, match="amplitudes shape"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes[:, :-1],
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )

    non_finite_spatial = spatial.copy()
    non_finite_spatial[0, 0] = np.nan
    with pytest.raises(ValueError, match="selected normalized spatial"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=non_finite_spatial,
            amplitudes=amplitudes,
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


def test_streaming_snr_rejects_prediction_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    monkeypatch.setattr(
        streaming_snr,
        "predict_points",
        lambda *args, **kwargs: np.zeros(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="prediction shape"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=3,
            device="cpu",
        )


@pytest.mark.parametrize("prediction_batch_size", [0, -1, True])
def test_streaming_snr_rejects_invalid_prediction_batch_size(
    prediction_batch_size: int,
) -> None:
    time, spatial, amplitudes, rows_by_ffid = _evaluation_inputs()
    with pytest.raises(ValueError, match="prediction_batch_size"):
        evaluate_model_global_snr_by_ffid(
            CoordinateModel(),
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            amplitudes=amplitudes,
            rows_by_ffid=rows_by_ffid,
            amplitude_rms=1.0,
            prediction_batch_size=prediction_batch_size,
            device="cpu",
        )
