from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.evaluation.metrics import signal_to_noise_ratio_db

_TIME_SELECTION = (2, 5)
_ARRAY_ROWS = np.array([[[[4, 1, 5, 0]]]], dtype=np.int64)
_OBSERVED_MASK = np.array([[[[True, False, True, False]]]], dtype=np.bool_)


def _volume(amplitudes: np.ndarray) -> ObservedC3Volume:
    time_start, time_stop = _TIME_SELECTION
    time_count = time_stop - time_start
    values = np.zeros((time_count, *_ARRAY_ROWS.shape), dtype=np.float32)
    flat_rows = _ARRAY_ROWS.reshape(-1)
    values[:, _OBSERVED_MASK] = amplitudes[
        flat_rows[_OBSERVED_MASK.reshape(-1)],
        time_start:time_stop,
    ].T
    return ObservedC3Volume(
        values=values,
        time_s=np.arange(time_start, time_stop, dtype=np.float64) * 0.008,
        array_rows=_ARRAY_ROWS.copy(),
        observed_trace_mask=_OBSERVED_MASK.copy(),
        evaluation_target_trace_mask=~_OBSERVED_MASK,
    )


def _trace_predictions(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).T.reshape((values.shape[1], 1, 1, 1, values.shape[0]))


def _metadata(time_selection: tuple[int, int] = _TIME_SELECTION) -> dict[str, object]:
    return {"selection": {"time": list(time_selection)}}


def _write_amplitudes(tmp_path: Path, amplitudes: np.ndarray) -> None:
    np.save(tmp_path / "amplitudes.npy", amplitudes)


def _metric_example(tmp_path: Path) -> tuple[np.ndarray, ObservedC3Volume]:
    amplitudes = np.full((7, 8), np.nan, dtype=np.float32)
    amplitudes[4, 2:5] = [2.0, 3.0, 4.0]
    amplitudes[5, 2:5] = [-2.0, -3.0, -4.0]
    amplitudes[1, 2:5] = [10.0, -10.0, 10.0]
    amplitudes[0, 2:5] = [1.0, 1.0, 1.0]
    _write_amplitudes(tmp_path, amplitudes)

    traces = amplitudes[_ARRAY_ROWS.reshape(-1), 2:5].copy()
    traces[1] = [9.0, -9.0, 9.0]
    traces[3] = [0.0, 0.0, 0.0]
    return _trace_predictions(traces), _volume(amplitudes)


def test_global_target_metrics_match_hand_calculation_and_existing_snr(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)
    target_reference = np.array([[10.0, -10.0, 10.0], [1.0, 1.0, 1.0]])
    target_prediction = np.array([[9.0, -9.0, 9.0], [0.0, 0.0, 0.0]])

    result = evaluate_c3_volume_prediction(
        prediction,
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata(),
    )

    target = result["evaluation_target"]
    assert result["evaluation_domain"] == "evaluation_target"
    assert result["amplitude_domain"] == "physical"
    assert target["snr_status"] == "finite"
    numeric_target = {key: value for key, value in target.items() if key != "snr_status"}
    assert numeric_target == pytest.approx(
        {
            "trace_count": 2,
            "sample_count": 6,
            "reference_energy": 303.0,
            "error_energy": 6.0,
            "snr_db": signal_to_noise_ratio_db(target_reference, target_prediction),
            "rmse": 1.0,
            "relative_l2": math.sqrt(6.0 / 303.0),
            "mean_trace_relative_mse": (0.01 + 1.0) / 2.0,
            "covered_target_trace_count": 2,
            "target_trace_count": 2,
        }
    )
    assert result["zero_fill"] == {
        "snr_db": 0.0,
        "snr_status": "finite",
        "rmse": pytest.approx(math.sqrt(303.0 / 6.0)),
    }
    assert result["observed_max_abs_error"] == 0.0
    json.dumps(result, allow_nan=False)


def test_observed_error_does_not_change_target_metrics(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)
    changed_observed = prediction.copy()
    changed_observed[:, volume.observed_trace_mask] += 7.0

    baseline = evaluate_c3_volume_prediction(
        prediction,
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata(),
    )
    changed = evaluate_c3_volume_prediction(
        changed_observed,
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata(),
    )

    assert changed["evaluation_target"] == baseline["evaluation_target"]
    assert changed["zero_fill"] == baseline["zero_fill"]
    assert baseline["observed_max_abs_error"] == 0.0
    assert changed["observed_max_abs_error"] == pytest.approx(7.0)


def test_explicit_coverage_must_include_every_target(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)
    coverage = np.ones_like(volume.evaluation_target_trace_mask)
    coverage.reshape(-1)[-1] = False

    with pytest.raises(ValueError, match="cover every evaluation target"):
        evaluate_c3_volume_prediction(
            prediction,
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
            target_coverage_mask=coverage,
        )


@pytest.mark.parametrize(
    "coverage",
    [
        np.ones((1, 1, 1, 3), dtype=np.bool_),
        np.ones((1, 1, 1, 4), dtype=np.int8),
    ],
)
def test_rejects_invalid_target_coverage_mask(
    tmp_path: Path,
    coverage: np.ndarray,
) -> None:
    prediction, volume = _metric_example(tmp_path)

    with pytest.raises(ValueError, match="target coverage mask"):
        evaluate_c3_volume_prediction(
            prediction,
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
            target_coverage_mask=coverage,
        )


class _GuardedAmplitudes(np.ndarray):
    reads: list[tuple[np.ndarray, slice]]

    def __new__(
        cls,
        values: np.ndarray,
        reads: list[tuple[np.ndarray, slice]],
    ) -> _GuardedAmplitudes:
        instance = np.asarray(values).view(cls)
        instance.reads = reads
        return instance

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        if source is not None:
            self.reads = getattr(source, "reads", [])

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple) and len(key) == 2:
            rows, time_selection = key
            if isinstance(rows, np.ndarray) and isinstance(time_selection, slice):
                self.reads.append((rows.copy(), time_selection))
        return super().__getitem__(key)


def test_reads_only_target_rows_and_selected_time_in_fixed_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_count = 1027
    array_rows = np.arange(trace_count + 20, 20, -1, dtype=np.int64).reshape(1, 1, 1, -1)
    observed_mask = np.zeros(array_rows.shape, dtype=np.bool_)
    observed_mask[..., 0] = True
    target_mask = ~observed_mask
    values = np.zeros((2, *array_rows.shape), dtype=np.float32)
    volume = ObservedC3Volume(
        values=values,
        time_s=np.array([0.016, 0.024]),
        array_rows=array_rows,
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=target_mask,
    )
    prediction = values.copy()
    amplitudes = np.full((trace_count + 21, 7), np.nan, dtype=np.float32)
    target_rows = array_rows[target_mask]
    target_reference = np.full((len(target_rows), 2), 2.0, dtype=np.float32)
    target_reference[1024:] = 10.0
    target_prediction = np.full((len(target_rows), 2), 1.5, dtype=np.float32)
    target_prediction[1024:] = 0.0
    amplitudes[target_rows, 2:4] = target_reference
    prediction[:, target_mask] = target_prediction.T
    reads: list[tuple[np.ndarray, slice]] = []
    guarded = _GuardedAmplitudes(amplitudes, reads)

    def load_guarded(path: Path, *, mmap_mode: str, allow_pickle: bool) -> np.ndarray:
        assert Path(path) == tmp_path / "amplitudes.npy"
        assert mmap_mode == "r"
        assert allow_pickle is False
        return guarded

    monkeypatch.setattr("seis_interp.evaluation.c3_volume_metrics.np.load", load_guarded)

    result = evaluate_c3_volume_prediction(
        prediction,
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata((2, 4)),
    )

    assert [len(rows) for rows, _ in reads] == [1024, 2]
    assert np.array_equal(np.concatenate([rows for rows, _ in reads]), target_rows)
    assert all(time_selection == slice(2, 4) for _, time_selection in reads)
    assert result["evaluation_target"]["trace_count"] == 1026
    assert result["evaluation_target"]["snr_db"] == pytest.approx(
        signal_to_noise_ratio_db(target_reference, target_prediction)
    )


@pytest.mark.parametrize(
    ("reference", "prediction", "status", "relative_l2"),
    [
        (np.array([3.0, -4.0, 1.0]), np.array([3.0, -4.0, 1.0]), "perfect_reconstruction", 0.0),
        (np.zeros(3), np.ones(3), "undefined_zero_reference", None),
    ],
)
def test_snr_special_cases_are_strict_json_safe(
    tmp_path: Path,
    reference: np.ndarray,
    prediction: np.ndarray,
    status: str,
    relative_l2: float | None,
) -> None:
    amplitudes = np.ones((7, 8), dtype=np.float32)
    amplitudes[1, 2:5] = reference
    amplitudes[0, 2:5] = reference
    _write_amplitudes(tmp_path, amplitudes)
    volume = _volume(amplitudes)
    traces = amplitudes[_ARRAY_ROWS.reshape(-1), 2:5].copy()
    traces[1] = prediction
    traces[3] = prediction

    result = evaluate_c3_volume_prediction(
        _trace_predictions(traces),
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata(),
    )

    target = result["evaluation_target"]
    assert target["snr_db"] is None
    assert target["snr_status"] == status
    assert target["relative_l2"] == relative_l2
    expected_trace_relative_mse = 0.0 if status == "perfect_reconstruction" else 1.0
    assert target["mean_trace_relative_mse"] == expected_trace_relative_mse
    if status == "undefined_zero_reference":
        assert result["zero_fill"] == {
            "snr_db": None,
            "snr_status": "undefined_zero_reference",
            "rmse": 0.0,
        }
    json.dumps(result, allow_nan=False)


def test_rejects_empty_target_set_before_reading_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amplitudes = np.ones((7, 8), dtype=np.float32)
    volume = _volume(amplitudes)
    all_observed = np.ones_like(volume.observed_trace_mask)
    volume = replace(
        volume,
        observed_trace_mask=all_observed,
        evaluation_target_trace_mask=~all_observed,
    )
    monkeypatch.setattr(
        "seis_interp.evaluation.c3_volume_metrics.np.load",
        lambda *_args, **_kwargs: pytest.fail("reference must not be read for an empty target"),
    )

    with pytest.raises(ValueError, match="at least one"):
        evaluate_c3_volume_prediction(
            volume.values.copy(),
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_rejects_nonfinite_target_prediction(tmp_path: Path, bad_value: float) -> None:
    prediction, volume = _metric_example(tmp_path)
    prediction[0, 0, 0, 0, 1] = bad_value

    with pytest.raises(ValueError, match="target predictions.*non-finite"):
        evaluate_c3_volume_prediction(
            prediction,
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
        )


def test_rejects_nonfinite_target_reference(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)
    amplitudes = np.load(tmp_path / "amplitudes.npy")
    amplitudes[1, 3] = np.nan
    _write_amplitudes(tmp_path, amplitudes)

    with pytest.raises(ValueError, match="reference amplitudes.*non-finite"):
        evaluate_c3_volume_prediction(
            prediction,
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
        )


def test_rejects_prediction_shape_and_mask_correspondence(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)
    with pytest.raises(ValueError, match="shape must match"):
        evaluate_c3_volume_prediction(
            prediction[:-1],
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
        )

    overlapping = replace(
        volume,
        evaluation_target_trace_mask=np.ones_like(volume.evaluation_target_trace_mask),
    )
    with pytest.raises(ValueError, match="disjointly cover"):
        evaluate_c3_volume_prediction(
            prediction,
            overlapping,
            interim_dir=tmp_path,
            volume_metadata=_metadata(),
        )


def test_rejects_time_selection_outside_source_amplitudes(tmp_path: Path) -> None:
    prediction, volume = _metric_example(tmp_path)

    with pytest.raises(ValueError, match="outside the source"):
        evaluate_c3_volume_prediction(
            prediction,
            volume,
            interim_dir=tmp_path,
            volume_metadata=_metadata((7, 10)),
        )


def test_zero_energy_trace_uses_divisor_one_and_boundary_targets_are_counted(
    tmp_path: Path,
) -> None:
    amplitudes = np.ones((7, 8), dtype=np.float32)
    amplitudes[4, 2:5] = 0.0
    amplitudes[0, 2:5] = [1.0, -1.0, 1.0]
    _write_amplitudes(tmp_path, amplitudes)
    volume = _volume(amplitudes)
    boundary_targets = np.array([True, False, False, True]).reshape(_ARRAY_ROWS.shape)
    volume = replace(
        volume,
        observed_trace_mask=~boundary_targets,
        evaluation_target_trace_mask=boundary_targets,
    )
    traces = amplitudes[_ARRAY_ROWS.reshape(-1), 2:5].copy()
    traces[0] = [2.0, -2.0, 2.0]
    traces[-1] = 0.0

    result = evaluate_c3_volume_prediction(
        _trace_predictions(traces),
        volume,
        interim_dir=tmp_path,
        volume_metadata=_metadata(),
        target_coverage_mask=np.ones_like(boundary_targets),
    )["evaluation_target"]

    assert result["trace_count"] == 2
    assert result["covered_target_trace_count"] == 2
    assert result["target_trace_count"] == 2
    assert result["mean_trace_relative_mse"] == pytest.approx((4.0 + 1.0) / 2.0)
