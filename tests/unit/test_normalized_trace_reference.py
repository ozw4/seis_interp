import json

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation.normalized_trace_reference import (
    evaluate_normalized_trace_reference,
    unit_rms_reference_traces,
)


def test_stable_unit_rms_and_zero_energy():
    rows = np.array([[0, 0], [1e300, -1e300], [1e-300, -1e-300]])
    np.testing.assert_array_equal(unit_rms_reference_traces(rows), [[0, 0], [1, -1], [1, -1]])


@pytest.mark.parametrize("bad", [np.zeros((0, 2)), np.zeros(2), [[np.nan, 1]], [[np.inf, 1]]])
def test_reject_invalid_reference(bad):
    with pytest.raises(ValueError):
        unit_rms_reference_traces(bad)


@pytest.mark.parametrize("chunk", [1, 2, 10])
def test_reference_score_ignores_physical_gain_but_penalizes_normalized_model_gain(tmp_path, chunk):
    values = np.array([[2, 10, 0], [-2, -10, 0]], dtype=np.float32).reshape(2, 1, 1, 1, 3)
    observed_mask = np.array([True, False, False]).reshape(1, 1, 1, 3)
    observed = ObservedC3Volume(
        values=values,
        time_s=np.array([0.0, 1.0]),
        array_rows=np.arange(3).reshape(1, 1, 1, 3),
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=~observed_mask,
    )
    reference = np.array([[np.nan, np.nan], [10, -10], [20, -20]], dtype=np.float64)
    np.save(tmp_path / "amplitudes.npy", reference)
    scale = np.array([2, 5, 100], dtype=np.float64).reshape(1, 1, 1, 3)
    normalized = np.array([[1, -1], [0.5, -0.5], [0.5, -0.5]])
    prediction = (normalized * scale.ravel()[:, None]).T.reshape(values.shape)
    result = evaluate_normalized_trace_reference(
        prediction,
        observed,
        scale,
        interim_dir=tmp_path,
        volume_metadata={"selection": {"time": [0, 2]}},
        chunk_size=chunk,
    )
    assert result["evaluation_target"]["snr_db"] == pytest.approx(10 * np.log10(4))
    assert result["evaluation_target"]["rmse"] == 0.5
    assert result["evaluation_target"]["trace_count"] == 2
    assert result["prediction_self_normalized"] is False
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(np.load(tmp_path / "amplitudes.npy"), reference)
    with pytest.raises(ValueError, match="positive"):
        evaluate_normalized_trace_reference(
            prediction,
            observed,
            scale * 0,
            interim_dir=tmp_path,
            volume_metadata={"selection": {"time": [0, 2]}},
        )
    corrupted = prediction.copy()
    corrupted[..., 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_normalized_trace_reference(
            corrupted,
            observed,
            scale,
            interim_dir=tmp_path,
            volume_metadata={"selection": {"time": [0, 2]}},
        )
