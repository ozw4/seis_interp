import numpy as np
import pytest

from seis_interp.evaluation.trace_snr import summarize_trace_snr, trace_snr_db


def test_trace_snr_is_mean_of_db_not_global_energy_ratio():
    reference = np.array([[1, 1], [10, 10]], dtype=np.float32)
    prediction = np.array([[0.5, 0.5], [9, 9]], dtype=np.float32)
    scores = trace_snr_db(reference, prediction)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(scores, [10 * np.log10(4), 20])
    assert summarize_trace_snr(scores)["mean_trace_snr_db"] == pytest.approx(scores.mean())
    assert scores.mean() != pytest.approx(10 * np.log10(202 / 2.5))


@pytest.mark.parametrize(
    "reference,prediction,status",
    [
        ([1], [1], "positive_infinity"),
        ([0], [1], "negative_infinity"),
        ([0], [0], "undefined"),
        ([1, 0], [1, 1], "undefined"),
    ],
)
def test_nonfinite_scores_are_reported_without_exclusion(reference, prediction, status):
    scores = trace_snr_db(np.array(reference)[:, None], np.array(prediction)[:, None])
    result = summarize_trace_snr(scores)
    assert result["mean_trace_snr_db"] is None
    assert result["mean_trace_snr_status"] == status
    assert result["trace_snr_trace_count"] == len(reference)


@pytest.mark.parametrize("reference,prediction", [([], []), ([[1]], [[1, 2]]), ([[np.nan]], [[0]])])
def test_bad_trace_inputs_fail(reference, prediction):
    with pytest.raises(ValueError):
        trace_snr_db(reference, prediction)
