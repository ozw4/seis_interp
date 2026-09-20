import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.forge_mvp_metrics import (
    evaluate_traces,
    paired_geometry_effects,
    validate_predictions,
    verify_saved_metrics,
)


def test_original_amplitude_metrics_match_independent_calculation():
    target = np.arange(1, 41, dtype=np.float32).reshape(8, 5)
    prediction = target * np.arange(1, 9, dtype=np.float32)[:, None]
    index = pd.DataFrame(
        {
            "trace_id": [str(i) for i in range(8)],
            "cell_id": np.arange(8),
            "clean_target": [True] * 6 + [False] * 2,
            "source_projection_distance_m": np.arange(8),
            "receiver_projection_distance_m": np.arange(8) / 10,
        }
    )
    metrics, rows = evaluate_traces(prediction, target, index)
    assert metrics["clean_target"]["masked_trace_relative_mse"] == pytest.approx(
        np.mean(np.arange(6) ** 2)
    )
    assert metrics["all_eligible"]["trace_count"] == 8
    expected_nmse = (
        np.square(prediction.astype(float) - target).sum() / np.square(target.astype(float)).sum()
    )
    assert metrics["all_eligible"]["global_nmse"] == pytest.approx(expected_nmse)
    assert metrics["all_eligible"]["correlation"]["median"] == pytest.approx(1)
    verify_saved_metrics(metrics, rows)
    metrics["clean_target"]["global_nmse"] += 0.1
    with pytest.raises(ValueError, match="recomputation"):
        verify_saved_metrics(metrics, rows)
    grid = rows.copy()
    grid["relative_mse"] += 1
    pair, effect, quartiles = paired_geometry_effects(rows, grid.iloc[::-1])
    assert pair.error_difference.eq(-1).all()
    assert effect["clean_target"]["fraction_real_lower_error"] == 1
    assert len(quartiles) == 7


def test_zero_energy_and_undefined_correlations_use_existing_contract():
    y = np.zeros((2, 3), dtype=np.float32)
    p = np.ones_like(y)
    metrics, rows = evaluate_traces(p, y, pd.DataFrame({"clean_target": [True, True]}))
    assert rows.relative_mse.eq(1).all()
    assert metrics["clean_target"]["undefined_correlation_count"] == 2
    assert metrics["clean_target"]["global_nmse"] is None


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "time"])
def test_prediction_coverage_gate(fault):
    mask = pd.DataFrame({"cell_id": [0, 1, 2], "split": ["test", "observed", "test"]})
    cells = np.array([2, 0])
    prediction = np.ones((2, 3), dtype=np.float32)
    if fault == "missing":
        cells[0] = 1
    if fault == "duplicate":
        cells[0] = 0
    if fault == "nan":
        prediction[0, 0] = np.nan
    if fault == "time":
        prediction = prediction[:, :2]
    with pytest.raises(ValueError):
        validate_predictions(prediction, cells, mask, 3)
