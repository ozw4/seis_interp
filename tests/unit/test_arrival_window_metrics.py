"""The pre-arrival window is geometry-only, and the split separates the two failures."""

import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.arrival_window_metrics import (
    ARRIVAL_WINDOWS,
    arrival_window_ends,
    windowed_waveform_columns,
)
from seis_interp.evaluation.study067_trace_metrics import waveform_columns
from seis_interp.pipelines.evaluate_arrival_windows import summarize

DT = 0.004


def test_the_window_follows_the_declared_line_and_nothing_else():
    """Offsets alone decide it: no waveform is read, so no target can leak into it."""
    offsets = np.array([0.0, 1100.0, 2200.0])
    ends = arrival_window_ends(
        offsets,
        intercept_s=0.22,
        velocity_m_per_s=5500.0,
        sample_interval_s=DT,
        sample_count=1001,
    )
    assert ends.tolist() == [55, 105, 155]
    # A faster declared velocity is the conservative direction: it shortens the window.
    faster = arrival_window_ends(
        offsets,
        intercept_s=0.22,
        velocity_m_per_s=11000.0,
        sample_interval_s=DT,
        sample_count=1001,
    )
    assert (faster <= ends).all() and faster[-1] < ends[-1]


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"intercept_s": 0.01}, "below the required"),
        ({"intercept_s": 5.0}, "no post-arrival samples"),
        ({"velocity_m_per_s": 0.0}, "velocity_m_per_s must be positive"),
        ({"intercept_s": -0.1}, "intercept_s must be nonnegative"),
        ({"sample_interval_s": 0.0}, "sample_interval_s must be positive"),
    ],
)
def test_an_unusable_window_is_refused_rather_than_shortened(overrides, message):
    settings = {
        "intercept_s": 0.22,
        "velocity_m_per_s": 5500.0,
        "sample_interval_s": DT,
        "sample_count": 1001,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        arrival_window_ends(np.array([0.0, 2200.0]), **settings)


def test_negative_offsets_are_refused():
    with pytest.raises(ValueError, match="nonnegative distances"):
        arrival_window_ends(
            np.array([-1.0]),
            intercept_s=0.22,
            velocity_m_per_s=5500.0,
            sample_interval_s=DT,
            sample_count=1001,
        )


def _split_case():
    """One trace that leaks before the arrival and one that only misses the signal."""
    target = np.zeros((2, 40))
    target[:, 20:] = np.sin(np.linspace(0, 6, 20))
    leaking = np.array(target[0])
    leaking[:20] = 0.5
    missing = np.zeros(40)
    return target, np.stack([leaking, missing])


def test_the_split_separates_leakage_from_a_missed_signal():
    target, prediction = _split_case()
    ends = np.full(2, 20, dtype=np.int64)
    columns = windowed_waveform_columns(prediction, target, ends)
    # The leaking trace reproduces the signal exactly and only errs before the arrival.
    assert columns["post_error_energy"][0] == pytest.approx(0.0)
    assert columns["post_snr_db"][0] == np.inf
    assert columns["pre_prediction_energy"][0] == pytest.approx(20 * 0.25)
    # The silent trace leaks nothing and misses everything.
    assert columns["pre_prediction_energy"][1] == 0.0
    assert columns["post_relative_mse"][1] == pytest.approx(1.0)
    assert columns["post_snr_db"][1] == pytest.approx(0.0)
    # A whole-trace score cannot tell them apart in the same way: both are nonzero error
    # for one and zero for the other only because the split assigns the error to a part.
    whole = waveform_columns(prediction, target)
    assert whole["error_energy"][0] == pytest.approx(columns["pre_prediction_energy"][0])


def test_a_silent_reference_window_leaves_the_excess_ratio_undefined():
    """Synthetic data is exactly silent before the arrival; the share stays defined."""
    target, prediction = _split_case()
    columns = windowed_waveform_columns(prediction, target, np.full(2, 20, dtype=np.int64))
    assert columns["pre_target_energy"].tolist() == [0.0, 0.0]
    assert np.isnan(columns["pre_excess_ratio"]).all()
    assert columns["pre_energy_ratio"][0] > 0 and columns["pre_energy_ratio"][1] == 0.0


def test_mismatched_windows_and_traces_are_refused():
    target, prediction = _split_case()
    with pytest.raises(ValueError, match="pre- and post-arrival part"):
        windowed_waveform_columns(prediction, target, np.array([0, 20]))
    with pytest.raises(ValueError, match="pre- and post-arrival part"):
        windowed_waveform_columns(prediction, target, np.array([20, 40]))
    with pytest.raises(ValueError, match="matching finite trace arrays"):
        windowed_waveform_columns(prediction[:1], target, np.array([20]))


def test_the_declared_windows_cover_both_datasets():
    assert set(ARRIVAL_WINDOWS) == {"forge_2017", "seg_c3_na"}
    for window in ARRIVAL_WINDOWS.values():
        assert set(window) == {"intercept_s", "velocity_m_per_s"}
        assert window["intercept_s"] > 0 and window["velocity_m_per_s"] > 0


def test_the_summary_reads_test_traces_only_and_matches_the_study_aggregation():
    """A grid run also predicts its validation traces, which took part in selection."""
    table = pd.DataFrame(
        {
            "post_snr_db": [1.0, 3.0, 100.0],
            "post_relative_mse": [0.4, 0.6, 0.0],
            "post_error_energy": [1.0, 3.0, 0.0],
            "post_target_energy": [4.0, 4.0, 4.0],
            "pre_prediction_energy": [1.0, 1.0, 50.0],
            "pre_target_energy": [2.0, 2.0, 2.0],
            "pre_energy_ratio": [0.1, 0.3, 9.0],
            "pre_excess_ratio": [0.5, 0.5, 25.0],
            "pre_target_share": [0.01, 0.01, 0.01],
            "clean_target": [True, True, True],
            "scored_domain": ["test", "test", "validation"],
        }
    )
    summary = summarize(table, "clean_target")
    assert summary["trace_count"] == 2
    assert summary["mean_post_snr_db"] == pytest.approx(2.0)
    # Mean of the per-trace ratio, as the studies report masked_trace_relative_mse.
    assert summary["mean_post_relative_mse"] == pytest.approx(0.5)
    assert summary["pooled_post_relative_mse"] == pytest.approx(0.5)
    assert summary["pooled_pre_excess_ratio"] == pytest.approx(0.5)
