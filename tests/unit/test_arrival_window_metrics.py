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


def test_the_invented_amplitude_is_a_plain_rms_in_the_data_units():
    """The headline number: what the method put there, with no weighting rule."""
    target, prediction = _split_case()
    ends = np.full(2, 20, dtype=np.int64)
    columns = windowed_waveform_columns(prediction, target, ends)
    # The leaking trace holds a constant 0.5 across its twenty pre-arrival samples.
    assert columns["pre_prediction_rms"].tolist() == [0.5, 0.0]
    assert columns["pre_target_rms"].tolist() == [0.0, 0.0]
    # Normalizing by the trace's own RMS keeps the same ordering without the units.
    trace_rms = np.sqrt(np.square(target).sum(axis=1) / target.shape[1])
    assert columns["pre_prediction_rms_over_trace_rms"][0] == pytest.approx(0.5 / trace_rms[0])
    assert columns["pre_prediction_rms_over_trace_rms"][1] == 0.0


def test_the_invented_amplitude_averages_without_a_weighting_rule():
    """A loud trace and a quiet one contribute equally to the reported mean."""
    target = np.zeros((2, 40))
    target[:, 20:] = 1.0
    target[0] *= 100.0
    prediction = np.array(target)
    prediction[0, :20] = 2.0
    prediction[1, :20] = 4.0
    columns = windowed_waveform_columns(prediction, target, np.full(2, 20, dtype=np.int64))
    assert columns["pre_prediction_rms"].tolist() == [2.0, 4.0]
    assert float(np.mean(columns["pre_prediction_rms"])) == pytest.approx(3.0)


def test_a_zero_prediction_scores_exactly_one_before_the_arrival():
    """Noise is a different realization per trace, so zero is the squared-error optimum.

    Reproducing its amplitude from an independent realization doubles the error, which
    is why a faithful-looking amplitude is not a virtue on real data.
    """
    rng = np.random.default_rng(17)
    target = rng.standard_normal((400, 40))
    ends = np.full(len(target), 20, dtype=np.int64)
    silent = windowed_waveform_columns(np.zeros_like(target), target, ends)
    np.testing.assert_allclose(silent["pre_relative_mse"], 1.0)
    # An independent draw of the same variance scores two, not one.
    independent = windowed_waveform_columns(rng.standard_normal(target.shape), target, ends)
    assert float(np.median(independent["pre_relative_mse"])) == pytest.approx(2.0, abs=0.25)
    # Reproducing the trace's own realization is what scores below one.
    assert windowed_waveform_columns(target, target, ends)["pre_relative_mse"].max() == 0.0


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
    # Three test traces so the median is an element, plus one validation trace that the
    # summary must ignore: it took part in selection.
    table = pd.DataFrame(
        {
            "post_snr_db": [1.0, 2.0, 3.0, 100.0],
            "post_relative_mse": [0.4, 0.5, 0.6, 0.0],
            "post_error_energy": [1.0, 2.0, 3.0, 0.0],
            "post_target_energy": [4.0, 4.0, 4.0, 4.0],
            "pre_prediction_energy": [1.0, 1.0, 1.0, 50.0],
            "pre_target_energy": [2.0, 2.0, 2.0, 2.0],
            "pre_error_energy": [1.0, 2.0, 3.0, 40.0],
            "pre_relative_mse": [0.5, 1.0, 1.5, 20.0],
            "pre_correlation": [0.05, 0.10, 0.15, 0.9],
            "pre_early_correlation": [0.04, 0.08, 0.12, 0.9],
            "pre_late_correlation": [0.06, 0.12, 0.18, 0.9],
            "pre_prediction_rms": [1.0, 1.5, 2.0, 50.0],
            "pre_target_rms": [3.0, 3.0, 3.0, 3.0],
            "pre_amplitude_ratio": [0.2, 0.25, 0.3, 9.0],
            "pre_excess_ratio": [0.04, 0.0625, 0.09, 81.0],
            "pre_prediction_rms_over_trace_rms": [0.1, 0.15, 0.2, 5.0],
            "pre_target_rms_over_trace_rms": [0.3, 0.3, 0.3, 0.3],
            "pre_energy_ratio": [0.1, 0.2, 0.3, 9.0],
            "pre_target_share": [0.01, 0.01, 0.01, 0.01],
            "clean_target": [True, True, True, True],
            "scored_domain": ["test", "test", "test", "validation"],
        }
    )
    summary = summarize(table, "clean_target")
    assert summary["trace_count"] == 3
    assert summary["mean_post_snr_db"] == pytest.approx(2.0)
    # Mean of the per-trace ratio, as the studies report masked_trace_relative_mse.
    assert summary["mean_post_relative_mse"] == pytest.approx(0.5)
    # The invented amplitude is summarized by the median, with the mean kept beside it:
    # trace amplitudes are skewed, so the mean describes the loudest few.
    assert summary["median_pre_prediction_rms"] == pytest.approx(1.5)
    assert summary["median_pre_target_rms"] == pytest.approx(3.0)
    assert summary["median_pre_prediction_rms_over_trace_rms"] == pytest.approx(0.15)
    assert summary["mean_pre_prediction_rms"] == pytest.approx(1.5)
    # The per-trace ratio's median, not the ratio of the two medians: those differ.
    assert summary["median_pre_amplitude_ratio"] == pytest.approx(0.25)
    # One is the zero predictor's score, so the median says the typical trace broke even.
    assert summary["median_pre_relative_mse"] == pytest.approx(1.0)
    assert summary["median_pre_correlation"] == pytest.approx(0.10)
    assert summary["median_pre_early_correlation"] == pytest.approx(0.08)
    assert summary["median_pre_late_correlation"] == pytest.approx(0.12)
    assert summary["median_pre_prediction_rms"] / summary["median_pre_target_rms"] == (
        pytest.approx(0.5)
    )
    # The energy form is its square, because the median commutes with squaring a ratio.
    assert summary["median_pre_excess_ratio"] == pytest.approx(0.25**2)


def test_a_clipped_arrival_shows_up_in_the_late_half_of_the_window():
    """The split tells an arrival at the window's end from a correlation spread through it."""
    rng = np.random.default_rng(23)
    target = rng.standard_normal((200, 40))
    ends = np.full(len(target), 20, dtype=np.int64)
    # An event the window failed to exclude sits in its late half and nowhere else.
    clipped = np.zeros_like(target)
    clipped[:, 10:20] = target[:, 10:20]
    late = windowed_waveform_columns(clipped, target, ends)
    assert np.nanmedian(late["pre_late_correlation"]) == pytest.approx(1.0)
    # Nothing was predicted in the early half, so its correlation is undefined, not zero.
    assert np.isnan(late["pre_early_correlation"]).all()
    # A weak correlation spread through the window looks different in the two halves.
    spread = 0.2 * target + rng.standard_normal(target.shape)
    even = windowed_waveform_columns(spread, target, ends)
    early = np.nanmedian(even["pre_early_correlation"])
    assert early > 0 and 0.5 < np.nanmedian(even["pre_late_correlation"]) / early < 2.0
