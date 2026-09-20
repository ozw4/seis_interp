from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.processing.forge_qc_review import (
    build_qc_review,
    measure_review_waveform,
    review_distributions,
    select_review_controls,
    select_review_examples,
    summarize_qc_review,
)


def config():
    result = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "studies/study_050_forge_qc_review/config.yaml"
        ).read_text()
    )
    result["minimum_reference_traces"] = 2
    return result


def table():
    return pd.DataFrame(
        {
            "source_file": ["a"] * 8,
            "trace_index": range(8),
            "ffid": 1,
            "receiver_line": 101,
            "receiver_point": range(501, 509),
            "receiver_x_m": np.arange(8) * 50.0,
            "receiver_y_m": 10.0,
            "offset_m": 200.0,
            "header_eligible": [True] * 7 + [False],
            "nonfinite_count": 0,
            "all_zero": [True] + [False] * 7,
            "constant": [True, True] + [False] * 6,
            "numerically_usable": [False, False] + [True] * 6,
            "std": [0, 0, 1e-9, 1, 2, 1, 1, 1000],
            "rms": [0, 3, 1, 1, 2, 1, 1, 1000],
            "dc_to_rms": [np.nan, 1, 1, 0, 0, 0, 0, 0],
            "rms_to_shot_offset_median": [np.nan, np.nan, 1000, 1, 2, 0.001, 1, 1000],
            "dc_review": [False, True, True, False, False, False, False, False],
            "amplitude_review": [False, False, True, False, False, True, False, False],
            "waveform_status": [
                "all_zero",
                "constant",
                "review",
                "passed_numeric_checks",
                "passed_numeric_checks",
                "review",
                "passed_numeric_checks",
                "header_excluded",
            ],
        }
    )


def test_only_authorized_failures_are_excluded_and_ac_reference_ignores_aux():
    raw = table()
    result = build_qc_review(raw, config())
    assert result.fixed_qc_excluded.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result.eligible_after_fixed_qc.sum() == 5
    assert result.near_constant_review.sum() == 1
    assert result.loc[2, "eligible_after_fixed_qc"]  # Review does not imply rejection.
    assert result.loc[2, "ac_rms_to_shot_offset_median"] == 1e-9
    assert result.loc[2, "ac_reference_count"] == 5
    assert result.exclusion_reason.iloc[7] == "header_excluded"
    summary = summarize_qc_review(result)
    assert summary["fixed_exclusions"] == {"all_zero": 1, "nonzero_constant": 1}
    assert summary["dc_review_count"] == 1  # Fixed constant is outside denominator.
    sensitivity, stations = review_distributions(result, config())
    assert sensitivity.retained_count.eq(5).all()
    assert stations.retained_count.sum() == 5
    assert "fixed_qc_excluded" not in raw


def test_reference_minimum_and_threshold_comparisons_are_explicit():
    cfg = config()
    cfg["minimum_reference_traces"] = 10
    result = build_qc_review(table(), cfg)
    assert result.ac_rms_to_shot_offset_median.isna().all()
    assert summarize_qc_review(result)["quantiles"]["ac_rms_to_shot_offset_median"]["0.5"] is None
    raw = table()
    raw.loc[2, "std"] = 0.01
    assert not build_qc_review(raw, config()).loc[2, "near_constant_review"]  # Strict <.


def test_nonfinite_or_duplicate_inputs_do_not_silently_enter_mask():
    raw = table()
    raw.loc[2, "nonfinite_count"] = 1
    with pytest.raises(ValueError, match="separate exclusion"):
        build_qc_review(raw, config())
    with pytest.raises(ValueError, match="duplicate"):
        build_qc_review(pd.concat([table(), table().iloc[:1]]), config())


def test_representative_selection_is_stable_and_controls_are_same_shot_line_unflagged():
    result = build_qc_review(table(), config())
    a = select_review_examples(result, config())
    b = select_review_examples(result.sample(frac=1, random_state=4), config())
    pd.testing.assert_frame_equal(a, b)
    controls = select_review_controls(result, result.loc[2], 2)
    assert controls.trace_index.tolist() == [3, 4]
    assert not controls.review_pending.any()
    assert controls.eligible_after_fixed_qc.all()
    result.loc[3, "source_file"] = "different"
    assert select_review_controls(result, result.loc[2], 2).trace_index.tolist() == [4, 6]


def test_selected_measurement_preserves_tiny_nonconstant_difference():
    x = np.ones(4001, dtype=np.float32) * 0.019
    x[-1] = np.nextafter(x[-1], np.float32(0))
    result = measure_review_waveform(x, 0.001)
    assert result["unique_sample_count"] == 2
    assert 0 < result["peak_to_peak"] < 1e-8
    assert result["last_half_second_mean"] < result["first_half_second_mean"]
