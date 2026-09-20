from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.processing.forge_waveform_qc import (
    flag_relative_amplitude,
    select_representative_shots,
    waveform_metrics,
)


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[2] / "studies/study_048_forge_waveform_qc/config.yaml"
    return yaml.safe_load(path.read_text())


def test_numeric_failures_are_not_hidden_by_finite_replacement(config):
    x = np.zeros((5, 4001))
    x[1] = 3
    x[2, 7] = np.nan
    x[3, 9] = np.inf
    x[4] = np.sin(2 * np.pi * 30 * np.arange(4001) * 0.001)
    stats, aggregate = waveform_metrics(x, 0.001, config)
    assert stats.all_zero.tolist() == [True, False, False, False, False]
    assert stats.constant.tolist() == [True, True, False, False, False]
    assert stats.numerically_usable.tolist() == [False, False, False, False, True]
    assert stats.nonfinite_count.tolist() == [0, 0, 1, 1, 0]
    assert stats.rms.iloc[1] == 3
    assert stats.rms.iloc[2:4].isna().all()
    assert aggregate["spectrum_count"] == 1
    assert aggregate["finite_trace_count"] == 3
    assert stats.power_fraction_sweep_4_84.iloc[:4].isna().all()


@pytest.mark.parametrize("samples", [4000, 4001])
def test_spectrum_distinguishes_sweep_and_out_of_band_and_masks_aux(config, samples):
    t = np.arange(samples) * 0.001
    x = np.array([np.sin(2 * np.pi * 30 * t), 1e6 * np.sin(2 * np.pi * 180 * t)])
    stats, spectra = waveform_metrics(x, 0.001, config, np.array([True, False]))
    assert stats.power_fraction_sweep_4_84.iloc[0] > 0.999
    assert stats.power_fraction_above_125.iloc[1] > 0.999
    assert spectra["spectrum_count"] == 1
    assert spectra["frequency_hz"][spectra["power_sum"].argmax()] == pytest.approx(30, abs=0.3)
    assert spectra["normalized_power_sum"].sum() == pytest.approx(1)
    expected = np.sum((x[0] - x[0].mean()) ** 2 * np.hanning(samples) ** 2) * samples
    assert spectra["power_sum"].sum() == pytest.approx(expected)


def test_exact_plateaus_differ_from_constant_and_single_peak(config):
    t = np.arange(4001) * 0.001
    sine = np.sin(2 * np.pi * 25 * t)
    x = np.array([sine, sine.copy(), np.zeros(4001), np.zeros(4001)])
    x[1, 100:104] = 2
    x[3, 900] = 100
    stats, _ = waveform_metrics(x, 0.001, config)
    assert stats.extreme_plateau_review.tolist() == [False, True, False, False]
    assert stats.impulsive_review.tolist() == [False, False, False, True]


def test_time_windows_measure_raw_energy(config):
    x = np.ones((1, 4001))
    x[:, 500:1000] = 2
    x[:, 1000:2000] = 3
    x[:, 2000:] = 4
    stats, _ = waveform_metrics(x, 0.001, config)
    assert [stats[f"window_{i}_rms"].iloc[0] for i in range(4)] == [1, 2, 3, 4]


def test_amplitude_reference_uses_candidates_in_same_shot_offset_bin(config):
    n = 50
    table = pd.DataFrame(
        {
            "source_file": ["a"] * 25 + ["b"] * 25,
            "offset_m": np.full(n, 100.0),
            "header_eligible": np.ones(n, bool),
            "numerically_usable": np.ones(n, bool),
            "rms": [1.0] * 24 + [1000.0] + [1000.0] * 25,
            "nonfinite_count": 0,
            "all_zero": False,
            "constant": False,
            "impulsive_review": False,
            "dc_review": False,
            "extreme_plateau_review": False,
        }
    )
    result = flag_relative_amplitude(table, config)
    assert result.amplitude_review.sum() == 1
    assert result.waveform_status.iloc[24] == "review"
    assert result.numerically_usable.all()
    assert result.rms_to_shot_offset_median.iloc[25] == 1


def test_representatives_are_geometry_based_and_order_independent():
    h = pd.DataFrame(
        {
            "ffid": [5, 4, 3, 2, 1],
            "source_line": [2, 2, 1, 1, 1],
            "source_point": [20, 10, 30, 10, 20],
        }
    )
    assert select_representative_shots(h) == [2, 1, 3, 4, 5]
    assert select_representative_shots(h.iloc[::-1]) == [2, 1, 3, 4, 5]
