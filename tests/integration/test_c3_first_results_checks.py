"""Actual suite readers and native zero-fill scoring retain selected times and roles."""

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from seis_interp.data import c3_benchmark_suite
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines.check_c3_first_results import (
    check_c3_first_results,
    zero_fill_c3_first_results,
)
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.processing.c3_first_results_preflight import preflight_classical_c3
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))


@pytest.fixture
def suite(tmp_path, request):
    interim = make_benchmark_interim(tmp_path / "source", time_origin=0.125)
    if getattr(request, "param", None) == "zero_reference":
        amplitudes = np.load(interim / "amplitudes.npy", allow_pickle=False)
        table = pd.read_parquet(interim / "traces.parquet")
        validation_rows = table["source_x_m"].ge(1480).to_numpy()
        amplitudes[validation_rows] = 0
        np.save(interim / "amplitudes.npy", amplitudes, allow_pickle=False)
    output = tmp_path / "suite"
    prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    return output


@pytest.mark.parametrize("suite", ["nonzero_reference", "zero_reference"], indirect=True)
def test_check_verifies_once_and_zero_fill_scores_all_targets_without_regeneration(
    suite, tmp_path, monkeypatch
):
    verify = Mock(wraps=c3_benchmark_suite.verify_c3_benchmark_suite)
    monkeypatch.setattr(c3_benchmark_suite, "verify_c3_benchmark_suite", verify)
    digest = file_sha256(suite / "benchmark_suite.json")
    kwargs = dict(
        suite_dir=suite,
        case_id="validation_random_trace",
        config={"execution": {"device": "cpu"}},
        expected_sha256=digest,
        dimensions=DIMENSIONS,
    )
    check = check_c3_first_results(**kwargs, output_dir=tmp_path / "check")
    assert verify.call_count == 1
    assert check["time_s"]["first"] == 0.125
    result = zero_fill_c3_first_results(**kwargs, output_dir=tmp_path / "zero")
    assert verify.call_count == 1
    metrics = result["evaluation_target"]
    assert metrics["trace_count"] == check["target_trace_count"]
    assert metrics["sample_count"] == 4 * check["target_trace_count"]
    assert metrics["reference_energy"] == metrics["error_energy"]
    if metrics["reference_energy"] > 0:
        assert metrics["snr_db"] == 0
        assert metrics["snr_status"] == "finite"
    else:
        assert metrics["snr_db"] is None
        assert metrics["snr_status"] == "undefined_zero_reference"
    assert result["observed_max_abs_error"] == 0
    assert file_sha256(suite / "benchmark_suite.json") == digest
    json.dumps(result, allow_nan=False)
    with pytest.raises(FileExistsError):
        zero_fill_c3_first_results(**kwargs, output_dir=tmp_path / "zero")


def test_wrong_expected_hash_is_rejected_before_zero_output(suite, tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        zero_fill_c3_first_results(
            suite_dir=suite,
            case_id="validation_random_trace",
            output_dir=tmp_path / "wrong",
            config={},
            expected_sha256="0" * 64,
            dimensions=DIMENSIONS,
        )
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("method", ["pocs", "drr"])
def test_classical_preflight_retains_full_volume_input_and_uses_all_drr_bins(suite, method):
    inputs = load_c3_benchmark_volume_inputs(
        suite, "validation_random_trace", dimensions=DIMENSIONS
    )
    values = inputs.observed_volume.values.copy()
    config = {
        "pocs": {
            "window_shape": [4, 1, 2, 2, 4],
            "overlap": [0, 0, 0, 0, 0],
            "n_iterations": 20,
            "threshold_start": 0.9,
            "threshold_end": 0.01,
        },
        "drr": {
            "spatial_window_shape": [1, 2, 2, 4],
            "spatial_overlap": [0, 0, 0, 0],
            "rank": 1,
            "damping_power": 4,
            "n_iterations": 5,
            "frequency_min_hz": 0.0,
            "frequency_max_hz": None,
        },
    }
    result = preflight_classical_c3(
        method=method, inputs=inputs, config=config, action_timeout_seconds=3600
    )
    assert result["scope"] == "single_window_resource_preflight"
    assert result["prediction_finite"]
    assert result["volume_shape"] == list(values.shape)
    assert result["window_count"] == 2
    np.testing.assert_array_equal(inputs.observed_volume.values, values)
    if method == "drr":
        assert result["fft_length"] == 4
        assert result["selected_bin_count"] == 3
        assert result["frequency_endpoints_hz"] == pytest.approx([0, 62.5])
        assert result["svd_input_dtype"] == "complex128"
    json.dumps(result, allow_nan=False)
