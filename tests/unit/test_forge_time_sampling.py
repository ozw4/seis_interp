from copy import deepcopy

import numpy as np
import pytest

from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces


def _contract():
    return {
        "time_sample_count": 2001,
        "sample_interval_us": 2000,
        "time_sampling": {
            "native_sample_count": 4001,
            "native_sample_interval_us": 1000,
            "decimation_factor": 2,
        },
    }


def test_resampling_preserves_origin_and_low_frequency_and_removes_alias():
    contract = _contract()
    before = deepcopy(contract)
    native = native_time_contract(contract)
    assert native["time_sample_count"] == 4001
    assert native["sample_interval_us"] == 1000
    time = np.arange(4001) * 0.001
    values = np.stack([np.cos(2 * np.pi * 10 * time), np.cos(2 * np.pi * 400 * time)]).astype(
        np.float32
    )
    original = values.copy()
    result = resample_forge_traces(values, contract)
    assert result.shape == (2, 2001)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 20:-20], values[0, ::2][20:-20], atol=0.004)
    assert np.sqrt(np.mean(result[1, 20:-20] ** 2)) < 0.005
    np.testing.assert_array_equal(values, original)
    assert contract == before


@pytest.mark.parametrize("sampling", [{}, {"time_sampling": None}])
def test_native_sampling_keeps_original_array_and_contract(sampling):
    contract = {"time_sample_count": 9, "sample_interval_us": 1000, **sampling}
    values = np.ones((3, 9), dtype=np.float32)
    assert native_time_contract(contract) is contract
    assert resample_forge_traces(values, contract) is values


@pytest.mark.parametrize("native_count", [8, 9])
def test_decimation_count_handles_even_and_odd_native_lengths(native_count):
    contract = _contract()
    contract["time_sampling"]["native_sample_count"] = native_count
    contract["time_sample_count"] = (native_count + 1) // 2
    result = resample_forge_traces(np.ones((1, native_count), dtype=np.float32), contract)
    assert result.shape == (1, (native_count + 1) // 2)


@pytest.mark.parametrize("value", [0, -1, True, 2.0, None])
def test_invalid_decimation_is_rejected(value):
    contract = _contract()
    contract["time_sampling"]["decimation_factor"] = value
    with pytest.raises(ValueError, match="positive integers"):
        native_time_contract(contract)


@pytest.mark.parametrize("key", ["time_sample_count", "sample_interval_us"])
def test_inconsistent_output_sampling_is_rejected(key):
    contract = _contract()
    contract[key] += 1
    with pytest.raises(ValueError, match="output time sampling"):
        native_time_contract(contract)


def test_resampling_rejects_wrong_native_waveform_length():
    with pytest.raises(ValueError, match="waveform length"):
        resample_forge_traces(np.ones((2, 2001), dtype=np.float32), _contract())
