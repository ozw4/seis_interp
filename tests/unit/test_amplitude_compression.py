import numpy as np
import pandas as pd
import pytest

from seis_interp.data.forge_compressed_inputs import compressed_model_inputs
from seis_interp.processing.amplitude_compression import (
    observed_percentile_scale,
    transform_amplitudes,
)


@pytest.mark.parametrize("mode", ["linear", "asinh"])
def test_signed_transform_roundtrip_preserves_physical_amplitudes(mode):
    values = np.array([[-1000, -2, 0, 2, 1000]], dtype=np.float32)
    transformed = transform_amplitudes(values, 2.5, mode)
    np.testing.assert_array_equal(np.sign(transformed), np.sign(values))
    restored = transform_amplitudes(transformed, 2.5, mode, inverse=True)
    np.testing.assert_allclose(restored, values, rtol=1e-6)
    assert transformed.dtype == restored.dtype == np.float32


def test_common_scale_preserves_trace_amplitude_differences_and_asinh_compresses_them():
    observed = np.array([[-1, 1], [-100, 100]], dtype=np.float32)
    scale = observed_percentile_scale(observed)
    assert scale == np.percentile(np.abs(observed), 90)
    linear = transform_amplitudes(observed, scale, "linear")
    compressed = transform_amplitudes(observed, scale, "asinh")
    assert linear[1, 1] / linear[0, 1] == pytest.approx(100)
    assert 1 < compressed[1, 1] / compressed[0, 1] < 100


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.array([[np.nan]]), np.empty((0, 2))])
def test_scale_rejects_undefined_observed_data(values):
    with pytest.raises(ValueError):
        observed_percentile_scale(values)


def test_inverse_overflow_is_reported_instead_of_clipping():
    with pytest.raises(FloatingPointError):
        transform_amplitudes(np.array([1000.0]), 1.0, "asinh", inverse=True)


@pytest.mark.parametrize("scale_multiplier", [0.1, 0.3, 1.0, 3.0, 10.0])
def test_compressed_runtime_accepts_only_observed_rows_and_has_no_trace_scales(scale_multiplier):
    mask = pd.DataFrame({"cell_id": [0, 1, 2], "split": ["observed", "test", "observed"]})
    geometry = pd.DataFrame({"cell_id": [0, 1, 2]})
    observed = np.array([[1, -1], [10, -10]], dtype=np.float32)
    inputs, scale = compressed_model_inputs(
        geometry,
        mask,
        observed,
        (1, 1, 1, 3),
        np.arange(2),
        "asinh",
        90,
        {},
        scale_multiplier=scale_multiplier,
    )
    assert scale == observed_percentile_scale(observed) * scale_multiplier
    assert inputs.global_rms == 1
    assert inputs.observed_rms is None and inputs.test_rms is None
    np.testing.assert_allclose(inputs.observed_values, np.arcsinh(observed / scale), rtol=1e-6)
    with pytest.raises(ValueError, match="outside outer observed"):
        inputs.read_observed(np.array([1]))
    with pytest.raises(ValueError, match="observed rows only"):
        compressed_model_inputs(
            geometry, mask, np.ones((3, 2)), (1, 1, 1, 3), np.arange(2), "asinh", 90, {}
        )


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf"), True])
def test_compressed_runtime_rejects_invalid_scale_multiplier(multiplier):
    mask = pd.DataFrame({"cell_id": [0, 1], "split": ["observed", "test"]})
    with pytest.raises(ValueError, match="scale_multiplier must be finite and positive"):
        compressed_model_inputs(
            mask[["cell_id"]],
            mask,
            np.ones((1, 2)),
            (1, 1, 1, 2),
            np.arange(2),
            "asinh",
            90,
            {},
            scale_multiplier=multiplier,
        )
