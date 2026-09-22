import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.forge_condition_inputs import (
    condition_model_inputs,
    downsample_traces,
    nested_condition_mask,
)


@pytest.mark.parametrize("factor, alias_hz", [(2, 400), (8, 200)])
def test_decimation_preserves_time_origin_and_rejects_aliasing(factor, alias_hz):
    t = np.arange(4001) / 1000
    values = np.stack(
        [np.ones_like(t), np.cos(2 * np.pi * 10 * t), np.cos(2 * np.pi * alias_hz * t)]
    ).astype(np.float32)
    result = downsample_traces(values, factor)
    assert result.shape == (3, len(t[::factor]))
    np.testing.assert_allclose(result[0], 1, atol=1e-6)
    np.testing.assert_allclose(
        result[1, 20:-20], np.cos(2 * np.pi * 10 * t[::factor])[20:-20], atol=0.004
    )
    assert np.sqrt(np.mean(result[2, 20:-20] ** 2)) < 0.005
    np.testing.assert_array_equal(downsample_traces(values, 1), values)


def test_nested_masks_keep_common_targets_hidden():
    mask = pd.DataFrame({"cell_id": np.arange(100), "hash_rank": np.arange(100, 0, -1)})
    smaller = nested_condition_mask(mask, 0.1)
    larger = nested_condition_mask(mask, 0.8)
    a = set(smaller.loc[smaller.split.eq("test"), "cell_id"])
    b = set(larger.loc[larger.split.eq("test"), "cell_id"])
    assert len(a) == 10 and len(b) == 80 and a < b
    assert "split" not in mask


@pytest.mark.parametrize("normalization", ["trace_oracle", "observed_global"])
def test_test_waveforms_cannot_change_observed_model_inputs(normalization):
    mask = pd.DataFrame(
        {"cell_id": [0, 1, 2, 3], "split": ["test", "observed", "test", "observed"]}
    )
    geometry = pd.DataFrame({"cell_id": [0, 1, 2, 3]})
    values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)

    def build(data):
        return condition_model_inputs(
            geometry, mask, data, (1, 1, 1, 4), np.arange(4), normalization, {}
        )

    first, scales = build(values)
    changed = values.copy()
    changed[[0, 2]] *= 100
    second, other_scales = build(changed)
    np.testing.assert_array_equal(first.observed_values, second.observed_values)
    assert first.global_rms == second.global_rms
    with pytest.raises(ValueError, match="outside outer observed"):
        first.read_observed(np.array([0]))
    if normalization == "trace_oracle":
        np.testing.assert_allclose(np.sqrt(np.mean(first.observed_values**2, axis=1)), 1, rtol=1e-6)
        np.testing.assert_allclose(other_scales[[0, 2]], 100 * scales[[0, 2]])
    else:
        np.testing.assert_array_equal(scales, other_scales)
