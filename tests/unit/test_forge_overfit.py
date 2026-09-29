from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.processing.forge_mvp_contract import make_geometry, validate_mapping
from seis_interp.training.forge_overfit import (
    NEURAL_METHODS,
    fixed_forge_batch,
    overfit_forge_batch,
    prediction_metrics,
)
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.fixture
def inputs_and_configs(tmp_path):
    _, contract, _, configs, mapping, stations, cells, grids, values = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, contract)
    observed = np.arange(0, 35, 2)
    rms = np.sqrt(np.square(values[observed].astype(float)).mean(axis=1))
    inputs = ForgeModelInputs(
        geometry=make_geometry(manifest, grids, "real"),
        observed_cells=observed,
        test_cells=np.arange(1, 35, 2),
        observed_values=values[observed],
        spatial_shape=(2, 3, 2, 3),
        time_s=np.arange(9) * 0.001,
        global_rms=1.0,
        binding={},
        observed_rms=rms,
        test_rms=np.ones(17),
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield inputs, configs
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", NEURAL_METHODS)
def test_fixed_batch_is_observed_only_and_hides_labels(inputs_and_configs, method):
    inputs, configs = inputs_and_configs
    batch = fixed_forge_batch(inputs, configs[method])
    assert set(batch.target_cells).issubset(inputs.observed_cells)
    assert set(batch.context_cells).issubset(inputs.observed_cells)
    assert not set(batch.target_cells) & set(batch.context_cells)
    expected = inputs.model_space().read_observed(batch.target_cells)
    np.testing.assert_array_equal(batch.target.numpy(), expected)
    changed = inputs.observed_values.copy()
    changed[np.isin(inputs.observed_cells, batch.target_cells)] *= -1
    other = fixed_forge_batch(replace(inputs, observed_values=changed), configs[method])
    np.testing.assert_array_equal(batch.target_cells, other.target_cells)
    torch.testing.assert_close(other.target, -batch.target)
    if method.startswith("regsi_"):
        torch.testing.assert_close(batch.model_input.waveforms, other.model_input.waveforms)
        assert (
            torch.count_nonzero(batch.model_input.waveforms[batch.model_input.query_indices]) == 0
        )
    else:
        torch.testing.assert_close(batch.model_input, other.model_input)


@pytest.mark.parametrize("method", NEURAL_METHODS)
def test_fixed_batch_updates_weights_and_saves_raw_and_ema_metrics(inputs_and_configs, method):
    inputs, configs = inputs_and_configs
    result = overfit_forge_batch(inputs, configs[method], steps=3, reporter=lambda _: None)
    assert result["history"][0]["gradient_norm"] > 0
    assert np.isfinite(result["predictions"]["raw"]).all()
    for variant, prediction in result["predictions"].items():
        expected = np.mean((prediction - result["target"]) ** 2)
        assert result["metrics"][variant]["relative_mse"] == pytest.approx(expected, rel=1e-5)
    assert ("ema" in result["predictions"]) == bool(configs[method]["training"].get("ema_decay"))
    assert result["metrics"]["raw"]["relative_mse"] != result["metrics"]["initial"]["relative_mse"]


def test_constant_predictions_have_undefined_correlation():
    target = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    result = prediction_metrics(torch.zeros_like(target), target)
    assert result["relative_mse"] == 1
    assert result["median_correlation"] is None
    assert result["undefined_correlation_count"] == 2
