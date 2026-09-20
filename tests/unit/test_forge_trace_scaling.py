from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from seis_interp.processing.forge_trace_scaling import prepare_trace_scales
from seis_interp.training.forge_mvp_methods import ForgeMethodResult, run_method

SETTINGS = {
    "geometry": "projected_grid",
    "radius": 2,
    "power": 2.0,
    "distance": "physical",
    "restoration": "multiply_once",
}
GRIDS = {
    role: {"across_spacing_m": 200.0, "along_spacing_m": 50.0} for role in ("source", "receiver")
}


def test_scalar_idw_uses_physical_grid_distance_and_observed_rms_only():
    scales, spacing = prepare_trace_scales(
        np.array([[2.0, -2.0], [8.0, -8.0]], dtype=np.float32),
        np.array([1, 2]),
        np.arange(4),
        (1, 1, 2, 2),
        GRIDS,
        SETTINGS,
    )
    assert spacing == (200.0, 50.0, 200.0, 50.0)
    np.testing.assert_allclose(
        scales.rms,
        [
            (2 / 50**2 + 8 / 200**2) / (1 / 50**2 + 1 / 200**2),
            2,
            8,
            (2 / 200**2 + 8 / 50**2) / (1 / 50**2 + 1 / 200**2),
        ],
    )
    assert scales.scale_source.tolist() == ["idw", "observed", "observed", "idw"]


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_invalid_observed_rms_stops(value):
    with pytest.raises(ValueError, match="observed trace RMS"):
        prepare_trace_scales(
            np.array([[value]], dtype=np.float32),
            np.array([0]),
            np.arange(4),
            (1, 1, 2, 2),
            GRIDS,
            SETTINGS,
        )


def test_restoration_multiplies_idw_once_without_forcing_prediction_rms():
    normalized = np.array([[0.1, -0.1], [0.0, 0.0]], dtype=np.float32)
    model_inputs = object()
    inputs = SimpleNamespace(
        model_space=lambda: model_inputs,
        test_rms=np.array([2.0, 9.0]),
        binding={"trace_scale_hash": "sealed"},
    )
    result = ForgeMethodResult(normalized.copy(), np.array([1, 2]), [], None, {})
    with patch("seis_interp.training.forge_mvp_methods._run_method", return_value=result) as worker:
        actual = run_method(inputs, {})
    worker.assert_called_once_with(model_inputs, {}, print)
    np.testing.assert_array_equal(
        actual.prediction, np.array([[0.2, -0.2], [0.0, 0.0]], dtype=np.float32)
    )
