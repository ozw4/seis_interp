import numpy as np
import pytest

from seis_interp.visualization.forge_condition_gathers import gather_panels


def test_gather_panels_keep_missing_predictions_and_preserve_observed_samples():
    target = np.array([[2, -4], [8, -4]], dtype=float)
    prediction = np.array([[100, 100], [4, -2]], dtype=float)
    panels = gather_panels(target, prediction, np.array([True, False]), np.array([2, 4]))
    np.testing.assert_array_equal(panels["Ground truth"], [[1, -2], [2, -1]])
    np.testing.assert_array_equal(panels["Observed input (missing = 0)"], [[1, -2], [0, 0]])
    np.testing.assert_array_equal(
        panels["Reconstruction (observed retained)"], [[1, -2], [1, -0.5]]
    )
    np.testing.assert_array_equal(panels["Error (reconstruction - truth)"], [[0, 0], [-1, 0.5]])
    np.testing.assert_array_equal(target, [[2, -4], [8, -4]])


def test_gather_display_rejects_zero_reference_rms():
    with pytest.raises(ValueError, match="finite and positive"):
        gather_panels(np.ones((2, 3)), np.ones((2, 3)), [True, False], [1, 0])
