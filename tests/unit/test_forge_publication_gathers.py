"""Fixed geometry, common gain, completed observed traces and residual polarity."""

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.visualization.forge_publication_gathers import (
    DISPLAY_METHODS,
    central_gather_cells,
    prepare_publication_gathers,
)


def test_central_cells_are_geometry_only_and_sorted():
    shape = (2, 3, 2, 4)
    cells = np.arange(np.prod(shape)).reshape(shape)
    index = pd.DataFrame({"cell_id": np.arange(cells.size)[::-1]})
    sections = central_gather_cells(index, shape)
    expected = [cells[1, 1, 1, :], cells[1, :, 1, 2], cells[:, 1, 1, 2], cells[1, 1, :, 2]]
    for section, selected in zip(sections, expected, strict=True):
        assert section["cell_ids"] == selected.tolist()
    with pytest.raises(ValueError, match="missing grid cells"):
        central_gather_cells(index.loc[index.cell_id.ne(int(expected[0][0]))], shape)


@pytest.mark.parametrize("mode", ["physical", "trace_rms"])
def test_panels_preserve_mask_prediction_gain_and_reference_minus_prediction(mode):
    index = pd.DataFrame({"cell_id": [7, 8, 9], "split": ["observed", "test", "test"]})
    truth = np.array([[0, 0, 4, 0], [0, 8, 0, 0], [0, 0, 0, 0]], np.float32)
    predictions = {method: truth * (i + 2) for i, method in enumerate(METHODS)}
    for values in predictions.values():
        values[2] = 1  # Silent reference: finite visible prediction, divisor 1.
    originals = {method: values.copy() for method, values in predictions.items()}
    sections = [{"cell_ids": [9, 7, 8], "varying_axis": "receiver_grid_point"}]
    prepared, clip = prepare_publication_gathers(
        index, truth, predictions, sections, normalization=mode
    )
    panels = prepared[0]["panels"]
    rows = [2, 0, 1]
    scale = np.array([2, 4, 1])[:, None] if mode == "trace_rms" else np.ones((3, 1))
    target = (truth / scale)[rows].T
    np.testing.assert_array_equal(panels[0], target)
    np.testing.assert_array_equal(panels[1], np.where(np.array([[False, True, False]]), target, 0))
    for i, method in enumerate(DISPLAY_METHODS):
        completed = predictions[method].copy()
        completed[0] = truth[0]
        expected = (completed / scale)[rows].T
        np.testing.assert_allclose(panels[i + 2], expected)
        np.testing.assert_allclose(panels[i + 8], target - expected)
        assert np.all(panels[i + 8][:, 1] == 0)
        np.testing.assert_array_equal(predictions[method], originals[method])
    assert clip == pytest.approx(np.percentile(np.abs(target), 99))
    assert prepared[0]["observed_count"] == 1 and prepared[0]["test_count"] == 2
