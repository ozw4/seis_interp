import numpy as np
import pandas as pd
import pytest

from seis_interp.data.forge_aligned_inputs import (
    align_model_inputs,
    moveout_canvas,
    shared_alignment,
    unalign_predictions,
    validate_alignment,
)
from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs


def _inputs(offsets=(0.0, 30.0, 10.0, 50.0), observed=(0, 2)):
    cells = np.arange(len(offsets))
    time_s = np.arange(5, dtype=np.float64) * 0.01
    geometry = pd.DataFrame({"cell_id": cells, "offset_m": offsets})
    observed = np.asarray(observed)
    test = np.setdiff1d(cells, observed)
    values = (np.arange(len(observed) * 5, dtype=np.float32) + 1).reshape(len(observed), 5)
    return ForgeModelInputs(
        geometry=geometry,
        observed_cells=observed,
        test_cells=test,
        observed_values=values,
        spatial_shape=(len(offsets),),
        time_s=time_s,
        global_rms=1.0,
        binding={},
    )


def test_observed_traces_move_to_geometry_shifts_on_the_shared_canvas():
    inputs = _inputs()
    aligned, canvas = align_model_inputs(inputs, {"velocity_m_per_s": 1000.0})
    # Delays are rint(offset / 1000 m/s / 0.01 s): 0, 3, 1, 5 samples.
    np.testing.assert_array_equal(canvas.delays, [0, 3, 1, 5])
    assert canvas.length == 10
    assert aligned.observed_values.shape == (2, 10)
    np.testing.assert_allclose(np.diff(aligned.time_s), 0.01)
    for row, cell in enumerate(inputs.observed_cells):
        shift = 5 - canvas.delays[cell]
        np.testing.assert_array_equal(
            aligned.observed_values[row, shift : shift + 5], inputs.observed_values[row]
        )
        assert np.count_nonzero(aligned.observed_values[row]) == 5
    np.testing.assert_array_equal(aligned.test_cells, inputs.test_cells)


def test_prediction_rows_return_exactly_to_their_original_samples():
    inputs = _inputs()
    _, canvas = align_model_inputs(inputs, {"velocity_m_per_s": 1000.0})
    original = np.random.default_rng(3).standard_normal((2, 5)).astype(np.float32)
    rows = np.zeros((2, canvas.length), dtype=np.float32)
    for row, cell in enumerate(inputs.test_cells):
        shift = 5 - canvas.delays[cell]
        rows[row, shift : shift + 5] = original[row]
        rows[row, :shift] = 99.0
    restored = unalign_predictions(rows, inputs.test_cells, canvas)
    np.testing.assert_array_equal(restored, original)
    with pytest.raises(ValueError):
        unalign_predictions(rows[:, :-1], inputs.test_cells, canvas)


def test_declared_max_delay_extends_the_canvas_and_must_cover_the_geometry():
    inputs = _inputs()
    canvas = moveout_canvas(inputs, {"velocity_m_per_s": 1000.0, "max_delay_samples": 7})
    assert canvas.length == 12
    with pytest.raises(ValueError):
        moveout_canvas(inputs, {"velocity_m_per_s": 1000.0, "max_delay_samples": 4})


def test_shared_alignment_covers_every_geometry():
    real = pd.DataFrame({"cell_id": [0, 1], "offset_m": [0.0, 44.0]})
    grid = pd.DataFrame({"cell_id": [0, 1], "offset_m": [0.0, 61.0]})
    alignment = shared_alignment(1000.0, [real, grid], 0.01)
    assert alignment == {"velocity_m_per_s": 1000.0, "max_delay_samples": 6}


@pytest.mark.parametrize(
    "alignment",
    [
        {},
        {"velocity_m_per_s": -1.0},
        {"velocity_m_per_s": True},
        {"velocity_m_per_s": 500.0, "extra": 1},
        {"velocity_m_per_s": 500.0, "max_delay_samples": -1},
        {"velocity_m_per_s": 500.0, "max_delay_samples": 1.5},
    ],
)
def test_invalid_alignment_is_rejected(alignment):
    with pytest.raises(ValueError):
        validate_alignment(alignment)
