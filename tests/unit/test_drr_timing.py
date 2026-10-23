import io
import json

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.evaluation import drr_timing
from seis_interp.processing import drr_batched
from seis_interp.training.c3_region_methods import run_c3_region_method


def timing_inputs():
    cells = np.arange(225)
    return ForgeModelInputs(
        geometry=pd.DataFrame({"cell_id": cells}),
        observed_cells=cells[::2],
        test_cells=cells[1::2],
        observed_values=np.random.default_rng(4)
        .normal(size=(len(cells[::2]), 4))
        .astype(np.float32),
        spatial_shape=(3, 5, 3, 5),
        time_s=np.arange(4) * 0.004,
        global_rms=1.0,
        binding={},
    )


def timing_config():
    return {
        "method_id": "drr_grid",
        "device": "cpu",
        "algorithm": {
            "rank": 1,
            "damping_power": 2,
            "n_iterations": 2,
            "frequency_min_hz": 0.0,
            "frequency_max_hz": None,
            "spatial_window_shape": [3, 3, 3, 3],
            "spatial_overlap": [0, 2, 0, 2],
        },
        "execution": {"workers": 1, "batch_matrix_limit": 1},
    }


def test_timing_observes_production_blocks_without_changing_predictions(monkeypatch):
    inputs, config = timing_inputs(), timing_config()
    expected = run_c3_region_method(inputs, config)
    reconstruct = drr_batched._reconstruct_batch
    predictions = []

    def capture(*args, **kwargs):
        result = run_c3_region_method(*args, **kwargs)
        predictions.append(result.prediction)
        return result

    monkeypatch.setattr(drr_timing, "run_c3_region_method", capture)
    log = io.StringIO()
    result = drr_timing.measure_drr_candidate(inputs, config, log)
    np.testing.assert_array_equal(predictions[0], expected.prediction)
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(records) == result["block_count"] == expected.diagnostics["block_count"] == 9
    assert result["nonempty_block_count"] == 9
    assert all(record["wall_seconds"] > 0 for record in records)
    assert result["process_peak_rss_bytes"] > 0
    assert result["test_scoring_calls"] == result["validation_scoring_calls"] == 0
    assert result["extrapolated_60_iteration_full_candidate_seconds"] == pytest.approx(
        result["non_block_overhead_seconds"] + result["block_wall_seconds_total"] * 30
    )
    assert drr_batched._reconstruct_batch is reconstruct


@pytest.mark.parametrize(
    "section,key,value",
    [("execution", "workers", 8), ("algorithm", "n_iterations", 60)],
)
def test_timing_rejects_full_runs_and_parallel_workers(section, key, value):
    config = timing_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="workers=1, and 1 or 2 iterations"):
        drr_timing.measure_drr_candidate(timing_inputs(), config, io.StringIO())


def test_timing_restores_hook_when_block_batch_is_not_one():
    config = timing_config()
    config["execution"]["batch_matrix_limit"] = 2_000_000
    reconstruct = drr_batched._reconstruct_batch
    with pytest.raises(ValueError, match="one spatial block per batch"):
        drr_timing.measure_drr_candidate(timing_inputs(), config, io.StringIO())
    assert drr_batched._reconstruct_batch is reconstruct
