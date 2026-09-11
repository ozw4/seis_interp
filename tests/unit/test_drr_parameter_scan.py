from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation import drr_parameter_scan
from seis_interp.processing.drr_windows import WindowedDrrResult


def test_pilot_uses_observations_and_aggregates_target_energies(monkeypatch):
    mask = np.ones((2, 2, 2, 3), dtype=bool)
    mask[..., 0] = False
    values = np.zeros((4, *mask.shape), dtype=np.float32)
    values[:, mask] = 2
    observed = ObservedC3Volume(
        values=values,
        time_s=np.arange(4),
        array_rows=np.arange(mask.size).reshape(mask.shape),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )
    calls = []

    def reconstruct(block, supplied_mask, time_s, **parameters):
        assert np.all(block[:, ~supplied_mask] == 0)
        assert np.all(block[:, supplied_mask] == 2)
        assert len(time_s) == 4
        prediction = block.copy()
        prediction[:, ~supplied_mask] = 1
        return WindowedDrrResult(prediction, 1, 0, 0)

    def evaluate(prediction, block, **kwargs):
        calls.append(block)
        assert np.all(prediction[:, block.evaluation_target_trace_mask] == 1)
        energy = 4.0 if len(calls) == 1 else 16.0
        return {
            "observed_max_abs_error": 0.0,
            "evaluation_target": {
                "trace_count": 4,
                "sample_count": 16,
                "reference_energy": energy,
                "error_energy": 1.0,
            },
        }

    monkeypatch.setattr(drr_parameter_scan, "interpolate_drr_volume", reconstruct)
    monkeypatch.setattr(drr_parameter_scan, "evaluate_c3_volume_prediction", evaluate)
    regions = [(slice(i, i + 1), slice(None), slice(None), slice(None)) for i in range(2)]
    result = drr_parameter_scan.score_drr_target_regions(
        observed,
        interim_dir=Path("unused"),
        volume_metadata={},
        parameters={},
        regions=regions,
    )
    assert result["trace_count"] == 8
    assert result["sample_count"] == 32
    assert result["snr_db"] == pytest.approx(10.0)
    np.testing.assert_array_equal(calls[1].array_rows, observed.array_rows[1:2])
    with pytest.raises(ValueError, match="must not overlap"):
        drr_parameter_scan.score_drr_target_regions(
            observed,
            interim_dir=Path("unused"),
            volume_metadata={},
            parameters={},
            regions=[regions[0], regions[0]],
        )


def test_full_run_config_changes_only_selected_drr_parameters():
    from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config

    reference = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_036_c3_random80_observed_only_poc/formal/drr.yaml"
    )
    selected = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_039_c3_drr_target_tuning/rank12_iterations60.yaml"
    )
    reference["drr"].update(rank=12, n_iterations=60, frequency_min_hz=0.0)
    assert selected == reference
