from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from seis_interp.data.experiment_imports import method_config_hash
from seis_interp.processing.c3_region_search import resolve_c3_candidate
from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_nonoverlapping import interpolate_drr_nonoverlapping
from seis_interp.training.c3_region_methods import run_c3_region_method


def drr_config(window=(2, 3, 2, 3)):
    return {
        "method_id": "drr_grid",
        "algorithm": {
            "rank": 1,
            "damping_power": 2,
            "n_iterations": 2,
            "frequency_min_hz": 0.0,
            "frequency_max_hz": None,
            "spatial_window_shape": list(window),
            "spatial_overlap": [0] * 4,
        },
        "execution": {"backend": "batched", "workers": 1},
    }


@pytest.mark.parametrize("shape", [(4, 6, 4, 6), (3, 5, 3, 5), (1, 4, 1, 4)])
@pytest.mark.parametrize("workers", [1, 2])
def test_grid_zero_overlap_is_bitwise_legacy_for_divisible_nondivisible_and_oversize_windows(
    shape, workers, monkeypatch
):
    from seis_interp.training import c3_region_methods

    def wrong_backend(*args, **kwargs):
        pytest.fail("zero overlap must use the legacy backend")

    monkeypatch.setattr(c3_region_methods, "interpolate_drr_volume_batched", wrong_backend)
    rng = np.random.default_rng(82)
    values = rng.normal(size=(7, *shape)).astype(np.float32)
    mask = rng.random(shape) < 0.5
    values[:, ~mask] = np.nan
    time_s = np.arange(7) * 0.008
    config = drr_config()
    config["execution"]["workers"] = workers
    inputs = SimpleNamespace(
        volume=lambda: SimpleNamespace(values=values, observed_trace_mask=mask),
        global_rms=2.0,
        time_s=time_s,
        test_cells=np.flatnonzero(~mask),
    )
    normalized = (values.astype(np.float64) / inputs.global_rms).astype(np.float32)
    previous = interpolate_drr_nonoverlapping(
        normalized, mask, time_s, **config["algorithm"], workers=workers
    )
    expected = previous.values.reshape(len(time_s), -1)[:, inputs.test_cells].T
    expected = (expected.astype(np.float64) * inputs.global_rms).astype(np.float32)
    actual = run_c3_region_method(inputs, config, reporter=None)
    np.testing.assert_array_equal(actual.prediction, expected)
    assert actual.diagnostics["block_count"] == previous.block_count


def test_grid_positive_overlap_uses_batched_clip_and_shift(monkeypatch):
    from seis_interp.training import c3_region_methods

    def wrong_backend(*args, **kwargs):
        pytest.fail("positive overlap must use the batched backend")

    monkeypatch.setattr(c3_region_methods, "interpolate_drr_nonoverlapping", wrong_backend)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 3, 5, 3, 5)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.5
    config = drr_config()
    config["algorithm"]["overlap_policy"] = "point_half"
    resolved = resolve_c3_candidate(config, int(mask.sum()), 64)
    assert resolved["algorithm"]["backend"] == "drr_batched"
    assert resolved["algorithm"]["boundary_rule"] == "clip_and_shift"
    assert resolved["algorithm"]["spatial_overlap"] == [0, 1, 0, 1]
    options = resolved["algorithm"].copy()
    options.pop("backend")
    time_s = np.arange(5) * 0.008
    expected = interpolate_drr_volume_batched(values, mask, time_s, **options)
    inputs = SimpleNamespace(
        volume=lambda: SimpleNamespace(values=values, observed_trace_mask=mask),
        global_rms=1.0,
        time_s=time_s,
        test_cells=np.flatnonzero(~mask),
    )
    actual = run_c3_region_method(inputs, resolved, reporter=None)
    np.testing.assert_array_equal(
        actual.prediction, expected.values.reshape(len(time_s), -1)[:, inputs.test_cells].T
    )


def test_legacy_identity_matches_only_the_zero_overlap_resolved_contract():
    legacy = drr_config()
    original = deepcopy(legacy)
    resolved = resolve_c3_candidate(legacy, 100, 64)
    assert legacy == original
    assert resolved["algorithm"]["backend"] == "drr_nonoverlapping"
    assert resolved["algorithm"]["boundary_rule"] == "unknown_padding"
    assert method_config_hash(legacy) == method_config_hash(resolved)
    overlapping = deepcopy(legacy)
    overlapping["algorithm"]["overlap_policy"] = "point_half"
    assert method_config_hash(overlapping) != method_config_hash(legacy)
    conflicting = deepcopy(resolved)
    conflicting["algorithm"]["boundary_rule"] = "clip_and_shift"
    assert method_config_hash(conflicting) != method_config_hash(resolved)
    with pytest.raises(ValueError, match="boundary_rule must be unknown_padding"):
        resolve_c3_candidate(conflicting, 100, 64)


@pytest.mark.parametrize("workers,matrix_limit", [(1, 1), (2, 400), (2, 2_000_000)])
def test_overlapping_grid_execution_settings_preserve_bits_and_import_identity(
    workers, matrix_limit
):
    rng = np.random.default_rng(91)
    values = rng.normal(size=(17, 3, 5, 3, 5)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.5
    values[:, ~mask] = np.nan
    inputs = SimpleNamespace(
        volume=lambda: SimpleNamespace(values=values, observed_trace_mask=mask),
        global_rms=2.0,
        time_s=np.arange(17) * 0.008,
        test_cells=np.flatnonzero(~mask),
    )
    baseline = drr_config()
    baseline["algorithm"]["spatial_overlap"] = [0, 1, 0, 1]
    changed = deepcopy(baseline)
    changed["execution"].update(workers=workers, batch_matrix_limit=matrix_limit)
    expected = run_c3_region_method(inputs, baseline, reporter=None)
    actual = run_c3_region_method(inputs, changed, reporter=None)
    np.testing.assert_array_equal(actual.prediction, expected.prediction)
    assert actual.diagnostics == expected.diagnostics
    assert method_config_hash(changed) == method_config_hash(baseline)


def test_overlapping_grid_import_identity_retains_scientific_settings():
    baseline = drr_config()
    baseline["algorithm"]["spatial_overlap"] = [0, 1, 0, 1]
    for key, value in (
        ("rank", 2),
        ("n_iterations", 30),
        ("spatial_overlap", [0, 0, 0, 0]),
        ("boundary_rule", "unknown_padding"),
    ):
        changed = deepcopy(baseline)
        changed["algorithm"][key] = value
        assert method_config_hash(changed) != method_config_hash(baseline)
