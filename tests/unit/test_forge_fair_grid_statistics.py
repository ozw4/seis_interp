from itertools import product

import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.forge_fair_grid_statistics import (
    FAIR_GRID_CONTRASTS,
    cluster_bootstrap_mean,
    cluster_bootstrap_weights,
    expected_best_curve,
    holm_adjust,
    paired_contrasts,
    scale_invariant_relative_mse,
    seed_average_errors,
    seed_mean_sd,
    wilcoxon_signed_rank,
)
from seis_interp.processing.forge_log_rms_idw import LOG_RMS_IDW_SETTINGS, log_rms_idw_scales

METHODS = ("pocs_grid", "drr_grid", "ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")


def _clustered_errors(seed=0, clusters=12, per_cluster=5):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(clusters) * 7, per_cluster)
    level = {m: 0.5 + 0.05 * i for i, m in enumerate(METHODS)}
    shared = rng.normal(0, 0.05, clusters)[ids // 7]
    errors = {m: level[m] + shared + rng.normal(0, 0.02, len(ids)) for m in METHODS}
    return errors, ids


def test_seed_mean_and_sample_sd():
    assert seed_mean_sd([1.0, 2.0, 3.0]) == (2.0, 1.0)
    assert seed_mean_sd([0.7]) == (0.7, None)
    with pytest.raises(ValueError):
        seed_mean_sd([])


def test_seed_average_errors_is_the_per_trace_mean():
    np.testing.assert_allclose(seed_average_errors([[1.0, 2.0], [3.0, 6.0]]), [2.0, 4.0])
    with pytest.raises(ValueError):
        seed_average_errors([[1.0, np.nan]])


def test_bootstrap_weights_are_reproducible_cluster_draws():
    first = cluster_bootstrap_weights(12, 500, 600703)
    np.testing.assert_array_equal(first, cluster_bootstrap_weights(12, 500, 600703))
    assert not np.array_equal(first, cluster_bootstrap_weights(12, 500, 600704))
    assert first.shape == (500, 12)
    assert np.all(first.sum(axis=1) == 12)
    with pytest.raises(ValueError):
        cluster_bootstrap_weights(0, 10, 1)


def test_bootstrap_resamples_whole_clusters():
    values = np.array([1.0, 3.0, 10.0, 10.0, 10.0])
    ids = np.array([4, 4, 9, 9, 9])
    # One resample of cluster 4 twice, one of cluster 9 twice.
    weights = np.array([[2, 0], [0, 2]])
    result = cluster_bootstrap_mean(values, ids, np.array([4, 9]), weights)
    assert result["estimate"] == pytest.approx(34 / 5)
    assert (result["ci95_low"], result["ci95_high"]) == pytest.approx(
        np.percentile([2.0, 10.0], [2.5, 97.5])
    )
    assert result["cluster_count"] == 2
    with pytest.raises(ValueError):
        cluster_bootstrap_mean(values, ids, np.array([4, 9, 11]), np.ones((2, 3)))


def test_bootstrap_p_value_is_null_centred():
    ids = np.repeat(np.arange(20), 3)
    weights = cluster_bootstrap_weights(20, 999, 3)
    shifted = cluster_bootstrap_mean(np.full(60, -1.0) + 1e-3 * ids, ids, np.arange(20), weights)
    assert shifted["bootstrap_p_value"] == pytest.approx(1 / 1000)
    assert shifted["ci95_high"] < 0
    zero = cluster_bootstrap_mean(np.zeros(60), ids, np.arange(20), weights)
    assert zero["bootstrap_p_value"] == 1.0


def test_holm_adjustment_matches_hand_computation():
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm_adjust([0.5, 0.6]), [1.0, 1.0])
    np.testing.assert_allclose(holm_adjust([0.2]), [0.2])
    with pytest.raises(ValueError):
        holm_adjust([1.2])


def test_wilcoxon_matches_scipy_normal_approximation():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    d = np.round(rng.normal(0.1, 1.0, 80), 1)
    d[:4] = 0.0
    ours = wilcoxon_signed_rank(d)
    reference = stats.wilcoxon(d, zero_method="wilcox", correction=False, method="approx")
    assert ours["nonzero_count"] == int(np.count_nonzero(d))
    assert ours["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert wilcoxon_signed_rank(np.zeros(3))["p_value"] == 1.0


def test_paired_contrasts_follow_the_six_declared_pairs():
    errors, ids = _clustered_errors()
    weights = cluster_bootstrap_weights(len(np.unique(ids)), 2000, 11)
    table = paired_contrasts(errors, ids, weights)
    assert list(table.contrast) == [c[0] for c in FAIR_GRID_CONTRASTS]
    for row, (_, left, right, role) in zip(table.itertuples(), FAIR_GRID_CONTRASTS, strict=True):
        difference = errors[left] - errors[right]
        assert row.estimate == pytest.approx(difference.mean())
        assert row.ci95_low <= row.estimate <= row.ci95_high
        assert row.role == role
        assert row.fraction_left_lower_error == pytest.approx(np.mean(difference < 0))
        assert row.wilcoxon_p_value == pytest.approx(wilcoxon_signed_rank(difference)["p_value"])
    np.testing.assert_allclose(table.holm_p_value, holm_adjust(table.bootstrap_p_value))
    assert np.all(table.holm_p_value >= table.bootstrap_p_value)
    pd.testing.assert_frame_equal(table, paired_contrasts(errors, ids, weights))
    with pytest.raises(KeyError):
        paired_contrasts({m: errors[m] for m in METHODS[:-1]}, ids, weights)


def test_scale_invariant_error_removes_any_per_trace_scalar():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(6, 40)).astype(np.float32)
    prediction = (target + rng.normal(0, 0.5, target.shape)).astype(np.float32)
    base = scale_invariant_relative_mse(prediction, target)
    scaled = prediction * np.array([3, -2, 0.1, 7, 1, -0.5], dtype=np.float32)[:, None]
    np.testing.assert_allclose(scale_invariant_relative_mse(scaled, target), base, rtol=1e-5)
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    cosine = np.sum(p * y, 1) / np.sqrt(np.sum(p * p, 1) * np.sum(y * y, 1))
    np.testing.assert_allclose(base, 1 - cosine**2, rtol=1e-10)
    np.testing.assert_allclose(scale_invariant_relative_mse(3 * target, target), 0, atol=1e-12)
    zero = scale_invariant_relative_mse(np.zeros_like(target), target)
    np.testing.assert_allclose(zero, 1.0)


def _brute_force_expected_best(scores, n):
    draws = np.array(list(product(scores, repeat=n)))
    best = draws.min(axis=1)
    return best.mean(), best.std()


def test_expected_best_curve_matches_enumeration_with_ties():
    scores = np.array([0.8, 0.7, 0.8, 0.75])
    curve = expected_best_curve(scores)
    assert list(curve.configurations) == [1, 2, 3, 4]
    assert curve.expected_best.iloc[0] == pytest.approx(scores.mean())
    for n in (1, 2, 3):
        mean, sd = _brute_force_expected_best(scores, n)
        assert curve.expected_best.iloc[n - 1] == pytest.approx(mean)
        assert curve.sd.iloc[n - 1] == pytest.approx(sd)
    assert np.all(np.diff(curve.expected_best) <= 0)
    assert curve.expected_best.iloc[-1] >= scores.min()
    maximum = expected_best_curve(scores, lower_is_better=False)
    assert maximum.expected_best.iloc[1] == pytest.approx(
        np.array(list(product(scores, repeat=2))).max(axis=1).mean()
    )


def _grids():
    return {
        role: {"across_spacing_m": 200.0, "along_spacing_m": 50.0}
        for role in ("source", "receiver")
    }


def test_log_rms_idw_is_the_geometric_idw_of_observed_rms():
    shape = (2, 3, 2, 3)
    rng = np.random.default_rng(4)
    observed = np.array([0, 5, 17, 22, 30, 35])
    query = np.setdiff1d(np.arange(36), observed)
    values = (rng.normal(size=(6, 11)) * np.array([0.1, 1, 10, 3, 0.5, 2])[:, None]).astype(
        np.float32
    )
    scales, spacing = log_rms_idw_scales(
        values, observed, query, shape, _grids(), LOG_RMS_IDW_SETTINGS
    )
    assert scales.shape == query.shape
    assert spacing == (200.0, 50.0, 200.0, 50.0)
    log_rms = np.log(np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1)))
    grid = np.array(np.unravel_index(np.arange(36), shape)).T
    expected = []
    for cell in query:
        offset = grid[observed] - grid[cell]
        weight = np.linalg.norm(offset * np.array(spacing), axis=1) ** -2.0
        weight[np.abs(offset).max(axis=1) > 2] = 0
        expected.append(np.exp(weight @ log_rms / weight.sum()))
    np.testing.assert_allclose(scales, expected, rtol=1e-12)


def test_log_rms_idw_rejects_other_settings_and_observed_queries():
    values = np.ones((2, 4), dtype=np.float32)
    constant, _ = log_rms_idw_scales(
        values * 3, [0, 1], [2, 3], (1, 2, 1, 2), _grids(), LOG_RMS_IDW_SETTINGS
    )
    np.testing.assert_allclose(constant, 3.0)
    with pytest.raises(ValueError):
        log_rms_idw_scales(
            values, [0, 1], [2], (1, 2, 1, 2), _grids(), {**LOG_RMS_IDW_SETTINGS, "radius": 3}
        )
    with pytest.raises(ValueError):
        log_rms_idw_scales(values, [0, 1], [1, 2], (1, 2, 1, 2), _grids(), LOG_RMS_IDW_SETTINGS)
