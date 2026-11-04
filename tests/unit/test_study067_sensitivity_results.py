"""CPU synthetic checks for the separate, frozen-machinery sensitivity report."""

import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights
from seis_interp.evaluation.study067_sensitivity_results import (
    c3_contrast,
    evaluate_sensitivity,
    paired_rows,
    primary_self_check,
    reused_weights,
    single_cell,
)
from seis_interp.evaluation.study067_statistics import adjust_families, contrast_table


def seed_tables():
    left, right = {}, {}
    for seed, cells in zip((7321, 7322, 7323), ([0, 1, 2, 3], [0, 2, 3], [0, 1, 3]), strict=True):
        left[seed] = pd.DataFrame(
            {
                "cell_id": cells,
                "trace_id": [f"trace_{cell}" for cell in cells],
                "source_cell": [cell // 2 for cell in cells],
                "clean_target": True,
                "target_energy": 1.0,
                "snr_db": np.arange(len(cells), dtype=float) + seed / 1000,
                "relative_mse": np.arange(len(cells), dtype=float) / 10 + 0.1,
            }
        )
        right[seed] = left[seed].copy()
        right[seed]["snr_db"] -= np.arange(len(cells)) / 5 + 0.1
        right[seed]["relative_mse"] += np.arange(len(cells)) / 100 + 0.01
        right[seed] = right[seed].iloc[::-1]
    return left, right


@pytest.mark.parametrize("validation_sets", [(), (["trace_1"],)])
def test_sensitivity_contrast_exactly_equals_primary_machinery(validation_sets):
    left, right = seed_tables()
    seeds, clusters = [7321, 7322, 7323], np.array([0, 1])
    weights = cluster_bootstrap_weights(2, 97, 670703)
    tests = {
        (method, seed): tables[seed]
        for method, tables in (("regsi_real", left), ("ccnet5d_grid", right))
        for seed in seeds
    }
    expected = contrast_table(
        tests,
        ["ccnet5d_grid"],
        seeds,
        clusters,
        weights,
        cluster_column="source_cell",
        validation_sets=validation_sets,
    ).assign(dataset="c3")
    actual = c3_contrast(left, right, seeds, clusters, weights, validation_sets)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.trace_count_seed_2.iloc[0] == 3
    assert "holm_p_value" not in actual


@pytest.mark.parametrize(
    "mismatch", ["cell_id", "trace_id", "source_cell", "clean_target", "target_energy", "duplicate"]
)
def test_pairing_guard_rejects_mismatched_rows(mismatch):
    left, right = seed_tables()
    a, b = left[7321].copy(), right[7321].copy()
    if mismatch == "duplicate":
        b = pd.concat([b, b.iloc[[0]]])
    else:
        b.loc[b.index[0], mismatch] = {
            "cell_id": 99,
            "trace_id": "different",
            "source_cell": 99,
            "clean_target": False,
            "target_energy": 2.0,
        }[mismatch]
    with pytest.raises(ValueError):
        paired_rows(a, b)
    right[7321] = b
    with pytest.raises(ValueError):
        c3_contrast(
            left,
            right,
            [7321, 7322, 7323],
            np.array([0, 1]),
            cluster_bootstrap_weights(2, 31, 670703),
        )


def test_weights_reuse_requires_exact_rng_and_source_order(tmp_path):
    path = tmp_path / "bootstrap_weights.npz"
    clusters = np.array([10, 20, 30])
    weights = cluster_bootstrap_weights(3, 31, 670703)
    np.savez(path, clusters=clusters, weights=weights)
    np.testing.assert_array_equal(reused_weights(path, clusters, resamples=31), weights)
    with pytest.raises(ValueError, match="universe"):
        reused_weights(path, clusters[::-1], resamples=31)
    changed = weights.copy()
    changed[0] = (changed[0] + 1) % 4
    np.savez(path, clusters=clusters, weights=changed)
    with pytest.raises(ValueError, match="weights differ"):
        reused_weights(path, clusters, resamples=31)
    np.savez(path, clusters=clusters, weights=weights)
    with pytest.raises(ValueError, match="weights differ"):
        reused_weights(path, clusters, resamples=32)


def test_primary_self_check_is_exact_including_original_holm(tmp_path):
    left, right = seed_tables()
    computed = c3_contrast(
        left, right, [7321, 7322, 7323], np.array([0, 1]), cluster_bootstrap_weights(2, 31, 670703)
    )
    families = {"primary_snr": "snr_db", "relmse": "relative_mse"}
    registered = adjust_families(
        pd.concat(
            [
                computed.assign(
                    dataset=dataset, baseline=baseline, contrast=f"regsi_real_minus_{baseline}"
                )
                for dataset in ("c3", "forge")
                for baseline in ("ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid")
            ],
            ignore_index=True,
        ),
        families,
    )
    path = tmp_path / "contrasts.csv"
    registered.to_csv(path, index=False)
    registered = pd.read_csv(path, float_precision="round_trip")
    actual = primary_self_check(computed, registered, families)
    assert actual.to_csv(index=False) == registered.iloc[:2].to_csv(index=False)
    computed.loc[0, "estimate"] = np.nextafter(computed.loc[0, "estimate"], np.inf)
    with pytest.raises(AssertionError):
        primary_self_check(computed, registered, families)


def test_forge_single_cell_uses_production_clean_population_and_paired_differences():
    left, right = seed_tables()
    a, b = left[7321].copy(), right[7321].sort_index().copy()
    a.loc[0, "clean_target"] = b.loc[0, "clean_target"] = False
    a.loc[0, "snr_db"] = -10000
    records = []
    for table in (a, b):
        selected = table.loc[table.clean_target]
        records.append(
            {
                "id": "a03_b02",
                "trace_seed": 600702,
                "metrics": {
                    "validation": {
                        "clean_target": {
                            "trace_count": len(selected),
                            "mean_trace_snr_db": float(selected.snr_db.mean()),
                            "masked_trace_relative_mse": float(selected.relative_mse.mean()),
                        }
                    }
                },
            }
        )
    clusters = np.array([0, 1])
    weights = cluster_bootstrap_weights(2, 101, 670703)
    result = single_cell(a, b, *records, "forge", "S-a", clusters, weights)
    delta = (a.snr_db - b.snr_db).loc[a.clean_target]
    assert result["trace_count"] == 3
    assert result["difference_db"] == pytest.approx(delta.mean())
    assert result["descriptive_paired_trace_mean_db"] == delta.mean()
    sums = np.array([delta.iloc[0], delta.iloc[1:].sum()])
    draws = (weights @ sums) / (weights @ np.array([1, 2]))
    np.testing.assert_array_equal(
        [result["descriptive_ci95_low"], result["descriptive_ci95_high"]],
        np.percentile(draws, [2.5, 97.5]),
    )


def test_output_guard_protects_primary_before_reading_any_inputs(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    for output in (primary, primary / "sensitivity"):
        with pytest.raises(ValueError, match="outside the primary analysis"):
            evaluate_sensitivity(tmp_path, {}, primary, output)
    assert list(primary.iterdir()) == []
