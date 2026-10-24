import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights, holm_adjust
from seis_interp.evaluation.study067_search_diagnostics import (
    ranking_summary,
    search_curve,
    winners_curse,
)
from seis_interp.evaluation.study067_statistics import (
    adjust_families,
    best_claims,
    bootstrap_seed_means,
    contrast_table,
    paired_seed_bootstrap,
    paired_t_interval,
    select_candidate,
    validation_free_mask,
)


def test_shared_weights_match_slow_reference_with_seed_specific_test_sets():
    clusters = np.array([10, 20, 30, 40])
    ids = [np.array([10, 10, 20, 30, 40]), np.array([10, 20, 20, 40]), np.array([10, 30, 40])]
    differences = [np.array([1, 2, 4, 7, 8.0]), np.array([-2, 5, 9, 6.0]), np.array([0, 2, 9.0])]
    weights = cluster_bootstrap_weights(4, 31, 670703)
    slow = []
    for weight in weights:
        seed_means = []
        for values, cells in zip(differences, ids, strict=True):
            repeated = []
            for cell, count in zip(clusters, weight, strict=True):
                repeated.extend(list(values[cells == cell]) * int(count))
            seed_means.append(np.mean(repeated))
        slow.append(seed_means)
    slow = np.asarray(slow)
    np.testing.assert_allclose(bootstrap_seed_means(differences, ids, clusters, weights), slow)
    result = paired_seed_bootstrap(differences, ids, clusters, weights)
    estimate = np.mean([x.mean() for x in differences])
    assert estimate != np.concatenate(differences).mean()
    assert result["estimate"] == pytest.approx(estimate)
    assert [result["ci95_low"], result["ci95_high"]] == pytest.approx(
        np.percentile(slow.mean(axis=1), [2.5, 97.5])
    )
    assert result["bootstrap_p_value"] == pytest.approx(
        (1 + (abs(slow.mean(axis=1) - estimate) >= abs(estimate)).sum()) / 32
    )
    for factor in (-2, 0.5):
        np.testing.assert_allclose(
            bootstrap_seed_means([factor * d for d in differences], ids, clusters, weights),
            factor * slow,
        )


def test_holm_and_separate_eight_contrast_families():
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    rows = [
        {"metric": metric, "bootstrap_p_value": p}
        for metric in ("snr_db", "relative_mse")
        for p in [0.001, 0.01, 0.02, 0.04, 0.2, 0.3, 0.5, 1]
    ]
    rows[8]["bootstrap_p_value"] = 0.9
    table = adjust_families(pd.DataFrame(rows), {"primary_snr": "snr_db", "relmse": "relative_mse"})
    for metric in ("snr_db", "relative_mse"):
        group = table.loc[table.metric.eq(metric)]
        np.testing.assert_allclose(group.holm_p_value, holm_adjust(group.bootstrap_p_value))
    assert table.holm_p_value.iloc[0] == 0.008
    with pytest.raises(ValueError, match="eight"):
        adjust_families(table.iloc[:-1], {"relmse": "relative_mse"})


def test_selection_ties_imports_and_divergence_rank_last():
    rows = [
        {"id": "a01_b01", "status": "diverged", "validation_score": -100},
        {"id": "a02_b01", "status": "completed", "validation_score": 0.3},
        {"id": "a01_b02", "status": "completed", "validation_score": 0.3, "imported_from": "pilot"},
    ]
    assert select_candidate(rows)["id"] == "a01_b02"
    assert select_candidate(rows[::-1])["id"] == "a01_b02"
    assert select_candidate([rows[0]])["status"] == "diverged"
    with pytest.raises(ValueError, match="unfinished"):
        select_candidate([dict(rows[1], status="running")])
    with pytest.raises(ValueError, match="duplicate"):
        select_candidate([rows[0], rows[0]])


def test_validation_free_subset_excludes_every_s1_validation_set():
    assert validation_free_mask(
        ["a", "b", "c", "d", "e"], [["a", "b"], ["b", "c"], ["e"]]
    ).tolist() == [False, False, False, True, False]


def test_paired_t_interval_df_two():
    result = paired_t_interval([1, 2, 3])
    assert result == pytest.approx(
        {"t_df": 2, "t_ci95_low": -0.484137711750329, "t_ci95_high": 4.484137711750329}
    )
    assert paired_t_interval([2, 2, 2])["t_ci95_low"] == 2
    with pytest.raises(ValueError, match="three"):
        paired_t_interval([1, 2])


def _tests():
    tables = {}
    for seed in (1, 2, 3):
        for method, offset in (("regsi_real", 2), ("baseline", 0)):
            tables[method, seed] = pd.DataFrame(
                {
                    "cell_id": [1, 2, 3],
                    "trace_id": ["a", "b", "c"],
                    "source_cell": [10, 10, 20],
                    "clean_target": True,
                    "snr_db": np.array([2, 5, 6]) + offset * seed,
                    "relative_mse": np.array([0.5, 0.4, 0.8]) - 0.05 * offset * seed,
                }
            )
    return tables


def test_contrasts_pair_by_identity_and_recompute_validation_free_subset():
    tables = _tests()
    # A method may save the same traces in a different order.
    tables["baseline", 2] = tables["baseline", 2].iloc[::-1]
    weights = cluster_bootstrap_weights(2, 20, 4)
    result = contrast_table(
        tables,
        ["baseline"],
        [1, 2, 3],
        [10, 20],
        weights,
        cluster_column="source_cell",
        validation_sets=[["a"]],
    )
    snr = result.loc[result.metric.eq("snr_db")].iloc[0]
    assert snr.estimate == 4
    assert snr.trace_count_seed_1 == 2
    assert snr.direction == "ReGSI 優位"
    assert snr.passes_best_condition
    tables["baseline", 1].loc[0, "trace_id"] = "wrong"
    with pytest.raises(ValueError, match="trace_id"):
        contrast_table(
            tables, ["baseline"], [1, 2, 3], [10, 20], weights, cluster_column="source_cell"
        )


def test_best_claim_requires_all_four_contrasts_and_all_seeds():
    table = pd.DataFrame(
        [
            {
                "dataset": "c3",
                "metric": metric,
                "passes_best_condition": True,
                "direction": "区別不能" if metric == "relative_mse" else "ReGSI 優位",
            }
            for metric in ("snr_db", "relative_mse")
            for _ in range(4)
        ]
    )
    assert best_claims(table).iloc[0].wording == "SNR では最良、relMSE では区別不能"
    table.loc[0, "passes_best_condition"] = False
    assert not best_claims(table).iloc[0].regsi_best_snr


def test_diverged_evaluation_seed_is_not_dropped_or_imputed():
    tables = _tests()
    tables["baseline", 2] = None
    result = contrast_table(
        tables,
        ["baseline"],
        [1, 2, 3],
        [10, 20],
        cluster_bootstrap_weights(2, 20, 4),
        cluster_column="source_cell",
    )
    assert result.estimate.isna().all()
    assert result.bootstrap_p_value.isna().all()
    assert result.delta_seed_1.tolist() == pytest.approx([2, -0.1])
    assert result.delta_seed_2.isna().all()
    assert result.direction.eq("算出不能").all()
    assert not result.passes_best_condition.any()
    # Do not invent a missing p-value or adjust a reduced family.
    complete = pd.DataFrame([{"metric": "snr_db", "bootstrap_p_value": 0.01}] * 7)
    family = adjust_families(
        pd.concat([complete, result.loc[result.metric.eq("snr_db")]], ignore_index=True),
        {"primary_snr": "snr_db"},
    )
    assert family.holm_p_value.isna().all()
    assert family.holm_status.str.contains("incomplete").all()


def test_bootstrap_rejects_empty_seed_replicate_without_redrawing():
    with pytest.raises(ValueError, match="no test traces"):
        bootstrap_seed_means([[1]], [[20]], [10, 20], np.array([[2, 0]]))


def test_search_diagnostics_keep_divergence_without_a_finite_penalty():
    records = [
        {"id": "a01_b01", "status": "completed", "validation_score": 0.2, "validation_snr_db": 7.0},
        {"id": "a01_b02", "status": "completed", "validation_score": 0.4, "validation_snr_db": 3.0},
        {
            "id": "a01_b03",
            "status": "diverged",
            "validation_score": -1,
            "validation_snr_db": np.nan,
        },
    ]
    curve = search_curve(records, "relative_mse")
    assert curve.probability_all_diverged.tolist() == pytest.approx([1 / 3, 1 / 9, 1 / 27])
    assert curve.conditional_expected_best.iloc[0] == pytest.approx(0.3)
    assert np.isposinf(curve.expected_best).all()
    assert ranking_summary(records)["spearman_rho"] == pytest.approx(1)


def test_winners_curse_selects_in_bag_and_scores_undrawn_cells():
    records = [{"id": str(i), "status": "completed"} for i in (0, 1)]
    validation = {
        str(i): pd.DataFrame(
            {
                "cell_id": [0, 1],
                "trace_id": ["a", "b"],
                "source_cell": [0, 1],
                "relative_mse": values,
            }
        )
        for i, values in enumerate(([0.0, 1.0], [1.0, 0.0]))
    }
    result = winners_curse(
        validation, records, "relative_mse", [0, 1], np.array([[2, 0], [0, 2]]), "source_cell"
    )
    assert result["bootstrap_inbag_mean"] == 0
    assert result["bootstrap_oob_mean"] == 1
    assert result["optimism_mean"] == 1
