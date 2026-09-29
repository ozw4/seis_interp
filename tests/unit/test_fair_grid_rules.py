from dataclasses import astuple
from math import inf
from pathlib import Path

import pytest
import yaml

from seis_interp.processing.fair_grid_rules import (
    AxisPosition,
    PlannedRun,
    RunRecord,
    apply_overrides,
    batch_aligned_fraction,
    best_pair,
    compose_stage,
    decay_key,
    e_decision,
    final_overrides,
    hidden_trace_count,
    is_divergence,
    merge_overrides,
    paired_seed_mean_choice,
    seed_mean_choice,
    select_final,
    top_runs,
)
from seis_interp.training.forge_mvp_methods import shadow_key

GRID = Path(__file__).resolve().parents[2] / "studies/study_060_forge_fair_grid/grid.yaml"
KEYS = ("0.999", "0.9995", "0.9998")


def _run(name, stage, *scores, loss=None):
    return RunRecord(name, stage, dict(zip(KEYS, scores, strict=True)), loss)


def _e(*args):
    return astuple(e_decision(*args))


def _axis(name, index, size=3, neighbours=None, extensions=(), midpoints=()):
    return AxisPosition(
        name, size, index, neighbours or {}, frozenset(extensions), frozenset(midpoints)
    )


def test_overrides_replace_dotted_paths_and_list_items_without_touching_the_input():
    config = {"model": {"frequency_base": [1.05, 1.05, 1.1, 1.1], "width": 128}}
    resolved = apply_overrides(config, {"model.frequency_base.3": 1.15, "model.width": 112})
    assert resolved == {"model": {"frequency_base": [1.05, 1.05, 1.1, 1.15], "width": 112}}
    assert config["model"]["frequency_base"][3] == 1.1
    with pytest.raises(ValueError, match="list index"):
        apply_overrides(config, {"model.frequency_base.4": 1.0})


def test_merged_overrides_let_later_layers_win_and_reject_nested_paths():
    assert merge_overrides({"a.b": 1, "a.c": 2}, {"a.b": 3}) == {"a.b": 3, "a.c": 2}
    with pytest.raises(ValueError, match="nests"):
        merge_overrides({"graph.relations": {}}, {"graph.relations.cmp": {}})


def test_winner_is_the_lowest_run_ema_pair_and_ties_follow_stage_then_name():
    runs = [_run("s2_b", "S2", 0.8, 0.7, 0.9), _run("s1_z", "S1", 0.9, 0.9, 0.7)]
    run, key = best_pair(runs)
    assert (run.name, key) == ("s1_z", "0.9998")
    tied = [_run("s1_b", "S1", 0.7, 0.7, 0.7), _run("s1_a", "S1", 0.8, 0.7, 0.9)]
    assert [(r.name, k) for r, k in [best_pair(tied)]] == [("s1_a", "0.9995")]


def test_top_runs_rank_by_best_ema_with_stable_ties():
    runs = [
        _run("s1_c", "S1", 0.90, 0.81, 0.85),
        _run("e_x", "E", 0.79, 0.95, 0.95),
        _run("s3_a", "S3", 0.80, 0.99, 0.99),
        _run("s1_b", "S1", 0.99, 0.99, 0.80),
    ]
    # s1_b and s3_a tie at 0.80: the earlier stage wins before the name.
    assert [run.name for run in top_runs(runs, 2)] == ["e_x", "s1_b"]
    with pytest.raises(ValueError, match="fewer"):
        top_runs(runs[:1], 2)


def test_non_finite_scores_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _run("s1_a", "S1", 0.9, float("nan"), 0.8)


def _diverged(name, stage):
    return RunRecord.diverged_run(name, stage, list(KEYS))


@pytest.mark.parametrize(
    "reason,diverged",
    [
        ("RuntimeError: non-finite CCNet-5D training loss at step 17", True),
        ("RuntimeError: non-finite PoC training loss", True),
        ("RuntimeError: nonfinite NeRSI loss at step 3", True),
        (
            "RuntimeError: The total norm of order 2.0 for gradients from `parameters` is "
            "non-finite, so it cannot be clipped.",
            True,
        ),
        ("ValueError: s1_a: non-finite prediction", True),
        ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", False),
        ("ValueError: scores must be a nonempty finite vector", False),
        (None, False),
    ],
)
def test_the_registered_pattern_recognises_every_divergence_failure(reason, diverged):
    grid = yaml.safe_load(GRID.read_text())
    pattern = grid["selection"]["diverged_runs"]["failure_reason_pattern"]
    assert is_divergence(reason, pattern) is diverged


def test_diverged_runs_count_but_rank_after_every_completed_run():
    with pytest.raises(ValueError, match="scores \\+inf"):
        RunRecord("s1_a", "S1", {"0.999": 0.9}, None, True)
    with pytest.raises(ValueError, match="scores \\+inf"):
        RunRecord("s1_a", "S1", {"0.999": inf}, 1.0, True)
    runs = [_diverged("s1_a", "S1"), _run("s1_b", "S1", 1.2, 1.3, 1.4)]
    assert best_pair(runs)[0].name == "s1_b"
    assert [run.name for run in top_runs(runs, 1)] == ["s1_b"]
    with pytest.raises(ValueError, match="fewer non-diverged runs"):
        top_runs(runs, 2)
    with pytest.raises(ValueError, match="every run to select from diverged"):
        best_pair(runs[:1])
    # A diverged seed makes its setting's mean +inf; all-diverged settings are refused.
    seeds = {("a", "x"): [0.5, inf, 0.5], ("b", "x"): [0.8, 0.8, 0.8]}
    assert seed_mean_choice(seeds) == (("b", "x"), pytest.approx(0.8))
    key, _ = paired_seed_mean_choice(seeds, {("a", "x"): [0.1], ("b", "x"): [0.9]})
    assert key == ("b", "x")
    with pytest.raises(ValueError, match="every setting has a diverged seed"):
        seed_mean_choice({("a", "x"): [inf]})
    with pytest.raises(ValueError, match="finite or"):
        seed_mean_choice({("a", "x"): [float("nan")]})


def test_a_diverged_wd_level_skips_the_wd_calibration():
    wd = _axis("weight_decay", 1, size=4, extensions=("upper",))
    edge = _axis("axis1", 2, extensions=("upper",))
    middle2 = _axis("axis2", 1, neighbours={0: 0.6, 2: 0.7}, midpoints={(0, 1)})
    assert _e(edge, middle2, wd, 1.0, 1.0, 0.05)[0] == "wd_calibration"
    for losses in ((1.0, None), (None, 1.0)):
        assert _e(edge, middle2, wd, *losses, 0.05) == ("edge_extension", "axis1", "upper")
    grid = yaml.safe_load(GRID.read_text())
    scores = {"s1_mid110_az110": 0.70}
    spec, planned, records = _searched(grid, "nersi_real", scores, {"s3_wd0.0001": 1.0})
    runs, decision = compose_stage("E", grid, spec, planned, records)
    assert decision["branch"] == "wd_calibration"
    records["s3_wd0.0001"] = _diverged("s3_wd0.0001", "S3")
    runs, decision = compose_stage("E", grid, spec, planned, records)
    assert (decision["branch"], runs[0].name) == ("edge_extension", "e_mid113")
    assert decision["train_loss_at_max_wd"] is None
    # A completed level without its training loss is still an error.
    records["s3_wd0.0001"] = _run("s3_wd0.0001", "S3", 0.9, 0.9, 0.9)
    with pytest.raises(ValueError, match="training loss of every S3 level"):
        compose_stage("E", grid, spec, planned, records)


def test_seed_mean_choice_and_regsi_real_grid_mean():
    real = {("a", "0.999"): [0.70, 0.72, 0.71], ("b", "0.999"): [0.72, 0.72, 0.72]}
    grid = {("a", "0.999"): [0.80, 0.80, 0.80], ("b", "0.999"): [0.74, 0.75, 0.76]}
    assert seed_mean_choice(real) == (("a", "0.999"), pytest.approx(0.71))
    key, score = paired_seed_mean_choice(real, grid)
    assert key == ("b", "0.999") and score == pytest.approx((0.72 + 0.75) / 2)
    tie = {("a", "x"): [0.5], ("b", "x"): [0.5]}
    assert seed_mean_choice(tie)[0] == ("a", "x")
    with pytest.raises(ValueError, match="same settings"):
        paired_seed_mean_choice(real, {("b", "0.999"): [0.7], ("a", "0.999"): [0.7]})


def test_e_rule_order_covers_every_branch():
    wd = _axis("weight_decay", 1, size=4, extensions=("upper",))
    wd_top = _axis("weight_decay", 3, size=4, extensions=("upper",))
    wd_bottom = _axis("weight_decay", 0, size=4, extensions=("lower", "upper"))
    middle = _axis("axis1", 1, neighbours={0: 0.8, 2: 0.7}, midpoints={(1, 2)})
    middle2 = _axis("axis2", 1, neighbours={0: 0.6, 2: 0.7}, midpoints={(0, 1)})
    # 1. calibration wins over every other branch; exactly +5% does not trigger it.
    edge = _axis("axis1", 2, extensions=("upper",))
    assert _e(edge, middle2, wd, 1.0, 1.049, 0.05)[0] == "wd_calibration"
    assert _e(edge, middle2, wd, 1.0, 1.05, 0.05) == ("edge_extension", "axis1", "upper")
    # 2. edge extension priority: axis 1, then axis 2, then wd; ends without values skip.
    lower_only_missing = _axis("axis1", 0, neighbours={1: 0.9}, extensions=("upper",))
    lower2 = _axis("axis2", 0, neighbours={1: 0.9}, extensions=("lower",))
    assert _e(lower_only_missing, lower2, wd, 1.0, 1.2, 0.05) == (
        "edge_extension",
        "axis2",
        "lower",
    )
    assert _e(middle, middle2, wd_top, 1.0, 1.2, 0.05) == (
        "edge_extension",
        "weight_decay",
        "upper",
    )
    assert _e(middle, middle2, wd_bottom, 1.0, 1.2, 0.05)[2] == "lower"
    # 3. midpoint towards the better neighbour on axis 1, else axis 2.
    assert _e(middle, middle2, wd, 1.0, 1.2, 0.05) == ("midpoint", "axis1", (1, 2))
    undefined = _axis("axis1", 1, neighbours={0: 0.6, 2: 0.7}, midpoints={(1, 2)})
    assert _e(undefined, middle2, wd, 1.0, 1.2, 0.05) == ("midpoint", "axis2", (0, 1))
    unordered = _axis("axis1", None)
    assert _e(unordered, middle2, wd, 1.0, 1.2, 0.05)[1] == "axis2"
    # 4. otherwise one wd step up; a missing upper value is an error.
    assert _e(unordered, _axis("axis2", None), wd, 1.0, 1.2, 0.05) == (
        "fallback_wd_extension",
        "weight_decay",
        "upper",
    )
    with pytest.raises(ValueError, match="no upper extension"):
        _e(unordered, _axis("axis2", None), _axis("weight_decay", 1), 1.0, 1.2, 0.05)


def test_regsi_inner_mask_pairs_follow_the_64_trace_rounding_rule():
    grid = yaml.safe_load(GRID.read_text())
    counts = grid["preparations"]["observed_trace_counts"]
    for pair in grid["methods"]["regsi"]["inner_mask_pairs"]:
        for domain, key in (("validation", "search"), ("final", "final")):
            fraction = batch_aligned_fraction(pair["nominal"], counts[domain], 64)
            assert fraction == pair[key]
            assert hidden_trace_count(counts[domain], fraction) % 64 == 0
    assert hidden_trace_count(counts["validation"], 0.0998) == 1324
    spec = grid["methods"]["regsi"]
    assert final_overrides(spec, {"training.inner_mask_fraction": 0.10127}) == {
        "training.inner_mask_fraction": 0.09983
    }
    with pytest.raises(ValueError, match="no pre-registered pair"):
        final_overrides(spec, {"training.inner_mask_fraction": 0.0998})


def test_decay_keys_match_the_tuner_shadow_keys():
    grid = yaml.safe_load(GRID.read_text())
    assert [decay_key(d) for d in grid["ema_eval_decays"]] == [
        shadow_key(d) for d in grid["ema_eval_decays"]
    ]


def test_every_neural_method_searches_sixteen_runs_and_classical_sixteen_points():
    grid = yaml.safe_load(GRID.read_text())
    for spec in grid["methods"].values():
        runs, _ = compose_stage("S1", grid, spec, {}, {})
        assert len({run.name for run in runs}) == len(runs)
        if "factors" in spec:
            assert len(runs) == 16
            with pytest.raises(ValueError, match="single stage"):
                compose_stage("S2", grid, spec, {}, {})
        else:
            assert len(runs) == 9
            assert len(spec["s2"]) == 3 and len(spec["weight_decay"]["s3"]) == 3


def test_s2_and_s3_compose_on_the_incumbent_and_regsi_s2_widens_the_one_round_topology():
    grid = yaml.safe_load(GRID.read_text())
    spec = grid["methods"]["regsi"]
    s1, _ = compose_stage("S1", grid, spec, {}, {})
    planned = {run.name: run for run in s1}
    records = {run.name: _run(run.name, "S1", 0.9, 0.9, 0.9) for run in s1}
    records["s1_k8r1w152_imf10"] = _run("s1_k8r1w152_imf10", "S1", 0.8, 0.7, 0.75)
    s2, decision = compose_stage("S2", grid, spec, planned, records)
    assert decision["base"] == {"run": "s1_k8r1w152_imf10", "ema_decay": "0.9995", "score": 0.7}
    fourier0 = next(run for run in s2 if run.name == "s2_fourier0")
    assert fourier0.overrides["model.width"] == 160
    assert fourier0.overrides["model.node_fourier_components"] == 0
    assert next(run for run in s2 if run.name == "s2_aniso2").overrides["model.width"] == 152
    planned |= {run.name: run for run in s2}
    records |= {run.name: _run(run.name, "S2", 0.9, 0.9, 0.9) for run in s2}
    records["s2_aniso2"] = _run("s2_aniso2", "S2", 0.69, 0.9, 0.9)
    s3, decision = compose_stage("S3", grid, spec, planned, records)
    assert decision["base"]["run"] == "s2_aniso2"
    assert [run.overrides["training.weight_decay"] for run in s3] == [0.01, 0.05, 0.2]
    assert all(run.overrides["graph.relations"]["cmp"]["midpoint_scale_m"] == 226.0 for run in s3)


def _searched(grid, method, scores, losses):
    """Plan S1-S3 with every run at 0.9 except the listed best-EMA scores."""
    spec = grid["methods"][method]
    planned, records = {}, {}
    for stage in ("S1", "S2", "S3"):
        runs, _ = compose_stage(stage, grid, spec, planned, records)
        for run in runs:
            planned[run.name] = run
            value = scores.get(run.name, 0.9)
            loss = losses.get(run.name, 1.0)
            records[run.name] = _run(run.name, stage, value, value + 0.01, value + 0.02, loss=loss)
    return spec, planned, records


def test_e_stage_midpoint_on_ccnet_axis_1_towards_the_better_neighbour():
    grid = yaml.safe_load(GRID.read_text())
    spec, planned, records = _searched(
        grid,
        "ccnet5d_grid",
        {"s1_p4848_f30_d2": 0.70, "s1_p4848_f40_d2": 0.75, "s1_p6686_f20_d2": 0.80},
        {"s3_wd0.0002": 1.2},
    )
    runs, decision = compose_stage("E", grid, spec, planned, records)
    assert decision["branch"] == "midpoint" and decision["target"] == [0, 1]
    assert [run.name for run in runs] == ["e_p4848_f35"]
    assert runs[0].overrides["patches.inner_mask_fraction"] == 0.35
    assert runs[0].overrides["model.time_dilation"] == 2


def test_e_stage_calibrates_weak_weight_decay_and_extends_edges():
    grid = yaml.safe_load(GRID.read_text())
    scores = {"s1_mid110_az110": 0.70}
    spec, planned, records = _searched(grid, "nersi_real", scores, {"s3_wd0.0001": 1.02})
    runs, decision = compose_stage("E", grid, spec, planned, records)
    assert decision["branch"] == "wd_calibration"
    assert decision["wd_level_runs"] == {
        "3e-05": "s3_wd3e-05",
        "5e-05": "s1_mid110_az110",
        "0.0001": "s3_wd0.0001",
        "7e-05": "s3_wd7e-05",
    }
    assert runs[0].name == "e_wd0.00015" and runs[0].overrides["model.frequency_base.0"] == 1.1
    spec, planned, records = _searched(grid, "nersi_real", scores, {"s3_wd0.0001": 1.2})
    runs, decision = compose_stage("E", grid, spec, planned, records)
    assert (decision["branch"], decision["axis"], runs[0].name) == (
        "edge_extension",
        "axis1",
        "e_mid113",
    )


def test_e_stage_skips_an_axis_that_s2_replaced():
    grid = yaml.safe_load(GRID.read_text())
    scores = {"s1_p4848_f30_d4": 0.70, "s2_tk3d1c58": 0.60}
    spec, planned, records = _searched(grid, "ccnet5d_grid", scores, {"s3_wd0.0002": 1.2})
    runs, decision = compose_stage("E", grid, spec, planned, records)
    positions = {p["axis"]: p["index"] for p in decision["axis_positions"]}
    assert positions == {"axis1": 1, "axis2": None, "weight_decay": 0}
    assert decision["e_base"]["run"] == "s2_tk3d1c58"
    # p4848_f30 neighbours score 0.9 each; the lower index wins the tie -> @0.35.
    assert runs[0].name == "e_p4848_f35" and runs[0].overrides["model.time_dilation"] == 1


def test_regsi_reseed_runs_and_final_selection_use_real_and_grid_means():
    grid = yaml.safe_load(GRID.read_text())
    scores = {"s1_k2r2w128_imf10": 0.70, "s1_k1r3w112_imf10": 0.71}
    spec, planned, records = _searched(grid, "regsi", scores, {"s3_wd0.2": 1.2})
    runs, decision = compose_stage("E", grid, spec, planned, records)
    # Topology is unordered and imf10 sits in the middle -> axis-2 midpoint towards imf05.
    assert decision["axis_positions"][0]["index"] is None
    assert runs[0].name == "e_imf075"
    assert runs[0].overrides["training.inner_mask_fraction"] == 0.07716
    planned[runs[0].name] = runs[0]
    records[runs[0].name] = _run(runs[0].name, "E", 0.95, 0.95, 0.95)
    reseed, decision = compose_stage("R", grid, spec, planned, records)
    assert [row["run"] for row in decision["top_runs"]] == list(scores)
    names = [run.name for run in reseed]
    assert names[:4] == [
        "r_s1_k2r2w128_imf10_seed7302",
        "r_s1_k2r2w128_imf10_seed7303",
        "r_s1_k1r3w112_imf10_seed7302",
        "r_s1_k1r3w112_imf10_seed7303",
    ]
    assert [run.method_id for run in reseed].count("regsi_grid") == 6
    for run in reseed:
        planned[run.name] = run
        grid_run = "_grid_" in run.name
        first = run.name.startswith("r_s1_k2r2w128")
        value = (0.80 if first else 0.72) if grid_run else (0.71 if first else 0.72)
        records[run.name] = _run(run.name, "R", value, value + 0.01, value + 0.02)
    choice, final = select_final(grid, spec, planned, records, decision)
    # Real-only means favour k2r2 (0.7067 vs 0.7167); real+grid favour k1r3.
    assert (choice["run"], choice["ema_decay"]) == ("s1_k1r3w112_imf10", "0.999")
    assert choice["score"] == pytest.approx(((0.71 + 0.72 + 0.72) / 3 + 0.72) / 2)
    assert [run.method_id for run in final] == ["regsi_real"] * 3 + ["regsi_grid"] * 3
    assert [run.overrides["model_seed"] for run in final] == [7311, 7312, 7313] * 2
    assert all(run.overrides["training.ema_decay"] == 0.999 for run in final)
    # Final runs predict the selected EMA only; the 10k snapshot is still saved.
    assert planned["s1_k1r3w112_imf10"].overrides["training.ema_eval_decays"] == [
        0.999,
        0.9995,
        0.9998,
    ]
    assert all("training.ema_eval_decays" not in run.overrides for run in final)
    assert all(run.overrides["training.ema_snapshot_updates"] == [10000] for run in final)
    assert final_overrides(spec, final[0].overrides)["training.inner_mask_fraction"] == 0.09983


def test_classical_selection_takes_the_lowest_validation_score():
    grid = yaml.safe_load(GRID.read_text())
    spec = grid["methods"]["drr_grid"]
    runs, _ = compose_stage("S1", grid, spec, {}, {})
    planned = {run.name: run for run in runs}
    records = {run.name: RunRecord(run.name, "S1", {"none": 0.8}) for run in runs}
    records["r3_it12_damp2"] = RunRecord("r3_it12_damp2", "S1", {"none": 0.7})
    choice, final = select_final(grid, spec, planned, records, None)
    assert choice["run"] == "r3_it12_damp2"
    assert final == [
        PlannedRun("final_r3_it12_damp2", "drr_grid", planned["r3_it12_damp2"].overrides)
    ]
