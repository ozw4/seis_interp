import json
import shutil
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.pipelines.fair_grid_stage import (
    FairGridContext,
    fair_grid_status,
    plan_fair_grid_stage,
    select_fair_grid,
)
from seis_interp.pipelines.tune_forge_matched import (
    TUNABLE_SECTIONS,
    matched_candidate_config,
    search_scoring,
)
from seis_interp.processing.fair_grid_rules import apply_overrides, final_overrides
from seis_interp.processing.forge_mvp_contract import METHODS

STUDY = Path(__file__).resolve().parents[2] / "studies/study_060_forge_fair_grid"
GRID = STUDY / "grid.yaml"
STUDY_RUNS = "runs/study_060_forge_fair_grid"
SHAPE = (6, 32, 8, 32)
PREPARATIONS = {
    "validation": {"path": f"{STUDY_RUNS}/validation_prep", "manifest_sha256": "a" * 64},
    "final": {"path": f"{STUDY_RUNS}/final_prep", "manifest_sha256": "b" * 64},
}
KEYS = ("0.999", "0.9995", "0.9998")
_STAMPS = count()


@pytest.fixture(scope="module")
def context():
    """Study method configs on random masks with the registered observed counts."""
    configs = {m: yaml.safe_load((STUDY / f"{m}.yaml").read_text()) for m in METHODS}
    budget = yaml.safe_load((STUDY / "config.yaml").read_text())["matched_training"]
    cells = np.random.default_rng(600701).permutation(int(np.prod(SHAPE)))[:14745]
    final = np.zeros(int(np.prod(SHAPE)), dtype=bool)
    final[cells] = True
    validation = final.copy()
    validation[cells[:1474]] = False
    return FairGridContext(
        PREPARATIONS,
        {"validation": configs, "final": configs},
        {"validation": validation.reshape(SHAPE), "final": final.reshape(SHAPE)},
        budget,
        4343,
    )


def _stage(repo, method, stage):
    return json.loads((repo / STUDY_RUNS / "fair_grid" / method / stage / "stage.json").read_text())


DIVERGED = "RuntimeError: non-finite CCNet-5D training loss at step 17"
FLOAT_LIKE_HASH = "8228172598e729840098733f451de5541f809dd7925f2e029355259e55d172a1"
OUT_OF_MEMORY = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


def _complete(
    repo,
    method,
    stage,
    score=lambda name: 0.9,
    loss=lambda name: 1.0,
    status=None,
    failure=None,
    only=None,
):
    """Write one matched-tuning record per planned run, as the tuner would.

    ``status`` and ``failure`` map run names to a tuner status and failure reason.
    """
    document = _stage(repo, method, stage)
    domain = "final" if stage == "final" else "validation"
    for run in document["runs"]:
        if only is not None and run["name"] not in only:
            continue
        output = repo / STUDY_RUNS / f"{next(_STAMPS):06d}_test_{run['method_id']}_matched_tuning"
        output.mkdir(parents=True)
        shutil.copyfile(repo / run["yaml"], output / "search.yaml")
        state = (status or {}).get(run["name"], "completed")
        manifest = {
            "status": state,
            "method_id": run["method_id"],
            "preparation": PREPARATIONS[domain]["path"],
        }
        if run["name"] in (failure or {}):
            manifest["failure_reason"] = failure[run["name"]]
        (output / "manifest.json").write_text(json.dumps(manifest))
        if state != "completed":
            continue
        classical = run["method_id"] in ("pocs_grid", "drr_grid")
        rows = [
            {
                "name": run["name"],
                "ema_decay": None if classical else key,
                "primary_relative_mse": score(run["name"]) + 0.01 * i,
                "final_training_loss_window_mean": loss(run["name"]),
                # A tuner hash that the pandas C parser would read as a huge float.
                **({} if classical else {"initialization_hash": FLOAT_LIKE_HASH}),
            }
            for i, key in enumerate([None] if classical else KEYS)
        ]
        pd.DataFrame(rows).to_csv(output / "search_results.csv", index=False)


def _grid_copy(tmp_path, edit):
    grid = yaml.safe_load(GRID.read_text())
    edit(grid["methods"])
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(grid, sort_keys=False))
    return path


@pytest.mark.parametrize("method", ["regsi", "ccnet5d_grid", "nersi_real", "pocs_grid", "drr_grid"])
def test_s1_yamls_pass_the_matched_tuner_and_record_their_checks(tmp_path, context, method):
    directory = plan_fair_grid_stage(tmp_path, GRID, method, "S1", context)
    document = _stage(tmp_path, method, "S1")
    assert document["grid_sha256"] == sha256_file(GRID)
    assert document["preparations"] == PREPARATIONS and document["rule_inputs"] == []
    assert len(document["runs"]) == len(list(directory.glob("*.yaml"))) in (9, 16)
    for run in document["runs"]:
        search = yaml.safe_load((tmp_path / run["yaml"]).read_text())
        assert sha256_file(tmp_path / run["yaml"]) == run["sha256"]
        assert search_scoring({}, search)["purpose"] == "selection" and search["evaluate"]
        training = search["candidates"][0].get("training", {})
        if method in ("pocs_grid", "drr_grid"):
            assert not training
        else:
            assert training["ema_eval_decays"] == [0.999, 0.9995, 0.9998]
            assert training["ema_snapshot_updates"] == [10000]
        base = context.base_configs["validation"][run["method_id"]]
        config, parameters = matched_candidate_config(
            base, search["candidates"][0], search["device"], context.budget, 4343, 13271
        )
        expected = apply_overrides(base, run["overrides"])
        for section in TUNABLE_SECTIONS[run["method_id"]]:
            assert config[section] == expected[section]
        checks = run["feasibility"]
        assert checks["validation"]["parameter_count"] == parameters
        assert {checks[d]["observed_trace_count"] for d in checks} == {13271, 14745}
        if method == "regsi":
            assert all(checks[d]["hidden_traces_per_episode"] % 64 == 0 for d in checks)
        if method == "ccnet5d_grid":
            assert all(checks[d]["min_hidden_traces_per_patch"] >= 64 for d in checks)
    with pytest.raises(FileExistsError):
        plan_fair_grid_stage(tmp_path, GRID, method, "S1", context)


def test_a_stage_waits_for_every_earlier_run_of_the_same_grid_and_preparations(tmp_path, context):
    with pytest.raises(ValueError, match="stage S1 has not been planned"):
        plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", context)
    plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S1", context)
    with pytest.raises(ValueError, match=r"incomplete \(s1_mid105_az105: pending\)"):
        plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", context)
    # A failure that is not a divergence (e.g. out of memory) waits for a re-run.
    _complete(
        tmp_path,
        "nersi_real",
        "S1",
        status={"s1_mid107_az110": "failed"},
        failure={"s1_mid107_az110": OUT_OF_MEMORY},
    )
    with pytest.raises(ValueError, match=r"s1_mid107_az110: failed"):
        plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", context)
    _complete(tmp_path, "nersi_real", "S1")
    with pytest.raises(ValueError, match="several completed tuning runs"):
        plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", context)


def test_a_diverged_run_counts_as_finished_ranks_last_and_is_never_rerun(tmp_path, context):
    plan_fair_grid_stage(tmp_path, GRID, "ccnet5d_grid", "S1", context)
    diverged = "s1_p4848_f40_d4"
    _complete(
        tmp_path,
        "ccnet5d_grid",
        "S1",
        score=lambda name: 0.7 if name == "s1_p4848_f30_d2" else 0.8,
        status={diverged: "failed"},
        failure={diverged: DIVERGED},
    )
    plan_fair_grid_stage(tmp_path, GRID, "ccnet5d_grid", "S2", context)
    document = _stage(tmp_path, "ccnet5d_grid", "S2")
    assert document["decision"]["base"]["run"] == "s1_p4848_f30_d2"
    inputs = {row["run"]: row for row in document["rule_inputs"]}
    assert len(inputs) == 9
    assert inputs[diverged]["status"] == "diverged"
    assert inputs[diverged]["scores"] is None and inputs[diverged]["train_loss"] is None
    assert inputs["s1_p4848_f30_d2"]["status"] == "completed"
    status = fair_grid_status(tmp_path, GRID).set_index("run")
    assert status.loc[diverged, "status"] == "diverged"
    assert np.isnan(status.loc[diverged, "best_score"])
    # Re-running the diverged yaml is refused, whatever its outcome.
    _complete(tmp_path, "ccnet5d_grid", "S1", only={diverged})
    with pytest.raises(ValueError, match=f"{diverged}: diverged and was re-run"):
        plan_fair_grid_stage(tmp_path, GRID, "ccnet5d_grid", "S3", context)


def test_changed_grid_or_preparations_are_refused(tmp_path, context):
    plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S1", context)
    _complete(tmp_path, "nersi_real", "S1")
    changed = _grid_copy(tmp_path, lambda m: m["nersi_real"]["s2"].pop())
    with pytest.raises(ValueError, match="grid.yaml changed"):
        plan_fair_grid_stage(tmp_path, changed, "nersi_real", "S2", context)
    other = FairGridContext(
        {**PREPARATIONS, "final": {"path": "elsewhere", "manifest_sha256": "c" * 64}},
        context.base_configs,
        context.observed_masks,
        context.budget,
        context.canvas_sample_count,
    )
    with pytest.raises(ValueError, match="other preparations"):
        plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", other)
    assert plan_fair_grid_stage(tmp_path, GRID, "nersi_real", "S2", context).exists()


def _wide_regsi(methods):
    methods["regsi"]["axis1"]["values"][0]["set"]["model.width"] = 200


def _unaligned_regsi(methods):
    methods["regsi"]["axis2"]["values"][1]["set"]["training.inner_mask_fraction"] = 0.0998
    methods["regsi"]["inner_mask_pairs"].append({"nominal": 0.1, "search": 0.0998, "final": 0.0998})


def _thin_ccnet(methods):
    methods["ccnet5d_grid"]["axis1"]["values"][1]["set"]["patches.inner_mask_fraction"] = 0.25


@pytest.mark.parametrize(
    ("method", "edit", "message"),
    [
        ("regsi", _wide_regsi, "outside the matched model size"),
        ("regsi", _unaligned_regsi, "hides 1324 of 13271 traces, not a multiple of 64"),
        ("ccnet5d_grid", _thin_ccnet, "hides .* < 64 traces at N=13271"),
    ],
)
def test_infeasible_grid_points_are_refused_before_anything_is_written(
    tmp_path, context, method, edit, message
):
    with pytest.raises(ValueError, match=message):
        plan_fair_grid_stage(tmp_path, _grid_copy(tmp_path, edit), method, "S1", context)
    assert not (tmp_path / STUDY_RUNS / "fair_grid" / method / "S1").exists()


def test_regsi_stages_replay_to_real_and_grid_final_runs(tmp_path, context):
    scores = {"s1_k2r2w128_imf10": 0.70, "s1_k1r3w112_imf10": 0.71}
    for stage in ("S1", "S2", "S3", "E"):
        plan_fair_grid_stage(tmp_path, GRID, "regsi", stage, context)
        _complete(
            tmp_path,
            "regsi",
            stage,
            score=lambda name: scores.get(name, 0.9),
            loss=lambda name: 1.2 if name == "s3_wd0.2" else 1.0,
        )
    e = _stage(tmp_path, "regsi", "E")
    assert (e["decision"]["branch"], e["decision"]["axis"]) == ("midpoint", "axis2")
    assert [run["name"] for run in e["runs"]] == ["e_imf075"]
    assert e["runs"][0]["feasibility"]["final"]["hidden_traces_per_episode"] == 1088
    assert len(e["rule_inputs"]) == 15 and e["rule_inputs"][0]["tuning_run"].startswith(STUDY_RUNS)
    plan_fair_grid_stage(tmp_path, GRID, "regsi", "R", context)
    reseed = _stage(tmp_path, "regsi", "R")
    assert [row["run"] for row in reseed["decision"]["top_runs"]] == list(scores)
    assert [run["method_id"] for run in reseed["runs"]].count("regsi_grid") == 6
    grid_run = next(run for run in reseed["runs"] if run["method_id"] == "regsi_grid")
    assert yaml.safe_load((tmp_path / grid_run["yaml"]).read_text())["method_id"] == "regsi_grid"

    def reseed_score(name):
        first = "k2r2w128" in name
        if "_grid_" in name:
            return 0.80 if first else 0.72
        return 0.71 if first else 0.72

    _complete(tmp_path, "regsi", "R", score=reseed_score)
    directory = select_fair_grid(tmp_path, GRID, "regsi", context)
    selection = _stage(tmp_path, "regsi", "final")
    assert (selection["decision"]["run"], selection["decision"]["ema_decay"]) == (
        "s1_k1r3w112_imf10",
        "0.999",
    )
    assert len(selection["rule_inputs"]) == 16 + 10
    spec = yaml.safe_load(GRID.read_text())["methods"]["regsi"]
    for run in selection["runs"]:
        search = yaml.safe_load((tmp_path / run["yaml"]).read_text())
        scoring = search_scoring({}, search)
        assert scoring == {
            "selection_domain": "final_test",
            "purpose": "final_evaluation",
            "evaluate": False,
            "independent_test_estimate": True,
        }
        candidate = search["candidates"][0]
        assert candidate["training"]["inner_mask_fraction"] == 0.09983
        assert "ema_eval_decays" not in candidate["training"]
        assert run["overrides"]["training.ema_decay"] == 0.999
        assert "training.ema_eval_decays" not in run["overrides"]
        assert run["final_overrides"] == final_overrides(spec, run["overrides"])
        assert set(run["feasibility"]) == {"final"}
    seeds = [
        yaml.safe_load((tmp_path / r["yaml"]).read_text())["candidates"][0]["model_seed"]
        for r in selection["runs"]
    ]
    assert seeds == [7311, 7312, 7313] * 2
    assert all(
        "--preparation runs/study_060_forge_fair_grid/final_prep" in c
        for c in selection["commands"]
    )
    status = fair_grid_status(tmp_path, GRID)
    assert len(status) == 9 + 3 + 3 + 1 + 10 + 6
    assert set(status.loc[status.stage.ne("final"), "status"]) == {"completed"}
    assert set(status.loc[status.stage.eq("final"), "status"]) == {"pending"}
    best = status.set_index("run").loc["s1_k2r2w128_imf10"]
    assert best.best_score == pytest.approx(0.70) and best["v_0.9998"] == pytest.approx(0.72)
    assert directory.name == "final"


def test_classical_method_selects_its_best_point_for_one_final_run(tmp_path, context):
    plan_fair_grid_stage(tmp_path, GRID, "drr_grid", "S1", context)
    _complete(
        tmp_path, "drr_grid", "S1", score=lambda name: 0.7 if name == "r3_it12_damp2" else 0.8
    )
    select_fair_grid(tmp_path, GRID, "drr_grid", context)
    selection = _stage(tmp_path, "drr_grid", "final")
    assert selection["decision"]["run"] == "r3_it12_damp2"
    (run,) = selection["runs"]
    search = yaml.safe_load((tmp_path / run["yaml"]).read_text())
    assert (search["backend"], search["workers"], search["evaluate"]) == ("batched", 32, False)
    assert search["candidates"][0]["algorithm"] == {
        "rank": 3,
        "damping_power": 2,
        "n_iterations": 12,
    }


def test_devices_assign_runs_round_robin_without_changing_candidates(tmp_path, context):
    default_root, assigned_root = tmp_path / "default", tmp_path / "assigned"
    plan_fair_grid_stage(default_root, GRID, "ccnet5d_grid", "S1", context)
    plan_fair_grid_stage(
        assigned_root, GRID, "ccnet5d_grid", "S1", context, devices=("cuda:0", "cuda:1")
    )
    default, assigned = (_stage(r, "ccnet5d_grid", "S1") for r in (default_root, assigned_root))
    assert "devices" not in default and assigned["devices"] == ["cuda:0", "cuda:1"]
    for position, (plain, placed) in enumerate(zip(default["runs"], assigned["runs"], strict=True)):
        plain_yaml = yaml.safe_load((default_root / plain["yaml"]).read_text())
        placed_yaml = yaml.safe_load((assigned_root / placed["yaml"]).read_text())
        assert plain_yaml["device"] == "cuda:0"
        assert placed_yaml["device"] == ("cuda:0", "cuda:1")[position % 2]
        assert {**placed_yaml, "device": plain_yaml["device"]} == plain_yaml
        assert placed["overrides"] == plain["overrides"]


@pytest.mark.parametrize(
    ("method", "devices"),
    [("pocs_grid", ("cuda:0",)), ("ccnet5d_grid", ()), ("ccnet5d_grid", ("cuda:0", "cuda:0"))],
)
def test_devices_are_rejected_for_classical_methods_and_malformed_lists(
    tmp_path, context, method, devices
):
    with pytest.raises(ValueError, match="devices must be distinct"):
        plan_fair_grid_stage(tmp_path, GRID, method, "S1", context, devices=devices)
    assert not (tmp_path / STUDY_RUNS / "fair_grid" / method / "S1").exists()
