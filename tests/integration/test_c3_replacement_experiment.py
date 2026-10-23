import json
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS, validate_c3_fair_grid_configs
from seis_interp.data import c3_region_inputs as data
from seis_interp.data.c3_experiment_sources import (
    c3_method_experiments,
    publish_c3_replacement_readiness,
)
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.pipelines import run_c3_region_experiment as pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training import ccnet5d_observed_training, relational_trace_graph_poc_trainer
from seis_interp.training import forge_mvp_methods as methods
from seis_interp.training.worker_processes import process_start_time
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.c3_replacement_experiment import tiny_c3_reference_study


@pytest.fixture(
    params=[("regsi_real", "nersi_profile"), ("regsi_real", "ccnet5d_grid", "nersi_profile")]
)
def prepared(tmp_path, monkeypatch, request):
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    previous, _, _ = tiny_c3_trace_study(tmp_path)
    original = data.prepare_c3_region_experiment(tmp_path, previous)
    study, contract, configs = tiny_c3_reference_study(tmp_path, original, request.param)
    replacement = data.prepare_c3_region_experiment(tmp_path, study)
    return tmp_path, original, replacement, study, contract, configs


def test_replacement_freezes_identical_inputs_and_declared_new_25_candidate_searches(prepared):
    _, original, replacement, _, contract, _ = prepared
    old = json.loads((original / "experiment.json").read_text())
    new = json.loads((replacement / "experiment.json").read_text())
    assert new["methods"] == contract["methods"]
    assert new["total_runs"] == 28 * len(contract["methods"])
    references = {"regsi_real": "a01_b03", "nersi_profile": "a01_b04"}
    if "ccnet5d_grid" in contract["methods"]:
        references["ccnet5d_grid"] = "a01_b01"
    assert new["reference_configurations"] == references
    for name, value in old["artifacts"].items():
        if name.startswith("input/"):
            assert new["artifacts"][name] == value
    sources = c3_method_experiments(replacement)
    assert set(sources) == set(C3_FAIR_GRID_METHODS)
    for method in C3_FAIR_GRID_METHODS:
        assert sources[method] == (replacement if method in contract["methods"] else original)
    assert contract["training_loss"] == "masked_trace_mse"


def _new_state(experiment, method, completed):
    write_json(
        experiment / "workers" / f"{method}.json",
        {
            "method_id": method,
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "status": "running",
            "s1_completed": completed,
            "completed_s1": 25 if completed else 0,
            "selected_configuration": "a01_b01" if completed else None,
        },
    )


def test_both_barriers_wait_for_new_searches_and_preserve_original_stop_records(prepared):
    _, original, replacement, _, contract, _ = prepared
    originals = {
        method: json.loads((original / "workers" / f"{method}.json").read_text())
        for method in contract["methods"]
    }
    for method in C3_FAIR_GRID_METHODS:
        if method not in contract["methods"]:
            _new_state(original, method, True)
        elif method != "nersi_profile":
            _new_state(replacement, method, True)
    _new_state(replacement, "nersi_profile", False)
    publish_c3_replacement_readiness(replacement)
    assert not pipeline.c3_s1_barrier_ready(original)
    assert not pipeline.c3_s1_barrier_ready(replacement)
    state = json.loads((original / "workers/regsi_real.json").read_text())
    assert state["status"] == "replaced" and state["replacement_completed_s1"] == 25
    assert state["completed_s1"] == originals["regsi_real"]["completed_s1"] == 2
    assert state["original_s1_completed"] is False
    _new_state(replacement, "nersi_profile", True)
    publish_c3_replacement_readiness(replacement)
    assert pipeline.c3_s1_barrier_ready(original)
    assert pipeline.c3_s1_barrier_ready(replacement)
    write_json(original / "workers/drr_grid.json", {"status": "failed", "failure_reason": "SVD"})
    assert pipeline.c3_s1_barrier_ready(original)
    assert pipeline.c3_s1_barrier_ready(replacement)


@pytest.mark.parametrize("method", ("regsi_real", "nersi_profile", "ccnet5d_grid"))
def test_replacement_methods_actually_optimize_mse_with_exact_trace_updates(
    prepared, monkeypatch, method
):
    repo, _, replacement, _, contract, configs = prepared
    if method not in configs:
        pytest.skip("method continues in the original experiment")
    captured = []
    loss_module = {
        "nersi_profile": methods,
        "ccnet5d_grid": ccnet5d_observed_training,
        "regsi_real": relational_trace_graph_poc_trainer,
    }[method]
    actual = loss_module.masked_trace_loss

    def recorded(prediction, target, *, loss_name, **options):
        assert loss_name == "masked_trace_mse"
        assert prediction.shape == target.shape == (2, 24)
        result = actual(prediction, target, loss_name=loss_name, **options)
        expected = torch.nn.functional.mse_loss(prediction.double(), target.double())
        torch.testing.assert_close(result, expected)
        captured.append(float(result.detach()))
        return result

    monkeypatch.setattr(loss_module, "masked_trace_loss", recorded)
    grid = yaml.safe_load((replacement / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    record = pipeline.execute_c3_region_run(repo, replacement, candidate, 21, "S1")
    assert record["status"] == "completed" and record["test_scoring_calls"] == 0
    assert record["observed_trace_count"] == 46
    assert np.isfinite(record["validation_score"])
    assert len(captured) == contract["matched_training"]["max_steps"]
    output = replacement / "method_runs" / method / "S1" / candidate["id"]
    checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert checkpoint["selection_rule"] == "fixed_final_ema"
    assert checkpoint["steps_completed"] == contract["matched_training"]["max_steps"]


def test_default_c3_loss_contract_still_rejects_unannounced_mse(prepared):
    _, _, _, _, contract, configs = prepared
    contract = {**contract}
    contract.pop("training_loss")
    with pytest.raises(ValueError, match="existing relative-MSE"):
        validate_c3_fair_grid_configs(configs, contract)


def test_replacement_refuses_a_changed_holdout_assignment(prepared):
    repo, original, _, study, _, _ = prepared
    config = yaml.safe_load((study / "config.yaml").read_text())
    config["validation_study_id"] = "different_namespace"
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="input hash mismatch"):
        data.prepare_c3_region_experiment(repo, study)
    state = json.loads((original / "workers/regsi_real.json").read_text())
    assert state["status"] == "stopped"


def test_replacement_refuses_to_restart_a_continuing_method(prepared):
    repo, _, replacement, _, _, _ = prepared
    with pytest.raises(ValueError, match="not active"):
        pipeline.run_c3_region_worker(repo, replacement, "pocs_grid")


def test_final_aggregation_reads_each_method_from_its_own_source(prepared):
    _, original, replacement, _, _, _ = prepared
    sources = c3_method_experiments(replacement)
    for method, source in sources.items():
        (source / "workers").mkdir(exist_ok=True)
        pd.DataFrame(
            [{"domain": "test", "metric": "rmse", "mean": 1, "sample_std": 0, "seed_count": 3}]
        ).to_csv(source / "workers" / f"{method}_summary.csv", index=False)
        for seed in (7311, 7312, 7313):
            write_json(
                source / "method_runs" / method / "evaluation" / f"seed_{seed}/manifest.json",
                {
                    "id": "a01_b01",
                    "trace_seed": seed,
                    "metrics": {"validation": {"rmse": 1}, "test": {"rmse": 1}},
                },
            )
    pipeline._write_c3_experiment_aggregates(replacement, sources)
    summary = pd.read_csv(replacement / "summary.csv")
    metrics = pd.read_csv(replacement / "seed_metrics.csv")
    assert set(summary.method_id) == set(C3_FAIR_GRID_METHODS)
    assert len(metrics) == 30 and set(metrics.method_id) == set(C3_FAIR_GRID_METHODS)
    assert not (original / "summary.csv").exists()


def test_worker_releases_each_configuration_slot_before_the_barrier_and_seed_runs(
    prepared, monkeypatch
):
    repo, _, replacement, _, contract, _ = prepared
    method = "regsi_real"
    for name, source in c3_method_experiments(replacement).items():
        if name != method:
            _new_state(source, name, True)
    held = False
    memory = []
    state_path = replacement / "workers" / f"{method}.json"

    @contextmanager
    def slot(directory, device, minimum, waiting):
        nonlocal held
        assert not held
        waiting({"device": device, "reason": "another_experiment_run"})
        assert json.loads(state_path.read_text())["phase"] == "waiting_for_gpu"
        held = True
        memory.append(minimum)
        try:
            yield
        finally:
            held = False

    def execute(repo, experiment, candidate, seed, stage, **options):
        assert held
        state = json.loads(state_path.read_text())
        assert state["phase"] == stage and "resource_wait" not in state
        return {"id": candidate["id"], "status": "completed", "validation_score": 1}

    barrier = pipeline.c3_s1_barrier_ready

    def unheld_barrier(experiment):
        assert not held
        return barrier(experiment)

    monkeypatch.setattr(pipeline, "cuda_run_slot", slot)
    monkeypatch.setattr(pipeline, "execute_c3_region_run", execute)
    monkeypatch.setattr(pipeline, "c3_s1_barrier_ready", unheld_barrier)
    monkeypatch.setattr(pipeline, "summarize_c3_region_seeds", lambda _, **kwargs: pd.DataFrame())
    pipeline.run_c3_region_worker(repo, replacement, method)
    requirements = contract["cuda_execution"]["minimum_free_memory_gib"][method]
    assert memory == list(requirements.values()) + [requirements["a01_b01"]] * 6
    state = json.loads(state_path.read_text())
    assert state["status"] == "completed"
    assert state["completed_s1"] == 25 and state["completed_evaluation"] == 3
    assert not held and "resource_wait" not in state
