import json
import os
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS
from seis_interp.data import c3_region_inputs as data
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.pipelines import run_c3_region_experiment as pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training.worker_processes import process_start_time
from tests.fixtures.c3_region_experiment import tiny_c3_region_study


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data, "execution_identity", lambda _: {"git_commit": "testcommit", "dirty_worktree": True}
    )
    study, contract, configs = tiny_c3_region_study(tmp_path)
    experiment = data.prepare_c3_region_experiment(tmp_path, study)
    return tmp_path, experiment, study, contract, configs


def _ready(experiment):
    for method in C3_FAIR_GRID_METHODS:
        write_json(
            experiment / "workers" / f"{method}.json",
            {
                "status": "running",
                "s1_completed": True,
                "pid": os.getpid(),
                "process_start_time": process_start_time(os.getpid()),
            },
        )


def test_frozen_inputs_keep_all_heldout_amplitudes_outside_models(prepared):
    _, experiment, study, contract, _ = prepared
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["total_runs"] == 140 and seal["target_waveforms_read"] == 0
    for seed in [
        contract["region_split"]["search_seed"],
        *contract["region_split"]["evaluation_seeds"],
    ]:
        selection, partition = data.load_c3_region_inputs(experiment, seed, ("validation",))
        final, _ = data.load_c3_region_inputs(experiment, seed, ("validation", "test"))
        visible = partition.loc[partition.input_observed]
        assert visible.region_role.eq("train").all()
        assert visible.outer_split.eq("observed").all()
        np.testing.assert_array_equal(selection.observed_cells, visible.cell_id)
        np.testing.assert_array_equal(selection.observed_values, final.observed_values)
        assert len(selection.test_cells) == 64 and len(final.test_cells) == 128
    (study / "config.yaml").write_text("changed after preparation")
    assert yaml.safe_load((experiment / "study/config.yaml").read_text()) == contract


@pytest.mark.parametrize("method", C3_FAIR_GRID_METHODS)
def test_each_method_runs_on_the_same_observed_only_partition(prepared, method):
    repo, experiment, _, contract, configs = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    record = pipeline.execute_c3_region_run(
        repo, experiment, candidate, contract["region_split"]["search_seed"], "S1"
    )
    assert record["status"] == "completed"
    assert np.isfinite(record["validation_score"])
    assert record["test_scoring_calls"] == 0
    expected = json.loads((experiment / "experiment.json").read_text())["partitions"]["21"][
        "observed"
    ]
    assert record["observed_trace_count"] == expected


@pytest.mark.parametrize("method", ("regsi_real", "ccnet5d_grid", "nersi_profile"))
def test_neural_methods_use_requested_adamw_for_every_update(prepared, monkeypatch, method):
    repo, experiment, _, contract, configs = prepared
    optimizers = []

    class RecordedAdamW(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_count = 0
            optimizers.append(self)

        def step(self, *args, **kwargs):
            self.update_count += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", RecordedAdamW)
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    record = pipeline.execute_c3_region_run(
        repo, experiment, candidate, contract["region_split"]["search_seed"], "S1"
    )
    assert record["status"] == "completed" and len(optimizers) == 1
    optimizer = optimizers[0]
    assert optimizer.defaults["lr"] == contract["matched_training"]["learning_rate"]
    assert optimizer.defaults["weight_decay"] == 0
    assert optimizer.update_count == contract["matched_training"]["max_steps"]
    run = experiment / "method_runs" / method / "S1" / candidate["id"]
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    assert checkpoint["selection_rule"] == "fixed_final_ema"
    assert checkpoint["steps_completed"] == optimizer.update_count


@pytest.mark.parametrize("device", ("cuda:0", "cuda:1"))
def test_cuda_is_initialized_before_memory_counters_are_reset(prepared, monkeypatch, device):
    repo, experiment, _, contract, configs = prepared
    events = []
    original = pipeline.run_c3_region_method

    def run_on_cpu(inputs, config, reporter, **options):
        assert events == [("initialize", device), ("reset", device)]
        config = deepcopy(config)
        config["device"] = "cpu"
        return original(inputs, config, reporter, **options)

    monkeypatch.setattr(
        torch.cuda, "set_device", lambda value: events.append(("initialize", value))
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda value: events.append(("reset", value))
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(pipeline, "peak_memory", lambda _: {})
    monkeypatch.setattr(pipeline, "run_c3_region_method", run_on_cpu)
    configs["nersi_profile"]["device"] = device
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs["nersi_profile"], grid["methods"]["nersi_profile"])[0]
    record = pipeline.execute_c3_region_run(
        repo, experiment, candidate, contract["region_split"]["search_seed"], "S1"
    )
    assert record["status"] == "completed"


def test_evaluation_waits_for_all_five_s1_searches(prepared):
    repo, experiment, _, contract, configs = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs["nersi_profile"], grid["methods"]["nersi_profile"])[0]
    with pytest.raises(RuntimeError, match="all five S1"):
        pipeline.execute_c3_region_run(repo, experiment, candidate, 7311, "evaluation")
    _ready(experiment)
    assert pipeline.c3_s1_barrier_ready(experiment)
    write_json(
        experiment / "workers/pocs_grid.json", {"status": "failed", "failure_reason": "interrupted"}
    )
    assert pipeline.c3_s1_barrier_ready(experiment)
    assert contract["region_split"]["search_seed"] not in [7311, 7312, 7313]


def test_new_models_and_finalized_predictions_precede_single_test_reads(prepared, monkeypatch):
    repo, experiment, _, contract, configs = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs["nersi_profile"], grid["methods"]["nersi_profile"])[0]
    calls = []
    original = pipeline.read_c3_region_targets

    def recorded(repo, experiment, partition, domain):
        seed = int(partition.region_seed.iloc[0])
        stage = "S1" if seed == contract["region_split"]["search_seed"] else "evaluation"
        name = candidate["id"] if stage == "S1" else f"seed_{seed}"
        run = experiment / "method_runs/nersi_profile" / stage / name
        assert (run / "prediction.npy").is_file() and (run / "final.pt").is_file()
        record = json.loads((run / "manifest.json").read_text())
        assert "prediction_finalized_at" in record
        calls.append((seed, domain))
        return original(repo, experiment, partition, domain)

    monkeypatch.setattr(pipeline, "read_c3_region_targets", recorded)
    search = pipeline.execute_c3_region_run(
        repo, experiment, candidate, contract["region_split"]["search_seed"], "S1"
    )
    assert search["test_scoring_calls"] == 0 and set(search["metrics"]) == {"validation"}
    _ready(experiment)
    evaluations = [
        pipeline.execute_c3_region_run(repo, experiment, candidate, seed, "evaluation")
        for seed in (7311, 7312, 7313)
    ]
    assert all(record["test_scoring_calls"] == 1 for record in evaluations)
    assert (
        len({record["diagnostics"]["initialization_hash"] for record in [search, *evaluations]})
        == 1
    )
    assert calls == [
        (21, "validation"),
        (7311, "validation"),
        (7311, "test"),
        (7312, "validation"),
        (7312, "test"),
        (7313, "validation"),
        (7313, "test"),
    ]
    summary = pipeline.summarize_c3_region_seeds(evaluations)
    assert len(summary) == 10 and summary.seed_count.eq(3).all()
    before = list(calls)
    pipeline.execute_c3_region_run(repo, experiment, candidate, 7311, "evaluation")
    assert calls == before
    manifests = list((experiment / "method_runs/nersi_profile").glob("*/*/manifest.json"))
    assert len(manifests) == 4
