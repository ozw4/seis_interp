import json
import os

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
from tests.fixtures.c3_region_experiment import (
    tiny_c3_trace_field_study,
    tiny_c3_trace_study,
)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    reads = []
    original = data.read_c3_amplitude_rows

    def read_observations(path, rows, time_count):
        reads.append(rows.copy())
        return original(path, rows, time_count)

    monkeypatch.setattr(data, "read_c3_amplitude_rows", read_observations)
    experiment = data.prepare_c3_region_experiment(tmp_path, study)
    assert len(reads) == 4 and all(len(rows) == 46 for rows in reads)
    return tmp_path, experiment, contract, configs


def test_full_crop_inputs_have_fixed_counts_and_new_trace_masks_for_each_seed(prepared):
    _, experiment, contract, _ = prepared
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["partition_unit"] == "trace" and seal["total_runs"] == 140
    assert seal["target_waveforms_read"] == 0
    observations = []
    for seed in [21, 7311, 7312, 7313]:
        inputs, partition = data.load_c3_region_inputs(experiment, seed, ("validation",))
        final, _ = data.load_c3_region_inputs(experiment, seed, ("validation", "test"))
        assert len(inputs.geometry) == contract["eligible_trace_count"]
        assert len(inputs.observed_cells) == 46
        assert len(inputs.test_cells) == 5 and len(final.test_cells) == 210
        assert partition.loc[partition.input_observed].trace_role.eq("train").all()
        assert partition.loc[partition.trace_role.eq("validation")].outer_split.eq("observed").all()
        assert partition.loc[partition.trace_role.eq("test")].outer_split.eq("test").all()
        assert "region_seed" not in inputs.binding and inputs.binding["trace_seed"] == seed
        np.testing.assert_array_equal(inputs.observed_values, final.observed_values)
        assert inputs.global_rms == final.global_rms
        observations.append(tuple(inputs.observed_cells))
    assert len(set(observations)) == 4


@pytest.mark.parametrize("method", C3_FAIR_GRID_METHODS)
def test_every_method_excludes_validation_and_test_inputs_and_scores_only_validation_in_s1(
    prepared, method
):
    repo, experiment, contract, configs = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    record = pipeline.execute_c3_region_run(repo, experiment, candidate, 21, "S1")
    assert record["status"] == "completed" and np.isfinite(record["validation_score"])
    assert record["observed_trace_count"] == 46
    assert record["trace_seed"] == 21 and "region_seed" not in record
    assert record["test_scoring_calls"] == 0 and set(record["metrics"]) == {"validation"}
    assert record["metrics"]["validation"]["trace_count"] == 5
    if method in ("ccnet5d_grid", "regsi_real", "nersi_profile"):
        output = experiment / "method_runs" / method / "S1" / candidate["id"]
        checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
        assert checkpoint["steps_completed"] == contract["matched_training"]["max_steps"]
        assert checkpoint["selection_rule"] == "fixed_final_ema"


def test_three_new_trace_assignments_use_fresh_models_and_read_test_after_s1(prepared, monkeypatch):
    repo, experiment, _, configs = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(configs["nersi_profile"], grid["methods"]["nersi_profile"])[0]
    with pytest.raises(RuntimeError, match="all five S1"):
        pipeline.execute_c3_region_run(repo, experiment, candidate, 7311, "evaluation")
    for method in C3_FAIR_GRID_METHODS:
        write_json(
            experiment / "workers" / f"{method}.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "process_start_time": process_start_time(os.getpid()),
                "s1_completed": True,
            },
        )
    calls = []
    original = pipeline.read_c3_region_targets

    def read_finalized(repo, experiment, partition, domain):
        seed = int(partition.trace_seed.iloc[0])
        output = experiment / "method_runs/nersi_profile/evaluation" / f"seed_{seed}"
        assert (output / "final.pt").is_file() and (output / "prediction.npy").is_file()
        assert "prediction_finalized_at" in json.loads((output / "manifest.json").read_text())
        calls.append((seed, domain))
        return original(repo, experiment, partition, domain)

    monkeypatch.setattr(pipeline, "read_c3_region_targets", read_finalized)
    records = [
        pipeline.execute_c3_region_run(repo, experiment, candidate, seed, "evaluation")
        for seed in (7311, 7312, 7313)
    ]
    assert calls == [
        (seed, domain) for seed in (7311, 7312, 7313) for domain in ("validation", "test")
    ]
    assert all(record["test_scoring_calls"] == 1 for record in records)
    assert all(record["metrics"]["test"]["trace_count"] == 205 for record in records)
    assert len({record["input_binding"]["partition_hash"] for record in records}) == 3
    assert len({record["diagnostics"]["initialization_hash"] for record in records}) == 1
    summary = pipeline.summarize_c3_region_seeds(records)
    assert len(summary) == 10 and summary.seed_count.eq(3).all()
    pipeline.execute_c3_region_run(repo, experiment, candidate, 7311, "evaluation")
    assert len(calls) == 6


def test_the_declared_trace_field_runs_the_s1_search_on_the_observed_partition(
    tmp_path, monkeypatch
):
    """A one-method study drives the same runner, barrier and parameter seal."""
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    study, contract, configs = tiny_c3_trace_field_study(tmp_path)
    experiment = data.prepare_c3_region_experiment(tmp_path, study)
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["total_runs"] == 28 and seal["target_waveforms_read"] == 0

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
    candidates = c3_search_candidates(configs["inr_real"], grid["methods"]["inr_real"])
    assert len(candidates) == 25
    record = pipeline.execute_c3_region_run(
        tmp_path, experiment, candidates[0], contract["trace_split"]["search_seed"], "S1"
    )
    assert record["status"] == "completed"
    assert np.isfinite(record["validation_score"])
    assert record["test_scoring_calls"] == 0
    assert record["parameter_count"] == seal["parameter_counts"]["inr_real"]
    assert len(optimizers) == 1
    optimizer = optimizers[0]
    assert optimizer.defaults["lr"] == contract["matched_training"]["learning_rate"]
    assert optimizer.update_count == contract["matched_training"]["max_steps"]
    manifest = json.loads(
        (experiment / "method_runs/inr_real/S1" / candidates[0]["id"] / "manifest.json").read_text()
    )
    gate = manifest["diagnostics"]["g0"]
    assert gate["zero_predictor_loss"] > 0 and gate["mean_loss"] is not None
    assert manifest["diagnostics"]["would_clip_threshold"] == 1.0
