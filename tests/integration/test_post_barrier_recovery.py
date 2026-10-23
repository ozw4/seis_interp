"""Reviewer repro: early post-barrier infrastructure faults must remain retryable."""

import json

import pytest
import yaml

from seis_interp.data import c3_region_inputs as c3_data
from seis_interp.data import forge_grid_inputs as forge_data
from seis_interp.data import forge_trace_experiment as single_data
from seis_interp.pipelines import run_c3_region_experiment as c3
from seis_interp.pipelines import run_forge_grid_experiment as forge
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training import run_recovery
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


@pytest.mark.parametrize("dataset", ["c3", "forge"])
@pytest.mark.parametrize("missing_metadata", [False, True])
def test_early_infra_failure_in_scoring_attempt(tmp_path, monkeypatch, dataset, missing_metadata):
    for module in (c3_data, forge_data, single_data):
        monkeypatch.setattr(
            module, "execution_identity", lambda _: {"git_commit": "f", "dirty_worktree": False}
        )
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    if dataset == "c3":
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        config = configs["pocs_grid"]
        prepare, execute, pipeline = (
            c3_data.prepare_c3_region_experiment,
            c3.execute_c3_region_run,
            c3,
        )
        barrier = "c3_s1_barrier_ready"
    else:
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        prepare, execute, pipeline = (
            forge_data.prepare_forge_grid_experiment,
            forge.execute_forge_grid_run,
            forge,
        )
        barrier = "forge_s1_barrier_ready"
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    experiment = prepare(tmp_path, study)
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["rehearsal"] is False
    seed = contract["trace_split"]["evaluation_seeds"][0]
    monkeypatch.setattr(pipeline, barrier, lambda _: True)
    predicted = execute(tmp_path, experiment, candidate, seed, "evaluation", prediction_only=True)
    assert predicted["status"] == "awaiting_barrier"

    # One transient infrastructure error (e.g. ENOSPC) right after the scoring attempt
    # wrote its start manifest and before prediction_finalized_at is rewritten.
    output = experiment / "method_runs/pocs_grid/evaluation" / f"seed_{seed}"
    if missing_metadata:
        prior = json.loads((output / "manifest.json").read_text())
        prior.pop("prediction_finalized_at")
        prior.pop("prediction_execution")
        (output / "manifest.json").write_text(json.dumps(prior))
    original = pipeline.validate_predictions
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        initial = json.loads((output / "manifest.json").read_text())
        assert initial["prediction_finalized_at"]
        if not missing_metadata:
            assert initial["prediction_execution"] == predicted["prediction_execution"]
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "validate_predictions", flaky)
    completed = execute(tmp_path, experiment, candidate, seed, "evaluation")
    assert completed["status"] == "completed" and completed["attempt"] == 1
    assert calls["n"] == 2
    if not missing_metadata:
        assert completed["prediction_finalized_at"] == predicted["prediction_finalized_at"]
        assert completed["prediction_execution"] == predicted["prediction_execution"]
    failed = json.loads((output / "attempts/1/manifest.json").read_text())
    assert failed["failure_kind"] == "infrastructure"
    assert failed["prediction_finalized_at"] == completed["prediction_finalized_at"]
    assert execute(tmp_path, experiment, candidate, seed, "evaluation")["status"] == "completed"
