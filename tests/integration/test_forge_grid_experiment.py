import json
from contextlib import contextmanager
from functools import partial
from itertools import combinations

import pandas as pd
import pytest
import yaml

from seis_interp.data import forge_grid_inputs as data
from seis_interp.data import forge_trace_experiment as single
from seis_interp.pipelines import run_forge_grid_experiment as pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.processing.forge_grid_partition import (
    partition_forge_traces,
    verify_forge_s1_partition,
)
from seis_interp.training.candidate_schedule import candidate_outcomes, ordered_candidates
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def identity(_):
        return {"git_commit": "testcommit"}

    monkeypatch.setattr(data, "execution_identity", identity)
    monkeypatch.setattr(single, "execution_identity", identity)
    study, contract, config, reference = tiny_forge_grid_study(tmp_path)
    experiment = data.prepare_forge_grid_experiment(tmp_path, study)
    return tmp_path, experiment, contract, config, reference


def test_seed_roles_are_deterministic_disjoint_and_exclude_natural_missing(prepared):
    _, experiment, contract, _, reference = prepared
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    observed_sets = []
    test_sets = []
    for seed in [19, 7311, 7312, 7313]:
        first = partition_forge_traces(manifest, contract, seed)
        pd.testing.assert_frame_equal(first, partition_forge_traces(manifest, contract, seed))
        inputs, _, saved = data.load_forge_grid_inputs(
            experiment, "pocs_grid", seed, ("validation",)
        )
        pd.testing.assert_frame_equal(first.fillna(""), saved.fillna(""))
        assert 35 not in first.cell_id.to_numpy()
        assert len(inputs.observed_cells) == 12
        assert len(inputs.test_cells) == 5 and len(inputs.excluded_cells) == 18
        assert set(inputs.observed_cells).isdisjoint(inputs.test_cells)
        assert set(inputs.test_cells).isdisjoint(inputs.excluded_cells)
        with pytest.raises(ValueError, match="outside outer observed"):
            inputs.read_observed(inputs.test_cells[:1])
        observed_sets.append(tuple(inputs.observed_cells))
        test_sets.append(frozenset(first.cell_id[first.role.eq("test")]))
        assert first.role.value_counts().to_dict() == {"test": 18, "train": 12, "validation": 5}
    assert len(set(observed_sets)) == 4
    assert all(left != right for left, right in combinations(test_sets, 2))
    reference_partition = pd.read_parquet(reference / "input/partition.parquet")
    verify_forge_s1_partition(partition_forge_traces(manifest, contract, 19), reference_partition)
    with pytest.raises(ValueError, match="cell IDs differ"):
        verify_forge_s1_partition(
            partition_forge_traces(manifest, contract, 7311), reference_partition
        )


def test_native_25_plus_three_flow_finalizes_before_scoring_and_obeys_barrier(
    prepared, monkeypatch
):
    repo, experiment, _, config, _ = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidates = c3_search_candidates(config, grid["methods"]["pocs_grid"])
    with pytest.raises(RuntimeError, match="finish S1"):
        pipeline.execute_forge_grid_run(repo, experiment, candidates[0], 7311, "evaluation")
    reads = []
    read = pipeline.read_forge_grid_targets
    run = pipeline.run_forge_c3_method

    def finalized(path, partition, domain):
        if domain == "test":
            assert pipeline.forge_s1_barrier_ready(path)
        seed = int(partition.trace_seed.iloc[0])
        stage = "S1" if seed == 19 else "evaluation"
        outputs = list((path / "method_runs/pocs_grid" / stage).glob("*/manifest.json"))
        assert any("prediction_finalized_at" in json.loads(item.read_text()) for item in outputs)
        reads.append((seed, domain))
        return read(path, partition, domain)

    def validation_only(inputs, config, reporter, **kwargs):
        if inputs.binding["trace_seed"] == 19:
            assert len(inputs.test_cells) == 5
            assert len(inputs.excluded_cells) == 18
        return run(inputs, config, reporter, **kwargs)

    monkeypatch.setattr(pipeline, "read_forge_grid_targets", finalized)
    monkeypatch.setattr(pipeline, "run_forge_c3_method", validation_only)
    pipeline.run_forge_grid_worker(repo, experiment, "pocs_grid")
    pipeline.summarize_forge_grid_experiment(experiment)
    state = json.loads((experiment / "workers/pocs_grid.json").read_text())
    assert state["completed_s1"] == 25 and state["completed_evaluation"] == 3
    assert state["status"] == "completed"
    assert reads.count((19, "validation")) == 25
    assert (19, "test") not in reads
    assert reads[-6:] == [
        (seed, domain) for seed in [7311, 7312, 7313] for domain in ["validation", "test"]
    ]
    records = [
        json.loads(path.read_text())
        for path in (experiment / "method_runs/pocs_grid").glob("*/*/manifest.json")
    ]
    assert len(records) == 28
    for record in records:
        assert record["runtime_seconds"] > 0 and record["peak_cpu_rss_bytes"] > 0
        assert record["peak_cuda_reserved_bytes"] == 0
        assert record["test_scoring_calls"] == (record["stage"] == "evaluation")
    metrics = pd.read_csv(experiment / "seed_metrics.csv")
    assert len(metrics) == 12
    assert {"rmse", "mean_trace_snr_db", "correlation_median"} <= set(metrics)
    summary = pd.read_csv(experiment / "summary.csv")
    assert len(summary) == 40 and summary.seed_count.eq(3).all()
    selected = metrics.loc[
        (metrics.domain == "test") & (metrics.population == "clean_target"), "rmse"
    ]
    row = summary.loc[
        (summary.domain == "test")
        & (summary.population == "clean_target")
        & (summary.metric == "rmse")
    ].iloc[0]
    assert row.sample_std == pytest.approx(selected.std(ddof=1))
    pipeline.run_forge_grid_worker(repo, experiment, "pocs_grid")
    assert len(reads) == 31


def test_cpu_candidate_priority_and_process_isolation(prepared):
    repo, experiment, _, config, _ = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidates = c3_search_candidates(config, grid["methods"]["pocs_grid"])
    ordered = ordered_candidates(candidates, ["a05_b05", "a01_b02"])
    assert [item["id"] for item in ordered[:3]] == ["a05_b05", "a01_b02", "a01_b01"]
    with pytest.raises(ValueError, match="existing"):
        ordered_candidates(candidates, ["missing"])
    execute = partial(pipeline._search_run, repo=repo, experiment=experiment, seed=19)
    children = []
    records = list(
        candidate_outcomes(execute, ordered[:2], parallel=2, children_changed=children.extend)
    )
    assert len({child["pid"] for child in children}) == 2
    assert all(child["process_start_time"] is not None for child in children)
    assert {row["id"] for row in records} == {"a05_b05", "a01_b02"}
    assert all(row["status"] == "completed" for row in records)


def test_interrupted_admission_leaves_no_run_and_can_resume(prepared, monkeypatch):
    repo, experiment, _, config, _ = prepared
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    admission = pipeline.candidate_admission

    @contextmanager
    def interrupted(*args, **kwargs):
        args[4]({"reason": "insufficient_free_memory"})
        raise KeyboardInterrupt("interrupted waiting")
        yield

    monkeypatch.setattr(pipeline, "candidate_admission", interrupted)
    with pytest.raises(KeyboardInterrupt, match="waiting"):
        pipeline.execute_forge_grid_run(repo, experiment, candidate, 19, "S1")
    assert not (experiment / "method_runs/pocs_grid/S1/a01_b01").exists()
    monkeypatch.setattr(pipeline, "candidate_admission", admission)
    record = pipeline.execute_forge_grid_run(repo, experiment, candidate, 19, "S1")
    assert record["status"] == "completed"


def test_diverged_s1_is_a_final_outcome_on_worker_resume(prepared, monkeypatch):
    repo, experiment, _, _, _ = prepared
    run = pipeline.run_forge_c3_method
    calls = []

    def diverge_once(inputs, config, reporter, **options):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("non-finite loss at step 1")
        return run(inputs, config, reporter, **options)

    monkeypatch.setattr(pipeline, "run_forge_c3_method", diverge_once)
    pipeline.run_forge_grid_worker(repo, experiment, "pocs_grid")
    assert len(calls) == 28
    failed_path = experiment / "method_runs/pocs_grid/S1/a01_b01/manifest.json"
    before = failed_path.read_bytes()
    pipeline.run_forge_grid_worker(repo, experiment, "pocs_grid")
    assert len(calls) == 28 and failed_path.read_bytes() == before
    assert json.loads((experiment / "workers/pocs_grid.json").read_text())["status"] == "completed"
