import json
import os

import numpy as np
import pytest
import yaml

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS
from seis_interp.data import c3_region_inputs as c3_data
from seis_interp.data import forge_grid_inputs as forge_data
from seis_interp.data import forge_trace_experiment as single_data
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.pipelines import run_c3_region_experiment as c3_pipeline
from seis_interp.pipelines import run_forge_grid_experiment as forge_pipeline
from seis_interp.pipelines import run_forge_trace_experiment as single_pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training.worker_processes import process_start_time
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


def identity(_):
    return {"git_commit": "testcommit"}


def ready(experiment, methods):
    for method in methods:
        write_json(
            experiment / "workers" / f"{method}.json",
            {
                "status": "running",
                "s1_completed": True,
                "pid": os.getpid(),
                "process_start_time": process_start_time(os.getpid()),
            },
        )


@pytest.mark.parametrize(
    "stage,seed,pilot", [("S1", 21, False), ("S1", 21, True), ("evaluation", 7311, False)]
)
def test_c3_candidate_and_seed_import_skip_training_and_rescore(
    tmp_path, monkeypatch, stage, seed, pilot
):
    monkeypatch.setattr(c3_data, "execution_identity", identity)
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    source = c3_data.prepare_c3_region_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs["pocs_grid"], grid["methods"]["pocs_grid"])[0]
    ready(source, C3_FAIR_GRID_METHODS)
    original = c3_pipeline.execute_c3_region_run(tmp_path, source, candidate, seed, stage)
    if pilot:
        (source / "candidates.json").write_text(json.dumps({"pocs_grid": [candidate]}))
        seal = json.loads((source / "experiment.json").read_text())
        (source / "experiment.json").unlink()
        write_json(source / "experiment.json", seal | {"pilot": "P1b", "validation_only": True})
        (source / "pocs_grid_summary.csv").write_text(
            "id,status,validation_score\na01_b01,completed,-999\n"
        )
    contract["import_from"] = [str(source)]
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    target = c3_data.prepare_c3_region_experiment(tmp_path, study)
    ready(target, C3_FAIR_GRID_METHODS)
    monkeypatch.setattr(
        c3_pipeline, "run_c3_region_method", lambda *a: pytest.fail("training repeated")
    )
    reads = []
    score = c3_pipeline.score_c3_region

    def rescore(*args, **kwargs):
        reads.append(True)
        return score(*args, **kwargs)

    monkeypatch.setattr(c3_pipeline, "score_c3_region", rescore)
    record = c3_pipeline.execute_c3_region_run(tmp_path, target, candidate, seed, stage)
    assert len(reads) == (1 if stage == "S1" else 2)
    assert record["metrics"] == original["metrics"]
    assert record["imported_from"]["source_input_hash"] == record["input_hash"]
    assert record["test_scoring_calls"] == (stage == "evaluation")


def test_forge_pilot_grid_import_skips_training_and_rescores_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(single_data, "execution_identity", identity)
    monkeypatch.setattr(forge_data, "execution_identity", identity)
    study, contract, config, _ = tiny_forge_grid_study(tmp_path)
    source = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    original = forge_pipeline.execute_forge_grid_run(tmp_path, source, candidate, 19, "S1")
    (source / "candidates.json").write_text(json.dumps({"pocs_grid": [candidate]}))
    seal = json.loads((source / "experiment.json").read_text())
    (source / "experiment.json").unlink()
    write_json(source / "experiment.json", seal | {"pilot": "P1b", "validation_only": True})
    (source / "pocs_grid_summary.csv").write_text(
        "id,status,validation_score\na01_b01,completed,-999\n"
    )
    contract["import_from"] = [str(source)]
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    target = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    monkeypatch.setattr(
        forge_pipeline, "run_forge_c3_method", lambda *a, **k: pytest.fail("training repeated")
    )
    domains = []
    read = forge_pipeline.read_forge_grid_targets

    def validation_only(experiment, partition, domain):
        domains.append(domain)
        assert domain == "validation"
        return read(experiment, partition, domain)

    monkeypatch.setattr(forge_pipeline, "read_forge_grid_targets", validation_only)
    record = forge_pipeline.execute_forge_grid_run(tmp_path, target, candidate, 19, "S1")
    assert domains == ["validation"] and record["test_scoring_calls"] == 0
    assert record["metrics"] == original["metrics"]
    assert record["imported_from"]["source_path"] == str(
        source / "method_runs/pocs_grid/S1/a01_b01"
    )


def test_completed_forge_single_config_imports_only_validation_and_rescores(tmp_path, monkeypatch):
    monkeypatch.setattr(single_data, "execution_identity", identity)
    monkeypatch.setattr(forge_data, "execution_identity", identity)
    study, contract, config, reference = tiny_forge_grid_study(tmp_path)
    single_pipeline.execute_forge_trace_method(tmp_path, reference, "pocs_grid")
    native = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[20]
    original = forge_pipeline.execute_forge_grid_run(tmp_path, native, candidate, 19, "S1")
    contract["import_from"] = [str(reference / "method_runs/pocs_grid")]
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    target = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    # The single configuration matches the last axis_a and first axis_b values.
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[20]
    monkeypatch.setattr(
        forge_pipeline, "run_forge_c3_method", lambda *a, **k: pytest.fail("training repeated")
    )
    domains = []
    read = forge_pipeline.read_forge_grid_targets

    def read_validation(experiment, partition, domain):
        domains.append(domain)
        return read(experiment, partition, domain)

    monkeypatch.setattr(forge_pipeline, "read_forge_grid_targets", read_validation)
    record = forge_pipeline.execute_forge_grid_run(tmp_path, target, candidate, 19, "S1")
    assert record["status"] == "completed" and "imported_from" in record
    assert domains == ["validation"] and record["test_scoring_calls"] == 0
    saved = target / "method_runs/pocs_grid/S1/a05_b01/cell_ids.npy"
    assert len(np.load(saved)) == 5
    native_output = native / "method_runs/pocs_grid/S1/a05_b01"
    np.testing.assert_array_equal(
        np.load(saved.parent / "prediction.npy"), np.load(native_output / "prediction.npy")
    )
    assert record["metrics"] == original["metrics"]


def test_forge_native_seed_import_is_recomputed_only_by_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(single_data, "execution_identity", identity)
    monkeypatch.setattr(forge_data, "execution_identity", identity)
    study, contract, config, _ = tiny_forge_grid_study(tmp_path)
    source = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    ready(source, ["pocs_grid"])
    original = forge_pipeline.execute_forge_grid_run(
        tmp_path, source, candidate, 7311, "evaluation"
    )
    contract["import_from"] = [str(source)]
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    target = forge_data.prepare_forge_grid_experiment(tmp_path, study)
    ready(target, ["pocs_grid"])
    monkeypatch.setattr(
        forge_pipeline, "run_forge_c3_method", lambda *a, **k: pytest.fail("training repeated")
    )
    record = forge_pipeline.execute_forge_grid_run(tmp_path, target, candidate, 7311, "evaluation")
    assert record["metrics"] == original["metrics"]
    assert record["test_scoring_calls"] == 1
    assert len(record["imported_from"]["source_snapshot_sha256"]) == 64
    assert (
        json.loads(
            (target / "method_runs/pocs_grid/evaluation/seed_7311/manifest.json").read_text()
        )["status"]
        == "completed"
    )


def test_profile_jitter_is_honoured_by_both_dataset_method_adapters(tmp_path, monkeypatch):
    import pandas as pd

    from seis_interp.training import forge_mvp_methods
    from seis_interp.training.c3_region_methods import run_c3_region_method
    from seis_interp.training.forge_c3_methods import run_forge_c3_method

    monkeypatch.setattr(c3_data, "execution_identity", identity)
    monkeypatch.setattr(single_data, "execution_identity", identity)
    monkeypatch.setattr(forge_data, "execution_identity", identity)
    c3_root, forge_root = tmp_path / "c3", tmp_path / "forge"
    c3_root.mkdir()
    forge_root.mkdir()
    study, _, configs = tiny_c3_trace_study(c3_root)
    c3 = c3_data.prepare_c3_region_experiment(c3_root, study)
    c3_inputs, _ = c3_data.load_c3_region_inputs(c3, 21, ("validation",))
    forge_study, _, _, _ = tiny_forge_grid_study(forge_root)
    forge = forge_data.prepare_forge_grid_experiment(forge_root, forge_study)
    # The prepared geometry is method-independent; select its real-coordinate table.
    forge_inputs, loaded, _ = forge_data.load_forge_grid_inputs(
        forge, "pocs_grid", 19, ("validation",)
    )
    from dataclasses import replace

    forge_inputs = replace(
        forge_inputs, geometry=pd.read_parquet(forge / "input/real_geometry.parquet")
    )
    calls = []
    original = forge_mvp_methods.ProfileCoordinateJitter

    def jitter(cells, shape, seed):
        calls.append((cells, shape, seed))
        return original(cells, shape, seed)

    monkeypatch.setattr(forge_mvp_methods, "ProfileCoordinateJitter", jitter)
    config = configs["nersi_profile"]
    config["training"]["coordinate_jitter_cells"] = 0.5
    first = run_c3_region_method(c3_inputs, config, lambda _: None)
    second = run_forge_c3_method(
        forge_inputs, config | {"alignment": loaded["alignment"]}, lambda _: None
    )
    assert len(calls) == 2 and all(item[0] == 0.5 for item in calls)
    assert len(first.prediction) == len(c3_inputs.test_cells)
    assert len(second.prediction) == len(forge_inputs.test_cells)
