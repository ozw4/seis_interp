"""Fixed-cell evaluation must retain production partitions and scoring artifacts."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from seis_interp.data import c3_region_inputs as data
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.pipelines import run_c3_fixed_cell_sensitivity as sensitivity
from seis_interp.pipelines import run_c3_region_experiment as production
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training.forge_mvp_methods import ForgeMethodResult
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    contract["trace_split"]["evaluation_seeds"] = [7321, 7322, 7323]
    contract["methods"] = list(configs)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    configs["ccnet5d_grid"]["patches"]["inner_mask_fraction"] = 0.25
    (study / "ccnet5d_grid.yaml").write_text(yaml.safe_dump(configs["ccnet5d_grid"]))
    parent = data.prepare_c3_region_experiment(tmp_path, study)
    write_json(parent / "workers/s1_barrier.json", {"passed": True, "methods": list(configs)})
    method = "ccnet5d_grid"
    configs[method]["patches"]["inner_mask_fraction"] = 0.1
    (study / f"{method}.yaml").write_text(yaml.safe_dump(configs[method]))
    parameters = json.loads((parent / "experiment.json").read_text())["parameter_counts"][method]

    def predict(inputs, config, report, **options):
        values = np.sin(inputs.test_cells[:, None] * 0.01 + inputs.time_s[None, :] * 10).astype(
            np.float32
        )
        return ForgeMethodResult(
            values, inputs.test_cells.copy(), [], None, {"parameter_count": parameters}
        )

    monkeypatch.setattr(production, "run_c3_region_method", predict)
    selection = tmp_path / "historical_selection.json"
    write_json(selection, {"winner": {"id": "a05_b02"}})
    contract["methods"] = [method]
    contract["fixed_cell_selection"] = {
        "path": str(selection.relative_to(tmp_path)),
        "sha256": sha256_file(selection),
    }
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    grid["methods"] = {method: grid["methods"][method]}
    (study / "grid.yaml").write_text(yaml.safe_dump(grid))
    experiment = data.prepare_c3_region_experiment(tmp_path, study)
    return tmp_path, experiment, parent


def test_fixed_seeds_match_production_scoring_rows_metrics_and_parquet_schema(
    prepared, monkeypatch
):
    repo, experiment, parent = prepared
    before = {str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()}
    calls = []
    original = production.read_c3_region_targets

    def read_targets(repo, root, partition, domain):
        rows, target = original(repo, root, partition, domain)
        calls.append((root, int(partition.trace_seed.iloc[0]), domain, rows.copy()))
        return rows, target

    monkeypatch.setattr(production, "read_c3_region_targets", read_targets)
    records = sensitivity.run_c3_fixed_cell_sensitivity(repo, experiment, parent)
    assert [row["trace_seed"] for row in records] == [7321, 7322, 7323]
    assert all(row["test_scoring_calls"] == 1 for row in records)
    assert not (experiment / "workers/ccnet5d_grid_selection.json").exists()
    assert not (experiment / "method_runs/ccnet5d_grid/S1").exists()
    assert not json.loads((experiment / "experiment.json").read_text()).get("rehearsal")
    assert before == {
        str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()
    }
    base = yaml.safe_load((parent / "study/ccnet5d_grid.yaml").read_text())
    grid = yaml.safe_load((parent / "study/grid.yaml").read_text())
    candidate = next(
        c
        for c in c3_search_candidates(base, grid["methods"]["ccnet5d_grid"])
        if c["id"] == "a05_b02"
    )
    for seed, record in zip((7321, 7322, 7323), records, strict=True):
        reference = production.execute_c3_region_run(repo, parent, candidate, seed, "evaluation")
        assert record["metrics"] == reference["metrics"]
        assert record["input_binding"] == reference["input_binding"]
        pd.testing.assert_frame_equal(
            pd.read_parquet(experiment / f"input/seed_{seed}/partition.parquet"),
            pd.read_parquet(parent / f"input/seed_{seed}/partition.parquet"),
        )
        for domain in ("validation", "test"):
            selected = [
                rows
                for root, s, d, rows in calls
                if s == seed and d == domain and root in (experiment, parent)
            ]
            assert len(selected) == 2
            pd.testing.assert_frame_equal(*selected)
            name = f"method_runs/ccnet5d_grid/evaluation/seed_{seed}/{domain}_per_trace.parquet"
            assert pq.read_schema(experiment / name).equals(
                pq.read_schema(parent / name), check_metadata=True
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(experiment / name), pd.read_parquet(parent / name)
            )
    calls.clear()
    sensitivity.run_c3_fixed_cell_sensitivity(repo, experiment, parent)
    assert not calls  # completed seeds are verified and reused without rescoring


@pytest.mark.parametrize(
    "fault", ["rehearsal", "barrier", "partition", "seeds", "model", "selection", "same_root"]
)
def test_fixed_cell_rejects_protocol_mismatches_before_evaluation(prepared, fault):
    repo, experiment, parent = prepared
    if fault == "same_root":
        experiment = parent
    elif fault in ("rehearsal", "partition"):
        path = experiment / "experiment.json"
        seal = json.loads(path.read_text())
        if fault == "rehearsal":
            seal["rehearsal"] = True
        else:
            seal["artifacts"]["input/seed_7321/partition.parquet"] = "0" * 64
        path.chmod(0o644)
        path.write_text(json.dumps(seal))
    elif fault == "barrier":
        write_json(parent / "workers/s1_barrier.json", {"passed": False})
    else:
        path = experiment / "study" / ("ccnet5d_grid.yaml" if fault == "model" else "config.yaml")
        settings = yaml.safe_load(path.read_text())
        if fault == "seeds":
            settings["trace_split"]["evaluation_seeds"] = [7311, 7312, 7313]
        elif fault == "model":
            settings["model"]["hidden_channels"] += 1
        else:
            settings["fixed_cell_selection"]["sha256"] = "0" * 64
        path.chmod(0o644)
        path.write_text(yaml.safe_dump(settings))
        seal_path = experiment / "experiment.json"
        seal = json.loads(seal_path.read_text())
        seal["artifacts"][str(path.relative_to(experiment))] = sha256_file(path)
        seal_path.chmod(0o644)
        seal_path.write_text(json.dumps(seal))
    with pytest.raises((ValueError, RuntimeError)):
        sensitivity.run_c3_fixed_cell_sensitivity(repo, experiment, parent)
    assert not (experiment / "method_runs/ccnet5d_grid/evaluation").exists()
