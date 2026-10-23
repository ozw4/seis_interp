"""Display pipeline consumes verified predictions without retraining or scoring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.forge_mvp_config import NEURAL_METHODS
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.visualize_forge_fair_grid import visualize_forge_fair_grid
from seis_interp.processing.forge_mvp_contract import METHODS
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    study, *_ = tiny_forge_study(tmp_path)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    preparation = prepare_forge_mvp(tmp_path, study)
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    test = mask.loc[mask.split.eq("test"), ["cell_id"]].iloc[::-1].copy()
    test["prediction_row"] = np.arange(len(test))
    runs = []
    for i, method in enumerate(METHODS):
        tuning = tmp_path / "runs" / method
        tuning.mkdir()
        candidate = tuning / "final"
        candidate.mkdir()
        test.to_parquet(tuning / "prediction_index.parquet", index=False)
        # The reversed index must map each saved row by cell, not array position.
        values = np.broadcast_to(test.cell_id.to_numpy()[:, None], (len(test), 9)).astype(
            np.float32
        )
        np.save(candidate / "prediction.npy", values + i)
        hashes = {
            name: sha256_file(tuning / name)
            for name in ("prediction_index.parquet", "final/prediction.npy")
        }
        write_json(tuning / "manifest.json", {"status": "completed", "artifacts": hashes})
        runs.append(
            {
                "method_id": method,
                "seed": 7311 if method in NEURAL_METHODS else None,
                "tuning_run": str(tuning.relative_to(tmp_path)),
                "candidate": "final",
                "prediction_hash": hashes["final/prediction.npy"],
            }
        )
    evaluation = tmp_path / "runs/evaluation"
    write_json(
        evaluation / "manifest.json",
        {
            "status": "completed",
            "selection_domain": "final_test",
            "final_runs": runs,
            "preparation": str(preparation.relative_to(tmp_path)),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
            "artifacts": {},
        },
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.visualize_forge_fair_grid.execution_identity",
        lambda _: {"git_commit": "test"},
    )
    # Source snapshots use real implementation files.
    repository = Path(__file__).resolve().parents[2]
    for module in (
        "data/forge_fair_grid_gathers.py",
        "visualization/forge_publication_gathers.py",
        "visualization/publication_sections.py",
        "pipelines/visualize_forge_fair_grid.py",
    ):
        target = tmp_path / "src/seis_interp" / module
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / "src/seis_interp" / module).read_bytes())
    return tmp_path, evaluation


def test_exports_four_gathers_and_shared_gain_without_new_scores(evaluated):
    repo, evaluation = evaluated
    output = visualize_forge_fair_grid(repo, evaluation, repo / "runs/figures", dpi=30)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["new_inference"] is False and record["new_test_scoring"] is False
    assert record["layout"] == [2, 6, 6] and record["display_seed"] == 7311
    assert len(list(output.glob("gather_*.png"))) == len(list(output.glob("gather_*.pdf"))) == 8
    assert not list(output.glob("*metrics*"))
    index = pd.read_parquet(output / "gather_index.parquet")
    data = np.load(output / "gathers.npz")
    test = index.split.eq("test").to_numpy()
    for i, method in enumerate(METHODS):
        np.testing.assert_array_equal(data[method][test, 0], index.loc[test, "cell_id"] + i)
        assert np.all(data[method][~test] == 0)


def test_incomplete_evaluation_is_rejected(evaluated):
    repo, evaluation = evaluated
    path = evaluation / "manifest.json"
    record = json.loads(path.read_text())
    record["status"] = "running"
    write_json(path, record)
    with pytest.raises(ValueError, match="completed final-test"):
        visualize_forge_fair_grid(repo, evaluation, repo / "runs/figures", dpi=30)


def test_changed_prediction_is_rejected_before_reading_display_targets(evaluated, monkeypatch):
    repo, evaluation = evaluated
    record = json.loads((evaluation / "manifest.json").read_text())
    run = record["final_runs"][0]
    path = repo / run["tuning_run"] / "final/prediction.npy"
    np.save(path, np.zeros((1, 9), np.float32))
    reads = []
    monkeypatch.setattr(
        "seis_interp.pipelines.visualize_forge_fair_grid.read_gather_reference",
        lambda *args: reads.append(args),
    )
    with pytest.raises(ValueError, match="input hash mismatch"):
        visualize_forge_fair_grid(repo, evaluation, repo / "runs/figures", dpi=30)
    assert reads == []
