from pathlib import PurePosixPath

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import read_poc_json

STUDY = REPOSITORY_ROOT / "studies/study_037_c3_neural_mse_loss_ablation"
METHODS = {"nersi", "ccnet5d", "relational_trace_graph"}


def test_frozen_results_bind_baseline_inputs_and_formal_configs():
    record = read_poc_json(STUDY / "stage_1b_results.lock.json")
    assert record["status"] == "frozen"
    assert record["run_directory"] == (
        "../../runs/study_037_c3_neural_mse_loss_ablation/20260911T095500Z_dcd38161ae39_formal"
    )
    baseline = STUDY / record["baseline"]["lock_path"]
    assert file_sha256(baseline) == record["baseline"]["sha256"]
    assert record["inputs_lock"] == read_poc_json(baseline)["inputs_lock"]
    assert set(record["config_sha256"]) == {
        f"formal/{name}.yaml" for name in ("nersi", "ccnet5d", "gnn")
    }
    for path, digest in record["config_sha256"].items():
        assert file_sha256(STUDY / path) == digest


def test_freeze_pins_all_three_runs_and_preserves_provenance():
    record = read_poc_json(STUDY / "stage_1b_results.lock.json")
    required = {
        f"{method}/{filename}"
        for method in METHODS
        for filename in (
            "config.resolved.yaml",
            "inputs.lock.json",
            "metadata.json",
            "metrics.json",
            "prediction.npy",
            "final.pt",
        )
    } | {
        "relational_trace_graph/artifacts/query_trace_ids.npy",
        "relational_trace_graph/artifacts/target_coverage.npy",
    }
    assert set(record["run_file_sha256"]) == required
    for path, digest in record["run_file_sha256"].items():
        assert not PurePosixPath(path).is_absolute()
        assert ".." not in PurePosixPath(path).parts
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert set(record["method_provenance"]) == METHODS
    for provenance in record["method_provenance"].values():
        assert provenance["git_commit"] == "5829a11482edd8adafd29b4be25f910fe5545425"
        assert provenance["git_worktree_dirty"] is False
    assert record["provenance_notes"]
    assert record["verification"]["fresh_checkpoint_reinference_performed"] is False


def test_mse_adoption_is_not_performance_selection_or_baseline_replacement():
    config = load_resolved_config(STUDY / "config.yaml")
    assert config["decision"] == {
        "adopted_loss": "masked_trace_mse",
        "basis": "user_directed_alignment_with_prior_research",
        "performance_based_selection": False,
        "replaces_stage_1_baseline": False,
    }
    record = read_poc_json(STUDY / "stage_1b_results.lock.json")
    assert {row["method"] for row in record["comparison"]} == METHODS
    for row in record["comparison"]:
        assert row["delta_snr_db"] == row["mse_snr_db"] - row["baseline_snr_db"]
        assert row["delta_snr_db"] < 0
        assert row["delta_rmse"] == row["mse_rmse"] - row["baseline_rmse"]
        assert row["delta_rmse"] > 0
