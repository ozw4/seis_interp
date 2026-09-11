"""The adopted baseline keeps explicit immutable references without requiring local run data."""

from pathlib import PurePosixPath

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.data.c3_poc_inputs import C3_RANDOM80_POC_BENCHMARK_ID, C3_RANDOM80_POC_SELECTION
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import read_poc_json

STUDY = REPOSITORY_ROOT / "studies/study_036_c3_random80_observed_only_poc"
METHODS = {"pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph"}


def test_baseline_identifies_the_completed_formal_run_and_full_input_lock():
    record = read_poc_json(STUDY / "stage_1_baseline.lock.json")
    assert record["baseline_id"] == "stage_1_baseline"
    assert record["status"] == "frozen"
    assert record["selection_basis"] == "user_designated_completed_run"
    assert record["run_directory"] == (
        "../../runs/study_036_c3_random80_observed_only_poc/20260911T081726Z_466b41fae15c_formal"
    )
    lock = record["inputs_lock"]
    assert record["benchmark_id"] == lock["benchmark_id"] == C3_RANDOM80_POC_BENCHMARK_ID
    assert lock["selection"] == C3_RANDOM80_POC_SELECTION
    assert lock["observed_trace_count"] == 26362
    assert lock["target_trace_count"] == 104710
    assert lock["benchmark_case"]["sha256"]
    assert lock["benchmark_volume"]["files"]
    assert lock["mask"]["files"]


def test_stage_1_formal_configs_remain_byte_identical_to_the_frozen_configs():
    record = read_poc_json(STUDY / "stage_1_baseline.lock.json")
    expected = {f"formal/{method}.yaml" for method in ("pocs", "drr", "nersi", "ccnet5d", "gnn")}
    assert set(record["config_sha256"]) == expected
    for relative, digest in record["config_sha256"].items():
        assert file_sha256(STUDY / relative) == digest, relative


def test_baseline_pins_metrics_configs_locks_predictions_checkpoints_and_logs():
    record = read_poc_json(STUDY / "stage_1_baseline.lock.json")
    files = record["run_file_sha256"]
    required = {
        "summary.json",
        "summary.csv",
        "logs/preflight.stdout.log",
        "logs/preflight.stderr.log",
    }
    for method in METHODS:
        required.update(
            f"{method}/{name}"
            for name in (
                "config.resolved.yaml",
                "inputs.lock.json",
                "metadata.json",
                "metrics.json",
                "prediction.npy",
            )
        )
        required.update(f"logs/{method}.{channel}.log" for channel in ("stdout", "stderr"))
        if method not in {"pocs", "drr"}:
            required.add(f"{method}/final.pt")
        else:
            assert f"{method}/final.pt" not in files
    required.update(
        f"relational_trace_graph/artifacts/{name}.npy"
        for name in ("query_trace_ids", "target_coverage")
    )
    assert set(files) == required
    for relative, digest in files.items():
        assert not PurePosixPath(relative).is_absolute()
        assert ".." not in PurePosixPath(relative).parts
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_baseline_preserves_per_method_provenance_and_verification_limits():
    record = read_poc_json(STUDY / "stage_1_baseline.lock.json")
    provenance = record["method_provenance"]
    assert set(provenance) == METHODS
    assert provenance["pocs"]["git_worktree_dirty"] is True
    assert provenance["pocs"]["git_commit"] == "466b41fae15c920b72fdc18974a88edf2e21b5cd"
    for method in METHODS - {"pocs"}:
        assert provenance[method]["git_worktree_dirty"] is False
        assert provenance[method]["git_commit"] == "d75ff667c4e689c5aefff561181b4d5fe69d8329"
    assert record["provenance_caveats"]
    verification = record["verification"]
    assert verification["fresh_checkpoint_reinference_performed"] is False
    assert all(
        value is True
        for key, value in verification.items()
        if key != "fresh_checkpoint_reinference_performed"
    )
