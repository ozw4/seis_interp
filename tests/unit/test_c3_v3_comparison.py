import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_interp.configuration import load_resolved_config
from seis_interp.evaluation.c3_v3_comparison import validate_v3_run_artifacts, write_v3_comparison
from seis_interp.pipelines import run_c3_v3_comparison as runner
from tests.fixtures.c3_poc_runs import poc_input_lock, write_poc_run_artifacts


def _v3_run(path, method="nersi"):
    lock = poc_input_lock()
    write_poc_run_artifacts(path, method, inputs_lock=lock)
    metadata = json.loads((path / "metadata.json").read_text())
    metadata.update(
        condition_id="c3_random80_v3",
        input_protocol="c3_random80_v3",
        primary_metric="mean_trace_snr_db",
    )
    (path / "metadata.json").write_text(json.dumps(metadata))
    metrics = json.loads((path / "metrics.json").read_text())
    metrics["evaluation_target"].update(
        mean_trace_snr_db=15.6,
        mean_trace_snr_status="finite",
        trace_snr_trace_count=lock["target_trace_count"],
        trace_snr_positive_infinity_count=0,
        trace_snr_negative_infinity_count=0,
        trace_snr_undefined_count=0,
    )
    (path / "metrics.json").write_text(json.dumps(metrics))
    return lock


@pytest.mark.parametrize("method", ["pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph"])
def test_v3_artifacts_and_csv_use_mean_trace_snr(tmp_path, method):
    run = tmp_path / "run"
    lock = _v3_run(run, method)
    row = validate_v3_run_artifacts(method, run, lock)
    assert row["mean_trace_snr_db"] == 15.6
    assert "global_snr_db" in row
    write_v3_comparison(tmp_path, {"runs": [row]})
    assert "mean_trace_snr_db" in (tmp_path / "summary.csv").read_text().splitlines()[0]


@pytest.mark.parametrize(
    "problem", ["nested_hash", "prediction_hash", "coverage", "trace_count", "nan"]
)
def test_v3_rejects_invalid_comparison(tmp_path, problem):
    run = tmp_path / "run"
    lock = _v3_run(run)
    if problem == "nested_hash":
        lock = deepcopy(lock)
        lock["benchmark_volume"]["files"]["volume.json"]["sha256"] = "e" * 64
    elif problem == "prediction_hash":
        (run / "prediction.npy").write_bytes(b"changed")
    else:
        file = run / ("metadata.json" if problem == "coverage" else "metrics.json")
        obj = json.loads(file.read_text())
        if problem == "coverage":
            obj["coverage"]["complete"] = False
        elif problem == "trace_count":
            obj["evaluation_target"]["trace_snr_trace_count"] -= 1
        else:
            obj["evaluation_target"]["mean_trace_snr_db"] = float("nan")
        file.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        validate_v3_run_artifacts("nersi", run, lock)


def test_v3_writer_rejects_nonfinite_json(tmp_path):
    with pytest.raises(ValueError):
        write_v3_comparison(tmp_path, {"runs": [{"mean_trace_snr_db": float("inf")}]})
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("failed_method", [None, "nersi"])
def test_runner_preserves_all_runs_and_continues(tmp_path, monkeypatch, failed_method):
    config = {"input_protocol": "c3_random80_v3", "input_conditions_lock": str(tmp_path / "lock")}
    (tmp_path / "lock").write_text("{}")
    monkeypatch.setattr(runner, "load_resolved_config", lambda _: config)
    monkeypatch.setattr(
        runner,
        "load_c3_random80_v3_inputs",
        lambda **_: SimpleNamespace(inputs_lock=poc_input_lock()),
    )
    calls = []

    def execute(arguments, **_):
        method = arguments[1].replace("-", "_")
        calls.append(method)
        if method == failed_method:
            raise ValueError("synthetic failure")
        _v3_run(Path(arguments[arguments.index("--output") + 1]), method)

    monkeypatch.setattr(runner, "run_poc_process", execute)
    output = tmp_path / "comparison"
    kwargs = dict(
        input_paths={}, configs=dict.fromkeys(runner.METHODS, tmp_path), output_dir=output
    )
    summary = runner.run_c3_v3_comparison(**kwargs)
    assert calls == list(runner.METHODS)
    assert summary["comparison_valid"] is (failed_method is None)
    assert sum(row["status"] == "success" for row in summary["runs"]) == (
        5 if failed_method is None else 4
    )
    assert all((output / method).is_dir() for method in runner.METHODS)
    with pytest.raises(FileExistsError):
        runner.run_c3_v3_comparison(**kwargs)


@pytest.mark.parametrize("method", ["pocs", "drr", "nersi", "ccnet5d", "gnn"])
def test_v3_configs_preserve_native_parameters(method):
    studies = Path("studies")
    baseline = (
        studies / "study_041_c3_nersi_translated_window" / "shot18_ry18.yaml"
        if method == "nersi"
        else studies
        / (
            "study_036_c3_random80_observed_only_poc"
            if method in {"pocs", "drr"}
            else "study_037_c3_neural_mse_loss_ablation"
        )
        / "formal"
        / f"{method}.yaml"
    )
    expected = load_resolved_config(baseline)
    actual = load_resolved_config(
        studies / "study_042_c3_v3_five_method_comparison" / f"{method}.yaml"
    )
    for key in (
        "input_protocol",
        "input_conditions_lock",
        "input_conditions_sha256",
        "benchmark_case",
        "benchmark_volume",
        "evaluation",
    ):
        expected.pop(key, None)
        actual.pop(key, None)
    if method == "gnn":
        expected["training"].setdefault("mixed_precision", "off")
    assert actual == expected


def test_v3_gnn_native_validator_accepts_only_complete_protocol_keys():
    from seis_interp.configuration import ConfigurationError
    from seis_interp.relational_trace_graph_poc_config import (
        validate_relational_trace_graph_poc_config,
    )

    config = load_resolved_config(Path("studies/study_042_c3_v3_five_method_comparison/gnn.yaml"))
    assert validate_relational_trace_graph_poc_config(config).training["max_steps"] == 5000
    for key in ("input_protocol", "input_conditions_lock", "input_conditions_sha256"):
        incomplete = deepcopy(config)
        incomplete.pop(key)
        with pytest.raises(ConfigurationError):
            validate_relational_trace_graph_poc_config(incomplete)
    config["unexpected"] = True
    with pytest.raises(ConfigurationError):
        validate_relational_trace_graph_poc_config(config)


def test_nersi_no_time_shear_changes_only_alignment():
    from seis_interp.nersi_config import validate_nersi_poc_config

    study = Path("studies/study_042_c3_v3_five_method_comparison")
    expected = load_resolved_config(study / "nersi.yaml")
    expected.pop("time_alignment")
    actual = load_resolved_config(study / "nersi_no_time_shear.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).time_alignment is None


def test_adopted_nersi_v3_result_and_runner_use_no_time_shear(tmp_path, monkeypatch):
    import runpy

    from seis_interp.data.file_checksums import file_sha256

    study = Path("studies/study_042_c3_v3_five_method_comparison")
    record = json.loads((study / "nersi_v3_result.lock.json").read_text())
    config_path = study / "nersi_no_time_shear.yaml"
    assert record["status"] == "adopted"
    assert record["time_shear"] is False
    assert record["primary_metric"] == "mean_trace_snr_db"
    assert record["config_path"] == str(config_path)
    assert record["config_sha256"] == file_sha256(config_path)
    assert record["evaluation_target"]["mean_trace_snr_db"] == 15.585628348312675
    assert record["independent_evaluation"] is False
    captured = {}

    def execute(**kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    main = runpy.run_path("scripts/run_c3_v3_comparison.py")["main"]
    monkeypatch.setitem(main.__globals__, "run_c3_v3_comparison", execute)
    monkeypatch.setattr("sys.argv", ["run_c3_v3_comparison.py", "--output", str(tmp_path)])
    assert main() == 0
    assert captured["configs"]["nersi"] == config_path
