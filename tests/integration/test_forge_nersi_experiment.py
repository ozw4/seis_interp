import json
import os
import re

import numpy as np
import pandas as pd
import pytest
import segyio
import torch
import yaml

from seis_interp.data.forge_aligned_inputs import moveout_canvas
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs
from seis_interp.evaluation.forge_mvp_metrics import validate_predictions, verify_saved_metrics
from seis_interp.pipelines import run_forge_nersi_experiment as pipeline
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_nersi_experiment import (
    check_forge_nersi_experiment,
    format_check_report,
    format_experiment_summary,
    run_forge_nersi_experiment,
    write_experiment_summary,
)
from seis_interp.training.forge_mvp_methods import (
    canvas_method_config,
    initialize_model,
    model_parameter_count,
    run_method,
)
from tests.fixtures.forge_mvp import tiny_forge_study

BUDGET = {
    "max_steps": 1,
    "learning_rate": 0.001,
    "loss_traces_per_update": 2,
    "parameter_count_target": 1300,
    "parameter_count_tolerance": 0.95,
}
RUN_FILES = (
    "config.requested.yaml",
    "config.resolved.yaml",
    "inputs.lock.json",
    "run.json",
    "metrics.json",
    "metrics_by_trace.parquet",
    "training_log.parquet",
    "training.log",
    "artifacts/final.pt",
    "artifacts/prediction.npy",
    "artifacts/prediction_index.parquet",
    "figures/training_loss.png",
    "figures/common_source.png",
    "figures/common_receiver.png",
    "figures/common_source_errors.png",
    "figures/common_receiver_errors.png",
)


def _quiet(message):
    return None


def _prepare_tiny(root, monkeypatch, variant):
    study, contract, _, configs, *_ = tiny_forge_study(root)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    if variant == "aligned":
        contract["preprocessing"]["alignment"] = {"velocity_m_per_s": 40000.0}
        contract["matched_training"] = BUDGET
        configs["nersi_real"]["training"]["batch_size"] = 2
        configs["ccnet5d_grid"]["training"]["supervised_traces_per_update"] = 2
        for name in ("regsi_real", "regsi_grid"):
            configs[name]["training"].update(inner_mask_fraction=0.5, query_batch_size=2)
    else:
        contract["time_sampling"] = {
            "native_sample_count": 9,
            "native_sample_interval_us": 1000,
            "decimation_factor": 2,
        }
        contract.update(time_sample_count=5, sample_interval_us=2000)
        configs["nersi_real"]["model"]["time_sample_count"] = 5
        configs["ccnet5d_grid"]["patches"]["shape"][0] = 5
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    for name, config in configs.items():
        (study / f"{name}.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.run_forge_nersi_experiment.execution_identity",
        lambda _: {
            "git_commit": "testcommit",
            "dirty_worktree": True,
            "hardware": {},
            "software_versions": {},
        },
    )
    preparation = prepare_forge_mvp(root, study)
    inputs = root / "inputs.yaml"
    inputs.write_text(
        yaml.safe_dump(
            {
                "preparation": str(preparation.relative_to(root)),
                "preparation_manifest_sha256": sha256_file(
                    preparation / "preparation_manifest.json"
                ),
            }
        )
    )
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(_tiny_config(variant), sort_keys=False))
    return {
        "root": root,
        "preparation": preparation,
        "config": config,
        "inputs": inputs,
        "output": root / "runs/study_forge_nersi_temp",
        "variant": variant,
    }


def _tiny_config(variant):
    return {
        "name": f"tiny_{variant}",
        "device": "cpu",
        "model_seed": 7301,
        "torch_threads": 1,
        "model": {
            "coordinate_frame": "midpoint_offset",
            "polar_offset": True,
            "moveout_velocity_m_per_s": None,
            "fourier_components": 2,
            "frequency_base": 1.25,
            "encoder_width": 4,
            "latent_channels": 2,
            "decoder_channels": [2, 2, 2],
            "kernel_size": 1,
            "head_slowness_ms_per_m": None,
            "static_shift_samples": 0.0,
            "residual_slowness_ms_per_m": 0.0,
            "output_band_hz": None,
            "output_window_mask": False,
        },
        "training": {
            "learning_rate": 0.001,
            "max_steps": 1,
            "batch_size": 2,
            "weight_decay": 0.0,
            "weight_decay_after": None,
            "ema_decay": None,
            "report_interval": 1,
        },
        "prediction": {"batch_size": 4},
    }


@pytest.fixture
def pinned_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["aligned", "decimated"])
def experiment(tmp_path, monkeypatch, pinned_threads, request):
    return _prepare_tiny(tmp_path, monkeypatch, request.param)


@pytest.fixture
def aligned(tmp_path, monkeypatch, pinned_threads):
    return _prepare_tiny(tmp_path, monkeypatch, "aligned")


def _run(setup):
    return run_forge_nersi_experiment(
        setup["root"], setup["config"], setup["inputs"], setup["output"], reporter=_quiet
    )


def _edit_config(setup, changes):
    """Rewrite the tiny config with ``{"section.key": value}`` changes."""
    config = yaml.safe_load(setup["config"].read_text())
    for path, value in changes.items():
        *parents, key = path.split(".")
        (config[parents[0]] if parents else config)[key] = value
    setup["config"].write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def _only_run_dir(setup):
    (run_dir,) = [path for path in setup["output"].iterdir() if path.is_dir()]
    return run_dir


def test_run_record_is_complete_and_reproducible(experiment):
    run_dir = _run(experiment)
    variant = experiment["variant"]
    assert re.fullmatch(rf"\d{{8}}T\d{{12}}Z_testcommit_tiny_{variant}", run_dir.name)
    for name in RUN_FILES:
        assert (run_dir / name).is_file(), name
    record = json.loads((run_dir / "run.json").read_text())
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert record["status"] == "completed"
    assert record["failed_stage"] is None and record["end_time"]
    assert record["evaluation"]["independent_test_estimate"] is False
    assert record["evaluation"]["amplitude_scaling"] == "trace_oracle"
    assert record["git_worktree_dirty"] is True
    assert record["requested_config"] == _tiny_config(variant)
    assert record["config_sha256"] == sha256_file(experiment["config"])
    assert (run_dir / "config.requested.yaml").read_bytes() == experiment["config"].read_bytes()
    assert "src/seis_interp/pipelines/run_forge_nersi_experiment.py" in record["source_hashes"]
    lock = json.loads((run_dir / "inputs.lock.json").read_text())
    assert set(lock) == {"preparation", "preparation_manifest_sha256", "input_binding"}
    assert lock["preparation_manifest_sha256"] == sha256_file(
        experiment["preparation"] / "preparation_manifest.json"
    )
    log = (run_dir / "training.log").read_text()
    assert "NeRSI step 1/1 minibatch_loss " in log
    assert "verifying the raw SEG-Y hashes" in log and "completed" in log
    for relative, digest in record["artifact_hashes"].items():
        assert sha256_file(run_dir / relative) == digest
    assert set(record["artifact_hashes"]) == set(RUN_FILES) - {"run.json"}
    canvas = record["canvas_sample_count"]
    assert record["parameter_count"] == model_parameter_count(resolved, canvas)
    if variant == "aligned":
        assert canvas == 15
        assert record["budget"]["budget_matched"] is True
    else:
        assert canvas == 5
        assert record["budget"] is None

    inputs, _ = load_model_inputs(experiment["preparation"], "nersi_real")
    expected = run_method(inputs, resolved, reporter=_quiet)
    mask = pd.read_parquet(experiment["preparation"] / "masks/m1_random80_mvp_v1.parquet")
    order = validate_predictions(expected.prediction, expected.cell_ids, mask, len(inputs.time_s))
    saved = np.load(run_dir / "artifacts/prediction.npy")
    np.testing.assert_array_equal(saved, expected.prediction[order])
    if variant == "decimated":
        assert saved.shape == (28, 5)
    checkpoint = torch.load(run_dir / "artifacts/final.pt", weights_only=False)
    assert checkpoint["steps_completed"] == 1
    assert checkpoint["selection_rule"] == "fixed_final"
    for key, value in expected.checkpoint["model_state_dict"].items():
        torch.testing.assert_close(checkpoint["model_state_dict"][key], value, rtol=0, atol=0)
    assert pd.read_parquet(run_dir / "training_log.parquet").step.iloc[-1] == 1
    metrics = json.loads((run_dir / "metrics.json").read_text())
    verify_saved_metrics(metrics, pd.read_parquet(run_dir / "metrics_by_trace.parquet"))

    summary = pd.read_csv(experiment["output"] / "summary.csv")
    assert list(summary.status) == ["completed"]
    assert summary.clean_target_relative_mse.iloc[0] == pytest.approx(
        metrics["clean_target"]["masked_trace_relative_mse"], rel=1e-15
    )
    assert "model.polar_offset" in summary.changed_keys.iloc[0].split("; ")
    assert summary["training.learning_rate"].iloc[0] == 0.001
    assert summary["model.decoder_channels"].iloc[0] == "[2, 2, 2]"
    assert summary["model.encoder_width"].iloc[0] == 4


def test_ema_and_moveout_run(aligned):
    _edit_config(
        aligned,
        {
            "training.ema_decay": 0.5,
            "training.max_steps": 3,
            "model.moveout_velocity_m_per_s": 5000.0,
        },
    )
    run_dir = _run(aligned)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["ema_initial_weight_fraction"] == pytest.approx(0.25)
    checkpoint = torch.load(run_dir / "artifacts/final.pt", weights_only=False)
    assert checkpoint["selection_rule"] == "fixed_final_ema"
    assert checkpoint["steps_completed"] == 3
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert resolved["model"]["moveout_samples_per_radius"] > 0
    resolved["model"]["moveout_samples_per_radius"] = 0.0
    assert record["parameter_count"] > model_parameter_count(resolved, 15)
    summary = pd.read_csv(aligned["output"] / "summary.csv")
    assert summary.ema_initial_weight_fraction.iloc[0] == pytest.approx(0.25)


# The tiny aligned preparation: 40000 m/s alignment (0.025 ms/m), 1 ms samples, T 9, canvas 15.
DECODER_PRIORS = {
    "model.head_slowness_ms_per_m": [0.02, 0.04],
    "model.static_shift_samples": 1.0,
    "model.residual_slowness_ms_per_m": 0.01,
    "model.output_band_hz": [20, 200],
    "model.output_window_mask": True,
    "training.max_steps": 3,
    "training.weight_decay_after": {"step": 2, "value": 1.0e-4},
}


def test_decoder_prior_run_matches_a_fresh_run_method(aligned):
    _edit_config(aligned, DECODER_PRIORS)
    run_dir = _run(aligned)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "completed"
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    model = resolved["model"]
    assert len(model["head_samples_per_radius"]) == 2
    assert model["head_samples_per_radius"][0] > 0 > model["head_samples_per_radius"][1]
    assert model["static_shift_samples"] == 1.0
    assert model["residual_slowness_samples_per_radius"] > 0
    assert model["spatial_shape"] == list(
        load_model_inputs(aligned["preparation"], "nersi_real")[0].spatial_shape
    )
    assert model["output_band_hz"] == [20, 200] and model["sample_interval_s"] == 0.001
    assert model["output_window"]["max_delay_samples"] == 6
    assert model["output_window"]["window_samples"] == 9
    assert resolved["training"]["weight_decay_after"] == {"step": 2, "value": 1.0e-4}
    assert record["parameter_count"] == model_parameter_count(resolved, 15)
    assert {"src_static", "rcv_static"} <= set(record["parameter_breakdown"])
    changes = record["changes_from_preparation_config"]
    assert any(item.startswith("training.weight_decay_after: null -> ") for item in changes)

    inputs, _ = load_model_inputs(aligned["preparation"], "nersi_real")
    expected = run_method(inputs, resolved, reporter=_quiet)
    mask = pd.read_parquet(aligned["preparation"] / "masks/m1_random80_mvp_v1.parquet")
    order = validate_predictions(expected.prediction, expected.cell_ids, mask, len(inputs.time_s))
    np.testing.assert_array_equal(
        np.load(run_dir / "artifacts/prediction.npy"), expected.prediction[order]
    )
    summary = pd.read_csv(aligned["output"] / "summary.csv")
    assert json.loads(summary["model.head_slowness_ms_per_m"].iloc[0]) == [0.02, 0.04]
    assert json.loads(summary["training.weight_decay_after"].iloc[0]) == {
        "step": 2,
        "value": 1.0e-4,
    }
    assert bool(summary["model.output_window_mask"].iloc[0]) is True


def test_decoder_prior_check_lines_and_window_match_the_constructed_model(aligned):
    _edit_config(aligned, DECODER_PRIORS)
    report = check_forge_nersi_experiment(aligned["root"], aligned["config"], aligned["inputs"])
    resolved = report["run_config"]
    inputs, _ = load_model_inputs(aligned["preparation"], "nersi_real")
    canvas = moveout_canvas(inputs, resolved["alignment"])
    model = initialize_model(canvas_method_config(resolved, canvas.length))
    shifts = resolved["model"]["head_samples_per_radius"]
    heads_line = (
        f"heads         2 heads, slowness 0.02..0.04 ms/m -> shifts {min(shifts):+.1f}.."
        f"{max(shifts):+.1f} samples per unit radius; pad {model.pad_left} + {model.pad_right}, "
        f"decoder length {model.decoder_time_count} (canvas {canvas.length})"
    )
    # The report's grids name the source and receiver roles of the model's own parameters.
    grids = (*model.src_static.shape[2:], *model.rcv_static.shape[2:])
    assert grids == tuple(inputs.spatial_shape) == tuple(report["statics"]["spatial_shape"])
    statics_line = (
        "statics       +-1 samples per source and per receiver (tanh-bounded, zero start), "
        "grids {}x{} + {}x{}".format(*grids)
    )
    lines = format_check_report(report).splitlines()
    assert heads_line in lines and statics_line in lines

    # The output window starts where the canvas placed each recorded trace.
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    table = inputs.geometry.set_index("cell_id")
    local = table[columns].to_numpy() / np.maximum(np.asarray(inputs.spatial_shape) - 1, 1)
    transform = model.coordinate_transform
    offset = torch.from_numpy(local.astype(np.float32)) @ transform[2:, :4].T + transform[2:, 4]
    radius = torch.linalg.vector_norm(offset, dim=1)
    keep = model._apply_window(torch.ones(len(table), canvas.length), radius).numpy()
    start = canvas.max_delay_samples - canvas.delays_for(table.index.to_numpy())
    rows = np.arange(canvas.length)[None, :]
    expected = (rows >= start[:, None]) & (rows < start[:, None] + canvas.time_sample_count)
    np.testing.assert_array_equal(keep.astype(bool), expected)


def test_statics_line_names_the_source_grid_before_the_receiver_grid():
    line = pipeline._statics_line({"samples": 6.0, "spatial_shape": [6, 32, 8, 32]})
    assert line.endswith("grids 6x32 + 8x32")


def test_targets_read_once_after_training(aligned, monkeypatch):
    calls = []
    original_run, original_read = pipeline.run_method, pipeline.read_selected_traces
    original_open, opened = segyio.open, []
    _edit_config(aligned, {"torch_threads": 2})

    def run(*args, **kwargs):
        calls.append("run_method")
        return original_run(*args, **kwargs)

    def read(*args, **kwargs):
        # Prediction, checkpoint and loss log are already on disk before any target read.
        (run_dir,) = [path for path in aligned["output"].iterdir() if path.is_dir()]
        for name in ("artifacts/prediction.npy", "artifacts/final.pt", "training_log.parquet"):
            assert (run_dir / name).is_file(), name
        calls.append("read_selected_traces")
        return original_read(*args, **kwargs)

    def segy_open(*args, **kwargs):
        assert "run_method" in calls, "SEG-Y opened before training finished"
        opened.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_method", run)
    monkeypatch.setattr(pipeline, "read_selected_traces", read)
    monkeypatch.setattr(segyio, "open", segy_open)
    run_dir = _run(aligned)
    assert calls == ["run_method", "read_selected_traces"]
    assert opened
    assert torch.get_num_threads() == 1
    npy_files = [path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*.npy")]
    assert npy_files == ["artifacts/prediction.npy"]


CHECK_CASES = {
    "aligned_default": (
        "aligned",
        {},
        [
            "canvas 15 samples (T 9 at 1000 us)",
            "matched yes",
            "moveout       off",
            "heads         off",
            "statics       off",
            "residual dm   off",
            "output band   off",
            "output window off",
            "weight decay  0 constant",
            "EMA           off",
        ],
    ),
    "decoder_priors": (
        "aligned",
        DECODER_PRIORS,
        [
            "heads         2 heads, slowness 0.02..0.04 ms/m -> shifts ",
            " samples per unit radius; pad ",
            "statics       +-1 samples per source and per receiver",
            "residual dm   +-0.01 ms/m per head -> ",
            "output band   20..200 Hz flat, zero-phase raised-cosine ramps 18..20 and 200..205 Hz",
            "output window zero outside [6 - rint(r * ",
            "), +9) samples",
            "weight decay  0 before step 2, 0.0001 from step 2",
            "moveout       off",
        ],
    ),
    "ema_moveout": (
        "aligned",
        {
            "training.ema_decay": 0.5,
            "training.max_steps": 3,
            "model.moveout_velocity_m_per_s": 5000.0,
        },
        [
            "moveout       5000.0 m/s -> ",
            "decoder-flat events have 4444.4 m/s on the raw gather",
            "EMA           weight left on first update 0.2500",
            "WARNING: EMA keeps 0.2500 of the step-1 weights",
            "steps 3!=1",
            "matched no",
        ],
    ),
    "source_receiver": (
        "aligned",
        {"model.coordinate_frame": "source_receiver", "model.polar_offset": False},
        ["source_receiver -> [source_line, source_point, receiver_line, receiver_point]"],
    ),
    "midpoint_cartesian": (
        "aligned",
        {"model.polar_offset": False},
        ["coordinates   midpoint_offset -> [cmp_x, cmp_y, half_offset_x, half_offset_y]"],
    ),
    "decimated": (
        "decimated",
        {},
        ["alignment none", "budget        none declared by the preparation", "canvas 5 samples"],
    ),
}


@pytest.mark.parametrize("case", sorted(CHECK_CASES))
def test_check_writes_nothing_and_reads_no_targets(tmp_path, monkeypatch, pinned_threads, case):
    variant, changes, expected = CHECK_CASES[case]
    setup = _prepare_tiny(tmp_path, monkeypatch, variant)
    _edit_config(setup, changes)

    def forbidden(*args, **kwargs):
        raise AssertionError("check must not train or read targets")

    monkeypatch.setattr(pipeline, "run_method", forbidden)
    monkeypatch.setattr(pipeline, "read_selected_traces", forbidden)
    before = sorted(setup["root"].rglob("*"))
    report = check_forge_nersi_experiment(setup["root"], setup["config"], setup["inputs"])
    assert sorted(setup["root"].rglob("*")) == before
    assert not setup["output"].exists()
    text = format_check_report(report)
    assert f"parameters    {report['parameter_count']:,} (" in text
    for line in expected:
        assert line in text, line
    assert text.endswith("VALID")


def test_invalid_config_creates_no_run_directory(aligned):
    config = yaml.safe_load(aligned["config"].read_text())
    config["training"]["batch_size"] = 8
    aligned["config"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="8 exceeds the 7 observed traces"):
        _run(aligned)
    config["training"]["batch_size"] = 2
    aligned["config"].write_text(yaml.safe_dump(config))
    inputs = yaml.safe_load(aligned["inputs"].read_text())
    inputs["preparation_manifest_sha256"] = "0" * 64
    aligned["inputs"].write_text(yaml.safe_dump(inputs))
    with pytest.raises(ValueError, match="preparation manifest hash mismatch.*the file has"):
        _run(aligned)
    assert not aligned["output"].exists()


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"device": "cuda:99"}, ValueError, "device: cuda:99 not available"),
        ({"model.fourier_components": 40, "model.frequency_base": 10.0}, ValueError, "model: "),
    ],
)
def test_preparation_dependent_errors_create_no_run_directory(aligned, changes, error, message):
    _edit_config(aligned, changes)
    with pytest.raises(error, match=message):
        _run(aligned)
    assert not aligned["output"].exists()


def test_missing_preparation_creates_no_run_directory(aligned):
    inputs = yaml.safe_load(aligned["inputs"].read_text())
    inputs["preparation"] = "runs/absent"
    aligned["inputs"].write_text(yaml.safe_dump(inputs))
    with pytest.raises(FileNotFoundError, match="preparation not found"):
        _run(aligned)
    assert not aligned["output"].exists()


def test_training_failure_is_recorded(aligned, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic OOM")

    _edit_config(aligned, {"torch_threads": 2})
    monkeypatch.setattr(pipeline, "run_method", fail)
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        _run(aligned)
    assert torch.get_num_threads() == 1
    run_dir = _only_run_dir(aligned)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["failed_stage"] == "training"
    assert record["failure_reason"] == "RuntimeError: synthetic OOM"
    assert record["end_time"]
    assert not (run_dir / "artifacts").exists()
    summary = pd.read_csv(aligned["output"] / "summary.csv")
    assert list(summary.status) == ["failed"]
    assert summary.clean_target_relative_mse.isna().all()


def test_interrupt_is_recorded_and_reraised(aligned, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_method", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(aligned)
    record = json.loads((_only_run_dir(aligned) / "run.json").read_text())
    assert record["status"] == "interrupted"
    assert record["failed_stage"] == "training"


def test_target_read_failure_keeps_the_saved_prediction(aligned, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("raw SEG-Y hash differs")

    monkeypatch.setattr(pipeline, "verify_hashes", fail)
    with pytest.raises(ValueError, match="raw SEG-Y hash differs"):
        _run(aligned)
    run_dir = _only_run_dir(aligned)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["failed_stage"] == "target_read"
    assert record["artifact_hashes"]["artifacts/prediction.npy"] == sha256_file(
        run_dir / "artifacts/prediction.npy"
    )


def test_summary_rebuild_statuses(tmp_path):
    output = tmp_path / "runs"
    records = {
        "20260101T000000000000Z_a_done": {
            "status": "completed",
            "requested_config": {"model": {"frequency_base": [1.1, 1.2, 1.3, 1.4]}},
            "changes_from_preparation_config": ["model.encoder_width: 150 -> 200"],
            "budget": {"parameter_ratio": 0.01, "budget_matched": True},
            "metrics_summary": {"clean_target_relative_mse": 0.51234567},
        },
        "20260101T000000000000Z_b_best": {
            "status": "completed",
            "changes_from_preparation_config": [],
            "parameter_count": 10,
            "budget": {"parameter_ratio": 0.0, "budget_matched": True},
            "metrics_summary": {
                "clean_target_relative_mse": 0.25,
                "all_eligible_relative_mse": 0.3,
                "clean_median_correlation": 0.9,
            },
        },
        "20260101T000000000001Z_a_failed": {
            "status": "failed",
            "failed_stage": "training",
            "metrics_summary": {"clean_target_relative_mse": 0.1},
        },
        "20260101T000000000002Z_a_running": {"status": "running", "pid": os.getpid()},
    }
    for name, record in records.items():
        (output / name).mkdir(parents=True)
        (output / name / "run.json").write_text(json.dumps({"run_id": name, **record}))
    (output / "20260101T000000000003Z_a_broken").mkdir()
    (output / "20260101T000000000003Z_a_broken/run.json").write_text("{not json")
    (output / "not_a_run").mkdir()

    path = write_experiment_summary(output)
    summary = pd.read_csv(path)
    assert list(summary.status) == ["completed", "completed", "failed", "running", "unreadable"]
    assert list(summary.clean_target_relative_mse.iloc[:2]) == [0.51234567, 0.25]
    assert summary.clean_target_relative_mse.iloc[2:].isna().all()
    assert json.loads(summary["model.frequency_base"].iloc[0]) == [1.1, 1.2, 1.3, 1.4]
    assert summary.changed_keys.iloc[0] == "model.encoder_width"
    assert summary.changes.iloc[0] == "model.encoder_width: 150 -> 200"
    assert not list(output.glob(".summary.*.tmp"))

    text = format_experiment_summary(path)
    lines = text.splitlines()
    best = next(i for i, line in enumerate(lines) if "_b_best" in line)
    done = next(i for i, line in enumerate(lines) if "_a_done" in line)
    assert best < done
    assert "0.5123" in lines[done] and "model.encoder_width: 150 -> 200" in lines[done]
    assert "NaN" not in lines[best]  # a run without changes prints a blank changes cell
    other = lines.index("other runs")
    for name, status in (
        ("a_failed", "failed"),
        ("a_running", "running"),
        ("a_broken", "unreadable"),
    ):
        row = next(line for line in lines[other:] if name in line)
        assert status in row

    empty = tmp_path / "empty"
    assert format_experiment_summary(write_experiment_summary(empty)).startswith("no runs yet")
