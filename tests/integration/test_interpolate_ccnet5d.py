from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_volume_run_inputs import load_c3_volume_run_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines.interpolate_ccnet5d import interpolate_ccnet5d_run
from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint
from tests.fixtures.ccnet5d_artifacts import prepare_ccnet5d_artifacts
from tests.fixtures.ccnet5d_runs import (
    ccnet5d_inference_config,
    ccnet5d_training_config,
    prepare_ccnet5d_benchmark,
    write_ccnet5d_config,
)


def _training(tmp_path, source, *, activation="linear"):
    config = ccnet5d_training_config(source)
    config["model"]["output_activation"] = activation
    path = write_ccnet5d_config(tmp_path / "train.yaml", config)
    output = tmp_path / "train"
    metrics = train_ccnet5d_run(
        config_path=path,
        interim_dir=source.interim,
        processed_dir=source.processed,
        output_dir=output,
    )
    return output, metrics


def _inference(artifacts, config, checkpoint, output, **kwargs):
    return interpolate_ccnet5d_run(
        config_path=config,
        checkpoint_path=checkpoint,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
        **kwargs,
    )


@pytest.mark.parametrize(
    "mask_kind,role,activation",
    [("random_trace", "best", "linear"), ("random_whole_ffid", "final", "relu")],
)
def test_frozen_checkpoint_inference_records_full_forward_and_truth_metric(
    tmp_path, monkeypatch, mask_kind, role, activation
):
    source = prepare_ccnet5d_artifacts(tmp_path / "data")
    artifacts = prepare_ccnet5d_benchmark(source, mask_kind=mask_kind)
    training, _ = _training(tmp_path, source, activation=activation)
    checkpoint = training / f"artifacts/{role}.pt"
    original_hash = file_sha256(checkpoint)
    loaded = load_ccnet5d_checkpoint(checkpoint)
    state = {name: tensor.clone() for name, tensor in loaded.model.state_dict().items()}
    config = ccnet5d_inference_config(artifacts)
    config["prediction"]["device"] = "cuda:0"
    path = write_ccnet5d_config(tmp_path / "infer.yaml", config)
    output = tmp_path / "infer"

    def no_optimizer(*args, **kwargs):
        pytest.fail("frozen inference must not create an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", no_optimizer)
    timestamps = []

    def timestamp():
        if timestamps:
            assert (output / "artifacts/prediction.npy").is_file()
        timestamps.append("2026-09-07T01:00:00Z")
        return timestamps[-1]

    monkeypatch.setattr(run_records, "utc_timestamp", timestamp)
    messages = []
    metrics = _inference(
        artifacts,
        path,
        checkpoint,
        output,
        device_override="cpu",
        progress_reporter=messages.append,
    )
    assert messages and len(timestamps) == 2
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) == [
        "artifacts/prediction.npy",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    records = {
        name: json.loads((output / f"{name}.json").read_text())
        for name in ("metrics", "run", "inputs.lock")
    }
    for record in records.values():
        json.dumps(record, allow_nan=False)
    assert records["metrics"] == metrics
    run = records["run"]
    assert run["device"] == "cpu" and run["status"] == "success"
    assert run["checkpoint"]["role"] == ("best_selection" if role == "best" else "final")
    assert run["amplitude"] == {
        "scale_source": "checkpoint_fit_region",
        "amplitude_rms": loaded.amplitude_rms,
    }
    assert run["prediction"]["halo_radius"] == 4
    assert run["prediction"]["tile_count"] == 16
    assert metrics["training_regime"] == "supervised_train_partition"
    assert metrics["method_variant"] == (
        "supervised_train_partition_linear_output"
        if activation == "linear"
        else "supervised_train_partition_paper_relu_output"
    )
    assert (
        records["inputs.lock"]["checkpoint"]["sha256"] == original_hash == file_sha256(checkpoint)
    )
    verified = load_c3_volume_run_inputs(
        config=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    assert {
        key: value for key, value in records["inputs.lock"].items() if key != "checkpoint"
    } == verified.inputs_lock
    observed = verified.observed_volume
    assert loaded.amplitude_rms != np.sqrt(
        np.mean(observed.values[:, observed.observed_trace_mask] ** 2, dtype=np.float64)
    )
    prediction = np.load(output / "artifacts/prediction.npy", allow_pickle=False)
    assert prediction.shape == observed.values.shape and prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    assert (
        prediction[:, observed.observed_trace_mask].tobytes()
        == observed.values[:, observed.observed_trace_mask].tobytes()
    )
    with torch.no_grad():
        direct = (
            loaded.model(torch.from_numpy(observed.values / loaded.amplitude_rms)[None, None])[
                0, 0
            ].numpy()
            * loaded.amplitude_rms
        )
    direct[:, observed.observed_trace_mask] = observed.values[:, observed.observed_trace_mask]
    np.testing.assert_allclose(prediction, direct, rtol=1e-6, atol=1e-3)
    target_rows = observed.array_rows[observed.evaluation_target_trace_mask]
    truth = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)[
        target_rows, 1:5
    ].T.astype(np.float64)
    errors = truth - prediction[:, observed.evaluation_target_trace_mask].astype(np.float64)
    target = metrics["evaluation_target"]
    assert target["snr_db"] == pytest.approx(10 * np.log10(np.sum(truth**2) / np.sum(errors**2)))
    assert target["sample_count"] == truth.size
    assert (
        metrics["observed_max_abs_error"]
        == metrics["uncovered_sample_count"]
        == metrics["uncovered_trace_count"]
        == 0
    )
    assert (
        metrics["amplitude_domain"] == "physical"
        and metrics["evaluation_domain"] == "evaluation_target"
    )
    reloaded = load_ccnet5d_checkpoint(checkpoint)
    for name, tensor in state.items():
        assert torch.equal(tensor, reloaded.model.state_dict()[name])


def test_benchmark_target_change_cannot_affect_training_rms_weights_or_prediction(tmp_path):
    runs = []
    for index, offset in enumerate((0.0, 50000.0)):
        root = tmp_path / str(index)
        root.mkdir()
        source = prepare_ccnet5d_artifacts(root / "data")
        artifacts = prepare_ccnet5d_benchmark(source, target_offset=offset)
        training, training_metrics = _training(root, source)
        checkpoint = training / "artifacts/best.pt"
        config = write_ccnet5d_config(root / "infer.yaml", ccnet5d_inference_config(artifacts))
        output = root / "infer"
        metrics = _inference(artifacts, config, checkpoint, output)
        runs.append(
            (
                training_metrics,
                load_ccnet5d_checkpoint(checkpoint),
                np.load(output / "artifacts/prediction.npy"),
                metrics,
            )
        )
    left, right = runs
    assert left[0] == right[0]
    assert left[1].amplitude_rms == right[1].amplitude_rms
    assert left[1].training_provenance != right[1].training_provenance
    for name, value in left[1].model.state_dict().items():
        assert torch.equal(value, right[1].model.state_dict()[name])
    np.testing.assert_array_equal(left[2], right[2])
    assert left[3]["evaluation_target"] != right[3]["evaluation_target"]
    assert (
        left[3]["observed_model_rmse_before_reinsertion"]
        == right[3]["observed_model_rmse_before_reinsertion"]
    )


@pytest.mark.parametrize(
    "problem",
    ["hash", "train_partition", "spatial_overlap", "dataset", "evaluation", "core", "constructor"],
)
def test_preflight_rejects_before_creating_output(tmp_path, problem):
    source = prepare_ccnet5d_artifacts(tmp_path / "data")
    partition = "train" if problem == "train_partition" else "validation"
    artifacts = prepare_ccnet5d_benchmark(source, partition=partition)
    training, _ = _training(tmp_path, source)
    checkpoint = training / "artifacts/best.pt"
    config = ccnet5d_inference_config(artifacts, partition=partition)
    if problem in ("hash", "spatial_overlap"):
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        lock = payload["training_provenance"]["source_inputs_lock"]
        if problem == "hash":
            lock["interim"]["amplitudes.npy"]["sha256"] = "f" * 64
        else:
            selection = deepcopy(artifacts.volume_metadata["selection"])
            selection["time"] = [0, 1]  # Time-disjoint is still forbidden for the same traces.
            lock["regions"]["fit"]["selection"] = selection
        checkpoint = tmp_path / "changed.pt"
        torch.save(payload, checkpoint)
    elif problem == "dataset":
        config["data"]["dataset_id"] = "wrong"
    elif problem == "evaluation":
        config["evaluation"]["domain"] = "all_traces"
    elif problem == "core":
        config["prediction"]["core_shape"] = [0, 1, 1, 1, 1]
    elif problem == "constructor":
        config["model"] = {"name": "ccnet5d"}
    path = write_ccnet5d_config(tmp_path / "infer.yaml", config)
    output = tmp_path / "infer"
    with pytest.raises(ValueError):
        _inference(artifacts, path, checkpoint, output)
    assert not output.exists()


def test_existing_output_is_unchanged(tmp_path):
    output = tmp_path / "keep"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        interpolate_ccnet5d_run(
            config_path=tmp_path,
            checkpoint_path=tmp_path,
            interim_dir=tmp_path,
            processed_dir=tmp_path,
            mask_dir=tmp_path,
            case_dir=tmp_path,
            volume_dir=tmp_path,
            output_dir=output,
        )
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "unchanged"


def test_interpolate_cli_end_to_end(tmp_path, capsys):
    source = prepare_ccnet5d_artifacts(tmp_path / "data")
    artifacts = prepare_ccnet5d_benchmark(source)
    training, _ = _training(tmp_path, source)
    path = write_ccnet5d_config(tmp_path / "infer.yaml", ccnet5d_inference_config(artifacts))
    output = tmp_path / "infer"
    args = ["interpolate", "ccnet5d"]
    for name, value in {
        "config": path,
        "checkpoint": training / "artifacts/best.pt",
        "interim": artifacts.interim,
        "processed": artifacts.processed,
        "mask": artifacts.mask,
        "case": artifacts.case,
        "volume": artifacts.volume,
        "output": output,
    }.items():
        args.extend([f"--{name}", str(value)])
    result = main([*args, "--device", "cpu", "--json"])
    captured = capsys.readouterr()
    assert result == 0
    summary = json.loads(captured.out)
    assert summary["observed_max_abs_error"] == 0
    assert captured.err
