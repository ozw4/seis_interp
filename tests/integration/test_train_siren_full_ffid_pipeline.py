from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.cli import main
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.train_siren import train_siren_run
from seis_interp.processing.normalization import read_normalization_parameters
from seis_interp.training.checkpoints import load_siren_checkpoint


def _build_full_ffid_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_a = tmp_path / "source_a.sgy"
    source_b = tmp_path / "source_b.sgy"
    source_a.write_bytes(b"synthetic source A")
    source_b.write_bytes(b"synthetic source B")
    interim = tmp_path / "interim"
    trace_count = 30
    sample_count = 4
    array_indices = np.arange(trace_count)
    trace_table = pd.DataFrame(
        {
            "source_file": np.repeat([source_a.name, source_b.name], [20, 10]),
            "trace_index": np.concatenate(
                [np.arange(20, dtype=np.int64), np.arange(10, dtype=np.int64)]
            ),
            "ffid": np.repeat([10, 20, 30], 10),
            "cmp_x_m": array_indices.astype(np.float64),
            "cmp_y_m": array_indices.astype(np.float64) * 2.0,
            "offset_m": 100.0 + array_indices.astype(np.float64),
            "azimuth_deg": array_indices.astype(np.float64) * 7.0,
            "sample_interval_s": np.full(trace_count, 0.008),
        }
    )
    time_s = np.arange(sample_count, dtype=np.float64) * 0.008
    amplitudes = (
        np.sin(array_indices[:, np.newaxis] * 0.2 + time_s[np.newaxis, :] * 10.0) + 1.5
    ).astype(np.float32)
    write_interim_trace_dataset(
        interim,
        trace_table,
        amplitudes,
        time_s,
        source_a,
        "synthetic",
        selection={"ffid_scope": "all", "include_incomplete_ffids": True},
    )
    metadata_path = interim / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("source_file")
    metadata.pop("source_sha256")
    metadata["source_files"] = [
        {"name": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        for source in (source_a, source_b)
    ]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.3,
        validation_fraction_of_holdout=0.5,
        random_seed=7,
        split_scope="per_ffid",
        config_source="studies/synthetic/config.yaml",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 7},
                "sampling": {
                    "random_trace_holdout_fraction": 0.3,
                    "validation_fraction_of_holdout": 0.5,
                    "split_scope": "per_ffid",
                },
                "normalization": {
                    "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
                    "amplitude": "train_global_rms",
                },
                "model": {
                    "name": "siren",
                    "input_features": 6,
                    "hidden_width": 8,
                    "hidden_layers": 1,
                    "omega_0": 10.0,
                    "hidden_omega": 1.0,
                },
                "training": {
                    "batch_mode": "full_ffid_epoch",
                    "loss": "l2",
                    "optimizer": "adam",
                    "learning_rate": 1.0e-3,
                    "max_epochs": 2,
                    "early_stopping_patience": 2,
                    "validation_batch_size": 5,
                    "device": "cpu",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed


def test_full_ffid_pipeline_streams_training_and_writes_the_run_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output = tmp_path / "run"
    import seis_interp.pipelines.train_siren as pipeline

    actual_load_interim = pipeline.load_interim_trace_dataset
    memory_map_flags: list[bool] = []

    def recording_load_interim(directory: Path, *, memory_map_amplitudes: bool = False):
        memory_map_flags.append(memory_map_amplitudes)
        return actual_load_interim(
            directory,
            memory_map_amplitudes=memory_map_amplitudes,
        )

    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.normalize_amplitudes",
        lambda *_args, **_kwargs: pytest.fail("full mode must not normalize the full array"),
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.build_trace_points",
        lambda *_args, **_kwargs: pytest.fail("full mode must not build all validation points"),
    )
    monkeypatch.setattr(pipeline, "load_interim_trace_dataset", recording_load_interim)
    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "artifacts/best.pt",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    assert metrics["batch_mode"] == "full_ffid_epoch"
    assert memory_map_flags == [True]
    assert metrics["training_ffid_count"] == metrics["validation_ffid_count"] == 3
    assert metrics["global_steps"] == metrics["epochs_completed"] * 3
    assert metrics["effective_steps_per_epoch"] == 3
    assert all("mean_ffid_batch_loss" in record for record in metrics["history"])
    checkpoint = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.validation_median_trace_snr_db is None
    assert checkpoint.validation_global_snr_db == pytest.approx(
        metrics["best_validation_global_snr_db"]
    )
    assert checkpoint.model.input_features == 6
    assert checkpoint.model.hidden_width == 8
    assert checkpoint.model.hidden_layers == 1
    assert checkpoint.model.omega_0 == 10.0
    assert checkpoint.model.hidden_omega == 1.0
    assert checkpoint.normalization == read_normalization_parameters(
        processed / "normalization.json"
    )

    inputs_lock_text = (output / "inputs.lock.json").read_text(encoding="utf-8")
    inputs_lock = json.loads(inputs_lock_text)
    assert str(tmp_path) not in inputs_lock_text
    assert [source["name"] for source in inputs_lock["source_files"]] == [
        "source_a.sgy",
        "source_b.sgy",
    ]
    assert inputs_lock["preparation"]["split_scope"] == "per_ffid"
    assert inputs_lock["training"] == {
        "batch_mode": "full_ffid_epoch",
        "training_ffid_count": 3,
        "validation_ffid_count": 3,
        "effective_steps_per_epoch": 3,
        "ffid_range": [10, 30],
        "training_traces_per_ffid": {"min": 7, "median": 7.0, "max": 7},
        "points_per_update": {"min": 28, "median": 28.0, "max": 28},
        "amplitude_scaling": "train_global_rms",
        "validation": "all_validation_traces_streamed",
    }
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["batch_mode"] == "full_ffid_epoch"
    assert run_metadata["effective_steps_per_epoch"] == 3


def test_full_ffid_pipeline_routes_no_test_rows_to_training_or_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    expected = {
        split: set(split_table.loc[split_table["split"] == split, "array_row"])
        for split in ("train", "validation", "test")
    }
    received: dict[str, set[int]] = {}
    import seis_interp.pipelines.train_siren as pipeline

    actual_sampler = pipeline.FullFfidBatchSampler
    actual_evaluate = pipeline.evaluate_model_global_snr_by_ffid

    def recording_sampler(*args: Any, **kwargs: Any):
        received["train"] = {int(row) for rows in args[3].values() for row in rows}
        return actual_sampler(*args, **kwargs)

    def recording_evaluate(*args: Any, **kwargs: Any) -> float:
        received["validation"] = {
            int(row) for rows in kwargs["rows_by_ffid"].values() for row in rows
        }
        return actual_evaluate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "FullFfidBatchSampler", recording_sampler)
    monkeypatch.setattr(pipeline, "evaluate_model_global_snr_by_ffid", recording_evaluate)
    train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
        progress_reporter=lambda _message: None,
    )

    assert received == {"train": expected["train"], "validation": expected["validation"]}
    assert expected["test"].isdisjoint(received["train"] | received["validation"])


def test_full_ffid_pipeline_requires_per_ffid_preparation(tmp_path: Path) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    preparation_path = processed / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["split_scope"] = "global"
    preparation_path.write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="split_scope"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )


def test_full_ffid_pipeline_ignores_inherited_random_point_step_controls(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["batch_size"] = "not used by full_ffid_epoch"
    config_payload["training"]["steps_per_epoch"] = -100
    config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
        progress_reporter=lambda _message: None,
    )

    assert metrics["effective_steps_per_epoch"] == 3


def test_full_ffid_pipeline_rejects_invalid_validation_batch_size_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["validation_batch_size"] = 0
    config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.train_siren_by_ffid",
        lambda *_args, **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match="training.validation_batch_size"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
            progress_reporter=lambda _message: None,
        )


def test_full_ffid_cli_keeps_json_clean_while_reporting_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["batch_mode"] == "full_ffid_epoch"
    assert "full_ffid_epoch 1/2 start" in captured.err


def test_full_ffid_cli_prints_the_human_training_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Batch mode: full_ffid_epoch" in captured.out
    assert "Best validation global S/N:" in captured.out
    assert "Epochs completed: 2" in captured.out
    assert "Optimizer steps: 6" in captured.out
    assert f"Checkpoint: {output / 'artifacts' / 'best.pt'}" in captured.out


def test_full_ffid_cli_serializes_perfect_validation_as_explicit_infinity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.evaluate_model_global_snr_by_ffid",
        lambda *_args, **_kwargs: float("inf"),
    )

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    summary = json.loads(captured.out)
    assert summary["best_validation_global_snr_db"] == "inf"
    assert all(record["validation_global_snr_db"] == "inf" for record in summary["history"])
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == summary
    assert load_siren_checkpoint(output / "artifacts" / "best.pt").validation_global_snr_db == (
        float("inf")
    )
