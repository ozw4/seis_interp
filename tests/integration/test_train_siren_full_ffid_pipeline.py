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
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig
from seis_interp.processing.trace_splits import EXCLUDED_SPLIT
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
    cmp_x_m = array_indices.astype(np.float64)
    cmp_y_m = array_indices.astype(np.float64) * 2.0
    offset_m = 100.0 + array_indices.astype(np.float64)
    azimuth_deg = array_indices.astype(np.float64) * 7.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    half_offset_x_m = 0.5 * offset_m * np.sin(azimuth_rad)
    half_offset_y_m = 0.5 * offset_m * np.cos(azimuth_rad)
    trace_table = pd.DataFrame(
        {
            "source_file": np.repeat([source_a.name, source_b.name], [20, 10]),
            "trace_index": np.concatenate(
                [np.arange(20, dtype=np.int64), np.arange(10, dtype=np.int64)]
            ),
            "ffid": np.repeat([10, 20, 30], 10),
            "source_x_m": cmp_x_m + half_offset_x_m,
            "source_y_m": cmp_y_m + half_offset_y_m,
            "receiver_x_m": cmp_x_m - half_offset_x_m,
            "receiver_y_m": cmp_y_m - half_offset_y_m,
            "cmp_x_m": cmp_x_m,
            "cmp_y_m": cmp_y_m,
            "offset_m": offset_m,
            "azimuth_deg": azimuth_deg,
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


@pytest.mark.parametrize(
    "batch_mode",
    ["random_points", "full_ffid_epoch", "random_complete_traces"],
)
def test_train_siren_batch_modes_support_cartesian_half_offset_coordinates(
    tmp_path: Path,
    batch_mode: str,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["model"].update(
        {
            "coordinate_features": "cmp_cartesian_half_offset",
            "input_features": 5,
        }
    )
    config_payload["training"].update(
        {
            "batch_mode": batch_mode,
            "max_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    if batch_mode == "random_points":
        config_payload["training"].update({"batch_size": 8, "steps_per_epoch": 1})
    elif batch_mode == "random_complete_traces":
        config_payload["training"].update({"traces_per_update": 2, "steps_per_epoch": 1})
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    normalization = read_normalization_parameters(processed / "normalization.json")
    half_offset_scale_m = 0.5 * normalization.coordinate_max[3]
    expected_contract = {
        "coordinate_features": "cmp_cartesian_half_offset",
        "coordinate_order": [
            "time_s",
            "cmp_x_m",
            "cmp_y_m",
            "half_offset_x_m",
            "half_offset_y_m",
        ],
        "coordinate_scale_min": [
            *normalization.coordinate_min[:3],
            -half_offset_scale_m,
            -half_offset_scale_m,
        ],
        "coordinate_scale_max": [
            *normalization.coordinate_max[:3],
            half_offset_scale_m,
            half_offset_scale_m,
        ],
        "half_offset_scale_m": half_offset_scale_m,
    }
    checkpoint = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.model.input_features == 5
    assert checkpoint.model_coordinates is not None
    assert checkpoint.model_coordinates.to_dict() == expected_contract
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert inputs_lock["model_coordinates"] == expected_contract
    assert run_metadata["model_coordinates"] == expected_contract


def test_full_ffid_pipeline_supports_per_trace_rms_as_a_training_target_scaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["amplitude_scaling"] = "per_trace_rms"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"
    import seis_interp.pipelines.train_siren as pipeline

    received_scaling: dict[str, str] = {}
    actual_sampler = pipeline.FullFfidBatchSampler
    actual_evaluate = pipeline.evaluate_model_global_snr_by_ffid
    actual_train = pipeline.train_siren_by_ffid

    def recording_sampler(*args: Any, **kwargs: Any):
        received_scaling["sampler"] = kwargs["amplitude_scaling"]
        return actual_sampler(*args, **kwargs)

    def recording_evaluate(*args: Any, **kwargs: Any) -> float:
        received_scaling["validation"] = kwargs["amplitude_scaling"]
        return actual_evaluate(*args, **kwargs)

    def recording_train(*args: Any, **kwargs: Any):
        received_scaling["trainer"] = kwargs["amplitude_scaling"]
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(pipeline, "FullFfidBatchSampler", recording_sampler)
    monkeypatch.setattr(pipeline, "evaluate_model_global_snr_by_ffid", recording_evaluate)
    monkeypatch.setattr(pipeline, "train_siren_by_ffid", recording_train)

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    assert metrics["amplitude_scaling"] == "per_trace_rms"
    assert received_scaling == {
        "sampler": "per_trace_rms",
        "validation": "per_trace_rms",
        "trainer": "per_trace_rms",
    }
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert metrics["validation_scale_source"] == "validation_trace_target_rms"
    checkpoint = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.amplitude_scaling == "per_trace_rms"
    assert checkpoint.validation_metric_domain == "oracle_per_trace_unit_rms"
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["normalization"]["amplitude"] == "train_global_rms"
    assert inputs_lock["training"]["amplitude_scaling"] == "per_trace_rms"
    assert inputs_lock["training"]["validation_metric_domain"] == ("oracle_per_trace_unit_rms")
    assert inputs_lock["training"]["validation_scale_source"] == "validation_trace_target_rms"
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["amplitude_scaling"] == "per_trace_rms"
    assert run_metadata["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert run_metadata["validation_scale_source"] == "validation_trace_target_rms"


@pytest.mark.parametrize("amplitude_scaling", ["train_global_rms", "per_trace_rms"])
def test_full_ffid_pipeline_forwards_and_records_active_trace_correlation_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amplitude_scaling: str,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "amplitude_scaling": amplitude_scaling,
            "correlation_weight": 0.1,
            "correlation_eps": 1.0e-4,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"
    import seis_interp.pipelines.train_siren as pipeline

    received: dict[str, float] = {}
    actual_train = pipeline.train_siren_by_ffid

    def recording_train(*args: Any, **kwargs: Any):
        received["correlation_weight"] = kwargs["correlation_weight"]
        received["correlation_eps"] = kwargs["correlation_eps"]
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(pipeline, "train_siren_by_ffid", recording_train)

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    expected_provenance = {
        "correlation_weight": 0.1,
        "correlation_eps": 1.0e-4,
        "loss_semantics": "mse_plus_trace_correlation",
    }
    assert received == {
        "correlation_weight": 0.1,
        "correlation_eps": 1.0e-4,
    }
    assert {key: metrics[key] for key in expected_provenance} == expected_provenance
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == metrics

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert {key: inputs_lock["training"][key] for key in expected_provenance} == expected_provenance
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert {key: run_metadata[key] for key in expected_provenance} == expected_provenance
    resolved_config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config["training"]["correlation_weight"] == 0.1
    assert resolved_config["training"]["correlation_eps"] == 1.0e-4
    assert inputs_lock["training"]["amplitude_scaling"] == amplitude_scaling
    if amplitude_scaling == "per_trace_rms":
        assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
        assert run_metadata["validation_metric_domain"] == "oracle_per_trace_unit_rms"


def test_full_ffid_pipeline_omits_trace_correlation_provenance_for_pure_mse(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output = tmp_path / "run"

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    resolved_config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    assert correlation_keys.isdisjoint(metrics)
    assert correlation_keys.isdisjoint(inputs_lock["training"])
    assert correlation_keys.isdisjoint(run_metadata)
    assert correlation_keys.isdisjoint(resolved_config["training"])


@pytest.mark.parametrize("amplitude_scaling", ["train_global_rms", "per_trace_rms"])
def test_full_ffid_pipeline_never_routes_amplitude_filtered_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amplitude_scaling: str,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    amplitudes_path = interim / "amplitudes.npy"
    amplitudes = np.load(amplitudes_path, allow_pickle=False)
    amplitudes[:5] = 0.0
    amplitudes[5:10] = 100.0
    np.save(amplitudes_path, amplitudes)
    trace_filter = TraceAmplitudeFilterConfig(
        exclude_all_zero=True,
        max_abs_amplitude=10.0,
    )
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.3,
        validation_fraction_of_holdout=0.5,
        random_seed=7,
        split_scope="per_ffid",
        trace_amplitude_filter=trace_filter,
        config_source="studies/synthetic/config.yaml",
        overwrite=True,
    )
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["sampling"]["trace_amplitude_filter"] = trace_filter.to_dict()
    config_payload["training"]["amplitude_scaling"] = amplitude_scaling
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

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

    output = tmp_path / "run"
    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    excluded_rows = set(split_table.loc[split_table["split"] == EXCLUDED_SPLIT, "array_row"])
    assert excluded_rows == set(range(10))
    assert excluded_rows.isdisjoint(received["train"] | received["validation"])
    assert metrics["training_ffid_count"] == metrics["validation_ffid_count"] == 2
    assert metrics["effective_steps_per_epoch"] == 2
    assert metrics["trace_quality"]["excluded_trace_count"] == 10
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["trace_amplitude_filter"] == trace_filter.to_dict()
    assert inputs_lock["preparation"]["trace_quality"]["fully_excluded_ffids"] == [10]


def test_full_ffid_pipeline_rejects_a_stale_pre_filter_preparation(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["sampling"]["trace_amplitude_filter"] = {
        "exclude_all_zero": True,
        "max_abs_amplitude": 10.0,
    }
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="trace_amplitude_filter"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
            progress_reporter=lambda _message: None,
        )


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


def test_full_ffid_pipeline_rejects_unknown_training_amplitude_scaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["amplitude_scaling"] = "global_rms"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.load_interim_trace_dataset",
        lambda *_args, **_kwargs: pytest.fail("data loading must not start"),
    )

    with pytest.raises(ValueError, match="training.amplitude_scaling must be one of"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )


def test_active_trace_correlation_loss_rejects_random_points_before_data_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": "random_points",
            "correlation_weight": 0.1,
            "correlation_eps": 1.0e-4,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.load_interim_trace_dataset",
        lambda *_args, **_kwargs: pytest.fail("data loading must not start"),
    )

    with pytest.raises(ValueError, match="correlation.*full_ffid_epoch"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
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


def test_full_ffid_cli_labels_per_trace_validation_as_oracle_normalized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["amplitude_scaling"] = "per_trace_rms"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

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
            str(tmp_path / "run"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Amplitude scaling: per_trace_rms" in captured.out
    assert "Validation metric domain: oracle per-trace unit RMS" in captured.out
    assert "Best oracle-normalized validation global S/N:" in captured.out


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
