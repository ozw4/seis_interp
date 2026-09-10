from __future__ import annotations

import json
from pathlib import Path

import yaml

from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_supervised_source
from seis_interp.data.file_checksums import file_sha256
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.training.ccnet5d_checkpoints import (
    load_ccnet5d_checkpoint,
    save_ccnet5d_checkpoint,
)
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))
FIT = {
    "time": [0, 4],
    "source_line": [0, 1],
    "shot_in_line": [0, 1],
    "relative_receiver_x": [3, 5],
    "relative_receiver_y": [32, 36],
}
SELECTION = {**FIT, "shot_in_line": [1, 2]}


def _normalization_checkpoint(suite: Path, path: Path) -> Path:
    source = load_c3_benchmark_supervised_source(
        suite,
        fit_region=FIT,
        selection_region=SELECTION,
        dimensions=DIMENSIONS,
    )
    model = CCNet5D(
        hidden_channels=2,
        intermediate_channels=2,
        kernel_size=3,
        output_activation="linear",
    )
    save_ccnet5d_checkpoint(
        path,
        model_config=model.constructor_config(),
        state_dict=model.state_dict(),
        amplitude_rms=source.amplitude_rms,
        training_provenance={
            "training_run": {"git_commit": "a" * 40, "git_worktree_dirty": False},
            "source_inputs_lock": source.inputs_lock,
            "patch_plan_sha256": "b" * 64,
            "patches_random_seed": 19,
            "training_random_seed": 19,
        },
        checkpoint_role="final",
        epoch=8,
        global_step=2048,
        selection_metrics={"relative_squared_error": 1.0},
    )
    return path


def test_full_training_dataset_pipeline_records_fixed_scale_plan_and_budget(tmp_path: Path) -> None:
    interim = make_benchmark_interim(tmp_path / "source")
    suite = tmp_path / "suite"
    prepare_c3_benchmark_artifacts(
        interim,
        suite,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    checkpoint = _normalization_checkpoint(suite, tmp_path / "study033_final.pt")
    config = {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3"},
        "model": {
            "name": "ccnet5d",
            "hidden_channels": 2,
            "intermediate_channels": 2,
            "kernel_size": 3,
            "output_activation": "linear",
        },
        "supervision": {
            "partition": "train",
            "sampling_domain": "qc_training_dataset",
            "amplitude_normalization": "fixed_checkpoint_fit_region_global_rms",
            "normalization_checkpoint_sha256": file_sha256(checkpoint),
            "normalization_checkpoint_global_step": 2048,
            "selection_region": SELECTION,
        },
        "patches": {
            "shape": [2, 1, 1, 2, 2],
            "fit_count": 8,
            "selection_count": 2,
            "missing_fraction": 0.5,
            "mask_kind": "random_trace",
            "random_seed": 19,
        },
        "training": {
            "optimizer": "adam",
            "loss": "mse_complete_patch",
            "random_seed": 19,
            "batch_size": 2,
            "max_epochs": 8,
            "learning_rate": 0.001,
            "decay_after_epochs": 6,
            "decay_factor": 0.1,
            "validate_every_steps": 32,
            "report_every_steps": 32,
            "device": "cpu",
            "cudnn_benchmark": False,
        },
        "selection": {
            "metric": "missing_global_snr_db",
            "domain": "held_out_train_partition_patch_instances",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    metrics = train_ccnet5d_run(
        config_path=config_path,
        interim_dir=interim,
        processed_dir=suite / "partition",
        suite_dir=suite,
        normalization_checkpoint_path=checkpoint,
        output_dir=output,
        dimensions=DIMENSIONS,
    )

    assert metrics["steps_completed"] == 32
    diagnostics = json.loads((output / "artifacts/patch_plan_diagnostics.json").read_text())
    assert diagnostics["fit_patch_count"] == diagnostics["unique_start_count"] == 8
    assert diagnostics["normalization_source"]["sha256"] == file_sha256(checkpoint)
    loaded = load_ccnet5d_checkpoint(output / "artifacts/final.pt")
    assert loaded.amplitude_rms == load_ccnet5d_checkpoint(checkpoint).amplitude_rms
    assert loaded.global_step == 32
    assert (
        loaded.training_provenance["source_inputs_lock"]["training_dataset"][
            "authorized_trace_count"
        ]
        == 1632
    )
    assert len(json.loads((output / "artifacts/patch_plan.json").read_text())["fit"]) == 8


def test_declared_candidate_budget_is_2048_updates() -> None:
    fit_count, batch_size, epochs = 512, 2, 8
    assert epochs * (fit_count // batch_size) == 2048
