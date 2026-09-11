from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_interp.data.c3_benchmark_suite import c3_suite_case, suite_path
from seis_interp.data.c3_volume_run_inputs import load_c3_volume_run_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.nersi_run_audit import audit_c3_nersi_run
from seis_interp.pipelines import interpolate_nersi as nersi_pipeline
from seis_interp.pipelines.interpolate_nersi import interpolate_nersi_run
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 8), (1, 2), (8, 2, 2, 2, 8))


@pytest.fixture(autouse=True)
def _use_tiny_synthetic_input_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nersi_pipeline,
        "load_c3_random80_poc_inputs",
        load_c3_volume_run_inputs,
    )


def test_audit_rescores_and_restores_a_complete_nersi_run(tmp_path: Path) -> None:
    interim = make_benchmark_interim(tmp_path / "source")
    suite = tmp_path / "suite"
    benchmark_config = synthetic_benchmark_config()
    benchmark_config["c3_benchmark"]["shape"] = list(DIMENSIONS.shape)
    benchmark_config["c3_benchmark"]["training"]["time_samples"] = [0, 8]
    benchmark_config["benchmark_volume"]["selection"]["time"] = [0, 8]
    manifest = prepare_c3_benchmark_artifacts(
        interim,
        suite,
        config=benchmark_config,
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    entry = c3_suite_case(manifest, "validation_random_trace")
    volume_dir = suite_path(suite, entry["volume_dir"])
    volume = json.loads((volume_dir / "volume.json").read_text(encoding="utf-8"))
    config = {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3"},
        "interpolation_mask": {
            key: entry[key] for key in ("partition", "kind", "missing_fraction")
        },
        "benchmark_case": {"id": entry["case_id"]},
        "benchmark_volume": {"id": volume["volume_id"], "selection": volume["selection"]},
        "model": {
            "name": "nersi",
            "coordinate_order": ["source_line", "shot_in_line", "relative_receiver_x"],
            "fourier_components": 2,
            "frequency_schedule": "exponential",
            "frequency_base": 1.25,
            "encoder_width": 8,
            "latent_channels": 2,
            "decoder_channels": [2, 2, 1],
            "upsample_scales": [2, 2, 2],
            "kernel_size": 1,
            "activation": "gelu",
            "output_activation": "linear",
        },
        "training": {
            "random_seed": 314,
            "optimizer": "adam",
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": 1.0e-3,
            "profiles_per_step": 1,
            "max_steps": 2,
            "report_interval": 1,
            "device": "cpu",
        },
        "prediction": {"batch_size": 5},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }
    config_path = tmp_path / "nersi.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run = tmp_path / "run"
    interpolate_nersi_run(
        config_path=config_path,
        interim_dir=suite_path(suite, manifest["interim"]),
        processed_dir=suite_path(suite, manifest["processed"]),
        mask_dir=suite_path(suite, entry["mask_dir"]),
        case_dir=suite_path(suite, entry["case_dir"]),
        volume_dir=volume_dir,
        output_dir=run,
    )

    report = audit_c3_nersi_run(
        suite_dir=suite,
        case_id=entry["case_id"],
        run_dir=run,
        output_dir=tmp_path / "audit",
        expected_suite_sha256=file_sha256(suite / "benchmark_suite.json"),
        restore_device="cpu",
        dimensions=DIMENSIONS,
    )

    assert report["status"] == "success"
    assert all(report["checks"].values())
    assert report["checkpoint_restoration"]["status"] == "passed"
    assert report["checkpoint_restoration"]["max_abs_error"] == 0.0
    assert report["independent_saved_output_metrics"]["observed_max_abs_error"] == 0.0
    prediction = np.load(run / "artifacts/prediction.npy", allow_pickle=False)
    assert report["independent_saved_output_metrics"]["evaluation_target"]["sample_count"] < (
        prediction.size
    )
    assert json.loads((tmp_path / "audit/result.json").read_text()) == report
