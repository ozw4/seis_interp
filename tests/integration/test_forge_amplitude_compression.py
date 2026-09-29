import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.pipelines.run_forge_amplitude_compression import run_forge_amplitude_compression
from seis_interp.processing.amplitude_compression import transform_amplitudes
from seis_interp.processing.forge_mvp_contract import make_geometry, validate_mapping
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize("variant", ["linear", "asinh"])
@pytest.mark.parametrize("scale_multiplier", [1.0, 0.1, 10.0])
def test_compression_experiment_never_uses_test_amplitudes_for_training_or_restoration(
    tmp_path, monkeypatch, variant, scale_multiplier
):
    repo = Path(__file__).resolve().parents[2]
    _, contract, _, configs, mapping, stations, cells, grids, values = tiny_forge_study(tmp_path)
    rows = validate_mapping(mapping, stations, cells, grids, contract)
    geometry = make_geometry(rows, grids, "real")
    cache = tmp_path / "cache"
    cache.mkdir()
    mask = rows.copy()
    mask["hash_rank"] = np.arange(1, len(mask) + 1)
    mask.to_parquet(cache / "base_mask.parquet", index=False)
    geometry.to_parquet(cache / "geometry.parquet", index=False)
    np.save(cache / "probe_cells.npy", np.array([0, 1]))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "config.yaml").write_text(yaml.safe_dump(configs["regsi_real"]))
    write_json(
        reference / "manifest.json",
        {
            "status": "completed",
            "artifacts": {"config.yaml": sha256_file(reference / "config.yaml")},
        },
    )
    study = tmp_path / "studies/compression"
    study.mkdir(parents=True)
    config = yaml.safe_load(
        (repo / "studies/study_055_forge_amplitude_compression/config.yaml").read_text()
    )
    config["study_id"] = "compression"
    config["training"].update(max_steps=2, query_batch_size=2)
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    (study / "inputs.yaml").write_text(
        yaml.safe_dump({"cache": "cache", "reference_run": "reference"})
    )
    script = repo / "scripts/forge_amplitude_compression.py"
    (tmp_path / "scripts/forge_amplitude_compression.py").write_bytes(script.read_bytes())
    monkeypatch.setattr(
        "seis_interp.pipelines.run_forge_amplitude_compression.execution_identity",
        lambda repo: {"git_commit": "test"},
    )
    observed = mask.hash_rank.to_numpy() > round(len(mask) * 0.8)
    outputs = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for target_multiplier in (1, 100):
            data = values[:35].copy()
            data[~observed] *= target_multiplier
            np.save(cache / "waveforms_q2.npy", data)
            artifacts = {
                p.name: sha256_file(p) for p in cache.iterdir() if p.name != "manifest.json"
            }
            write_json(
                cache / "manifest.json",
                {
                    "status": "prepared",
                    "spatial_shape": [2, 3, 2, 3],
                    "original_sample_interval_us": 1000,
                    "artifacts": artifacts,
                },
            )
            output = run_forge_amplitude_compression(
                tmp_path, study, variant, device="cpu", scale_multiplier=scale_multiplier
            )
            outputs.append(output)
            record = json.loads((output / "manifest.json").read_text())
            assert record["status"] == "completed"
            assert record["test_waveforms_read_only_after_prediction"]
            assert not record["per_trace_rms"] and not record["test_rms"]
            assert record["loss"] == "masked_trace_mse"
            assert record["scale_multiplier"] == scale_multiplier
            assert record["scale"] == (
                np.percentile(np.abs(data[observed].astype(float)), 90) * scale_multiplier
            )
            resolved = yaml.safe_load((output / "config.resolved.yaml").read_text())
            assert resolved["global_rms"] == 1.0
            assert resolved["amplitude_transform"]["scale_multiplier"] == scale_multiplier
            assert resolved["amplitude_transform"]["fitted_scale"] == record["scale"]
            assert resolved["model"]["amplitude_mode"] == "train_global_rms"
            verify_hashes(output, record["artifacts"])
            target = np.load(output / "target.npy").astype(float)
            np.testing.assert_array_equal(target, data[:2])
            for name in ("raw", "ema"):
                prediction = np.load(output / f"prediction_{name}.npy")
                transformed = np.load(output / f"prediction_transformed_{name}.npy")
                np.testing.assert_array_equal(
                    prediction,
                    transform_amplitudes(transformed, record["scale"], variant, inverse=True),
                )
                score = json.loads((output / f"metrics_{name}.json").read_text())["clean_target"]
                error = np.mean(
                    np.sum((prediction - target) ** 2, axis=1) / np.sum(target**2, axis=1)
                )
                assert score["masked_trace_relative_mse"] == pytest.approx(error, rel=1e-10)
    finally:
        torch.set_num_threads(previous)
    for name in ("raw", "ema"):
        np.testing.assert_array_equal(
            np.load(outputs[0] / f"prediction_{name}.npy"),
            np.load(outputs[1] / f"prediction_{name}.npy"),
        )
    a = pd.read_parquet(outputs[0] / "training_history.parquet")
    b = pd.read_parquet(outputs[1] / "training_history.parquet")
    np.testing.assert_array_equal(a.loss, b.loss)
