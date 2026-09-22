import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.pipelines.search_forge_conditions import (
    prepare_forge_conditions,
    run_forge_condition,
)
from seis_interp.processing.forge_condition_inputs import condition_model_inputs, downsample_traces
from seis_interp.processing.forge_mvp_contract import make_geometry, validate_mapping
from seis_interp.training.forge_condition_prediction import predict_regsi_condition
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize("source_decimation", [1, 2])
def test_preparation_creates_requested_sampling_rates_without_changing_source(
    tmp_path, monkeypatch, source_decimation
):
    _, contract, _, configs, mapping, stations, cells, grids, values = tiny_forge_study(tmp_path)
    rows = validate_mapping(mapping, stations, cells, grids, contract)
    if source_decimation > 1:
        contract["time_sampling"] = {
            "native_sample_count": contract["time_sample_count"],
            "native_sample_interval_us": contract["sample_interval_us"],
            "decimation_factor": source_decimation,
        }
        contract["time_sample_count"] = len(range(0, values.shape[1], source_decimation))
        contract["sample_interval_us"] *= source_decimation
    preparation = tmp_path / "preparation"
    for directory in ("input", "masks", "features", "configs"):
        (preparation / directory).mkdir(parents=True)
    rows.to_parquet(preparation / "input/m1_trace_manifest.parquet", index=False)
    mask = rows.copy()
    mask["hash_rank"] = np.arange(1, len(mask) + 1)
    mask.to_parquet(preparation / "masks/m1_random80_mvp_v1.parquet", index=False)
    make_geometry(rows, grids, "real").to_parquet(
        preparation / "features/real_geometry.parquet", index=False
    )
    write_json(preparation / "input/input_contract.json", contract)
    write_json(
        preparation / "input/raw_input_hashes.json",
        {
            "dataset_root": "raw",
            "hashes": {p.name: sha256_file(p) for p in (tmp_path / "raw").glob("*.sgy")},
        },
    )
    (preparation / "configs/regsi_real.yaml").write_text(yaml.safe_dump(configs["regsi_real"]))
    source_hashes = {
        str(p.relative_to(preparation)): sha256_file(p)
        for p in preparation.rglob("*")
        if p.is_file()
    }
    write_json(preparation / "preparation_manifest.json", {"artifacts": source_hashes})
    script = Path(__file__).resolve().parents[2] / "scripts/forge_regsi_conditions.py"
    (tmp_path / "scripts/forge_regsi_conditions.py").write_bytes(script.read_bytes())
    monkeypatch.setattr(
        "seis_interp.pipelines.search_forge_conditions.execution_identity",
        lambda repo: {"git_commit": "test"},
    )
    cache = prepare_forge_conditions(tmp_path, preparation, factors=(1, 2))
    record = json.loads((cache / "manifest.json").read_text())
    assert record["status"] == "prepared"
    assert record["downsampling"]["factors"] == [1, 2]
    assert record["original_sample_interval_us"] == 1000
    assert record["original_time_sample_count"] == values.shape[1]
    np.testing.assert_array_equal(np.load(cache / "waveforms_q1.npy"), values[:35])
    np.testing.assert_array_equal(
        np.load(cache / "waveforms_q2.npy"), downsample_traces(values[:35], 2)
    )
    assert not (cache / "waveforms_q8.npy").exists()
    verify_hashes(cache, record["artifacts"])
    verify_hashes(preparation, source_hashes)


@pytest.mark.parametrize("normalization", ["trace_oracle", "observed_global"])
@pytest.mark.parametrize("factor", [2, 8])
def test_condition_run_keeps_probe_hidden_and_recomputes_saved_scores(
    tmp_path, monkeypatch, normalization, factor
):
    _, contract, _, configs, mapping, stations, cells, grids, values = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, contract)
    geometry = make_geometry(manifest, grids, "real")
    cache = tmp_path / "cache"
    cache.mkdir()
    geometry.to_parquet(cache / "geometry.parquet", index=False)
    mask = manifest.copy()
    mask["hash_rank"] = np.arange(1, len(mask) + 1)
    mask["split"] = "test"
    mask.to_parquet(cache / "base_mask.parquet", index=False)
    np.save(cache / "probe_cells.npy", np.array([0, 1]))
    np.save(cache / f"waveforms_q{factor}.npy", values[:35])
    (cache / "regsi_real.yaml").write_text(yaml.safe_dump(configs["regsi_real"]))
    hashes = {p.name: sha256_file(p) for p in cache.iterdir()}
    write_json(
        cache / "manifest.json",
        {
            "status": "prepared",
            "artifacts": hashes,
            "spatial_shape": [2, 3, 2, 3],
            "original_sample_interval_us": 1000,
        },
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_regsi_conditions.py"
    (tmp_path / "scripts/forge_regsi_conditions.py").write_bytes(script.read_bytes())
    monkeypatch.setattr(
        "seis_interp.pipelines.search_forge_conditions.execution_identity",
        lambda repo: {"git_commit": "test"},
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = run_forge_condition(
            tmp_path,
            cache,
            steps=2,
            batch_size=2,
            device="cpu",
            normalization=normalization,
            factor=factor,
        )
    finally:
        torch.set_num_threads(previous)
    record = json.loads((run / "manifest.json").read_text())
    assert record["status"] == "completed"
    assert record["factor"] == factor
    assert record["oracle_test_rms"] == (normalization == "trace_oracle")
    assert record["evaluated_trace_count"] == 2
    resolved = yaml.safe_load((run / "config.yaml").read_text())
    assert resolved["amplitude_scaling"] == normalization
    assert resolved["mask_hash"] == sha256_file(run / "mask.parquet")
    assert resolved["checkpoint_rule"] == "fixed_final_raw_and_ema"
    verify_hashes(run, record["artifacts"])
    saved_mask = pd.read_parquet(run / "mask.parquet").set_index("cell_id")
    index = pd.read_parquet(run / "evaluation_index.parquet")
    assert saved_mask.loc[index.cell_id, "split"].eq("test").all()
    target = np.load(run / "target.npy").astype(float)
    np.testing.assert_array_equal(target, values[:2])
    inputs, scales = condition_model_inputs(
        geometry,
        saved_mask.reset_index(),
        values[:35],
        (2, 3, 2, 3),
        np.arange(values.shape[1]) * factor * 0.001,
        normalization,
        {},
    )
    torch.set_num_threads(1)
    try:
        frozen = predict_regsi_condition(
            inputs,
            resolved,
            index.cell_id.to_numpy(),
            torch.load(run / "state_ema.pt", map_location="cpu", weights_only=True),
        ).prediction
        with pytest.raises(ValueError, match="held out"):
            predict_regsi_condition(inputs, resolved, inputs.observed_cells[:1], {})
    finally:
        torch.set_num_threads(previous)
    if normalization == "trace_oracle":
        frozen = frozen * scales[:2, None]
    np.testing.assert_allclose(frozen, np.load(run / "prediction_ema.npy"), atol=1e-6)
    for variant in ("raw", "ema"):
        pred = np.load(run / f"prediction_{variant}.npy").astype(float)
        error = np.mean(np.sum((pred - target) ** 2, axis=1) / np.sum(target**2, axis=1))
        metrics = json.loads((run / f"metrics_{variant}.json").read_text())
        assert metrics["clean_target"]["masked_trace_relative_mse"] == pytest.approx(
            error, rel=1e-10
        )
