import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes, write_json
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_nersi import _candidate_config, tune_forge_nersi
from seis_interp.processing.nersi_trace_coordinates import midpoint_offset_transform
from seis_interp.training.forge_mvp_methods import run_method
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize("polar", [False, True])
def test_nersi_tuning_preserves_inputs_and_records_reproducible_checkpoint(
    tmp_path, monkeypatch, polar
):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_nersi.execution_identity",
        lambda _: {"git_commit": "testcommit", "dirty_worktree": True},
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_nersi_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    preparation = prepare_forge_mvp(tmp_path, study)
    source = tmp_path / "original_run"
    write_json(
        source / "matrix_manifest.json",
        {
            "preparation": str(preparation.relative_to(tmp_path)),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
        },
    )
    write_json(source / "runs/nersi_real/metrics.json", {"original": True})
    hashes = {str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    candidate = dict(
        name="candidate",
        model=dict(
            frequency_base=[1.05, 1.15, 1.05, 1.15],
            coordinate_transform=midpoint_offset_transform(
                pd.read_parquet(preparation / "features/real_geometry.parquet"),
                contract["spatial_shape"],
                polar=polar,
            ).tolist(),
            polar_offset=polar,
            moveout_samples_per_radius=2.5 if polar else 0.0,
        ),
        training=dict(batch_size=2, learning_rate=0.0001, max_steps=3, ema_decay=0.9),
        prediction=dict(batch_size=4),
    )
    search = tmp_path / "search.yaml"
    search.write_text(
        yaml.safe_dump(
            dict(
                target_relative_mse=10.0,
                stop_when_target_met=True,
                device="cpu",
                candidates=[candidate, dict(name="must_not_run", algorithm={})],
            )
        )
    )
    output = tune_forge_nersi(tmp_path, source, search)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(source, hashes)
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["target_met"] is True
    assert record["independent_test_estimate"] is False
    assert not (output / "must_not_run").exists()
    inputs, config = load_model_inputs(preparation, "nersi_real")
    assert all(config[key] == value for key, value in configs["nersi_real"].items())
    for section in ("model", "training", "prediction"):
        config[section].update(candidate[section])
    expected = run_method(inputs, config)
    np.testing.assert_array_equal(np.load(output / "candidate/prediction.npy"), expected.prediction)
    checkpoint = torch.load(output / "candidate/final.pt", weights_only=False)
    assert checkpoint["steps_completed"] == 3
    assert checkpoint["input_binding"] == inputs.binding
    for name, tensor in expected.checkpoint["model_state_dict"].items():
        torch.testing.assert_close(checkpoint["model_state_dict"][name], tensor, rtol=0, atol=0)
    history = pd.read_parquet(output / "candidate/training_log.parquet")
    assert history.step.iloc[-1] == 3


@pytest.mark.parametrize(
    "override",
    [
        {"preprocessing": {}},
        {"model": {"time_sample_count": 3}},
        {"model": {"input_features": 3}},
        {"model": {"unsupported": 4}},
        {"training": {"learning_rate_schedule": "cosine"}},
        {"training": {"optimizer": "adamw"}},
        {"training": {"mixed_precision": "bf16"}},
    ],
)
def test_tuning_rejects_changes_outside_supported_nersi_parameters(tmp_path, override):
    _, _, _, configs, *_ = tiny_forge_study(tmp_path)
    original = deepcopy(configs["nersi_real"])
    with pytest.raises(ValueError):
        _candidate_config(configs["nersi_real"], dict(name="candidate", **override), "cpu")
    assert configs["nersi_real"] == original
