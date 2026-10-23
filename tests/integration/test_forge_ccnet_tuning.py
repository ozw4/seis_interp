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
from seis_interp.pipelines.tune_forge_ccnet import _candidate_config, tune_forge_ccnet
from seis_interp.training.forge_mvp_methods import run_method
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize("record_kernels", [False, True])
def test_ccnet_tuning_preserves_inputs_and_records_reproducible_checkpoint(
    tmp_path, monkeypatch, record_kernels
):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_ccnet.execution_identity",
        lambda _: {"git_commit": "testcommit", "dirty_worktree": True},
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_ccnet_tuning.py"
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
    write_json(source / "runs/ccnet5d_grid/metrics.json", {"original": True})
    hashes = {str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    candidate = dict(
        name="candidate",
        model=dict(hidden_channels=2, intermediate_channels=2, time_kernel_size=5, time_dilation=2),
        training=dict(
            learning_rate=0.0001, max_steps=3, ema_decay=0.9, supervised_traces_per_update=2
        ),
        patches=dict(inner_mask_fraction=0.5),
        prediction=dict(core_shape=[4, 2, 2, 2, 2]),
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
    kernels = ["sm80_xmma_fprop_tf32", "sm80_xmma_dgrad_tf32", "sm80_xmma_wgrad_tf32"]
    if record_kernels:

        def recorded(inputs, config, reporter):
            result = run_method(inputs, config, reporter)
            result.diagnostics.update(
                conv_kernels_first_update=kernels, time_conv_tensor_cores=True
            )
            return result

        monkeypatch.setattr("seis_interp.pipelines.tune_forge_ccnet.run_method", recorded)
    output = tune_forge_ccnet(tmp_path, source, search)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(source, hashes)
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["target_met"] is True
    assert record["independent_test_estimate"] is False
    assert not (output / "must_not_run").exists()
    resolved = yaml.safe_load((output / "candidate/resolved_config.yaml").read_text())
    assert resolved["checkpoint_rule"] == "fixed_final_ema"
    inputs, config = load_model_inputs(preparation, "ccnet5d_grid")
    assert all(config[key] == value for key, value in configs["ccnet5d_grid"].items())
    for section in ("model", "training", "patches", "prediction"):
        config[section].update(candidate[section])
    expected = run_method(inputs, config)
    np.testing.assert_array_equal(np.load(output / "candidate/prediction.npy"), expected.prediction)
    checkpoint = torch.load(output / "candidate/final.pt", weights_only=False)
    assert checkpoint["steps_completed"] == 3
    assert checkpoint["input_binding"] == inputs.binding
    assert checkpoint["model_config"]["time_kernel_size"] == 5
    for name, tensor in expected.checkpoint["model_state_dict"].items():
        torch.testing.assert_close(checkpoint["model_state_dict"][name], tensor, rtol=0, atol=0)
    history = pd.read_parquet(output / "candidate/training_log.parquet")
    assert history.step.iloc[-1] == 3
    assert history.supervised_trace_presentations.iloc[-1] == 6
    summary = json.loads((output / "candidate/run_manifest.json").read_text())
    csv = pd.read_csv(output / "search_results.csv")
    assert set(csv.columns) == set(summary) - {"conv_kernels_first_update"}
    if record_kernels:
        assert summary["conv_kernels_first_update"] == kernels
        assert summary["time_conv_tensor_cores"] is True
        assert bool(csv.loc[0, "time_conv_tensor_cores"]) is True


@pytest.mark.parametrize(
    "override",
    [
        {"algorithm": {}},
        {"model": {"output_activation": "relu"}},
        {"model": {"unsupported": 4}},
        {"training": {"learning_rate_schedule": "cosine"}},
        {"training": {"optimizer": "adamw"}},
        {"training": {"mixed_precision": "bf16"}},
        {"training": {"batch_size": 4}},
        {"patches": {"shape": [4, 2, 2, 2, 2]}},
        {"prediction": {"query_batch_size": 4}},
    ],
)
def test_tuning_rejects_changes_outside_supported_ccnet_parameters(tmp_path, override):
    _, _, _, configs, *_ = tiny_forge_study(tmp_path)
    original = deepcopy(configs["ccnet5d_grid"])
    with pytest.raises(ValueError):
        _candidate_config(configs["ccnet5d_grid"], dict(name="candidate", **override), "cpu")
    assert configs["ccnet5d_grid"] == original


def test_alignment_candidate_uses_geometry_only_delays_and_covers_every_sample(
    tmp_path, monkeypatch
):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    preparation = prepare_forge_mvp(tmp_path, study)
    inputs, base = load_model_inputs(preparation, "ccnet5d_grid")
    config = _candidate_config(
        base,
        dict(
            name="aligned",
            model=dict(hidden_channels=2, intermediate_channels=2),
            training=dict(max_steps=2),
            alignment=dict(velocity_m_per_s=50.0),
        ),
        "cpu",
    )
    assert config["alignment"] == {"velocity_m_per_s": 50.0}
    result = run_method(inputs, config)
    assert result.prediction.shape == (len(inputs.test_cells), len(inputs.time_s))
    assert np.isfinite(result.prediction).all()
    assert result.checkpoint["alignment"] == {"velocity_m_per_s": 50.0}
    delays = np.rint(inputs.geometry.offset_m.to_numpy() / 50.0 / 0.001)
    assert delays.max() > 0
    with pytest.raises(ValueError):
        _candidate_config(base, dict(name="bad", alignment=dict(velocity=50.0)), "cpu")
    with pytest.raises(ValueError):
        run_method(inputs, {**config, "alignment": {"velocity_m_per_s": 50.0, "extra": 1}})


def test_candidate_without_ema_keeps_fixed_final_checkpoint_rule(tmp_path):
    _, _, _, configs, *_ = tiny_forge_study(tmp_path)
    config = _candidate_config(
        configs["ccnet5d_grid"],
        dict(name="candidate", model=dict(kernel_size=5), training=dict(max_steps=2)),
        "cpu",
    )
    assert config["checkpoint_rule"] == "fixed_final"
    assert config["model"]["kernel_size"] == 5
    assert config["device"] == "cpu"
