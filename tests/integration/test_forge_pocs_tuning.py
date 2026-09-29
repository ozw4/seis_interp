import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes, write_json
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_classical import tune_forge_classical
from seis_interp.pipelines.tune_forge_pocs import tune_forge_pocs
from seis_interp.training.forge_mvp_methods import run_method
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize("backend", ["numpy", "batched"])
@pytest.mark.parametrize("method_id", ["pocs_grid", "drr_grid"])
@pytest.mark.parametrize("stop_on_target", [False, True])
def test_tuning_preserves_mask_restoration_and_original_run(
    tmp_path, monkeypatch, backend, method_id, stop_on_target
):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    identity = {"git_commit": "testcommit", "dirty_worktree": True}
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_classical.execution_identity", lambda _: identity
    )
    script_name = f"forge_{method_id.removesuffix('_grid')}_tuning.py"
    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    (tmp_path / "scripts" / script_name).write_bytes(script.read_bytes())
    preparation = prepare_forge_mvp(tmp_path, study)
    source = tmp_path / "original_run"
    write_json(
        source / "matrix_manifest.json",
        {
            "preparation": str(preparation.relative_to(tmp_path)),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
        },
    )
    write_json(source / "runs" / method_id / "metrics.json", {"original": True})
    original_hashes = {
        str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()
    }
    algorithm = configs[method_id]["algorithm"]
    candidates = [dict(name="candidate", algorithm=algorithm)]
    if stop_on_target:
        candidates.append(dict(name="must_not_run", algorithm={}))
    threshold = 10.0 if stop_on_target else 0.8
    config_path = tmp_path / "search.yaml"
    config_path.write_text(
        yaml.safe_dump(
            dict(
                target_relative_mse=threshold,
                stop_when_target_met=stop_on_target,
                backend=backend,
                device="cpu",
                candidates=candidates,
            )
        )
    )
    if method_id == "pocs_grid":
        output = tune_forge_pocs(tmp_path, source, config_path)
    else:
        output = tune_forge_classical(tmp_path, source, config_path, method_id=method_id)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(source, original_hashes)
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["independent_test_estimate"] is False
    assert record["target_met"] == (record["best_candidate"]["primary_relative_mse"] <= threshold)
    assert not (output / "must_not_run").exists()
    inputs, config = load_model_inputs(preparation, method_id)
    expected = run_method(inputs, config)
    actual = np.load(output / "candidate/prediction.npy")
    np.testing.assert_allclose(actual, expected.prediction, rtol=1e-5, atol=1e-6)
    index = pd.read_parquet(output / "prediction_index.parquet")
    np.testing.assert_array_equal(index.cell_id, expected.cell_ids)
