import json
from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs
from seis_interp.forge_nersi_experiment_config import (
    NERSI_TRACE_DEFAULTS,
    load_experiment_config,
    resolve_run_config,
)
from seis_interp.training.forge_mvp_methods import model_parameter_count

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPOSITORY_ROOT / "studies/study_forge_nersi_temp"
# Pinned here, not read from the user-editable config.yaml and inputs.yaml.
DEFAULT_PREPARATION = (
    "runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011629803645Z_25c5193588b2"
)
DEFAULT_MANIFEST_SHA256 = "0f52058e10bcb340d4f166e0a856fa7bb4eada53ec99884b097bc3e456ede966"


def test_default_workspace_config_reproduces_the_sealed_nersi():
    preparation = REPOSITORY_ROOT / DEFAULT_PREPARATION
    if not preparation.is_dir():
        pytest.skip(f"default preparation is not present locally: {DEFAULT_PREPARATION}")
    manifest = preparation / "preparation_manifest.json"
    assert sha256_file(manifest) == DEFAULT_MANIFEST_SHA256
    inputs, sealed = load_model_inputs(preparation, "nersi_real")
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    requested = load_experiment_config(WORKSPACE / "config.default.yaml")
    resolved = resolve_run_config(
        requested, sealed, contract, inputs.geometry, inputs.spatial_shape
    )
    for key in ("model_seed", "loss", "checkpoint_rule"):
        assert resolved[key] == sealed[key], key
    for section in ("model", "training", "prediction"):
        assert set(resolved[section]) >= set(sealed[section]), section
        for key, value in resolved[section].items():
            expected = sealed[section].get(key, NERSI_TRACE_DEFAULTS.get(key))
            if key == "coordinate_transform":
                assert np.array_equal(np.asarray(value), np.asarray(expected))
            else:
                assert value == expected, f"{section}.{key}"
    seal = json.loads(manifest.read_text())
    canvas = seal.get("canvas_sample_count", seal["time_sample_count"])
    assert model_parameter_count(resolved, canvas) == seal["parameter_counts"]["nersi_real"]
