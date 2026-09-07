from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from seis_interp.models.ccnet5d import CCNet5D, ccnet5d_method_variant
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training.ccnet5d_checkpoints import (
    BEST_SELECTION_CHECKPOINT_ROLE,
    FINAL_CHECKPOINT_ROLE,
    load_ccnet5d_checkpoint,
    save_ccnet5d_checkpoint,
)


def _model(output_activation: str = "linear") -> CCNet5D:
    torch.manual_seed(7)
    return CCNet5D(
        hidden_channels=2,
        intermediate_channels=3,
        kernel_size=1,
        output_activation=output_activation,
    )


def _provenance() -> dict[str, object]:
    fit_selection = {
        "time": [0, 2],
        "source_line": [0, 2],
        "shot_in_line": [0, 2],
        "relative_receiver_x": [0, 2],
        "relative_receiver_y": [0, 2],
    }
    held_out_selection = {**fit_selection, "shot_in_line": [2, 4]}
    return {
        "training_run": {"git_commit": "f" * 40, "git_worktree_dirty": False},
        "source_inputs_lock": {
            "dataset_id": "synthetic",
            "partition": "train",
            "partition_random_seed": 42,
            "interim": {"amplitudes.npy": {"sha256": "a" * 64}},
            "processed": {"trace_splits.parquet": {"sha256": "b" * 64}},
            "canonical_policy": "keep_lowest_array_row",
            "array_rows_hash_rule": "sha256_shape_le_int64_then_c_order_le_int64",
            "regions": {
                "fit": {
                    "selection": fit_selection,
                    "shape": [2, 2, 2, 2, 2],
                    "array_rows_sha256": "c" * 64,
                },
                "selection": {
                    "selection": held_out_selection,
                    "shape": [2, 2, 2, 2, 2],
                    "array_rows_sha256": "d" * 64,
                },
            },
        },
        "patch_plan_sha256": "e" * 64,
        "patches_random_seed": 13,
        "training_random_seed": 17,
    }


def _selection_metrics() -> dict[str, object]:
    return {
        "patch_count": 2,
        "missing_sample_count": 16,
        "reference_energy": 8.0,
        "error_energy": 2.0,
        "relative_squared_error": 0.25,
        "snr_db": 6.020599913279624,
        "snr_status": "finite",
        "metric_scope": "selection_patch_instances_missing_only",
    }


def _save(path: Path, model: CCNet5D, **overrides: object) -> None:
    arguments = {
        "model_config": model.constructor_config(),
        "state_dict": model.state_dict(),
        "amplitude_rms": 2.5,
        "training_provenance": _provenance(),
        "checkpoint_role": BEST_SELECTION_CHECKPOINT_ROLE,
        "epoch": 2,
        "global_step": 7,
        "selection_metrics": _selection_metrics(),
    }
    arguments.update(overrides)
    save_ccnet5d_checkpoint(path, **arguments)


@pytest.mark.parametrize(
    ("activation", "role"),
    [("linear", BEST_SELECTION_CHECKPOINT_ROLE), ("relu", FINAL_CHECKPOINT_ROLE)],
)
def test_round_trip_restores_function_constructor_rms_and_provenance(
    tmp_path: Path, activation: str, role: str
) -> None:
    model = _model(activation)
    values = torch.randn(1, 1, 2, 2, 2, 2, 2)
    expected_normalized = model(values).detach()
    path = tmp_path / f"{role}.pt"

    _save(path, model, checkpoint_role=role)
    loaded = load_ccnet5d_checkpoint(path, device="cpu")

    torch.testing.assert_close(loaded.model(values), expected_normalized, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.model(values) * loaded.amplitude_rms,
        expected_normalized * 2.5,
        rtol=0,
        atol=0,
    )
    assert loaded.model.constructor_config() == {
        "hidden_channels": 2,
        "intermediate_channels": 3,
        "kernel_size": 1,
        "output_activation": activation,
    }
    assert loaded.amplitude_rms == 2.5
    assert loaded.training_provenance == _provenance()
    assert loaded.checkpoint_role == role
    assert (loaded.epoch, loaded.global_step) == (2, 7)
    assert loaded.selection_metrics == _selection_metrics()


def test_payload_is_a_cpu_snapshot_without_optimizer_state(tmp_path: Path) -> None:
    model = _model()
    saved_state = copy.deepcopy(model.state_dict())
    path = tmp_path / "best.pt"

    _save(path, model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {
        "model_type",
        "method_variant",
        "model_config",
        "state_dict",
        "axis_order",
        "normalization",
        "training_provenance",
        "checkpoint_role",
        "epoch",
        "global_step",
        "selection_metrics",
    }
    assert payload["model_type"] == "ccnet5d"
    assert payload["method_variant"] == ccnet5d_method_variant("linear")
    assert payload["axis_order"] == list(VOLUME_AXIS_ORDER)
    assert payload["normalization"] == {
        "kind": "fit_region_global_rms",
        "amplitude_rms": 2.5,
    }
    assert "optimizer" not in payload
    for name, tensor in payload["state_dict"].items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, saved_state[name], rtol=0, atol=0)

    loaded = load_ccnet5d_checkpoint(path)
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, saved_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"), True])
def test_save_rejects_nonpositive_or_nonfinite_rms(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(ValueError, match="amplitude_rms"):
        _save(tmp_path / "bad.pt", _model(), amplitude_rms=invalid)


@pytest.mark.parametrize("role", ["best", "unknown", "", None])
def test_save_rejects_unknown_checkpoint_role(tmp_path: Path, role: object) -> None:
    with pytest.raises(ValueError, match="checkpoint_role"):
        _save(tmp_path / "bad.pt", _model(), checkpoint_role=role)


def test_load_rejects_variant_constructor_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    payload["method_variant"] = ccnet5d_method_variant("relu")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="method_variant"):
        load_ccnet5d_checkpoint(path)


def test_load_requires_every_constructor_field_and_strict_weights(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["output_activation"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="constructor fields"):
        load_ccnet5d_checkpoint(path)

    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    del payload["state_dict"][next(iter(payload["state_dict"]))]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_ccnet5d_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("model_type", "other", "model_type"),
        ("axis_order", ["wrong"], "axis_order"),
        ("checkpoint_role", "best", "checkpoint_role"),
        ("normalization", {"kind": "fit_region_global_rms", "amplitude_rms": 0}, "amplitude_rms"),
        ("selection_metrics", {"snr_db": float("nan")}, "strict JSON"),
    ],
)
def test_load_rejects_corrupt_metadata(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    path = tmp_path / "best.pt"
    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    payload[field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_ccnet5d_checkpoint(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("training_run",), "required fields"),
        (("training_random_seed",), "required fields"),
        (("source_inputs_lock", "partition"), "partition"),
        (("source_inputs_lock", "regions", "selection"), "fit and selection"),
        (
            (
                "source_inputs_lock",
                "regions",
                "fit",
                "selection",
                "relative_receiver_y",
            ),
            "must contain exactly",
        ),
    ],
)
def test_load_rejects_incomplete_training_provenance(
    tmp_path: Path, mutation: tuple[str, ...], message: str
) -> None:
    path = tmp_path / "best.pt"
    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    target = payload["training_provenance"]
    for key in mutation[:-1]:
        target = target[key]
    del target[mutation[-1]]
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_ccnet5d_checkpoint(path)


@pytest.mark.parametrize(
    "training_run",
    [
        None,
        {"git_commit": "a" * 40},
        {"git_worktree_dirty": False},
        {"git_commit": "a" * 40, "git_worktree_dirty": False, "extra": True},
        {"git_commit": "", "git_worktree_dirty": False},
        {"git_commit": 42, "git_worktree_dirty": False},
        {"git_commit": "a" * 40, "git_worktree_dirty": "false"},
        {"git_commit": "a" * 40, "git_worktree_dirty": 0},
    ],
)
def test_save_and_load_require_training_git_metadata(tmp_path, training_run):
    path = tmp_path / "best.pt"
    provenance = _provenance()
    provenance["training_run"] = training_run
    with pytest.raises(ValueError, match="training_provenance.training_run"):
        _save(path, _model(), training_provenance=provenance)
    assert not path.exists()

    _save(path, _model())
    payload = torch.load(path, weights_only=True)
    payload["training_provenance"]["training_run"] = training_run
    torch.save(payload, path)
    with pytest.raises(ValueError, match="training_provenance.training_run"):
        load_ccnet5d_checkpoint(path)
