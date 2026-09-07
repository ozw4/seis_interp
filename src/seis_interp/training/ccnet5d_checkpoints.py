"""Save and restore supervised CCNet-5D checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.ccnet5d import CCNet5D, ccnet5d_method_variant
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER, validated_index_range

CCNET5D_MODEL_TYPE = "ccnet5d"
FIT_REGION_GLOBAL_RMS = "fit_region_global_rms"
BEST_SELECTION_CHECKPOINT_ROLE = "best_selection"
FINAL_CHECKPOINT_ROLE = "final"
_CHECKPOINT_ROLES = {BEST_SELECTION_CHECKPOINT_ROLE, FINAL_CHECKPOINT_ROLE}
_MODEL_CONFIG_FIELDS = {
    "hidden_channels",
    "intermediate_channels",
    "kernel_size",
    "output_activation",
}
_TRAINING_PROVENANCE_FIELDS = {
    "source_inputs_lock",
    "patch_plan_sha256",
    "patches_random_seed",
    "training_random_seed",
}
_SOURCE_INPUT_LOCK_FIELDS = {
    "dataset_id",
    "partition",
    "partition_random_seed",
    "interim",
    "processed",
    "canonical_policy",
    "regions",
    "array_rows_hash_rule",
}


@dataclass(frozen=True)
class LoadedCCNet5DCheckpoint:
    """A restored network and the supervised-training metadata it needs."""

    model: CCNet5D
    amplitude_rms: float
    training_provenance: dict[str, object]
    checkpoint_role: str
    epoch: int
    global_step: int
    selection_metrics: dict[str, object]


def save_ccnet5d_checkpoint(
    path: Path,
    *,
    model_config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    amplitude_rms: float,
    training_provenance: Mapping[str, object],
    checkpoint_role: str,
    epoch: int,
    global_step: int,
    selection_metrics: Mapping[str, object],
) -> None:
    """Save a non-resumable CPU snapshot and its complete supervised provenance."""
    config, method_variant = _validated_model_config(model_config)
    rms = _positive_finite_float(amplitude_rms, "amplitude_rms")
    provenance = _validated_training_provenance(training_provenance)
    role = _validated_checkpoint_role(checkpoint_role)
    epoch_value = _positive_integer(epoch, "epoch")
    step_value = _positive_integer(global_step, "global_step")
    metrics = _strict_json_mapping(selection_metrics, "selection_metrics")
    snapshot = _cpu_state_dict(state_dict)
    payload = {
        "model_type": CCNET5D_MODEL_TYPE,
        "method_variant": method_variant,
        "model_config": config,
        "state_dict": snapshot,
        "axis_order": list(VOLUME_AXIS_ORDER),
        "normalization": {
            "kind": FIT_REGION_GLOBAL_RMS,
            "amplitude_rms": rms,
        },
        "training_provenance": provenance,
        "checkpoint_role": role,
        "epoch": epoch_value,
        "global_step": step_value,
        "selection_metrics": metrics,
    }
    torch.save(payload, Path(path))


def load_ccnet5d_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCCNet5DCheckpoint:
    """Rebuild a CCNet-5D from complete constructor metadata and strict weights."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
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
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("CCNet-5D checkpoint is missing required fields")
    if payload["model_type"] != CCNET5D_MODEL_TYPE:
        raise ValueError("checkpoint model_type must be 'ccnet5d'")

    config, expected_variant = _validated_model_config(payload["model_config"])
    if payload["method_variant"] != expected_variant:
        raise ValueError("checkpoint method_variant does not match model_config.output_activation")
    if payload["axis_order"] != list(VOLUME_AXIS_ORDER):
        raise ValueError("checkpoint axis_order must match the C3 volume axis order")

    normalization = payload["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError("checkpoint normalization must be a mapping")
    if normalization.get("kind") != FIT_REGION_GLOBAL_RMS:
        raise ValueError("checkpoint normalization.kind must be 'fit_region_global_rms'")
    try:
        amplitude_rms = _positive_finite_float(
            normalization["amplitude_rms"], "normalization.amplitude_rms"
        )
    except KeyError as error:
        raise ValueError("checkpoint normalization is missing amplitude_rms") from error

    provenance = _validated_training_provenance(payload["training_provenance"])
    role = _validated_checkpoint_role(payload["checkpoint_role"])
    epoch = _positive_integer(payload["epoch"], "epoch")
    global_step = _positive_integer(payload["global_step"], "global_step")
    metrics = _strict_json_mapping(payload["selection_metrics"], "selection_metrics")

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(tensor, torch.Tensor) for tensor in state_dict.values()
    ):
        raise ValueError("checkpoint state_dict must map names to tensors")
    try:
        model = CCNet5D(**config)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model_config contains invalid constructor fields") from error
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return LoadedCCNet5DCheckpoint(
        model=model,
        amplitude_rms=amplitude_rms,
        training_provenance=provenance,
        checkpoint_role=role,
        epoch=epoch,
        global_step=global_step,
        selection_metrics=metrics,
    )


def _validated_model_config(value: object) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_CONFIG_FIELDS:
        raise ValueError(
            "CCNet-5D model_config must contain exactly the required constructor fields"
        )
    for name in ("hidden_channels", "intermediate_channels", "kernel_size"):
        field = value[name]
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ValueError("CCNet-5D model_config contains invalid constructor fields")
    if value["kernel_size"] % 2 != 1:
        raise ValueError("CCNet-5D model_config contains invalid constructor fields")
    try:
        variant = ccnet5d_method_variant(value["output_activation"])
    except ValueError as error:
        raise ValueError("CCNet-5D model_config contains invalid constructor fields") from error
    return dict(value), variant


def _validated_training_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("training_provenance is missing required fields")
    missing_provenance = sorted(_TRAINING_PROVENANCE_FIELDS.difference(value))
    if missing_provenance:
        raise ValueError(f"training_provenance is missing required fields: {missing_provenance!r}")
    source_lock = value["source_inputs_lock"]
    if not isinstance(source_lock, Mapping):
        raise ValueError("training_provenance source_inputs_lock is missing required fields")
    missing_source_lock = sorted(_SOURCE_INPUT_LOCK_FIELDS.difference(source_lock))
    if missing_source_lock:
        raise ValueError(
            "training_provenance source_inputs_lock is missing required fields: "
            f"{missing_source_lock!r}"
        )
    if source_lock["partition"] != "train":
        raise ValueError("training_provenance source partition must be 'train'")
    dataset_id = source_lock["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("training_provenance source dataset_id must be a non-empty string")
    _nonnegative_integer(
        source_lock["partition_random_seed"],
        "source_inputs_lock.partition_random_seed",
    )
    for name in ("interim", "processed"):
        files = source_lock[name]
        if not isinstance(files, Mapping) or not files:
            raise ValueError(
                f"training_provenance source_inputs_lock.{name} must identify input files"
            )
    regions = source_lock["regions"]
    if not isinstance(regions, Mapping) or not {"fit", "selection"}.issubset(regions):
        raise ValueError("training_provenance must identify fit and selection regions")
    for name in ("fit", "selection"):
        region = regions[name]
        if not isinstance(region, Mapping) or not {
            "selection",
            "shape",
            "array_rows_sha256",
        }.issubset(region):
            raise ValueError(
                f"training_provenance source {name} region must identify its selection"
            )
        selection = region["selection"]
        if not isinstance(selection, Mapping) or set(selection) != set(VOLUME_AXIS_ORDER):
            raise ValueError(
                f"training_provenance source {name} region selection must contain "
                f"exactly {list(VOLUME_AXIS_ORDER)!r}"
            )
        for axis in VOLUME_AXIS_ORDER:
            validated_index_range(
                selection[axis],
                name=f"training_provenance.source_inputs_lock.regions.{name}.selection.{axis}",
            )
    patch_plan_sha256 = value["patch_plan_sha256"]
    if (
        not isinstance(patch_plan_sha256, str)
        or len(patch_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in patch_plan_sha256)
    ):
        raise ValueError("training_provenance patch_plan_sha256 must be lowercase SHA-256")
    _nonnegative_integer(value["patches_random_seed"], "patches_random_seed")
    _nonnegative_integer(value["training_random_seed"], "training_random_seed")
    return _strict_json_mapping(value, "training_provenance")


def _strict_json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    converted = dict(value)
    try:
        json.dumps(converted, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict JSON values") from error
    return converted


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict must be a mapping")
    snapshot: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map string names to tensors")
        snapshot[name] = tensor.detach().cpu().clone()
    return snapshot


def _validated_checkpoint_role(value: object) -> str:
    if not isinstance(value, str) or value not in _CHECKPOINT_ROLES:
        raise ValueError(f"checkpoint_role must be one of {sorted(_CHECKPOINT_ROLES)!r}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
