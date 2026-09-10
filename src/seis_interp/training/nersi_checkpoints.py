"""Strict fixed-final checkpoint storage for profile-wise NeRSI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.nersi import Nersi
from seis_interp.training.c3_volume_nersi_data import (
    PROFILE_AXIS_ORDER,
    PROFILE_COORDINATE_ORDER,
    C3VolumeNersiData,
)

FIXED_STEP_FINAL_CHECKPOINT_ROLE = "fixed_step_final"
NERSI_METHOD_VARIANT = "profile_wise_per_volume_internal_learning_fixed_steps_reimplementation"
COORDINATE_NORMALIZATION = "local_regular_grid_index_minmax_0_1"
AMPLITUDE_SCALING = "observed_volume_global_rms"
TRAINING_DOMAIN = "benchmark_observed_samples"

_MODEL_CONFIG_FIELDS = {
    "input_features",
    "fourier_components",
    "frequency_base",
    "encoder_width",
    "latent_channels",
    "decoder_channels",
    "profile_shape",
    "upsample_scales",
    "kernel_size",
    "activation",
    "output_activation",
}


@dataclass(frozen=True)
class LoadedFixedStepNersiCheckpoint:
    """Restored final NeRSI function and its volume-local preprocessing contract."""

    model: Nersi
    coordinate_order: tuple[str, str, str]
    profile_axis_order: tuple[str, str]
    spatial_shape: tuple[int, int, int, int]
    profile_shape: tuple[int, int]
    amplitude_scaling: str
    amplitude_scale: float
    global_step: int
    final_batch_loss: float
    input_binding: dict[str, object]


def save_fixed_step_nersi_checkpoint(
    path: Path,
    model: Nersi,
    data: C3VolumeNersiData,
    *,
    global_step: int,
    final_batch_loss: float,
    input_binding: Mapping[str, object],
) -> None:
    """Save a non-resumable final function and observed-only preprocessing metadata."""
    if not isinstance(model, Nersi):
        raise TypeError("model must be a Nersi")
    if not isinstance(data, C3VolumeNersiData):
        raise TypeError("data must be C3VolumeNersiData")
    spatial_shape, profile_shape, scale = _validated_data_contract(model, data)
    step, loss = _validated_training_values(global_step, final_batch_loss)
    binding = _validated_input_binding(input_binding)
    model_config = model.constructor_config()
    if not isinstance(model_config, Mapping) or set(model_config) != _MODEL_CONFIG_FIELDS:
        raise ValueError("model constructor_config must contain every NeRSI constructor field")
    payload = {
        "model_type": "nersi",
        "checkpoint_role": FIXED_STEP_FINAL_CHECKPOINT_ROLE,
        "method_variant": NERSI_METHOD_VARIANT,
        "training_domain": TRAINING_DOMAIN,
        "input_binding": binding,
        "model_config": dict(model_config),
        "model_state_dict": _cpu_state_dict(model.state_dict()),
        "preprocessing": {
            "coordinate_order": list(PROFILE_COORDINATE_ORDER),
            "coordinate_normalization": COORDINATE_NORMALIZATION,
            "profile_axis_order": list(PROFILE_AXIS_ORDER),
            "spatial_shape": list(spatial_shape),
            "profile_shape": list(profile_shape),
            "amplitude_scaling": AMPLITUDE_SCALING,
            "amplitude_scale": scale,
        },
        "training": {"global_step": step, "final_batch_loss": loss},
    }
    torch.save(payload, Path(path))


def load_fixed_step_nersi_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedFixedStepNersiCheckpoint:
    """Rebuild a NeRSI from complete metadata and load its weights strictly."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "model_type",
        "checkpoint_role",
        "method_variant",
        "training_domain",
        "input_binding",
        "model_config",
        "model_state_dict",
        "preprocessing",
        "training",
    }
    if not isinstance(payload, Mapping):
        raise ValueError("fixed-step NeRSI checkpoint must be a mapping")
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"fixed-step NeRSI checkpoint is missing required fields: {missing!r}")
    for key, expected in (
        ("model_type", "nersi"),
        ("checkpoint_role", FIXED_STEP_FINAL_CHECKPOINT_ROLE),
        ("method_variant", NERSI_METHOD_VARIANT),
        ("training_domain", TRAINING_DOMAIN),
    ):
        if payload[key] != expected:
            raise ValueError(f"checkpoint {key} must be {expected!r}")

    model_config = payload["model_config"]
    if not isinstance(model_config, Mapping) or set(model_config) != _MODEL_CONFIG_FIELDS:
        raise ValueError("checkpoint model_config must contain every NeRSI constructor field")
    try:
        model = Nersi(**dict(model_config))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model_config contains invalid constructor fields") from error
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state_dict.items()
    ):
        raise ValueError("checkpoint model_state_dict must map names to tensors")
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "checkpoint model_state_dict does not strictly match model_config"
        ) from error

    preprocessing = payload["preprocessing"]
    preprocessing_fields = {
        "coordinate_order",
        "coordinate_normalization",
        "profile_axis_order",
        "spatial_shape",
        "profile_shape",
        "amplitude_scaling",
        "amplitude_scale",
    }
    if not isinstance(preprocessing, Mapping) or not preprocessing_fields.issubset(preprocessing):
        raise ValueError("checkpoint preprocessing is missing required fields")
    if preprocessing["coordinate_order"] != list(PROFILE_COORDINATE_ORDER):
        raise ValueError("checkpoint coordinate_order does not match the NeRSI contract")
    if preprocessing["coordinate_normalization"] != COORDINATE_NORMALIZATION:
        raise ValueError("checkpoint coordinate_normalization does not match the NeRSI contract")
    if preprocessing["profile_axis_order"] != list(PROFILE_AXIS_ORDER):
        raise ValueError("checkpoint profile_axis_order does not match the NeRSI contract")
    if preprocessing["amplitude_scaling"] != AMPLITUDE_SCALING:
        raise ValueError("checkpoint amplitude_scaling does not match the NeRSI contract")
    spatial_shape = _positive_shape(preprocessing["spatial_shape"], 4, "spatial_shape")
    profile_shape = _positive_shape(preprocessing["profile_shape"], 2, "profile_shape")
    if spatial_shape[-1] != profile_shape[-1]:
        raise ValueError("checkpoint spatial_shape receiver-y must match profile_shape")
    if tuple(model.profile_shape) != profile_shape:
        raise ValueError("checkpoint model profile_shape does not match preprocessing")
    scale = _positive_finite_float(preprocessing["amplitude_scale"], "amplitude_scale")

    training = payload["training"]
    if not isinstance(training, Mapping) or not {"global_step", "final_batch_loss"}.issubset(
        training
    ):
        raise ValueError("checkpoint training is missing required fields")
    step, loss = _validated_training_values(training["global_step"], training["final_batch_loss"])
    binding = _validated_input_binding(payload["input_binding"])
    model.to(device)
    return LoadedFixedStepNersiCheckpoint(
        model=model,
        coordinate_order=PROFILE_COORDINATE_ORDER,
        profile_axis_order=PROFILE_AXIS_ORDER,
        spatial_shape=spatial_shape,
        profile_shape=profile_shape,
        amplitude_scaling=AMPLITUDE_SCALING,
        amplitude_scale=scale,
        global_step=step,
        final_batch_loss=loss,
        input_binding=binding,
    )


def nersi_checkpoint_input_binding(inputs_lock: Mapping[str, object]) -> dict[str, object]:
    """Select the immutable case and volume identities needed by a per-volume model."""
    if not isinstance(inputs_lock, Mapping):
        raise ValueError("inputs_lock must be a mapping")
    case = inputs_lock.get("benchmark_case")
    volume = inputs_lock.get("benchmark_volume")
    if not isinstance(case, Mapping) or not isinstance(volume, Mapping):
        raise ValueError("inputs_lock must contain benchmark_case and benchmark_volume mappings")
    files = volume.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("inputs_lock benchmark_volume.files must be a mapping")
    if any(
        not isinstance(name, str) or not isinstance(record, Mapping)
        for name, record in files.items()
    ):
        raise ValueError("benchmark volume file records must map names to hashes")
    file_hashes: dict[str, str] = {}
    for name, record in sorted(files.items()):
        file_hashes[name] = _sha256(record.get("sha256"), f"benchmark_volume.files.{name}")
    return _validated_input_binding(
        {
            "case_id": case.get("case_id"),
            "volume_id": volume.get("volume_id"),
            "input_hashes": {
                "benchmark_case": _sha256(case.get("sha256"), "benchmark_case.sha256"),
                "benchmark_volume": file_hashes,
            },
        }
    )


def validate_fixed_step_nersi_checkpoint_input_binding(
    checkpoint: LoadedFixedStepNersiCheckpoint,
    inputs_lock: Mapping[str, object],
    data: C3VolumeNersiData,
) -> None:
    """Reject pairing a volume-local checkpoint with any other verified input."""
    if not isinstance(checkpoint, LoadedFixedStepNersiCheckpoint):
        raise TypeError("checkpoint must be a LoadedFixedStepNersiCheckpoint")
    expected = nersi_checkpoint_input_binding(inputs_lock)
    if checkpoint.input_binding != expected:
        raise ValueError("checkpoint case/volume input binding differs from current inputs")
    if not isinstance(data, C3VolumeNersiData):
        raise TypeError("data must be C3VolumeNersiData")
    if (
        checkpoint.coordinate_order != PROFILE_COORDINATE_ORDER
        or checkpoint.profile_axis_order != PROFILE_AXIS_ORDER
        or checkpoint.spatial_shape != tuple(data.spatial_shape)
        or checkpoint.profile_shape != tuple(data.profile_shape)
        or checkpoint.amplitude_scale != float(data.amplitude_scale)
    ):
        raise ValueError("checkpoint preprocessing differs from current NeRSI data")


def _validated_data_contract(
    model: Nersi, data: C3VolumeNersiData
) -> tuple[tuple[int, int, int, int], tuple[int, int], float]:
    spatial_shape = _positive_shape(data.spatial_shape, 4, "spatial_shape")
    profile_shape = _positive_shape(data.profile_shape, 2, "profile_shape")
    if tuple(model.profile_shape) != profile_shape:
        raise ValueError("model profile_shape must match data profile_shape")
    if spatial_shape[-1] != profile_shape[-1]:
        raise ValueError("spatial_shape receiver-y must match profile_shape")
    profile_count = math.prod(spatial_shape[:-1])
    if data.normalized_coordinates.shape != (profile_count, len(PROFILE_COORDINATE_ORDER)):
        raise ValueError("data profile count does not match spatial_shape")
    if data.normalized_profiles.shape != (profile_count, 1, *profile_shape):
        raise ValueError("normalized_profiles do not match spatial_shape and profile_shape")
    if data.observed_trace_mask.shape != (profile_count, spatial_shape[-1]):
        raise ValueError("observed_trace_mask does not match spatial_shape")
    return (
        spatial_shape,
        profile_shape,
        _positive_finite_float(data.amplitude_scale, "amplitude_scale"),
    )


def _validated_input_binding(value: object) -> dict[str, object]:
    required = {"case_id", "volume_id", "input_hashes"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("input_binding must contain exactly case_id, volume_id, and input_hashes")
    case_id = _nonempty_string(value["case_id"], "input_binding.case_id")
    volume_id = _nonempty_string(value["volume_id"], "input_binding.volume_id")
    hashes = value["input_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "benchmark_case",
        "benchmark_volume",
    }:
        raise ValueError(
            "input_binding.input_hashes must contain benchmark_case and benchmark_volume"
        )
    volume_hashes = hashes["benchmark_volume"]
    if not isinstance(volume_hashes, Mapping) or not volume_hashes:
        raise ValueError("input_binding benchmark_volume hashes must be a nonempty mapping")
    if any(not isinstance(name, str) for name in volume_hashes):
        raise ValueError("input_binding benchmark volume file names must be strings")
    normalized_volume_hashes = {
        _nonempty_string(name, "input_binding benchmark volume file name"): _sha256(
            digest, f"input_binding benchmark_volume.{name}"
        )
        for name, digest in sorted(volume_hashes.items())
    }
    return {
        "case_id": case_id,
        "volume_id": volume_id,
        "input_hashes": {
            "benchmark_case": _sha256(hashes["benchmark_case"], "input_binding benchmark_case"),
            "benchmark_volume": deepcopy(normalized_volume_hashes),
        },
    }


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_shape(value: object, length: int, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != length
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or int(item) <= 0
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain {length} positive integers")
    return tuple(int(item) for item in value)


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _validated_training_values(step: object, loss: object) -> tuple[int, float]:
    if isinstance(step, bool) or not isinstance(step, Integral) or int(step) <= 0:
        raise ValueError("global_step must be a positive integer")
    if isinstance(loss, bool) or not isinstance(loss, Real):
        raise ValueError("final_batch_loss must be nonnegative and finite")
    converted_loss = float(loss)
    if not math.isfinite(converted_loss) or converted_loss < 0.0:
        raise ValueError("final_batch_loss must be nonnegative and finite")
    return int(step), converted_loss


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to(device="cpu").clone() for name, tensor in state_dict.items()}
