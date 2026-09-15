"""Strict fixed-final checkpoint storage for profile-wise NeRSI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from seis_interp.models.nersi import Nersi
from seis_interp.processing.trace_time_alignment import (
    alignment_time_padding,
    validate_time_alignment,
)
from seis_interp.training.c3_volume_nersi_data import (
    SPATIAL_AXIS_ORDER,
    C3VolumeNersiData,
    ProfileCoordinateBounds,
    profile_coordinate_bounds,
    profile_coordinate_order,
    validate_profile_axis,
    validate_trace_amplitude_scale,
)
from seis_interp.training.nersi_optimization import validate_nersi_optimization

FIXED_STEP_FINAL_CHECKPOINT_ROLE = "fixed_step_final"
NERSI_METHOD_VARIANT = "profile_wise_per_volume_internal_learning_fixed_steps_reimplementation"
COORDINATE_NORMALIZATION = "fixed_analysis_domain_index_bounds_to_unit_interval"
AMPLITUDE_SCALING = "observed_volume_global_rms"
TRACE_RMS_IDW_SCALING = "observed_trace_rms_idw"
NERSI_IDW_METHOD_VARIANT = f"{NERSI_METHOD_VARIANT}_trace_rms_idw"
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
_OPTIONAL_MODEL_CONFIG_FIELDS = {
    "coordinate_mapping",
    "axis_frequency_limits",
    "decoder_convolution",
    "profile_embedding_channels",
    "profile_grid_shape",
    "latent_spatial_rank",
    "temporal_basis_components",
}


@dataclass(frozen=True)
class LoadedFixedStepNersiCheckpoint:
    """Restored final NeRSI function and its volume-local preprocessing contract."""

    model: Nersi
    coordinate_order: tuple[str, str, str]
    profile_axis_order: tuple[str, str]
    coordinate_bounds: ProfileCoordinateBounds
    spatial_shape: tuple[int, int, int, int]
    profile_shape: tuple[int, int]
    profile_axis: str
    amplitude_scaling: str
    amplitude_scale: float
    global_step: int
    final_batch_loss: float
    input_binding: dict[str, object]
    model_initialization_seed: int
    sampling_seed: int
    trace_amplitude_scale: np.ndarray | None = None
    time_alignment: dict[str, object] | None = None
    optimization: dict | None = None


def nersi_method_variant(
    *,
    trace_rms_idw: bool,
    time_alignment: Mapping[str, object] | None,
    profile_axis: str = "relative_receiver_y",
) -> str:
    """Identify the amplitude and time preprocessing used by a NeRSI function."""
    variant = NERSI_IDW_METHOD_VARIANT if trace_rms_idw else NERSI_METHOD_VARIANT
    axis = validate_profile_axis(profile_axis)
    if axis != "relative_receiver_y":
        variant = f"{variant}_{axis}_profiles"
    if time_alignment is None:
        return variant
    boundary = validate_time_alignment(time_alignment)["boundary"]
    suffix = {
        "circular": "circular_time_alignment",
        "zero_pad": "zero_padded_time_alignment",
        "fourier_periodic": "fourier_time_alignment",
    }[boundary]
    return f"{variant}_{suffix}"


def save_fixed_step_nersi_checkpoint(
    path: Path,
    model: Nersi,
    data: C3VolumeNersiData,
    *,
    global_step: int,
    final_batch_loss: float,
    input_binding: Mapping[str, object],
    model_initialization_seed: int,
    sampling_seed: int,
    optimization: dict | None = None,
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
    if (
        not isinstance(model_config, Mapping)
        or set(model_config) - _OPTIONAL_MODEL_CONFIG_FIELDS != _MODEL_CONFIG_FIELDS
    ):
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
            "coordinate_order": list(data.coordinate_order),
            "coordinate_normalization": COORDINATE_NORMALIZATION,
            "coordinate_bounds": [list(bounds) for bounds in data.coordinate_bounds],
            "profile_axis_order": list(data.profile_axis_order),
            "spatial_shape": list(spatial_shape),
            "profile_shape": list(profile_shape),
            "amplitude_scaling": AMPLITUDE_SCALING,
            "amplitude_scale": scale,
        },
        "training": {
            "global_step": step,
            "final_batch_loss": loss,
            "model_initialization_seed": _seed(
                model_initialization_seed, "model_initialization_seed"
            ),
            "sampling_seed": _seed(sampling_seed, "sampling_seed"),
        },
    }
    if data.trace_amplitude_scale is not None:
        validate_trace_amplitude_scale(data.trace_amplitude_scale, spatial_shape)
        payload["method_variant"] = NERSI_IDW_METHOD_VARIANT
        payload["preprocessing"]["amplitude_scaling"] = TRACE_RMS_IDW_SCALING
        payload["preprocessing"]["trace_amplitude_scale"] = torch.from_numpy(
            data.trace_amplitude_scale.copy()
        )
    if data.time_alignment is not None:
        payload["preprocessing"]["time_alignment"] = validate_time_alignment(data.time_alignment)
    payload["method_variant"] = nersi_method_variant(
        trace_rms_idw=data.trace_amplitude_scale is not None,
        time_alignment=data.time_alignment,
        profile_axis=data.profile_axis,
    )
    if optimization is not None:
        payload["training"]["optimization"] = deepcopy(optimization)
        payload["training"]["prediction_weights"] = (
            "final_ema" if optimization["ema_decay"] is not None else "final_raw"
        )
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
        ("training_domain", TRAINING_DOMAIN),
    ):
        if payload[key] != expected:
            raise ValueError(f"checkpoint {key} must be {expected!r}")

    model_config = payload["model_config"]
    if (
        not isinstance(model_config, Mapping)
        or set(model_config) - _OPTIONAL_MODEL_CONFIG_FIELDS != _MODEL_CONFIG_FIELDS
    ):
        raise ValueError("checkpoint model_config must contain every NeRSI constructor field")
    try:
        model = Nersi(**dict(model_config))
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("checkpoint model_config contains invalid constructor fields") from error
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state_dict.items()
    ):
        raise ValueError("checkpoint model_state_dict must map names to tensors")
    try:
        for name, expected in model.named_buffers():
            actual = state_dict.get(name)
            if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
                raise ValueError(
                    f"checkpoint model_state_dict fixed encoding buffer mismatch: {name}"
                )
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "checkpoint model_state_dict does not strictly match model_config"
        ) from error

    preprocessing = payload["preprocessing"]
    preprocessing_fields = {
        "coordinate_order",
        "coordinate_normalization",
        "coordinate_bounds",
        "profile_axis_order",
        "spatial_shape",
        "profile_shape",
        "amplitude_scaling",
        "amplitude_scale",
    }
    if not isinstance(preprocessing, Mapping) or not preprocessing_fields.issubset(preprocessing):
        raise ValueError("checkpoint preprocessing is missing required fields")
    profile_axis_order = preprocessing["profile_axis_order"]
    if (
        not isinstance(profile_axis_order, list)
        or len(profile_axis_order) != 2
        or profile_axis_order[0] != "time"
    ):
        raise ValueError("checkpoint profile_axis_order does not match the NeRSI contract")
    try:
        profile_axis = validate_profile_axis(profile_axis_order[1])
    except ValueError as error:
        raise ValueError(
            "checkpoint profile_axis_order does not match the NeRSI contract"
        ) from error
    coordinate_order = profile_coordinate_order(profile_axis)
    if preprocessing["coordinate_order"] != list(coordinate_order):
        raise ValueError("checkpoint coordinate_order does not match the NeRSI contract")
    if preprocessing["coordinate_normalization"] != COORDINATE_NORMALIZATION:
        raise ValueError("checkpoint coordinate_normalization does not match the NeRSI contract")
    scaling = preprocessing["amplitude_scaling"]
    if scaling not in (AMPLITUDE_SCALING, TRACE_RMS_IDW_SCALING):
        raise ValueError("checkpoint amplitude_scaling does not match the NeRSI contract")
    spatial_shape = _positive_shape(preprocessing["spatial_shape"], 4, "spatial_shape")
    profile_axis_index = SPATIAL_AXIS_ORDER.index(profile_axis)
    coordinate_shape = tuple(
        value for index, value in enumerate(spatial_shape) if index != profile_axis_index
    )
    if model.coordinate_mapping is not None and model.coordinate_grid_shape != coordinate_shape:
        raise ValueError("checkpoint coordinate grid differs from spatial_shape")
    if model.profile_grid_shape is not None and model.profile_grid_shape != coordinate_shape:
        raise ValueError("checkpoint profile embedding grid differs from spatial_shape")
    trace_scale = None
    if scaling == TRACE_RMS_IDW_SCALING:
        tensor = preprocessing.get("trace_amplitude_scale")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float64:
            raise ValueError("checkpoint trace_amplitude_scale must be a float64 tensor")
        trace_scale = tensor.numpy().copy()
        validate_trace_amplitude_scale(trace_scale, spatial_shape)
    else:
        if "trace_amplitude_scale" in preprocessing:
            raise ValueError("global RMS checkpoint must not contain trace_amplitude_scale")
    alignment = (
        validate_time_alignment(preprocessing["time_alignment"])
        if "time_alignment" in preprocessing
        else None
    )
    if alignment is not None and profile_axis != "relative_receiver_y":
        raise ValueError("checkpoint time_alignment requires relative_receiver_y profiles")
    expected_variant = nersi_method_variant(
        trace_rms_idw=trace_scale is not None,
        time_alignment=alignment,
        profile_axis=profile_axis,
    )
    if payload["method_variant"] != expected_variant:
        raise ValueError("checkpoint method_variant does not match preprocessing")
    profile_shape = _positive_shape(preprocessing["profile_shape"], 2, "profile_shape")
    if alignment is not None and profile_shape[0] <= alignment_time_padding(
        spatial_shape[-1], alignment
    ):
        raise ValueError("checkpoint alignment padding leaves no physical time samples")
    if spatial_shape[profile_axis_index] != profile_shape[-1]:
        raise ValueError("checkpoint spatial_shape generated axis must match profile_shape")
    if tuple(model.profile_shape) != profile_shape:
        raise ValueError("checkpoint model profile_shape does not match preprocessing")
    coordinate_bounds = _coordinate_bounds(
        preprocessing["coordinate_bounds"],
        spatial_shape=spatial_shape,
        profile_axis=profile_axis,
    )
    scale = _positive_finite_float(preprocessing["amplitude_scale"], "amplitude_scale")

    training = payload["training"]
    if not isinstance(training, Mapping) or not {"global_step", "final_batch_loss"}.issubset(
        training
    ):
        raise ValueError("checkpoint training is missing required fields")
    step, loss = _validated_training_values(training["global_step"], training["final_batch_loss"])
    binding = _validated_input_binding(payload["input_binding"])
    optimization = training.get("optimization")
    if optimization is not None:
        # Initial LR lives in resolved config; inference requires the option structure only.
        optimization = validate_nersi_optimization(
            optimization, optimization["minimum_learning_rate"]
        )
        expected_weights = "final_ema" if optimization["ema_decay"] is not None else "final_raw"
        if training.get("prediction_weights") != expected_weights:
            raise ValueError("checkpoint prediction_weights differs from optimization")
    model.to(device)
    return LoadedFixedStepNersiCheckpoint(
        model=model,
        coordinate_order=coordinate_order,
        profile_axis_order=("time", profile_axis),
        coordinate_bounds=coordinate_bounds,
        spatial_shape=spatial_shape,
        profile_shape=profile_shape,
        profile_axis=profile_axis,
        amplitude_scaling=scaling,
        trace_amplitude_scale=trace_scale,
        time_alignment=alignment,
        optimization=optimization,
        amplitude_scale=scale,
        global_step=step,
        final_batch_loss=loss,
        input_binding=binding,
        model_initialization_seed=_seed(
            training.get("model_initialization_seed"), "model_initialization_seed"
        ),
        sampling_seed=_seed(training.get("sampling_seed"), "sampling_seed"),
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
        checkpoint.coordinate_order != data.coordinate_order
        or checkpoint.profile_axis_order != data.profile_axis_order
        or checkpoint.profile_axis != data.profile_axis
        or checkpoint.coordinate_bounds != data.coordinate_bounds
        or checkpoint.spatial_shape != tuple(data.spatial_shape)
        or checkpoint.profile_shape != tuple(data.profile_shape)
        or checkpoint.amplitude_scale != float(data.amplitude_scale)
        or not np.array_equal(checkpoint.trace_amplitude_scale, data.trace_amplitude_scale)
        or checkpoint.time_alignment != data.time_alignment
    ):
        raise ValueError("checkpoint preprocessing differs from current NeRSI data")


def _validated_data_contract(
    model: Nersi, data: C3VolumeNersiData
) -> tuple[tuple[int, int, int, int], tuple[int, int], float]:
    spatial_shape = _positive_shape(data.spatial_shape, 4, "spatial_shape")
    profile_shape = _positive_shape(data.profile_shape, 2, "profile_shape")
    if tuple(model.profile_shape) != profile_shape:
        raise ValueError("model profile_shape must match data profile_shape")
    profile_axis_index = SPATIAL_AXIS_ORDER.index(data.profile_axis)
    if spatial_shape[profile_axis_index] != profile_shape[-1]:
        raise ValueError("spatial_shape generated axis must match profile_shape")
    profile_count = math.prod(
        value for index, value in enumerate(spatial_shape) if index != profile_axis_index
    )
    coordinate_shape = tuple(
        value for index, value in enumerate(spatial_shape) if index != profile_axis_index
    )
    if model.profile_grid_shape is not None and model.profile_grid_shape != coordinate_shape:
        raise ValueError("model profile embedding grid must match data spatial_shape")
    if data.normalized_coordinates.shape != (profile_count, len(data.coordinate_order)):
        raise ValueError("data profile count does not match spatial_shape")
    if data.normalized_profiles.shape != (profile_count, 1, *profile_shape):
        raise ValueError("normalized_profiles do not match spatial_shape and profile_shape")
    if data.observed_trace_mask.shape != (profile_count, spatial_shape[profile_axis_index]):
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


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


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


def _coordinate_bounds(
    value: object,
    *,
    spatial_shape: tuple[int, int, int, int],
    profile_axis: str,
) -> ProfileCoordinateBounds:
    expected = profile_coordinate_bounds(spatial_shape, profile_axis=profile_axis)
    if not isinstance(value, (tuple, list)) or len(value) != len(expected):
        raise ValueError("checkpoint coordinate_bounds must contain three index bounds")
    normalized: list[tuple[int, int]] = []
    for bounds in value:
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in bounds)
        ):
            raise ValueError("checkpoint coordinate_bounds must contain integer pairs")
        normalized.append((int(bounds[0]), int(bounds[1])))
    result = tuple(normalized)
    if result != expected:
        raise ValueError("checkpoint coordinate_bounds must match the full analysis domain")
    return result  # type: ignore[return-value]


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
