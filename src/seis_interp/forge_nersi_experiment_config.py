"""Strict YAML config for scratch FORGE NeRSI experiments on a sealed preparation.

The workspace config names every tunable NeRSI knob. Data binding (alignment,
mask, amplitude scaling, geometry) comes only from the sealed preparation, and
settings that the code fixes are rejected instead of silently ignored. Nothing
here reads waveforms or builds a model.
"""

import difflib
import hashlib
import math
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from seis_interp.models.nersi_trace import BAND_HIGH_RAMP_HZ, BAND_LOW_RAMP_HZ
from seis_interp.processing.nersi_trace_coordinates import (
    head_samples_per_radius,
    linear_moveout_samples_per_radius,
    midpoint_offset_transform,
    slowness_samples_per_radius,
)
from seis_interp.training.trace_graph_ema import validate_trace_graph_ema_decay

TOP_LEVEL_KEYS = (
    "name",
    "device",
    "model_seed",
    "torch_threads",
    "model",
    "training",
    "prediction",
)
SECTION_KEYS = {
    "model": (
        "coordinate_frame",
        "polar_offset",
        "moveout_velocity_m_per_s",
        "fourier_components",
        "frequency_base",
        "encoder_width",
        "latent_channels",
        "decoder_channels",
        "kernel_size",
        "head_slowness_ms_per_m",
        "static_shift_samples",
        "residual_slowness_ms_per_m",
        "output_band_hz",
        "output_window_mask",
    ),
    "training": (
        "learning_rate",
        "max_steps",
        "batch_size",
        "weight_decay",
        "weight_decay_after",
        "ema_decay",
        "report_interval",
    ),
    "prediction": ("batch_size",),
}
_PREPARATION_REASON = (
    "fixed by the preparation (see --check); choose another preparation in inputs.yaml"
)
_NERSI_TRACE_REASON = "fixed by NersiTrace (4 / [2,2,2] / gelu / linear)"
_NOT_FORWARDED_REASON = "not forwarded by NersiTrace; changing it needs a code change"
REJECTED_KEYS = {
    "training.optimizer": (
        "fixed: _fit_nersi always uses torch.optim.Adam with coupled L2 weight_decay"
    ),
    "training.learning_rate_schedule": "fixed: constant learning rate",
    "training.mixed_precision": "fixed: fp32 parameters, no AMP",
    "loss": "fixed: masked_trace_relative_mse on the aligned canvas",
    "model.input_features": _NERSI_TRACE_REASON,
    "model.upsample_scales": _NERSI_TRACE_REASON,
    "model.activation": _NERSI_TRACE_REASON,
    "model.output_activation": _NERSI_TRACE_REASON,
    "model.time_sample_count": (
        "derived from the preparation contract; run_method extends it to the alignment canvas"
    ),
    "checkpoint_rule": "derived from training.ema_decay",
    "model.coordinate_transform": (
        "derived from model.coordinate_frame and model.polar_offset using the preparation geometry"
    ),
    "model.moveout_samples_per_radius": "use model.moveout_velocity_m_per_s",
    "model.head_samples_per_radius": "use model.head_slowness_ms_per_m",
    "model.residual_slowness_samples_per_radius": "use model.residual_slowness_ms_per_m",
    "model.output_window": "derived from model.output_window_mask and the preparation alignment",
    "model.spatial_shape": "derived from the preparation geometry when statics are on",
    "model.sample_interval_s": "derived from the preparation contract when the band is on",
    "model.axis_frequency_limits": _NOT_FORWARDED_REASON,
    "model.decoder_convolution": _NOT_FORWARDED_REASON,
    **{
        key: _PREPARATION_REASON
        for key in (
            "method_id",
            "geometry_mode",
            "alignment",
            "amplitude_scaling",
            "trace_scale_hash",
            "mask_hash",
            "mask_id",
            "mask_seed",
            "global_rms",
            "preparation",
        )
    },
}
BINDING_KEYS = (
    "alignment",
    "amplitude_scaling",
    "trace_scale_hash",
    "mask_hash",
    "mask_id",
    "mask_seed",
    "global_rms",
)
COORDINATE_FRAMES = ("midpoint_offset", "source_receiver")
FRAME_AXES = {
    ("source_receiver", False): ["source_line", "source_point", "receiver_line", "receiver_point"],
    ("midpoint_offset", False): ["cmp_x", "cmp_y", "half_offset_x", "half_offset_y"],
    ("midpoint_offset", True): ["cmp_x", "cmp_y", "radius", "azimuth/pi"],
}
RESERVED_NAMES = {"latest", "temp", "tmp", "final", "final2", "new"}
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
DEVICE_PATTERN = re.compile(r"^cuda:(\d+)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# PyYAML resolves YAML 1.1 octal (07301), base-60 (20:00), hex and binary scalars as ints.
PLAIN_INTEGER_PATTERN = re.compile(r"[-+]?(0|[1-9][0-9]*)")
BUDGET_KEYS = (
    "max_steps",
    "learning_rate",
    "loss_traces_per_update",
    "parameter_count_target",
    "parameter_count_tolerance",
)
# NersiTrace defaults for keys that older sealed configs omit.
NERSI_TRACE_DEFAULTS = {
    "coordinate_transform": None,
    "polar_offset": False,
    "moveout_samples_per_radius": 0.0,
    "head_samples_per_radius": None,
    "static_shift_samples": 0.0,
    "residual_slowness_samples_per_radius": 0.0,
    "output_band_hz": None,
    "output_window": None,
}


def load_experiment_config(path: Path, *, device_override: str | None = None) -> dict:
    """Read, statically validate and return the requested config (device overridden)."""
    return parse_experiment_config(Path(path).read_bytes(), path, device_override=device_override)


def parse_experiment_config(
    content: bytes, path: Path, *, device_override: str | None = None
) -> dict:
    """Validate config file bytes already read from ``path`` (hash the same bytes)."""
    requested, errors = _load_strict_yaml(content, path)
    errors += validate_experiment_config(requested)
    if device_override is not None:
        errors += [f"--device: {e.split(': ', 1)[1]}" for e in _device_errors(device_override)]
    if errors:
        raise ValueError(f"{path}: invalid NeRSI experiment config\n" + "\n".join(errors))
    requested = deepcopy(requested)
    if device_override is not None:
        requested["device"] = device_override
    return requested


def load_experiment_inputs(path: Path) -> dict:
    """Return the preparation path and the sha256 of its preparation manifest."""
    inputs, errors = _load_strict_yaml(Path(path).read_bytes(), path)
    if errors:
        raise ValueError(f"{path}: invalid inputs\n" + "\n".join(errors))
    keys = {"preparation", "preparation_manifest_sha256"}
    if not isinstance(inputs, dict) or set(inputs) != keys:
        raise ValueError(f"{path}: inputs must contain exactly {sorted(keys)}")
    preparation = inputs["preparation"]
    if (
        not isinstance(preparation, str)
        or not preparation
        or Path(preparation).is_absolute()
        or ".." in Path(preparation).parts
    ):
        raise ValueError(f"{path}: preparation must be a repository-relative path without '..'")
    digest = inputs["preparation_manifest_sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{path}: preparation_manifest_sha256 must be 64 lowercase hex characters")
    return dict(inputs)


def config_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_strict_yaml(content: bytes, path: Path) -> tuple[object, list[str]]:
    """Safe-load YAML, reporting duplicate keys and YAML 1.1 integer spellings by key path."""
    loader = yaml.SafeLoader(content)
    try:
        node = loader.get_single_node()
        errors = _node_errors(node, "", loader) if node is not None else []
        value = loader.construct_document(node) if node is not None else None
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: YAML syntax error\n{error}") from error
    finally:
        loader.dispose()
    return value, errors


def _node_errors(node, path: str, loader) -> list[str]:
    if isinstance(node, yaml.MappingNode):
        errors, lines = [], {}
        for key_node, value_node in node.value:
            key = key_node.value if isinstance(key_node, yaml.ScalarNode) else str(key_node)
            child = f"{path}.{key}" if path else key
            line = key_node.start_mark.line + 1
            if key in lines:
                errors.append(f"{child}: duplicate key (lines {lines[key]} and {line})")
            lines.setdefault(key, line)
            errors += _node_errors(value_node, child, loader)
        return errors
    if isinstance(node, yaml.SequenceNode):
        return [
            e
            for i, item in enumerate(node.value)
            for e in _node_errors(item, f"{path}[{i}]", loader)
        ]
    if node.tag == "tag:yaml.org,2002:int" and not PLAIN_INTEGER_PATTERN.fullmatch(node.value):
        value = loader.construct_yaml_int(node)
        return [
            f"{path}: {node.value!r} is not a plain decimal integer; PyYAML reads it as {value} "
            "(YAML 1.1 octal, base-60, hex or binary), write decimal digits without leading zeros"
        ]
    if node.tag == "tag:yaml.org,2002:float" and ":" in node.value:
        return [f"{path}: {node.value!r} is a YAML 1.1 base-60 number; write a plain decimal"]
    return []


def validate_experiment_config(config) -> list[str]:
    """Return every static key, type, range and cross-field error by key path."""
    if not isinstance(config, dict):
        return ["config must be a mapping"]
    errors = _key_errors(config, TOP_LEVEL_KEYS, "")
    sections = {}
    for section in SECTION_KEYS:
        if section not in config:
            continue
        if not isinstance(config[section], dict):
            errors.append(f"{section} must be a mapping")
            continue
        sections[section] = config[section]
        errors += _key_errors(config[section], SECTION_KEYS[section], f"{section}.")
    errors += _value_errors(config, sections)
    return errors


def resolve_run_config(requested, sealed, contract, geometry, spatial_shape) -> dict:
    """Return the complete ``run_method`` config; neither input is modified."""
    if sealed.get("method_id") != "nersi_real" or sealed.get("geometry_mode") != "real":
        raise ValueError("preparation config must be nersi_real with real geometry")
    model, training = requested["model"], requested["training"]
    polar = model["polar_offset"]
    transform = None
    if model["coordinate_frame"] == "midpoint_offset":
        transform = midpoint_offset_transform(geometry, spatial_shape, polar=polar).tolist()
    interval = contract["sample_interval_us"] / 1e6
    moveout = 0.0
    if model["moveout_velocity_m_per_s"] is not None:
        moveout = linear_moveout_samples_per_radius(
            geometry, model["moveout_velocity_m_per_s"], interval
        )
    ema_decay = validate_trace_graph_ema_decay(training["ema_decay"])
    config = {
        "method_id": "nersi_real",
        "geometry_mode": "real",
        "device": requested["device"],
        "model_seed": requested["model_seed"],
        "loss": "masked_trace_relative_mse",
        "checkpoint_rule": "fixed_final_ema" if ema_decay is not None else "fixed_final",
        "model": {
            "time_sample_count": int(contract["time_sample_count"]),
            "input_features": 4,
            "fourier_components": model["fourier_components"],
            "frequency_base": deepcopy(model["frequency_base"]),
            "encoder_width": model["encoder_width"],
            "latent_channels": model["latent_channels"],
            "decoder_channels": list(model["decoder_channels"]),
            "upsample_scales": [2, 2, 2],
            "kernel_size": model["kernel_size"],
            "activation": "gelu",
            "output_activation": "linear",
            "coordinate_transform": transform,
            "polar_offset": polar,
            "moveout_samples_per_radius": float(moveout),
            **_resolve_decoder_priors(model, sealed, contract, geometry, spatial_shape),
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": training["learning_rate"],
            "max_steps": training["max_steps"],
            "batch_size": training["batch_size"],
            "weight_decay": training["weight_decay"],
            "weight_decay_after": deepcopy(training["weight_decay_after"]),
            "ema_decay": ema_decay,
            "report_interval": training["report_interval"],
            "learning_rate_schedule": "constant",
            "mixed_precision": "off",
        },
        "prediction": {"batch_size": requested["prediction"]["batch_size"]},
    }
    config.update({key: deepcopy(sealed[key]) for key in BINDING_KEYS if key in sealed})
    return config


def _resolve_decoder_priors(model, sealed, contract, geometry, spatial_shape) -> dict:
    """NersiTrace kwargs for slowness heads, statics, residual slowness, band and window."""
    interval = contract["sample_interval_us"] / 1e6
    alignment = sealed.get("alignment")
    velocity = alignment["velocity_m_per_s"] if alignment else None
    heads = None
    if model["head_slowness_ms_per_m"] is not None:
        heads = head_samples_per_radius(
            geometry, model["head_slowness_ms_per_m"], interval, velocity
        )
    residual = 0.0
    if model["residual_slowness_ms_per_m"]:
        residual = slowness_samples_per_radius(
            geometry, model["residual_slowness_ms_per_m"], interval
        )
    resolved = {
        "head_samples_per_radius": heads,
        "static_shift_samples": float(model["static_shift_samples"]),
        "residual_slowness_samples_per_radius": float(residual),
        "output_band_hz": deepcopy(model["output_band_hz"]),
        "output_window": _resolve_output_window(model, alignment, contract, geometry, interval),
    }
    if model["static_shift_samples"]:
        resolved["spatial_shape"] = [int(n) for n in spatial_shape]
    if model["output_band_hz"] is not None:
        nyquist = 0.5 / interval
        if model["output_band_hz"][1] + BAND_HIGH_RAMP_HZ >= nyquist:
            raise ValueError(
                f"model.output_band_hz: hi + {BAND_HIGH_RAMP_HZ} Hz must be below the "
                f"{nyquist:g} Hz Nyquist frequency of the preparation"
            )
        resolved["sample_interval_s"] = interval
    return resolved


def _resolve_output_window(model, alignment, contract, geometry, interval):
    """Aligned-canvas window of the recorded samples: start ``max_delay - rint(r * delay)``."""
    if not model["output_window_mask"]:
        return None
    if not alignment or alignment.get("max_delay_samples") is None:
        raise ValueError(
            "model.output_window_mask: needs a preparation aligned with a declared "
            "max_delay_samples"
        )
    return {
        "max_delay_samples": int(alignment["max_delay_samples"]),
        "window_samples": int(contract["time_sample_count"]),
        "delay_samples_per_radius": linear_moveout_samples_per_radius(
            geometry, alignment["velocity_m_per_s"], interval
        ),
    }


def validate_against_preparation(run_config: dict, observed_trace_count: int) -> None:
    batch_size = run_config["training"]["batch_size"]
    if batch_size > observed_trace_count:
        raise ValueError(
            f"training.batch_size: {batch_size} exceeds the {observed_trace_count} observed "
            "traces (sampling without replacement)"
        )


def validate_device_available(device: str) -> None:
    if device == "cpu":
        return
    match = DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError(f"device: must be cpu or cuda:<index>, got {device!r}")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if int(match.group(1)) >= count:
        raise ValueError(f"device: {device} not available ({count} CUDA devices)")


def budget_comparison(run_config: dict, parameter_count: int, budget: dict | None) -> dict | None:
    """Matched-budget flags computed from the preparation's declaration; never enforced."""
    if budget is None:
        return None
    missing = [key for key in BUDGET_KEYS if key not in budget]
    if missing:
        raise ValueError(
            f"matched_training lacks {', '.join(missing)} (older budget schema); not compared"
        )
    training = run_config["training"]
    target, tolerance = budget["parameter_count_target"], budget["parameter_count_tolerance"]
    ratio = parameter_count / target - 1
    flags = {
        "steps_equal": training["max_steps"] == budget["max_steps"],
        "learning_rate_equal": training["learning_rate"] == budget["learning_rate"],
        "batch_equal": training["batch_size"] == budget["loss_traces_per_update"],
        "parameters_within_tolerance": abs(ratio) <= tolerance,
    }
    return {
        "max_steps": budget["max_steps"],
        "learning_rate": budget["learning_rate"],
        "batch_size": budget["loss_traces_per_update"],
        "parameter_count": parameter_count,
        "parameter_count_target": target,
        "parameter_count_tolerance": tolerance,
        "parameter_band": _parameter_band(target, tolerance),
        "parameter_ratio": ratio,
        **flags,
        "budget_matched": all(flags.values()),
    }


def _parameter_band(target, tolerance) -> list[int]:
    """Integer counts that pass the same ratio test as ``validate_parameter_counts``."""
    low, high = math.ceil(target * (1 - tolerance)), math.floor(target * (1 + tolerance))
    while abs(low / target - 1) > tolerance:
        low += 1
    while abs(high / target - 1) > tolerance:
        high -= 1
    return [low, high]


def ema_initial_weight_fraction(training: dict) -> float | None:
    """Weight of the first-update parameters left in the final EMA average."""
    if training["ema_decay"] is None:
        return None
    return float(training["ema_decay"]) ** (training["max_steps"] - 1)


def config_differences(run_config: dict, sealed: dict) -> list[str]:
    """Knob changes against the sealed method config, as ``key: sealed -> run`` items."""
    differences = [
        f"{key}: {_describe(sealed.get(key))} -> {_describe(run_config.get(key))}"
        for key in ("model_seed", "loss", "checkpoint_rule")
        if sealed.get(key) != run_config.get(key)
    ]
    for section in ("model", "training", "prediction"):
        before, after = sealed.get(section, {}), run_config[section]
        for key in [*after, *(key for key in before if key not in after)]:
            default = NERSI_TRACE_DEFAULTS.get(key) if section == "model" else None
            old, new = before.get(key, default), after.get(key)
            if not _same_value(old, new):
                differences.append(f"{section}.{key}: {_describe(old)} -> {_describe(new)}")
    return differences


def fourier_max_angular_frequencies(model: dict) -> list[float]:
    """Largest Fourier angular frequency per axis, pi * base**F."""
    base = model["frequency_base"]
    bases = base if isinstance(base, list) else [base] * 4
    return [math.pi * float(value) ** model["fourier_components"] for value in bases]


def _key_errors(mapping: dict, allowed: tuple, prefix: str) -> list[str]:
    errors = [f"missing required key {prefix}{key}" for key in allowed if key not in mapping]
    for key in mapping:
        path = f"{prefix}{key}"
        if path in REJECTED_KEYS:
            errors.append(f"{path}: {REJECTED_KEYS[path]}")
        elif key not in allowed:
            errors.append(_unknown_key_error(str(key), allowed, prefix))
    return errors


def _unknown_key_error(key: str, allowed: tuple, prefix: str) -> str:
    # Initialisms such as ``lr`` are too short for difflib's similarity ratio.
    matches = [name for name in allowed if "".join(p[0] for p in name.split("_")) == key]
    matches = matches or difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
    hint = f"; did you mean {prefix}{matches[0]}?" if matches else ";"
    listed = ", ".join(f"{prefix}{name}" for name in allowed)
    return f"unknown key {prefix}{key}{hint} allowed: {listed}"


def _value_errors(config: dict, sections: dict) -> list[str]:
    errors = []

    def check(path, value, rule):
        if (message := rule(value)) is not None:
            errors.append(f"{path}: {message}")
            return False
        return True

    if "name" in config:
        check("name", config["name"], _name_error)
    if "device" in config:
        errors += _device_errors(config["device"])
    if "model_seed" in config:
        check("model_seed", config["model_seed"], _integer_rule(0, 2**32 - 1))
    if "torch_threads" in config:
        check("torch_threads", config["torch_threads"], _integer_rule(1))
    model = sections.get("model", {})
    rules = {
        "model.coordinate_frame": _choice_rule(COORDINATE_FRAMES),
        "model.polar_offset": _boolean_error,
        "model.moveout_velocity_m_per_s": _optional_real_rule(minimum=0, exclusive=True),
        "model.fourier_components": _integer_rule(1),
        "model.frequency_base": _frequency_base_error,
        "model.encoder_width": _integer_rule(1),
        "model.latent_channels": _integer_rule(1),
        "model.decoder_channels": _decoder_channels_error,
        "model.kernel_size": _kernel_size_error,
        "model.head_slowness_ms_per_m": _head_slowness_error,
        "model.static_shift_samples": _real_rule(minimum=0, exclusive=False),
        "model.residual_slowness_ms_per_m": _real_rule(minimum=0, exclusive=False),
        "model.output_band_hz": _output_band_error,
        "model.output_window_mask": _boolean_error,
        "training.learning_rate": _real_rule(minimum=0, exclusive=True),
        "training.max_steps": _integer_rule(1),
        "training.batch_size": _integer_rule(1),
        "training.weight_decay": _real_rule(minimum=0, exclusive=False),
        "training.weight_decay_after": _weight_decay_after_error,
        "training.ema_decay": _ema_decay_error,
        "training.report_interval": _integer_rule(1),
        "prediction.batch_size": _integer_rule(1),
    }
    valid = {}
    for path, rule in rules.items():
        section, key = path.split(".")
        if key in sections.get(section, {}):
            valid[path] = check(path, sections[section][key], rule)
    polar_valid = valid.get("model.polar_offset", False)
    if (
        polar_valid
        and valid.get("model.coordinate_frame")
        and model["polar_offset"]
        and model["coordinate_frame"] != "midpoint_offset"
    ):
        errors.append("model.polar_offset: polar_offset needs the midpoint_offset frame")
    if (
        polar_valid
        and valid.get("model.moveout_velocity_m_per_s")
        and model["moveout_velocity_m_per_s"] is not None
        and not model["polar_offset"]
    ):
        errors.append("model.moveout_velocity_m_per_s: moveout requires polar_offset: true")
    errors += _decoder_prior_errors(model, valid)
    errors += _weight_decay_schedule_errors(sections.get("training", {}), valid)
    return errors


def _decoder_prior_errors(model: dict, valid: dict) -> list[str]:
    """Cross-field rules for slowness heads, statics, residual slowness and the window."""
    errors = []
    heads_valid = valid.get("model.head_slowness_ms_per_m", False)
    heads = heads_valid and model["head_slowness_ms_per_m"] is not None
    polar = valid.get("model.polar_offset", False) and model["polar_offset"]
    if heads and not (
        polar
        and valid.get("model.coordinate_frame")
        and model["coordinate_frame"] == "midpoint_offset"
    ):
        errors.append(
            "model.head_slowness_ms_per_m: slowness heads require coordinate_frame "
            "midpoint_offset and polar_offset: true"
        )
    if (
        heads
        and valid.get("model.moveout_velocity_m_per_s")
        and model["moveout_velocity_m_per_s"] is not None
    ):
        errors.append(
            "model.head_slowness_ms_per_m: exclusive with model.moveout_velocity_m_per_s; "
            "set one of them to null"
        )
    for key in ("static_shift_samples", "residual_slowness_ms_per_m"):
        if heads_valid and not heads and valid.get(f"model.{key}") and model[key]:
            errors.append(f"model.{key}: requires model.head_slowness_ms_per_m (0 = off)")
    if (
        valid.get("model.output_window_mask")
        and model["output_window_mask"]
        and valid.get("model.polar_offset")
        and not model["polar_offset"]
    ):
        errors.append("model.output_window_mask: the window requires polar_offset: true")
    return errors


def _weight_decay_schedule_errors(training: dict, valid: dict) -> list[str]:
    schedule = training.get("weight_decay_after")
    if not valid.get("training.weight_decay_after") or schedule is None:
        return []
    if valid.get("training.max_steps") and schedule["step"] >= training["max_steps"]:
        return [
            f"training.weight_decay_after: step must be < training.max_steps "
            f"({training['max_steps']}), got {schedule['step']}"
        ]
    return []


def _name_error(value):
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        return f"must match {NAME_PATTERN.pattern} (lowercase snake_case), got {value!r}"
    if value in RESERVED_NAMES:
        return f"{value!r} is reserved; choose a descriptive name"
    return None


def _device_errors(value) -> list[str]:
    if value == "cpu" or (isinstance(value, str) and DEVICE_PATTERN.fullmatch(value)):
        return []
    if value == "cuda":
        return ["device: bare 'cuda' is ambiguous; give an explicit index such as cuda:0"]
    return [f"device: must be cpu or cuda:<index>, got {value!r}"]


def _type_name(value) -> str:
    return f"{type(value).__name__} {value!r}"


def _number_string_hint(value, *, integer=False):
    """Hint for numbers that PyYAML 1.1 leaves as strings, such as ``1e-3`` or ``1.0e3``.

    PyYAML reads exponent notation as a float only with both a decimal point and a
    signed exponent (``1.0e-3``). Integer keys get only the plain integer spelling,
    and nonintegral or nonfinite strings get no hint.
    """
    if not isinstance(value, str):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number) or (integer and not number.is_integer()):
        return None
    mantissa, _, exponent = value.lower().partition("e")
    if integer:
        spelling = str(int(number))
    else:
        spelling = np.format_float_positional(number, trim="-")
        if exponent:
            mantissa = mantissa if "." in mantissa else f"{mantissa}.0"
            exponent = exponent if exponent[0] in "+-" else f"+{exponent}"
            spelling += f" or {mantissa}e{exponent}"
    reason = (
        "PyYAML reads exponent notation as a number only with a decimal point and a signed exponent"
        if exponent
        else "quoted numbers stay strings"
    )
    return f"got string {value!r}; {reason}, write {spelling}"


def _integer_rule(minimum, maximum=None):
    def rule(value):
        if (hint := _number_string_hint(value, integer=True)) is not None:
            return hint
        if type(value) is not int:
            return f"must be an integer, got {_type_name(value)}"
        if value < minimum or (maximum is not None and value > maximum):
            upper = f" and <= {maximum}" if maximum is not None else ""
            return f"must be >= {minimum}{upper}, got {value}"
        return None

    return rule


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _real_rule(*, minimum, exclusive):
    def rule(value):
        if (hint := _number_string_hint(value)) is not None:
            return hint
        if not _is_real(value):
            return f"must be a finite number, got {_type_name(value)}"
        if value < minimum or (exclusive and value == minimum):
            return f"must be {'>' if exclusive else '>='} {minimum}, got {value}"
        return None

    return rule


def _optional_real_rule(*, minimum, exclusive):
    real = _real_rule(minimum=minimum, exclusive=exclusive)
    return lambda value: None if value is None else real(value)


def _choice_rule(choices):
    def rule(value):
        if value not in choices:
            return f"must be one of {', '.join(choices)}, got {value!r}"
        return None

    return rule


def _boolean_error(value):
    if not isinstance(value, bool):
        return f"must be true or false, got {_type_name(value)}"
    return None


def _frequency_base_error(value):
    if (hint := _number_string_hint(value)) is not None:
        return hint
    values = value if isinstance(value, list) else [value]
    if isinstance(value, list) and len(value) != 4:
        return f"must be one number or a list of exactly 4 numbers, got {len(value)} items"
    if not all(_is_real(item) and item > 1 for item in values):
        return f"every base must be a finite number > 1, got {value!r}"
    return None


def _decoder_channels_error(value):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(type(item) is int and item >= 1 for item in value)
    ):
        return f"must be a list of exactly 3 positive integers, got {value!r}"
    return None


def _kernel_size_error(value):
    if type(value) is not int or value < 1 or value % 2 == 0:
        return f"must be an odd positive integer, got {_type_name(value)}"
    return None


def _head_slowness_error(value):
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        return f"must be null or a nonempty list of slownesses in ms/m, got {_type_name(value)}"
    for item in value:
        if (hint := _number_string_hint(item)) is not None:
            return hint
    if not all(_is_real(item) and item > 0 for item in value):
        return f"every slowness must be a finite number > 0 (ms/m), got {value!r}"
    if len(set(value)) != len(value):
        return f"slownesses must be distinct, got {value!r}"
    return None


def _output_band_error(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return f"must be null or [lo, hi] in Hz, got {_type_name(value)}"
    for item in value:
        if (hint := _number_string_hint(item)) is not None:
            return hint
    if not all(_is_real(item) for item in value) or not BAND_LOW_RAMP_HZ < value[0] < value[1]:
        return f"needs finite {BAND_LOW_RAMP_HZ} < lo < hi (Hz), got {value!r}"
    return None


def _weight_decay_after_error(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"step", "value"}:
        return f"must be null or a mapping with exactly step and value, got {value!r}"
    if (message := _integer_rule(1)(value["step"])) is not None:
        return f"step {message}"
    if (message := _real_rule(minimum=0, exclusive=False)(value["value"])) is not None:
        return f"value {message}"
    return None


def _ema_decay_error(value):
    if value is None:
        return None
    if (hint := _number_string_hint(value)) is not None:
        return hint
    if _is_real(value) and value == 0:
        return "0 would silently disable EMA; use null to disable EMA"
    try:
        validate_trace_graph_ema_decay(value)
    except ValueError as error:
        return f"{error}, got {value!r}"
    return None


def _same_value(old, new) -> bool:
    if isinstance(old, (list, np.ndarray)) or isinstance(new, (list, np.ndarray)):
        if old is None or new is None:
            return False
        return np.array_equal(np.asarray(old), np.asarray(new))
    return old == new


def _describe(value) -> str:
    array = np.asarray(value) if isinstance(value, list) else None
    if array is not None and array.ndim == 2:
        return f"{array.shape[0]}x{array.shape[1]} matrix"
    if array is not None and array.ndim == 1 and array.dtype.kind == "f":
        return "[" + ", ".join(f"{item:.6g}" for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return "null" if value is None else str(value)
