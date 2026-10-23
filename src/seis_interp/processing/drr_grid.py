"""Resolve the DRR grid's backend and boundary rule from its effective overlap."""

from copy import deepcopy

from seis_interp.processing.drr_windows import _validated_window_spec


def resolve_drr_grid_config(config):
    """Make the legacy zero-overlap and overlapping computations explicit."""
    resolved = deepcopy(config)
    algorithm = resolved["algorithm"]
    policy = algorithm.setdefault("overlap_policy", "explicit")
    if policy not in ("explicit", "point_half"):
        raise ValueError("DRR overlap_policy must be explicit or point_half")
    window, overlap = _validated_window_spec(
        algorithm["spatial_window_shape"], algorithm["spatial_overlap"]
    )
    if policy == "point_half":
        if window is None:
            raise ValueError("point_half overlap requires spatial_window_shape")
        overlap = (0, window[1] // 2, 0, window[3] // 2)
    overlapping = overlap is not None and any(overlap)
    backend = "drr_batched" if overlapping else "drr_nonoverlapping"
    boundary = "clip_and_shift" if overlapping else "unknown_padding"
    for key, expected in (("backend", backend), ("boundary_rule", boundary)):
        if key in algorithm and algorithm[key] != expected:
            raise ValueError(f"DRR {key} must be {expected} for the effective overlap")
        algorithm[key] = expected
    if overlap is not None:
        algorithm["spatial_overlap"] = list(overlap)
    resolved.setdefault("execution", {}).setdefault("batch_matrix_limit", 2_000_000)
    return resolved
