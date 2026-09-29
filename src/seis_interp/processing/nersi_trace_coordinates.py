"""Geometry-only affine coordinates for trace-wise NeRSI Fourier features."""

import math

import numpy as np


def midpoint_offset_transform(geometry, spatial_shape, *, polar=False):
    """Return an exact affine map from local indices to normalized CMP/half-offset.

    The map is accepted only when the input coordinates reproduce the actual
    source/receiver geometry. Geometry bounds include all eligible cells; no
    waveform values or observation roles are used.
    """
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    local = geometry[columns].to_numpy(dtype=np.float64)
    local /= np.maximum(np.asarray(spatial_shape) - 1, 1)
    source = geometry[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64)
    receiver = geometry[["receiver_x_m", "receiver_y_m"]].to_numpy(dtype=np.float64)
    physical = np.column_stack(((source + receiver) / 2, (receiver - source) / 2))
    if not np.isfinite(local).all() or not np.isfinite(physical).all() or not len(local):
        raise ValueError("finite nonempty coordinates required")
    lower = physical.min(axis=0)
    span = np.ptp(physical, axis=0)
    normalized = (physical - lower) / np.where(span > 0, span, 1)
    if polar:
        # Equal scale and zero origin preserve physical distances and azimuths.
        maximum_radius = np.linalg.norm(physical[:, 2:], axis=1).max()
        normalized[:, 2:] = physical[:, 2:] / (maximum_radius or 1)
    design = np.column_stack((local, np.ones(len(local))))
    transform = np.linalg.lstsq(design, normalized, rcond=None)[0].T
    if not np.allclose(design @ transform.T, normalized, rtol=0, atol=1e-8):
        raise ValueError("actual midpoint/offset geometry is not affine in the local coordinates")
    return transform


def linear_moveout_samples_per_radius(geometry, velocity_m_per_s, sample_interval_s):
    """Return linear-moveout samples per unit polar radius for one velocity.

    The unit radius is the largest actual half-offset, the same radius that
    ``midpoint_offset_transform(..., polar=True)`` normalizes by, so a trace at
    normalized radius ``r`` is delayed by ``r * 2 * R_max / (velocity * dt)``.
    """
    velocity = float(velocity_m_per_s)
    if not math.isfinite(velocity) or velocity <= 0:
        raise ValueError("moveout velocity must be finite and positive")
    interval = _sample_interval(sample_interval_s)
    maximum_radius = _maximum_half_offset(geometry, "moveout")
    return 2 * maximum_radius / (velocity * interval)


def head_samples_per_radius(
    geometry, slowness_ms_per_m, sample_interval_s, alignment_velocity_m_per_s=None
):
    """Signed decoder shifts per unit polar radius for absolute apparent slownesses.

    Head ``k`` reads the decoder at ``t + m_k * r`` on a canvas already aligned
    at ``p_align = 1 / alignment velocity`` (0 without alignment), so an event
    with raw slowness ``p_k`` is flat in that head:
    ``m_k = (p_align - p_k) * 2 * R_max / dt``. Slownesses stay in ms/m until the
    final ``1e-3`` factor.
    """
    interval = _sample_interval(sample_interval_s)
    maximum_radius = _maximum_half_offset(geometry, "slowness heads")
    aligned = 0.0
    if alignment_velocity_m_per_s is not None:
        velocity = float(alignment_velocity_m_per_s)
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError("alignment velocity must be finite and positive")
        aligned = 1e3 / velocity
    shifts = []
    for value in slowness_ms_per_m:
        slowness = float(value)
        if not math.isfinite(slowness) or slowness <= 0:
            raise ValueError("head slownesses must be finite and positive")
        shifts.append((aligned - slowness) * 1e-3 * 2 * maximum_radius / interval)
    return shifts


def slowness_samples_per_radius(geometry, slowness_ms_per_m, sample_interval_s):
    """Decoder samples per unit polar radius for a slowness increment in ms/m."""
    slowness = float(slowness_ms_per_m)
    if not math.isfinite(slowness) or slowness < 0:
        raise ValueError("slowness increment must be finite and nonnegative")
    interval = _sample_interval(sample_interval_s)
    maximum_radius = _maximum_half_offset(geometry, "slowness")
    return slowness * 1e-3 * 2 * maximum_radius / interval


def _sample_interval(sample_interval_s):
    interval = float(sample_interval_s)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("sample interval must be finite and positive")
    return interval


def _maximum_half_offset(geometry, purpose):
    source = geometry[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64)
    receiver = geometry[["receiver_x_m", "receiver_y_m"]].to_numpy(dtype=np.float64)
    maximum_radius = float(np.linalg.norm((receiver - source) / 2, axis=1).max(initial=0.0))
    if not math.isfinite(maximum_radius) or maximum_radius == 0:
        raise ValueError(f"{purpose} requires a nonzero finite maximum half-offset")
    return maximum_radius
