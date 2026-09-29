"""Geometry-only affine coordinates for trace-wise NeRSI Fourier features."""

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
