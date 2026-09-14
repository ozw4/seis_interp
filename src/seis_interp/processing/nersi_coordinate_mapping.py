"""Geometry-only profile anchors and lattice Nyquist limits for NeRSI."""

import numpy as np


def nersi_profile_encoding(index_table, spatial_shape, *, cartesian=False, fractions=None):
    """Represent each profile by CMP-x, first-receiver CMP-y, and half-offset-x.

    Receiver-y remains the decoder axis: its half-offset-y and CMP-y increment
    are recorded explicitly. This is a profile parameterization, not a point INR.
    Irregular source placement is preserved without fitting an affine approximation.
    """
    shape = tuple(spatial_shape)
    grid = np.indices(shape[:3]).reshape(3, -1).T.astype(np.float64)
    counts = np.array(shape[:3])
    normalized = grid / np.maximum(counts - 1, 1)
    encoding = {}
    if cartesian:
        columns = ["source_x_m", "source_y_m", "relative_receiver_x_m", "relative_receiver_y_m"]
        raw = index_table[columns].to_numpy(dtype=np.float64)
        if raw.shape != (int(np.prod(shape)), 4) or not np.isfinite(raw).all():
            raise ValueError("profile coordinates require complete finite geometry in volume order")
        raw = raw.reshape(*shape, 4)
        anchor = raw[..., 0, :].reshape(-1, 4)
        if not np.allclose(raw[..., :3], raw[..., :1, :3], rtol=0, atol=1e-8):
            raise ValueError("source and receiver-x geometry must be constant within each profile")
        hy = raw[..., 3] / 2
        if not np.allclose(hy, hy[0, 0, 0], rtol=0, atol=1e-8):
            raise ValueError("half-offset-y must share a fixed decoder grid across profiles")
        if not np.allclose(np.diff(hy[0, 0, 0]), np.diff(hy[0, 0, 0])[0], rtol=0, atol=1e-8):
            raise ValueError("half-offset-y must be regularly sampled")
        physical = np.column_stack(
            (anchor[:, 0] + anchor[:, 2] / 2, anchor[:, 1] + anchor[:, 3] / 2, anchor[:, 2] / 2)
        )
        lower = physical.min(axis=0)
        upper = physical.max(axis=0)
        target = (physical - lower) / np.where(upper > lower, upper - lower, 1)
        normalized = target
        encoding["coordinate_mapping"] = {
            "normalized_anchors": normalized.tolist(),
            "profile_grid_shape": list(shape[:3]),
            "axis_order": ["cmp_x_m", "cmp_y_at_first_receiver_m", "half_offset_x_m"],
            "physical_min": lower.tolist(),
            "physical_max": upper.tolist(),
            "decoder_half_offset_y_m": hy[0, 0, 0].tolist(),
        }
    if fractions is not None:
        lattice = normalized.reshape(*shape[:3], 3)
        maximum_steps = np.zeros(3)
        for axis, count in enumerate(shape[:3]):
            if count > 1:
                maximum_steps = np.maximum(
                    maximum_steps, np.abs(np.diff(lattice, axis=axis)).reshape(-1, 3).max(0)
                )
        encoding["axis_frequency_limits"] = (
            np.pi * np.asarray(fractions) / np.where(maximum_steps > 0, maximum_steps, np.inf)
        ).tolist()
    return encoding
