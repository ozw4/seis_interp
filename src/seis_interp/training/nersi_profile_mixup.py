"""Construct observed-only mixtures between neighboring NeRSI profiles."""

from numbers import Real

import numpy as np

from seis_interp.training.c3_volume_nersi_data import C3VolumeNersiData


def observed_neighbor_profile_mixup(
    data: C3VolumeNersiData,
    selected: np.ndarray,
    *,
    max_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic profiles using only the two parents' common O traces.

    Each selected profile is paired with one adjacent receiver-x profile. The
    coordinate and complete waveform are mixed with the same uniform fraction.
    Profiles with no common observed trace produce no synthetic training row.
    This assumes local amplitude smoothness; it is not an exact wave symmetry.
    """
    if not isinstance(data, C3VolumeNersiData):
        raise TypeError("data must be C3VolumeNersiData")
    if (
        isinstance(max_fraction, bool)
        or not isinstance(max_fraction, Real)
        or not np.isfinite(max_fraction)
        or not 0 < max_fraction <= 0.5
    ):
        raise ValueError("max_fraction must be finite and in (0, 0.5]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    if (
        not isinstance(selected, np.ndarray)
        or selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or np.any(selected >= len(data.normalized_coordinates))
        or np.any(selected < 0)
    ):
        raise ValueError("selected must contain valid integer profile indices")
    coordinates = []
    profiles = []
    masks = []
    receiver_x_count = data.spatial_shape[2]
    for index in selected:
        receiver_x = int(index) % receiver_x_count
        neighbors = []
        if receiver_x > 0:
            neighbors.append(int(index) - 1)
        if receiver_x + 1 < receiver_x_count:
            neighbors.append(int(index) + 1)
        if not neighbors:
            continue
        neighbor = int(rng.choice(neighbors))
        fraction = float(rng.uniform(0.0, max_fraction))
        mask = data.observed_trace_mask[index] & data.observed_trace_mask[neighbor]
        if not mask.any():
            continue
        first = data.normalized_profiles[index, 0, :, mask]
        second = data.normalized_profiles[neighbor, 0, :, mask]
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("mixup parent observed traces must be finite")
        profile = np.zeros_like(data.normalized_profiles[index])
        profile[0, :, mask] = (1 - fraction) * first + fraction * second
        coordinates.append(
            (1 - fraction) * data.normalized_coordinates[index]
            + fraction * data.normalized_coordinates[neighbor]
        )
        profiles.append(profile)
        masks.append(mask)
    count = len(profiles)
    return (
        np.asarray(coordinates, dtype=data.normalized_coordinates.dtype).reshape(count, 3),
        np.asarray(profiles, dtype=data.normalized_profiles.dtype).reshape(
            count, 1, *data.profile_shape
        ),
        np.asarray(masks, dtype=np.bool_).reshape(count, data.profile_shape[-1]),
    )
