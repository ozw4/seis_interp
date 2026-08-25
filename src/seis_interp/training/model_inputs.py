"""Convert stored coordinate and amplitude arrays into model input tensors."""

from __future__ import annotations

import numpy as np
import torch


def to_model_coordinate_tensor(
    coordinates: np.ndarray,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return coordinates as a contiguous ``float32`` model input tensor."""
    coordinate_array = np.asarray(coordinates)
    if coordinate_array.ndim != 2:
        raise ValueError(
            f"coordinates must be two-dimensional, got {coordinate_array.ndim} dimensions"
        )
    return torch.as_tensor(coordinate_array, dtype=torch.float32, device=device).contiguous()


def to_model_tensors(
    coordinates: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``coordinates`` and ``targets`` as contiguous ``float32`` tensors.

    ``coordinates`` must have shape ``(n_points, n_features)`` and ``targets``
    either ``(n_points,)`` or ``(n_points, 1)``; the returned targets always
    have shape ``(n_points, 1)`` so that they match the model output. Tensors
    are created on ``device``, or on the CPU when it is ``None``.

    This is the boundary where the ``float64`` model features derived from
    ``data/interim`` become the ``float32`` the model parameters use;
    ``Siren.forward()`` performs no dtype conversion. The inputs are not
    modified, but an input that is already ``float32`` may share its memory
    with the returned tensor.
    """
    target_array = np.asarray(targets)

    coordinate_tensor = to_model_coordinate_tensor(coordinates, device=device)
    if target_array.ndim not in (1, 2) or (target_array.ndim == 2 and target_array.shape[1] != 1):
        raise ValueError(
            f"targets must have shape (n_points,) or (n_points, 1), got {target_array.shape}"
        )
    if coordinate_tensor.shape[0] != target_array.shape[0]:
        raise ValueError(
            f"coordinates and targets must cover the same number of points, got "
            f"{coordinate_tensor.shape[0]} and {target_array.shape[0]}"
        )

    target_tensor = torch.as_tensor(target_array, dtype=torch.float32, device=device)
    return coordinate_tensor, target_tensor.reshape(-1, 1).contiguous()
