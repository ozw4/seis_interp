"""Batched model prediction for coordinate points."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import torch

from seis_interp.training.model_inputs import to_model_coordinate_tensor


def predict_points(
    model: torch.nn.Module,
    coordinates: np.ndarray,
    *,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Predict coordinate points in bounded batches while preserving order."""
    coordinate_array = np.asarray(coordinates)
    if coordinate_array.ndim != 2:
        raise ValueError(f"coordinates must be two-dimensional, got {coordinate_array.shape}")
    model_input_features = getattr(model, "input_features", None)
    if model_input_features is not None and coordinate_array.shape[1] != model_input_features:
        raise ValueError(
            f"model expects {model_input_features} input features but coordinates have "
            f"{coordinate_array.shape[1]}"
        )
    if coordinate_array.shape[0] == 0:
        raise ValueError("coordinates must not be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    was_training = model.training
    predictions: list[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(coordinate_array), int(batch_size)):
                batch = to_model_coordinate_tensor(
                    coordinate_array[start : start + int(batch_size)], device=device
                )
                output = model(batch)
                if output.ndim != 2 or output.shape != (len(batch), 1):
                    raise ValueError(
                        f"model output must have shape (n_points, 1), got {tuple(output.shape)}"
                    )
                predictions.append(
                    output.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
                )
    finally:
        model.train(was_training)

    return np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
