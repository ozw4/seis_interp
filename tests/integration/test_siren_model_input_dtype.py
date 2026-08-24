from __future__ import annotations

import numpy as np
import torch

from seis_interp.models import Siren
from seis_interp.training.model_inputs import to_model_tensors


def test_interim_style_float64_coordinates_reach_the_siren() -> None:
    coordinates = np.array(
        [
            [0.000, 1000.0, 2000.0, 500.0, 90.0],
            [0.008, 1010.0, 2000.0, 520.0, 91.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([0.1, -0.2], dtype=np.float32)

    coordinate_tensor, target_tensor = to_model_tensors(coordinates, targets)
    torch.manual_seed(0)
    model = Siren(input_features=5, hidden_width=8, hidden_layers=1, output_features=1)

    prediction = model(coordinate_tensor)

    assert prediction.shape == (2, 1)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert target_tensor.shape == prediction.shape
    assert target_tensor.dtype == prediction.dtype
