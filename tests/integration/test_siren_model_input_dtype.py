from __future__ import annotations

import numpy as np
import torch

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models import Siren
from seis_interp.training.model_inputs import to_model_tensors


def test_float64_encoded_model_coordinates_reach_the_default_siren() -> None:
    coordinates = np.array(
        [
            [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, -1.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([0.1, -0.2], dtype=np.float32)

    coordinate_tensor, target_tensor = to_model_tensors(coordinates, targets)
    torch.manual_seed(0)
    model = Siren(hidden_width=8, hidden_layers=1, output_features=1)

    prediction = model(coordinate_tensor)

    assert coordinates.shape[1] == len(MODEL_COORDINATE_ORDER) == 6
    assert coordinate_tensor.shape == (2, 6)
    assert prediction.shape == (2, 1)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert target_tensor.shape == prediction.shape
    assert target_tensor.dtype == prediction.dtype
