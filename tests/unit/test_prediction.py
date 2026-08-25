from __future__ import annotations

import numpy as np
import torch

from seis_interp.models.siren import Siren
from seis_interp.training.prediction import predict_points


def test_chunked_prediction_matches_direct_prediction_and_restores_training_mode() -> None:
    torch.manual_seed(2)
    model = Siren(hidden_width=8, hidden_layers=1)
    coordinates = np.arange(66, dtype=np.float64).reshape(11, 6) / 20.0
    expected = model(torch.as_tensor(coordinates, dtype=torch.float32)).detach().numpy().reshape(-1)

    actual = predict_points(model, coordinates, batch_size=4, device="cpu")

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (11,)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_prediction_preserves_eval_mode_and_accepts_large_batch() -> None:
    model = Siren(hidden_width=8, hidden_layers=1).eval()
    coordinates = np.zeros((3, 6), dtype=np.float64)

    assert predict_points(model, coordinates, batch_size=20, device="cpu").shape == (3,)
    assert not model.training
