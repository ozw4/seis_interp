"""Save and restore self-contained SIREN inference checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters


@dataclass(frozen=True)
class LoadedSirenCheckpoint:
    """A restored SIREN and its training-time inference metadata."""

    model: Siren
    normalization: NormalizationParameters
    epoch: int
    global_step: int
    validation_median_trace_snr_db: float
    validation_global_snr_db: float


def save_siren_checkpoint(
    path: Path,
    model: Siren,
    normalization: NormalizationParameters,
    *,
    epoch: int,
    global_step: int,
    validation_median_trace_snr_db: float,
    validation_global_snr_db: float,
) -> None:
    """Save constructor values, CPU weights, normalization, and best-epoch metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {
        "model_type": "siren",
        "model_config": {
            "input_features": model.input_features,
            "hidden_width": model.hidden_width,
            "hidden_layers": model.hidden_layers,
            "output_features": model.output_features,
            "omega_0": model.omega_0,
            "hidden_omega": model.hidden_omega,
        },
        "model_state_dict": state_dict,
        "normalization": normalization.to_dict(),
        "training": {
            "epoch": epoch,
            "global_step": global_step,
            "validation_median_trace_snr_db": validation_median_trace_snr_db,
            "validation_global_snr_db": validation_global_snr_db,
        },
    }
    torch.save(payload, checkpoint_path)


def load_siren_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedSirenCheckpoint:
    """Rebuild a SIREN from its saved constructor values and strict weights."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("model_type") != "siren":
        raise ValueError("checkpoint model_type must be 'siren'")
    model = Siren(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    normalization = NormalizationParameters.from_dict(payload["normalization"])
    training = payload["training"]
    return LoadedSirenCheckpoint(
        model=model,
        normalization=normalization,
        epoch=training["epoch"],
        global_step=training["global_step"],
        validation_median_trace_snr_db=training["validation_median_trace_snr_db"],
        validation_global_snr_db=training["validation_global_snr_db"],
    )
