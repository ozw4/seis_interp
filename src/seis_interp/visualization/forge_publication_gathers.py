"""Fixed central FORGE gathers with reference-scaled predictions and residuals."""

import numpy as np
import pandas as pd

from seis_interp.processing.forge_mvp_contract import METHODS

AXES = ("source_grid_line", "source_grid_point", "receiver_grid_line", "receiver_grid_point")
LABELS = {
    "source_grid_line": "Source line\nindex",
    "source_grid_point": "Source point\nindex",
    "receiver_grid_line": "Receiver line\nindex",
    "receiver_grid_point": "Receiver point\nindex",
}
METHOD_TITLES = ("POCS", "DRR", "NeRSI", "CCNet-5D", "ReGSI\n(real)", "ReGSI\n(grid)")
DISPLAY_METHODS = (
    "pocs_grid",
    "drr_grid",
    "nersi_real",
    "ccnet5d_grid",
    "regsi_real",
    "regsi_grid",
)
PANEL_TITLES = (
    "Reference",
    "Masked\ninput",
    *METHOD_TITLES,
    "Residual\n(POCS)",
    "Residual\n(DRR)",
    "Residual\n(NeRSI)",
    "Residual\n(CCNet-5D)",
    "Residual\n(ReGSI-real)",
    "Residual\n(ReGSI-grid)",
)


def central_gather_cells(index, spatial_shape):
    """Select four complete central slices using geometry alone, in display order."""
    centers = np.asarray(spatial_shape) // 2
    coordinates = np.asarray(np.unravel_index(index.cell_id.to_numpy(), spatial_shape)).T
    sections = []
    for axis in (3, 1, 0, 2):
        keep = np.all(np.delete(coordinates, axis, 1) == np.delete(centers, axis), axis=1)
        positions = np.flatnonzero(keep)
        positions = positions[np.argsort(coordinates[positions, axis])]
        if not np.array_equal(coordinates[positions, axis], np.arange(spatial_shape[axis])):
            raise ValueError(f"central {AXES[axis]} gather contains missing grid cells")
        sections.append(
            {
                "varying_axis": AXES[axis],
                "fixed_local_indices": {
                    name: int(centers[i]) for i, name in enumerate(AXES) if i != axis
                },
                "cell_ids": index.iloc[positions].cell_id.tolist(),
            }
        )
    return sections


def prepare_publication_gathers(index, reference, predictions, sections, *, normalization):
    """Reference/input, six completed gathers, then reference-minus-prediction.

    Predictions retain their physical amplitude. Observed columns are copied from
    the reference; only test columns are filled from saved predictions. The RMS
    view divides every panel by the same full-duration reference RMS per trace.
    One pooled reference-only 99th-percentile clip serves all four gathers.
    """
    if normalization not in {"physical", "trace_rms"}:
        raise ValueError("normalization must be physical or trace_rms")
    if set(predictions) != set(METHODS):
        raise ValueError("predictions must contain all six methods")
    if (
        reference.ndim != 2
        or len(reference) != len(index)
        or any(values.shape != reference.shape for values in predictions.values())
    ):
        raise ValueError("reference and predictions must align with the gather index")
    scale = np.ones((len(reference), 1))
    if normalization == "trace_rms":
        rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=1, keepdims=True))
        scale = np.where(rms > 0, rms, 1.0)
    truth = reference / scale
    observed = index.split.eq("observed").to_numpy()
    completed = {
        method: np.where(observed[:, None], reference, predictions[method]) / scale
        for method in METHODS
    }
    lookup = pd.Index(index.cell_id)
    result = []
    for section in sections:
        rows = lookup.get_indexer(section["cell_ids"])
        if (rows < 0).any():
            raise ValueError("section cells are absent from the gather index")
        target = truth[rows].T
        values = [completed[method][rows].T for method in DISPLAY_METHODS]
        result.append(
            {
                **section,
                "observed_count": int(observed[rows].sum()),
                "test_count": int((~observed[rows]).sum()),
                "panels": [
                    target,
                    np.where(observed[rows][None, :], target, 0),
                    *values,
                    *(target - value for value in values),
                ],
            }
        )
    clip = float(
        np.percentile(np.concatenate([np.abs(s["panels"][0]).ravel() for s in result]), 99)
    )
    return result, clip if clip > 0 else 1.0
