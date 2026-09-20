"""Extract fixed central C3 sections without choosing geometry from amplitudes."""

from collections.abc import Mapping

import numpy as np

SPATIAL_AXES = ("source_line", "shot_in_line", "relative_receiver_x", "relative_receiver_y")


def prepare_central_sections(
    *,
    reference_amplitudes: np.ndarray,
    array_rows: np.ndarray,
    observed_values: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    time_selection: tuple[int, int],
) -> tuple[list[dict], float]:
    """Return reference/input, predictions, residuals in row-major panel order.

    Contrast is the pooled reference 99th percentile over the three sections,
    never a per-method or per-trace normalization. Only selected traces are read.
    """
    if array_rows.ndim != 4:
        raise ValueError("array_rows must have four spatial dimensions")
    start, stop = time_selection
    expected_shape = (stop - start, *array_rows.shape)
    if observed_values.shape != expected_shape or any(
        values.shape != expected_shape for values in predictions.values()
    ):
        raise ValueError("observed values and predictions must cover the selected volume")
    centers = tuple(size // 2 for size in array_rows.shape)
    sections = []
    for axis in (3, 1, 0):
        selection = tuple(
            slice(None) if index == axis else center for index, center in enumerate(centers)
        )
        rows = array_rows[selection]
        reference = np.asarray(reference_amplitudes[rows, start:stop], dtype=np.float64).T
        predicted = [
            np.asarray(values[(slice(None), *selection)], dtype=np.float64)
            for values in predictions.values()
        ]
        sections.append(
            {
                "varying_axis": SPATIAL_AXES[axis],
                "fixed_local_indices": {
                    name: centers[index] for index, name in enumerate(SPATIAL_AXES) if index != axis
                },
                "array_rows": rows.tolist(),
                "panels": [
                    reference,
                    np.asarray(observed_values[(slice(None), *selection)]),
                    *predicted,
                    *(reference - values for values in predicted),
                ],
            }
        )
    clip = float(
        np.percentile(np.concatenate([np.abs(s["panels"][0]).ravel() for s in sections]), 99)
    )
    return sections, clip if clip > 0 else 1.0
