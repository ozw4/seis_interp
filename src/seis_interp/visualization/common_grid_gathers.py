"""Central common-grid gathers: completed predictions and residuals, geometry-only."""

import numpy as np

AXES = ("source_grid_line", "source_grid_point", "receiver_grid_line", "receiver_grid_point")
LABELS = {
    "source_grid_line": "Source line\nindex",
    "source_grid_point": "Source point\nindex",
    "receiver_grid_line": "Receiver line\nindex",
    "receiver_grid_point": "Receiver point\nindex",
}
# The proposed method is shown last, as in the earlier five-method figures.
DISPLAY_METHODS = ("pocs_grid", "drr_grid", "nersi_profile", "ccnet5d_grid", "regsi_real")
METHOD_TITLES = ("POCS", "DRR", "NeRSI", "CCNet-5D", "ReGSI")
# The neural-implicit panel reads observed traces instead of fitting weights alone.
FIELD_DISPLAY_METHODS = ("pocs_grid", "drr_grid", "cinr_real", "ccnet5d_grid", "regsi_real")
FIELD_METHOD_TITLES = ("POCS", "DRR", "cINR", "CCNet-5D", "ReGSI")


def panel_titles(method_titles=METHOD_TITLES, *, residuals=True):
    """Row titles for one five-method layout: inputs, predictions, then residuals.

    Without residuals the figure keeps the same panels in the same order, minus the
    residual row, so the seven remaining panels fit two rows at nearly twice the area.
    """
    titles = tuple(method_titles)
    if len(titles) != 5:
        raise ValueError("the publication layout shows exactly five methods")
    return (
        "Reference",
        "Masked\ninput",
        *titles,
        *((f"Residual\n({title})" for title in titles) if residuals else ()),
    )


PANEL_TITLES = panel_titles()
SECTION_AXES = {"c3": (3, 1, 0), "forge": (3, 1, 0, 2)}


def central_sections(cell_ids, spatial_shape, axes):
    """Complete central slices chosen from the grid shape alone, in the given axis order."""
    shape = tuple(int(size) for size in spatial_shape)
    if len(shape) != len(AXES):
        raise ValueError("spatial_shape must have four grid axes")
    present = set(np.asarray(cell_ids).tolist())
    centers = np.asarray(shape) // 2
    sections = []
    for axis in axes:
        coordinates = np.tile(centers, (shape[axis], 1))
        coordinates[:, axis] = np.arange(shape[axis])
        cells = np.ravel_multi_index(tuple(coordinates.T), shape)
        if any(int(cell) not in present for cell in cells):
            raise ValueError(f"central {AXES[axis]} gather contains missing grid cells")
        sections.append(
            {
                "varying_axis": AXES[axis],
                "fixed_local_indices": {
                    name: int(centers[i]) for i, name in enumerate(AXES) if i != axis
                },
                "cell_ids": [int(cell) for cell in cells],
            }
        )
    return sections


def completed_section_panels(
    reference,
    observed,
    predictions,
    *,
    normalization,
    display_methods=DISPLAY_METHODS,
    residuals=True,
):
    """Reference, masked input, completed predictions, then reference minus completion.

    Arrays are [trace, sample] for one section. Observed traces keep the
    reference, so their residual is zero; hidden traces come from the saved
    physical predictions. The RMS view divides every panel by the same
    full-duration reference RMS per trace. Panels are returned as [sample, trace].
    """
    if normalization not in {"physical", "trace_rms"}:
        raise ValueError("normalization must be physical or trace_rms")
    display_methods = tuple(display_methods)
    if tuple(predictions) != display_methods:
        raise ValueError(f"predictions must follow the display order {display_methods}")
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if reference.ndim != 2 or observed.shape != (len(reference),):
        raise ValueError("reference must be [trace, sample] with one observed flag per trace")
    if any(np.shape(values) != reference.shape for values in predictions.values()):
        raise ValueError("predictions must match the reference section")
    scale = np.ones((len(reference), 1))
    if normalization == "trace_rms":
        rms = np.sqrt(np.mean(np.square(reference), axis=1, keepdims=True))
        scale = np.where(rms > 0, rms, 1.0)
    truth = reference / scale
    completed = [
        np.where(observed[:, None], reference, np.asarray(predictions[method], dtype=np.float64))
        / scale
        for method in display_methods
    ]
    if not all(np.isfinite(values).all() for values in completed):
        raise ValueError("hidden traces need finite predictions")
    return [
        truth.T,
        np.where(observed[:, None], truth, 0.0).T,
        *(values.T for values in completed),
        *(((truth - values).T for values in completed) if residuals else ()),
    ]


def pooled_reference_clip(panel_sets):
    """One symmetric clip: the 99th percentile of |reference| pooled over sections."""
    values = np.concatenate([np.abs(panels[0]).ravel() for panels in panel_sets])
    clip = float(np.percentile(values, 99))
    return clip if clip > 0 else 1.0
