"""Publication-sized seismic sections with shared axes and amplitude scale."""

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np

FIGURE_STYLE = {
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "grid.linewidth": 0.5,
}


def create_publication_section(
    panels: Sequence[np.ndarray],
    *,
    time_s: np.ndarray,
    spatial_label: str,
    clip: float,
    font_family: str = "Arial",
    panel_titles: Sequence[str] | None = None,
    display: str = "image",
    columns: int = 2,
    show_colorbar: bool = True,
    layout: str = "grid",
):
    """Return a 178 mm figure with panels supplied in row-major order.

    Images use interpolation='None' and one symmetric color scale.
    Panel letters sit outside the upper-left corner, beside optional subtitles.
    """
    import matplotlib.pyplot as plt

    if display not in {"image", "wiggle"}:
        raise ValueError("display must be image or wiggle")
    if layout not in {"grid", "2-5-5"} or (layout == "2-5-5" and len(panels) != 12):
        raise ValueError("layout must be grid or 2-5-5 with twelve panels")
    if not panels or len(panels) % 2:
        raise ValueError("panels must contain a nonempty even number of sections")
    if columns not in {2, 6} or len(panels) % columns:
        raise ValueError("columns must be 2 or 6 and divide the panel count")
    if panel_titles is not None and len(panel_titles) != len(panels):
        raise ValueError("panel_titles must match the number of panels")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    time_s = np.asarray(time_s)
    shape = np.asarray(panels[0]).shape
    if len(shape) != 2 or any(np.asarray(panel).shape != shape for panel in panels):
        raise ValueError("panels must be two-dimensional sections of equal shape")
    if time_s.ndim != 1 or len(time_s) != shape[0] or len(time_s) < 2:
        raise ValueError("time_s must match the sections and contain at least two samples")
    rows = len(panels) // columns
    with mpl.rc_context({**FIGURE_STYLE, "font.family": font_family, "pdf.fonttype": 42}):
        if layout == "2-5-5":
            figure = plt.figure(figsize=(178 / 25.4, 142 / 25.4))
            grid = figure.add_gridspec(
                3, 1, left=0.045, right=0.995, top=0.935, bottom=0.125, hspace=0.27
            )
            plot_axes = []
            for row, count in enumerate((2, 5, 5)):
                row_grid = grid[row].subgridspec(1, 5, wspace=0.20)
                for column in range(count):
                    first = plot_axes[0] if plot_axes else None
                    plot_axes.append(
                        figure.add_subplot(row_grid[column], sharex=first, sharey=first)
                    )
            y_label_indices = {0, 2, 7}
            x_label_indices = {7, 8, 9, 10, 11}
            for index, axis in enumerate(plot_axes):
                axis.tick_params(
                    labelleft=index in y_label_indices,
                    labelbottom=index < 2 or index in x_label_indices,
                )
        else:
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(178 / 25.4, ((44 if columns == 6 else 36) * rows + 24) / 25.4),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            plot_axes = list(axes.flat)
            y_label_indices = set(range(0, len(panels), columns))
            x_label_indices = set(range(len(panels) - columns, len(panels)))
        if layout == "2-5-5":
            pass  # The nested grids above own this layout's margins.
        elif columns == 6 and display == "image" and not show_colorbar:
            figure.subplots_adjust(
                left=0.05, right=0.995, top=0.91, bottom=0.15, hspace=0.36, wspace=0.28
            )
        elif columns == 6:
            figure.subplots_adjust(
                left=0.08,
                right=0.99,
                top=0.87,
                bottom=0.20 if display == "wiggle" else 0.27,
                hspace=0.60,
                wspace=0.40,
            )
        else:
            figure.subplots_adjust(
                left=0.09, right=0.985, top=0.965, bottom=0.12, hspace=0.26, wspace=0.13
            )
        step = float(time_s[1] - time_s[0])
        extent = (-0.5, shape[1] - 0.5, time_s[-1] + step / 2, time_s[0] - step / 2)
        for index, (axis, values) in enumerate(zip(plot_axes, panels, strict=True)):
            if display == "wiggle":
                _plot_wiggle_traces(axis, values, time_s, clip)
                axis.set_xlim(extent[:2])
                axis.set_ylim(extent[2:])
            else:
                artist = axis.imshow(
                    values,
                    cmap="seismic",
                    vmin=-clip,
                    vmax=clip,
                    aspect="auto",
                    extent=extent,
                    interpolation="None",
                    rasterized=True,
                )
            tick_count = min(shape[1], 3 if columns == 6 or layout == "2-5-5" else 6)
            axis.set_xticks(np.unique(np.linspace(0, shape[1] - 1, tick_count, dtype=int)))
            axis.tick_params(axis="both", which="both", length=0)
            axis.minorticks_off()
            if index in y_label_indices:
                axis.set_ylabel("Time (s)", labelpad=2)
            if index in x_label_indices:
                axis.set_xlabel(spatial_label)
            if panel_titles is not None:
                axis.set_title(panel_titles[index], pad=6)
            axis.annotate(
                f"({chr(97 + index)})",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(1, 6),
                textcoords="offset points",
                va="bottom",
                ha="right",
                annotation_clip=False,
                fontsize=10,
                fontweight="bold",
            )
        if display == "image" and show_colorbar:
            color_axis = figure.add_axes((0.30, 0.105 if columns == 6 else 0.047, 0.48, 0.010))
            colorbar = figure.colorbar(
                artist, cax=color_axis, orientation="horizontal", extend="both"
            )
            colorbar.set_label("Amplitude", fontsize=9)
            colorbar.set_ticks([-clip, 0, clip], labels=[f"−{clip:.2f}", "0", f"{clip:.2f}"])
    return figure


def _plot_wiggle_traces(axis, values: np.ndarray, time_s: np.ndarray, clip: float) -> None:
    """Draw each trace with fixed gain, display clipping, and positive-lobe fill."""
    for position, trace in enumerate(np.asarray(values).T):
        offset = 0.45 * (np.clip(trace, -clip, clip) / clip)
        x = position + offset
        axis.plot(x, time_s, color="black", linewidth=0.35)
        axis.fill_betweenx(
            time_s,
            position,
            x,
            where=offset > 0,
            interpolate=True,
            color="black",
            linewidth=0,
        )
