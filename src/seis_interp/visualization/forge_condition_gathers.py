"""Consistently scaled gather comparisons with explicit observed/test columns."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def gather_panels(target, prediction, observed, display_scale):
    """Normalize all panels by one reference scale; retain observed data exactly."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    scale = np.asarray(display_scale, dtype=float)
    if target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("target and prediction must be matching [trace, time] arrays")
    if observed.shape != (len(target),) or scale.shape != (len(target),):
        raise ValueError("observed and display scale must have one entry per trace")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("display scales must be finite and positive")
    truth = target / scale[:, None]
    reconstructed = np.where(observed[:, None], target, prediction) / scale[:, None]
    return {
        "Ground truth": truth,
        "Observed input (missing = 0)": np.where(observed[:, None], truth, 0),
        "Reconstruction (observed retained)": reconstructed,
        "Error (reconstruction - truth)": reconstructed - truth,
    }


def draw_gather(ax, values, time_s, cells, *, clip, error=False):
    """Keep the declared cell order, marking each 32-point survey-line boundary."""
    image = ax.imshow(
        values.T,
        extent=(cells[0] - 0.5, cells[-1] + 0.5, time_s[-1], time_s[0]),
        aspect="auto",
        interpolation="nearest",
        cmap="seismic" if error else "gray",
        vmin=-clip,
        vmax=clip,
    )
    for boundary in np.arange(32, cells[-1] + 1, 32):
        ax.axvline(boundary - 0.5, color="#2086ba", lw=0.5, alpha=0.65)
    ax.set_xlabel("Receiver cell" if len(cells) == 256 else "Source cell")
    ax.set_ylabel("Time (s)")
    return image


def plot_condition_gather(
    target,
    prediction,
    observed,
    time_s,
    cells,
    display_scale,
    output: Path,
    *,
    title,
    scale_label,
    clip,
):
    """Plot input, completed gather and residual with one common amplitude scale."""
    panels = gather_panels(target, prediction, observed, display_scale)
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), layout="constrained", sharey=True)
    for ax, (name, values) in zip(axes, panels.items(), strict=True):
        plot = draw_gather(ax, values, time_s, cells, clip=clip, error=name.startswith("Error"))
        ax.set_title(name, fontsize=11)
        strip = ax.inset_axes([0, 1.015, 1, 0.018])
        strip.imshow(
            np.asarray(observed)[None, :],
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["#e69f00", "#009e73"]),
            vmin=0,
            vmax=1,
        )
        strip.set_axis_off()
        fig.colorbar(plot, ax=ax, shrink=0.65, pad=0.01, label=scale_label)
    fig.suptitle(
        f"{title}\nGreen: observed; orange: predicted missing; blue lines: survey-line boundaries"
        f" | shared clip +/-{clip:.3g}",
        fontsize=13,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)
