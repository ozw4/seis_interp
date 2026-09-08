"""Deterministic gather and time-energy figures for a fixed C3 crop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_c3_crop_qc(
    amplitudes: np.ndarray, index: pd.DataFrame, time_s: np.ndarray, summary: dict, output_dir: Path
) -> dict[str, object]:
    """Display the first local line/shot/x gather with a fixed 99th-percentile clip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "matplotlib_unavailable"}
    gather = index.loc[
        index["source_line_index"].eq(0)
        & index["shot_in_line_index"].eq(0)
        & index["relative_receiver_x_index"].eq(0)
    ].sort_values("relative_receiver_y_index")
    start, stop = summary["selection"]["time"]
    values = np.asarray(amplitudes[gather["array_row"].to_numpy(), start:stop], dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    clip = float(np.percentile(finite, 99)) if len(finite) else 0.0
    display_clip = clip if clip > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(
        values.T,
        cmap="gray",
        vmin=-display_clip,
        vmax=display_clip,
        aspect="auto",
        extent=[0, len(gather), time_s[stop - 1], time_s[start]],
    )
    axes[0].set(
        xlabel="Local receiver y index", ylabel="Time (s)", title="First line / shot / receiver x"
    )
    energy = summary["signal"]["energy_by_sample"]
    if energy is not None:
        axes[1].plot(time_s[start:stop], energy)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Sum of squared physical amplitudes",
        title="Fixed crop time energy",
    )
    fig.savefig(output_dir / "crop_qc.png", dpi=150)
    plt.close(fig)
    return {
        "status": "created",
        "file": "crop_qc.png",
        "gather_local_indices": [0, 0, 0],
        "clip_percentile": 99,
        "clip_abs_amplitude": clip,
    }
