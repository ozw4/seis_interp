"""Static, explicitly scaled diagnostic plots for FORGE waveform QC."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gather(values: np.ndarray, headers: pd.DataFrame, dt: float, output: Path) -> None:
    """Show all candidate traces plus near/middle/far receiver-line sections."""
    valid = headers.header_eligible.to_numpy() & np.isfinite(values).all(axis=1)
    rows = np.flatnonzero(valid)
    x = values.astype(float)
    rms = np.sqrt(np.mean(x**2, axis=1))
    normalized = np.divide(x, rms[:, None], out=np.zeros_like(x), where=rms[:, None] > 0)
    line_order = headers.loc[valid].groupby("receiver_line").offset_m.median().sort_values()
    lines = line_order.index[sorted({0, len(line_order) // 2, len(line_order) - 1})]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].imshow(
        normalized[rows, ::4].T,
        cmap="gray",
        vmin=-3,
        vmax=3,
        aspect="auto",
        extent=[0, len(rows), (x.shape[1] - 1) * dt, 0],
    )
    axes[0, 0].set(title="All candidate traces; per-trace RMS display", xlabel="Trace order")
    for ax, line in zip(axes.flat[1:], lines, strict=False):
        subset = headers[valid & headers.receiver_line.eq(line)].sort_values("receiver_point")
        indices = subset.index.to_numpy()
        ax.imshow(
            normalized[indices, ::2].T,
            cmap="gray",
            vmin=-3,
            vmax=3,
            aspect="auto",
            extent=[
                subset.receiver_point.min() - 0.5,
                subset.receiver_point.max() + 0.5,
                (x.shape[1] - 1) * dt,
                0,
            ],
        )
        ax.set(
            title=f"Receiver line {int(line)}; RMS display",
            xlabel="Receiver point (gaps compressed if present)",
        )
    for ax in axes.flat:
        ax.set_ylabel("Time (s)")
    fig.suptitle(f"FFID {int(headers.ffid.iloc[0])}; display only, no filtering or AGC")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_qc_overview(table: pd.DataFrame, spectra: dict, output: Path) -> None:
    """Compare physical stored amplitudes, time energy and two spectral weightings."""
    candidate = table[table.header_eligible & table.numerically_usable]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].hist(np.log10(candidate.rms), bins=120)
    axes[0, 0].set(xlabel="log10(RMS), stored amplitude units", ylabel="Trace count")
    for line, group in candidate.groupby("source_line"):
        med = group.groupby("ffid").rms.median()
        axes[0, 1].plot(med.index, med, ".", label=str(int(line)), markersize=3)
    axes[0, 1].set(yscale="log", xlabel="FFID", ylabel="Median trace RMS")
    axes[0, 1].legend(title="Source line")
    f = spectra["frequency_hz"]
    for name, label in [
        ("power_sum", "Energy weighted"),
        ("normalized_power_sum", "Equal trace weight"),
    ]:
        power = spectra[name]
        db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
        axes[1, 0].plot(f, db, label=label)
    axes[1, 0].set(
        xlim=(0, 150), ylim=(-100, 1), xlabel="Frequency (Hz)", ylabel="Power / maximum (dB)"
    )
    axes[1, 0].legend()
    rms = np.sqrt(spectra["time_energy_sum"] / spectra["finite_trace_count"])
    axes[1, 1].plot(np.arange(len(rms)) * spectra["dt_s"], rms)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Ensemble RMS (stored units)", yscale="log")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_review_traces(
    values: np.ndarray, headers: pd.DataFrame, rows: list[int], dt: float, output: Path
) -> None:
    """Raw amplitude time series for flagged/representative traces, never rescaled."""
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(12, 2.1 * len(rows)), constrained_layout=True, squeeze=False
    )
    for ax, row in zip(axes[:, 0], rows, strict=True):
        ax.plot(np.arange(values.shape[1]) * dt, values[row], linewidth=0.6)
        h = headers.iloc[row]
        ax.set(
            title=f"FFID {int(h.ffid)}, trace_index {row}, receiver "
            f"{int(h.receiver_line)}/{int(h.receiver_point)}",
            ylabel="Stored amplitude",
            xlabel="Time (s)",
        )
    fig.savefig(output, dpi=150)
    plt.close(fig)
