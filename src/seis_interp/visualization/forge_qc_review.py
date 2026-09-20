"""Review distributions and raw/centered waveform comparisons without filtering."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_review_distributions(table, stations, output: Path) -> None:
    c = table[table.eligible_after_fixed_qc]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].hist(np.log10(c.std_to_rms), bins=np.linspace(-10, 0.001, 101), log=True)
    axes[0, 0].axvline(-2, color="red", ls="--", label="Diagnostic q < 0.01")
    axes[0, 0].set(xlabel="log10(q): standard deviation / raw RMS", ylabel="Trace count")
    axes[0, 0].legend()
    axes[0, 1].hist(c.dc_to_rms, bins=100, log=True)
    axes[0, 1].axvline(0.1, color="red", ls="--")
    axes[0, 1].set(xlabel="abs(mean) / raw RMS", ylabel="Trace count")
    for name, label in [
        ("rms_to_shot_offset_median", "Raw RMS"),
        ("ac_rms_to_shot_offset_median", "Centered RMS"),
    ]:
        values = c[name].dropna()
        axes[1, 0].hist(
            np.log10(values), bins=np.linspace(-7, 5, 121), histtype="step", label=label, log=True
        )
    axes[1, 0].set(
        xlabel="log10(amplitude / same-shot, same-offset-bin median)", ylabel="Trace count"
    )
    axes[1, 0].legend()
    for bound in [-2, 2]:
        axes[1, 0].axvline(bound, color="red", ls="--", lw=0.7)
    valid = c.rms_to_shot_offset_median.notna()
    hb = axes[1, 1].hexbin(
        c.loc[valid, "dc_to_rms"],
        np.log10(c.loc[valid, "rms_to_shot_offset_median"]),
        gridsize=75,
        bins="log",
        mincnt=1,
    )
    fig.colorbar(hb, ax=axes[1, 1], label="Trace count")
    axes[1, 1].set(xlabel="abs(mean) / raw RMS", ylabel="log10(raw relative amplitude)")
    fig.suptitle("After fixed zero/constant exclusions; diagnostic thresholds do not reject traces")
    fig.savefig(output / "distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, label in zip(axes, ["dc", "amplitude"], strict=True):
        pts = ax.scatter(
            stations.x_m,
            stations.y_m,
            c=stations[label + "fraction"] * 100,
            s=9,
            cmap="magma",
            vmin=0,
            vmax=100,
        )
        fig.colorbar(pts, ax=ax, label="Flagged / retained at this station (%)")
        ax.set(
            title=label.upper() + " review rate",
            xlabel="UTM easting (m)",
            ylabel="UTM northing (m)",
            aspect="equal",
        )
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="x", rotation=30)
    fig.savefig(output / "station_rates.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, label in zip(axes, ["dc_review", "amplitude_review"], strict=True):
        rates = c.groupby(["source_line", "receiver_line"])[label].mean().unstack() * 100
        im = ax.imshow(
            rates,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=max(1, rates.max().max()),
        )
        ax.set_xticks(
            np.arange(len(rates.columns)), rates.columns.astype(int), rotation=90, fontsize=7
        )
        ax.set_yticks(np.arange(len(rates.index)), rates.index.astype(int))
        ax.set(title=label, xlabel="Receiver line", ylabel="Source line")
        fig.colorbar(im, ax=ax, label="Flagged / retained in line pair (%)")
    fig.savefig(output / "line_pair_rates.png", dpi=160)
    plt.close(fig)


def plot_station_sequence(table, output: Path) -> None:
    """Show the concentration previously found at receiver 101/579, including rejects."""
    c = table[
        table.header_eligible & table.receiver_line.eq(101) & table.receiver_point.eq(579)
    ].sort_values("ffid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(c.ffid, c["mean"], ".", ms=3, label="Mean")
    axes[0].plot(c.ffid, c["std"], ".", ms=3, label="Centered RMS")
    axes[0].set(ylabel="Stored amplitude units")
    axes[0].legend()
    valid = c.eligible_after_fixed_qc
    axes[1].scatter(c.loc[valid, "ffid"], c.loc[valid, "std_to_rms"], s=8, label="Retained")
    rejected = c.fixed_qc_excluded
    axes[1].scatter(
        c.loc[rejected, "ffid"],
        np.full(rejected.sum(), 1e-10),
        s=14,
        marker="x",
        color="red",
        label="Fixed excluded (marker at 1e-10)",
    )
    axes[1].axhline(0.01, color="black", ls="--", label="Diagnostic q = 0.01")
    axes[1].set(yscale="log", ylabel="std / RMS", xlabel="FFID (shot identifier; not elapsed time)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Receiver 101/579; missing/dead headers are not filled")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_review_example(values, gather, target, controls, dt: float, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    rows = [int(target.trace_index), *controls.trace_index.astype(int).tolist()]
    for j, index in enumerate(rows):
        h = gather.iloc[index]
        raw = values[index].astype(float)
        centered = raw - raw.mean()
        rms = np.sqrt(np.mean(centered**2))
        label = (
            "Target" if j == 0 else "Control"
        ) + f" R{int(h.receiver_line)}/{int(h.receiver_point)}"
        color = "#c23b22" if j == 0 else ["#2166ac", "#4d9221"][(j - 1) % 2]
        time = np.arange(len(raw)) * dt
        axes[0, 0].plot(time, raw, color=color, lw=0.7, alpha=0.85, label=label)
        axes[0, 1].plot(time, centered, color=color, lw=0.7, alpha=0.85, label=label)
        if rms > 0:
            axes[1, 0].plot(time, centered / rms + j * 8, color=color, lw=0.6, label=label)
            power = np.abs(np.fft.rfft(centered * np.hanning(len(raw)))) ** 2
            frequency = np.fft.rfftfreq(len(raw), dt)
            if power.max() > 0:
                axes[1, 1].plot(
                    frequency,
                    10 * np.log10(np.maximum(power / power.max(), 1e-12)),
                    color=color,
                    lw=0.9,
                    label=label,
                )
    axes[0, 0].set(
        title="Raw amplitude (shared scale; no automatic offset)",
        xlabel="Time (s)",
        ylabel="Stored amplitude",
    )
    axes[0, 1].set(
        title="Mean removed, display only (shared scale)",
        xlabel="Time (s)",
        ylabel="Stored amplitude",
    )
    axes[1, 0].set(
        title="Centered / own RMS + display separation",
        xlabel="Time (s)",
        ylabel="Normalized display only",
    )
    axes[1, 1].set(
        title="Centered Hann power / each trace maximum",
        xlabel="Frequency (Hz)",
        ylabel="dB",
        xlim=(0, 100),
        ylim=(-100, 2),
    )
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="best")
    for ax in axes[0]:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
    fig.suptitle(
        f"{target.review_stratum} / {target.selection}; FFID {int(target.ffid)}, "
        f"trace {int(target.trace_index)}\n"
        f"q={target.std_to_rms:.4g}, DC/RMS={target.dc_to_rms:.4g}, "
        f"raw ratio={target.rms_to_shot_offset_median:.4g}, "
        f"centered ratio={target.ac_rms_to_shot_offset_median:.4g}; controls={len(controls)}"
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)
