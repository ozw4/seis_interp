"""Geometry, acquisition availability and sparse-grid diagnostics for FORGE."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_station_layout(sources, receivers, navigation, availability, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax = axes[0]
    ax.scatter(navigation.x_m, navigation.y_m, s=7, c="lightgray", label="Source navigation")
    for line, group in sources.groupby("source_line"):
        ax.plot(group.source_x_m, group.source_y_m, ".-", ms=4, lw=0.7, label=f"Source {line}")
    ax.set_title("Source positions: local files vs navigation")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax = axes[1]
    known = receivers.receiver_x_m.notna()
    colors = (availability >= 3).mean(axis=0)
    scatter = ax.scatter(
        receivers.loc[known, "receiver_x_m"],
        receivers.loc[known, "receiver_y_m"],
        s=7,
        c=colors[known],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(scatter, ax=ax, label="Numerically usable / recorded source count")
    ax.set_title(f"Receiver positions ({int((~known).sum())} stations have unknown XY)")
    for ax in axes:
        ax.set(xlabel="UTM 12N easting (m)", ylabel="UTM 12N northing (m)", aspect="equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="x", rotation=30)
    _save(fig, output)


def plot_station_spacing(spacing: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, role in enumerate(["source", "receiver"]):
        along, across = spacing[role]
        axes[0, col].hist(along.loc[along.point_id_step.eq(1), "distance_m"], bins=60)
        axes[0, col].set(
            title=f"{role.title()}: adjacent point IDs only",
            xlabel="Distance (m)",
            ylabel="Pair count",
        )
        groups = list(across.groupby(["line_a", "line_b"]))
        if groups:
            med = [g.distance_m.median() for _, g in groups]
            low = [g.distance_m.quantile(0.05) for _, g in groups]
            high = [g.distance_m.quantile(0.95) for _, g in groups]
            axes[1, col].errorbar(
                np.arange(len(groups)),
                med,
                yerr=np.array([np.subtract(med, low), np.subtract(high, med)]),
                fmt=".",
            )
            axes[1, col].set_xticks(
                np.arange(len(groups)),
                [f"{a}-{b}" for (a, b), _ in groups],
                rotation=90,
                fontsize=7,
            )
        axes[1, col].set(
            title="Adjacent observed lines: median and 5-95% at matched point IDs",
            xlabel="Line IDs (unobserved lines are not filled)",
            ylabel="Euclidean separation (m)",
        )
    _save(fig, output)


def plot_availability(matrix, sources, receivers, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for ax, values, title in [
        (axes[0], matrix < 3, "Missing / header excluded / waveform unusable"),
        (axes[1], matrix == 3, "Numerically usable but DC/amplitude review retained"),
    ]:
        ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["#f1f1f1", "#c23b22"]),
            vmin=0,
            vmax=1,
        )
        ax.set(title=title, ylabel="Source order by (line, point)")
        yticks = sources.groupby("source_line").source_id.mean()
        xticks = receivers.groupby("receiver_line").receiver_id.mean()
        ax.set_yticks(yticks.to_numpy(), yticks.index.astype(str))
        ax.set_xticks(xticks.to_numpy(), xticks.index.astype(str), rotation=45, fontsize=8)
        for bound in sources.groupby("source_line").source_id.min().to_numpy()[1:]:
            ax.axhline(bound - 0.5, color="gray", lw=0.4)
    axes[1].set_xlabel(
        "Receiver order by (line, point); ticks show line IDs; station-ID gaps compressed"
    )
    _save(fig, output)


def plot_station_id_support(sources, receivers, steps, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    for ax, role, stations, line_step, point_step, opposite_count in [
        (axes[0], "source", sources, steps[0], steps[1], len(receivers)),
        (axes[1], "receiver", receivers, steps[2], steps[3], len(sources)),
    ]:
        line, point = stations[f"{role}_line"], stations[f"{role}_point"]
        li = ((line - line.min()) / line_step).astype(int)
        pi = ((point - point.min()) / point_step).astype(int)
        values = np.full((li.max() + 1, pi.max() + 1), np.nan)
        values[li, pi] = stations.numerically_usable_traces / opposite_count
        im = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[
                point.min() - point_step / 2,
                point.max() + point_step / 2,
                line.min() - line_step / 2,
                line.max() + line_step / 2,
            ],
        )
        ax.set(
            title=f"{role.title()} station-ID support (gray = no local station)",
            xlabel="Point ID",
            ylabel="Line ID",
        )
        fig.colorbar(im, ax=ax, label="Numerically usable pairs / opposite station count")
    _save(fig, output)


def plot_trace_geometry(observed, fold_width: float, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].hist(observed.offset_m, bins=80)
    axes[0, 0].set(xlabel="Full horizontal offset (m)", ylabel="Trace count")
    axes[0, 1].hist(observed.azimuth_deg.dropna(), bins=np.arange(0, 361, 10))
    axes[0, 1].set(
        xlabel="Source-to-receiver azimuth (degrees clockwise from north)",
        ylabel="Trace count",
        xlim=(0, 360),
    )
    for ax, x, y, width, title, labels in [
        (
            axes[1, 0],
            observed.midpoint_x_m,
            observed.midpoint_y_m,
            fold_width,
            f"Midpoint fold ({fold_width:g} m bins; all offsets)",
            ("Midpoint easting (m)", "Midpoint northing (m)"),
        ),
        (
            axes[1, 1],
            observed.offset_x_m,
            observed.offset_y_m,
            fold_width * 2,
            f"Offset-vector density ({fold_width * 2:g} m bins)",
            ("Receiver - source easting (m)", "Receiver - source northing (m)"),
        ),
    ]:
        edges = [
            np.arange(np.floor(v.min() / width), np.floor(v.max() / width) + 2) * width
            for v in [x, y]
        ]
        hist = ax.hist2d(x, y, bins=edges, norm=LogNorm(), cmap="viridis")
        fig.colorbar(hist[3], ax=ax, label="Trace count")
        ax.set(title=title, xlabel=labels[0], ylabel=labels[1], aspect="equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Numerically usable traces; review flags retained")
    _save(fig, output)


def plot_grid_comparison(table: pd.DataFrame, output: Path) -> None:
    base = table[table.phase_bins.eq(0)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    labels = base.frame + "/" + base.grid_id
    for ax, column, title in [
        (axes[0], "bbox_fill_fraction", "4D bounding-box fill (%)"),
        (axes[1], "reference_fill_fraction", "Known-station support fill (%)"),
        (axes[2], "collision_trace_fraction", "Traces sharing a 4D cell (%)"),
    ]:
        ax.barh(np.arange(len(base)), base[column] * 100)
        ax.set_yticks(np.arange(len(base)), labels, fontsize=8)
        ax.set(title=title, xlim=(0, 100))
        ax.invert_yaxis()
    axes[0].set(xscale="log", xlim=(0.01, 100), xlabel="Log scale")
    axes[1].set(xlim=(99.5, 100), xlabel="Expanded range: 99.5-100%")
    fig.suptitle("Phase 0 bins; no amplitudes averaged, no empty cells synthesized")
    _save(fig, output)
