"""Actual/grid coordinates and occupancy evidence for shortlisted FORGE regions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_region_geometry(source, receiver, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, region in zip(axes, [source, receiver], strict=True):
        m, p = region.metrics, region.stations
        ii, jj = np.indices(m["shape"])
        direction, normal = np.array(m["direction"]), np.array(m["normal"])
        xy = (
            np.array(m["origin_xy_m"])
            + (m["first_u_m"] + jj.ravel()[:, None] * m["along_spacing_m"]) * direction
            + (m["first_v_m"] + ii.ravel()[:, None] * m["across_spacing_m"]) * normal
        )
        ax.scatter(xy[:, 0], xy[:, 1], marker="+", color="gray", s=22, label="Grid nodes")
        for line, group in p.groupby("line"):
            group = group.sort_values("point")
            ax.plot(group.x_m, group.y_m, lw=0.6, alpha=0.4)
            ax.annotate(str(int(line)), (group.x_m.iloc[0], group.y_m.iloc[0]), fontsize=7)
        pts = ax.scatter(p.x_m, p.y_m, c=p.distance_m, s=14, cmap="magma", label="Actual stations")
        ax.quiver(
            p.x_m,
            p.y_m,
            p.grid_x_m - p.x_m,
            p.grid_y_m - p.y_m,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            color="#2166ac",
        )
        fig.colorbar(pts, ax=ax, label="Projection distance (m)", shrink=0.65)
        ax.set(
            title=f"{m['role']}: {m['shape']}; p95={m['distance_p95_m']:.2f} m",
            xlabel="UTM easting (m)",
            ylabel="UTM northing (m)",
            aspect="equal",
        )
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_region_occupancy(source, receiver, counts, recorded, support, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, region in zip(axes[:2], [source, receiver], strict=True):
        n = np.bincount(region.stations.cell, minlength=region.metrics["cell_count"])
        im = ax.imshow(
            n.reshape(region.metrics["shape"]),
            origin="lower",
            aspect="auto",
            vmin=0,
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Known stations per node", shrink=0.7)
        ax.set(title=region.metrics["role"], xlabel="Grid along index", ylabel="Grid line index")
    labels = np.full(len(counts), 3, dtype=np.uint8)
    labels[support == 0] = 0
    labels[(support > 0) & (recorded == 0)] = 1
    labels[(recorded > 0) & (counts == 0)] = 2
    labels[counts > 1] = 4
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#eeeeee", "#999999", "#d95f02", "#1b9e77", "#7570b3"])
    im = axes[2].imshow(
        labels.reshape(source.metrics["cell_count"], receiver.metrics["cell_count"]),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, 5), cmap.N),
    )
    cb = fig.colorbar(im, ax=axes[2], ticks=range(5), shrink=0.7)
    cb.ax.set_yticklabels(
        ["No station support", "No record", "QC empty", "One retained", "Collision"]
    )
    axes[2].set(
        title="4D cells flattened (line-major order)",
        xlabel="Receiver grid cell",
        ylabel="Source grid cell",
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_search_tradeoffs(table, finalists, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for size, group in table.groupby("size", sort=False):
        axes[0].scatter(
            group.normalized_p95, group.fill_fraction * 100, s=12, alpha=0.3, label=size
        )
        axes[1].scatter(group.grid_cells, group.normalized_p95, s=12, alpha=0.3, label=size)
    axes[0].scatter(
        finalists.normalized_p95,
        finalists.fill_fraction * 100,
        marker="x",
        color="black",
        s=35,
        label="Finalists",
    )
    axes[0].set(xlabel="Worst station-role normalized p95 displacement", ylabel="4D fill (%)")
    axes[1].set(
        xlabel="Spatial grid cells",
        ylabel="Worst station-role normalized p95 displacement",
        xscale="log",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output, dpi=160)
    plt.close(fig)
