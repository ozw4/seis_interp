"""Fixed geometry/mask displays and predeclared FORGE sections."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_preparation(manifest, mask, features, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, role in zip(axes, ("source", "receiver"), strict=True):
        for mode, marker in (("real", "."), ("grid", "+")):
            xy = features[mode][[f"{role}_x_m", f"{role}_y_m"]].drop_duplicates()
            ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], s=12, marker=marker, label=mode)
        ax.set(title=role, xlabel="UTM easting [m]", ylabel="UTM northing [m]", aspect="equal")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.tight_layout()
    fig.savefig(output / "geometry.png", dpi=150)
    plt.close(fig)
    source_count = int(manifest.source_cell.max()) + 1
    receiver_count = int(manifest.receiver_cell.max()) + 1
    image = np.full((source_count, receiver_count), -1, dtype=np.int8)
    image[mask.source_cell, mask.receiver_cell] = mask.split.eq("observed").astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot = ax.imshow(image, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap="viridis")
    ax.set(xlabel="receiver cell", ylabel="source cell", title="Fixed random 80% outer mask")
    bar = fig.colorbar(plot, ax=ax, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["natural missing", "test", "observed"])
    fig.tight_layout()
    fig.savefig(output / "outer_mask.png", dpi=150)
    plt.close(fig)


def plot_sections(manifest, mask, truth, predictions, contract, output):
    """Call only after evaluation opens test amplitudes; fixed source/receiver IDs."""
    time = np.arange(contract["time_sample_count"]) * contract["sample_interval_us"] / 1e6
    settings = contract["visualization"]
    keep = (time >= settings["time_range_s"][0]) & (time <= settings["time_range_s"][1])
    observed = mask.split.eq("observed").to_numpy()
    clip = float(np.percentile(np.abs(truth[observed]), settings["clip_observed_percentile"]))
    clip = clip if clip > 0 else 1.0
    for role in ("source", "receiver"):
        rows = np.flatnonzero(manifest[f"{role}_cell"].eq(settings[f"{role}_cell"]))
        observed_panel = np.where(observed[rows, None], truth[rows], 0)
        panels = {"ground truth": truth[rows], "observed input": observed_panel}
        panels.update({name: value[rows] for name, value in predictions.items()})
        fig, axes = plt.subplots(2, 4, figsize=(18, 10), squeeze=False)
        for ax, (name, values) in zip(axes.flat, panels.items(), strict=False):
            ax.imshow(
                values[:, keep].T,
                aspect="auto",
                cmap="gray",
                vmin=-clip,
                vmax=clip,
                extent=(0, len(rows), time[keep][-1], time[keep][0]),
                interpolation="nearest",
            )
            ax.set(title=name, xlabel="trace in fixed section", ylabel="time [s]")
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f"common_{role}.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
        for ax, (name, values) in zip(axes.flat, predictions.items(), strict=False):
            ax.imshow(
                (values[rows] - truth[rows])[:, keep].T,
                aspect="auto",
                cmap="seismic",
                vmin=-clip,
                vmax=clip,
                extent=(0, len(rows), time[keep][-1], time[keep][0]),
                interpolation="nearest",
            )
            ax.set(title=name + " error", xlabel="trace in fixed section", ylabel="time [s]")
        for ax in list(axes.flat)[len(predictions) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f"common_{role}_errors.png", dpi=150)
        plt.close(fig)


def plot_projection_errors(quartiles, output):
    fig, ax = plt.subplots(figsize=(7, 4))
    clean = quartiles.loc[quartiles.domain.eq("clean_target")]
    for mode in ("real", "grid"):
        ax.plot(clean.quartile, clean[f"relative_mse_{mode}"], "o-", label="ReGSI " + mode)
    ax.set(
        xlabel="Source projection distance quartile (fixed test set)",
        ylabel="Clean-target relative MSE",
        xticks=[1, 2, 3, 4],
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "projection_quartiles.png", dpi=150)
    plt.close(fig)
