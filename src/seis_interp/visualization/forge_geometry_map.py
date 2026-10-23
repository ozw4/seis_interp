"""Actual FORGE station coordinates with the trace mask of one fixed source."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def prepare_source_mask_map(manifest, mask, *, source_cell):
    """Join by trace cell, then assign receiver states for only the selected source.

    A station has no survey-wide masked state: the outer mask belongs to a
    source/receiver pair. Absent pairs remain natural missing, never artificial test.
    Grid coordinates are intentionally not used.
    """
    if manifest.cell_id.duplicated().any() or mask.cell_id.duplicated().any():
        raise ValueError("geometry and mask require unique trace cells")
    if set(manifest.cell_id) != set(mask.cell_id):
        raise ValueError("geometry and mask trace cells must match")
    if not mask.split.isin(["observed", "test"]).all():
        raise ValueError("source mask map requires an observed/test outer mask")
    if manifest.duplicated(["source_cell", "receiver_cell"]).any():
        raise ValueError("duplicate source/receiver pair")
    sources = _actual_stations(manifest, "source")
    receivers = _actual_stations(manifest, "receiver")
    if source_cell not in set(sources.cell_id):
        raise ValueError("selected source cell is absent from geometry")
    selected = manifest.loc[manifest.source_cell.eq(source_cell), ["cell_id", "receiver_cell"]]
    selected = selected.merge(mask[["cell_id", "split"]], on="cell_id", validate="one_to_one")
    receivers = receivers.merge(
        selected[["receiver_cell", "split"]],
        left_on="cell_id",
        right_on="receiver_cell",
        how="left",
        validate="one_to_one",
    ).drop(columns="receiver_cell")
    receivers["split"] = receivers.split.fillna("natural_missing")
    return sources, receivers


def _actual_stations(manifest, role):
    columns = [f"{role}_cell", f"{role}_x_m", f"{role}_y_m"]
    stations = (
        manifest[columns]
        .drop_duplicates()
        .rename(columns=dict(zip(columns, ["cell_id", "x_m", "y_m"], strict=True)))
    )
    if stations.cell_id.duplicated().any():
        raise ValueError(f"inconsistent actual {role} coordinates")
    if not np.isfinite(stations[["x_m", "y_m"]].to_numpy()).all():
        raise ValueError(f"nonfinite actual {role} coordinates")
    return stations.sort_values("cell_id").reset_index(drop=True)


def create_source_mask_map(sources, receivers, *, source_cell, missing_fraction, mask_seed):
    """One equal-scale UTM map, with distinct source and receiver markers."""
    selected = sources.loc[sources.cell_id.eq(source_cell)]
    if len(selected) != 1:
        raise ValueError("map requires exactly one selected source")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(178 / 25.4, 178 / 25.4))
        other = sources.loc[~sources.cell_id.eq(source_cell)]
        ax.scatter(
            other.x_m,
            other.y_m,
            marker="^",
            s=15,
            facecolors="#8c8c8c",
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
            label=f"Other sources ({len(other)})",
        )
        for split, label, color, filled in (
            ("observed", "Receivers: observed", "#0072B2", True),
            ("test", "Receivers: masked", "#D55E00", False),
        ):
            points = receivers.loc[receivers.split.eq(split)]
            ax.scatter(
                points.x_m,
                points.y_m,
                marker="o",
                s=22,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
                label=f"{label} ({len(points)})",
            )
        absent = receivers.loc[receivers.split.eq("natural_missing")]
        if len(absent):
            ax.scatter(
                absent.x_m,
                absent.y_m,
                marker="x",
                s=25,
                color="black",
                linewidths=1.0,
                zorder=4,
                label=f"Receivers: natural missing ({len(absent)})",
            )
        point = selected.iloc[0]
        ax.scatter(
            [point.x_m],
            [point.y_m],
            marker="*",
            s=150,
            facecolors="#E69F00",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label=f"Selected source (cell {source_cell})",
        )
        ax.annotate(
            f"S{source_cell}",
            (point.x_m, point.y_m),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            zorder=6,
        )
        ax.set(
            xlabel="UTM easting [m]",
            ylabel="UTM northing [m]",
            aspect="equal",
        )
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax.tick_params(labelsize=8)
        ax.grid(color="#e5e5e5", linewidth=0.5, zorder=0)
        ax.margins(0.05)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.88)
        fig.text(
            0.56,
            0.95,
            "FORGE M1: actual source / receiver geometry",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.56,
            0.915,
            f"Trace mask: {missing_fraction:.0%} missing (seed {mask_seed}); "
            f"fixed source {source_cell}",
            ha="center",
            fontsize=8,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.54, 0.055),
            ncol=2,
            fontsize=8,
            frameon=False,
            columnspacing=1.6,
            handletextpad=0.7,
        )
    return fig
