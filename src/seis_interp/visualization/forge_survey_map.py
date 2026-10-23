"""Locate a FORGE analysis subset in the actual full acquisition geometry."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


def prepare_survey_map(full_geometry, navigation, analysis):
    """Match stations by line/point and verify their actual coordinates.

    Navigation covers source positions without local recordings as well. Receiver
    positions come from the full geometry audit; unknown coordinates are not filled.
    """
    sources = navigation[["line", "point", "x_m", "y_m"]].copy()
    _validate_stations(sources, "source navigation")
    recorded = _survey_stations(full_geometry, "source")
    sources = _mark_analysis(sources, recorded).rename(columns={"in_analysis": "recorded"})
    sources = _mark_analysis(sources, _survey_stations(analysis, "source"))
    receivers = _mark_analysis(
        _survey_stations(full_geometry, "receiver"), _survey_stations(analysis, "receiver")
    )
    return sources, receivers


def _survey_stations(geometry, role):
    names = [f"{role}_line", f"{role}_point", f"{role}_x_m", f"{role}_y_m"]
    table = (
        geometry[names]
        .drop_duplicates()
        .rename(columns=dict(zip(names, ["line", "point", "x_m", "y_m"], strict=True)))
    )
    _validate_stations(table, role)
    return table.sort_values(["line", "point"]).reset_index(drop=True)


def _validate_stations(table, name):
    if table.empty or table.duplicated(["line", "point"]).any():
        raise ValueError(f"empty or inconsistent {name} station keys")
    if not np.isfinite(table[["line", "point", "x_m", "y_m"]].to_numpy()).all():
        raise ValueError(f"nonfinite {name} station coordinates")


def _mark_analysis(survey, selected):
    keys = ["line", "point"]
    if not set(map(tuple, selected[keys].to_numpy())).issubset(
        set(map(tuple, survey[keys].to_numpy()))
    ):
        raise ValueError("analysis station is absent from full survey geometry")
    joined = survey.merge(selected, on=keys, how="left", suffixes=("", "_analysis"))
    selected_rows = joined.x_m_analysis.notna()
    if not np.allclose(
        joined.loc[selected_rows, ["x_m", "y_m"]],
        joined.loc[selected_rows, ["x_m_analysis", "y_m_analysis"]],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("analysis station coordinates differ from full survey geometry")
    joined["in_analysis"] = selected_rows
    return (
        joined.drop(columns=["x_m_analysis", "y_m_analysis"])
        .sort_values(keys)
        .reset_index(drop=True)
    )


def create_survey_map(sources, receivers, *, unknown_receiver_count=0):
    """One UTM plan showing all stations and the selected source/receiver subset."""
    selected_sources = sources.loc[sources.in_analysis]
    selected_receivers = receivers.loc[receivers.in_analysis]
    if selected_sources.empty or selected_receivers.empty:
        raise ValueError("survey map requires analysis sources and receivers")
    selected = pd.concat(
        [selected_sources[["x_m", "y_m"]], selected_receivers[["x_m", "y_m"]]],
        ignore_index=True,
    )
    lower, upper = selected.min(), selected.max()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(178 / 25.4, 160 / 25.4))
        for table, label, marker, size, color, order in (
            (sources, "Survey sources", "^", 8, "#bdbdbd", 1),
            (receivers, "Survey receivers", "o", 5, "#d9d9d9", 1),
            (selected_sources, "M1 sources", "^", 20, "#D55E00", 3),
            (selected_receivers, "M1 receivers", "o", 16, "#0072B2", 4),
        ):
            ax.scatter(
                table.x_m,
                table.y_m,
                marker=marker,
                s=size,
                facecolors=color,
                edgecolors="none",
                zorder=order,
                label=f"{label} ({len(table):,})",
            )
        ax.add_patch(
            Rectangle(
                (lower.x_m, lower.y_m),
                upper.x_m - lower.x_m,
                upper.y_m - lower.y_m,
                fill=False,
                edgecolor="#333333",
                linewidth=1.2,
                linestyle="--",
                zorder=5,
            )
        )
        ax.annotate(
            "Analysis M1",
            ((lower.x_m + upper.x_m) / 2, upper.y_m),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
            zorder=6,
        )
        ax.set(xlabel="UTM 12N easting [m]", ylabel="UTM 12N northing [m]", aspect="equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.tick_params(labelsize=8)
        ax.grid(color="#eeeeee", linewidth=0.5, zorder=0)
        ax.margins(0.04)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.88)
        fig.text(
            0.56,
            0.95,
            "FORGE 2017: full acquisition and analysis subset",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.56,
            0.915,
            "Dashed box: combined XY extent of the M1 analysis stations",
            ha="center",
            fontsize=8,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.55, 0.06),
            ncol=2,
            fontsize=8,
            frameon=False,
            columnspacing=2.0,
        )
        note = "Source positions include navigation stations without local recordings."
        if unknown_receiver_count:
            note += f" {unknown_receiver_count} receiver without XY omitted."
        fig.text(0.55, 0.025, note, ha="center", fontsize=6.5)
    return fig
