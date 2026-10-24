"""Static, exportable validation heatmaps and S1 search diagnostic curves."""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def grid_position(candidate):
    match = re.fullmatch(r"a(\d+)_b(\d+)", candidate)
    if match is None:
        raise ValueError(f"not a common-grid candidate ID: {candidate}")
    return int(match[1]) - 1, int(match[2]) - 1


def grid_edge(candidate, grid):
    a, b = grid_position(candidate)
    return a in (0, len(grid["axis_a"]) - 1) or b in (0, len(grid["axis_b"]) - 1)


def validation_heatmaps(records, grid, selected_id, ranking, destination, title):
    for metric, key in (("relative_mse", "validation_score"), ("snr_db", "validation_snr_db")):
        image = np.full((len(grid["axis_a"]), len(grid["axis_b"])), np.nan)
        fig, ax = plt.subplots(figsize=(7, 6))
        for row in records:
            a, b = grid_position(row["id"])
            if row["status"] == "completed":
                image[a, b] = row[key]
            else:
                ax.text(b, a, "D", ha="center", va="center")
        colour = ax.imshow(image, cmap="viridis_r" if metric == "relative_mse" else "viridis")
        fig.colorbar(colour, ax=ax, label=metric)
        for candidate, marker, label, size in (
            (selected_id, "s", "relMSE winner", 240),
            (ranking["best_snr_id"], "*", "best SNR", 150),
        ):
            if candidate:
                a, b = grid_position(candidate)
                ax.scatter(
                    b,
                    a,
                    marker=marker,
                    s=size,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.8,
                    label=label,
                )
        ax.set_xticks(
            range(len(grid["axis_b"])), [v["id"] for v in grid["axis_b"]], rotation=25, ha="right"
        )
        ax.set_yticks(range(len(grid["axis_a"])), [v["id"] for v in grid["axis_a"]])
        rho = ranking["spearman_rho"]
        ax.set(
            title=f"{title}: validation {metric}\nSpearman (relMSE rank, SNR rank) = {rho:.4f}"
            if rho is not None
            else f"{title}: validation {metric}; Spearman undefined",
            xlabel="axis b",
            ylabel="axis a",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.24), ncols=2)
        fig.tight_layout()
        fig.savefig(destination / f"validation_{metric}.png", dpi=160)
        plt.close(fig)


def search_figures(curves, curse, destination, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("relative_mse", "snr_db"), strict=True):
        table = curves.loc[curves.metric.eq(metric)]
        ax.plot(table.configurations, table.conditional_expected_best)
        ax.fill_between(
            table.configurations,
            table.conditional_expected_best - table.sd,
            table.conditional_expected_best + table.sd,
            alpha=0.2,
        )
        ax.set(
            xlabel="k draws with replacement",
            ylabel=metric,
            title=f"{title}: expected best of k\nconditional on a finite candidate",
        )
    fig.tight_layout()
    fig.savefig(destination / "expected_best_of_k.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ("relative_mse", "snr_db"), strict=True):
        row = curse.loc[curse.metric.eq(metric)].iloc[0]
        if row.status == "available":
            ax.bar(["in bag", "out of bag"], [row.bootstrap_inbag_mean, row.bootstrap_oob_mean])
        else:
            ax.text(0.5, 0.5, row.status, ha="center", wrap=True)
        ax.set(title=f"{title}: S1 winner's curse", ylabel=metric)
    fig.tight_layout()
    fig.savefig(destination / "winners_curse.png", dpi=160)
    plt.close(fig)
