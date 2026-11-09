"""Acquisition geometry and predetermined whole-shot gathers for a fixed comparison."""

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.processing.forge_shot_mask import selected_shot_gathers


def plot_shot_acquisition(manifest, partition, counts, output):
    mask_title = (
        f"Random whole-shot50, seed {counts['mask_seed']}"
        if counts.get("mask_family") == "random_shot"
        else f"Periodic shot50, phase {counts['observed_phase']}"
    )
    sources = manifest.drop_duplicates("source_cell").set_index("source_cell")
    role = partition.drop_duplicates("source_cell").set_index("source_cell").role
    receivers = manifest.drop_duplicates("receiver_cell")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(
        receivers.receiver_x_m,
        receivers.receiver_y_m,
        s=10,
        c="0.65",
        marker="^",
        label=f"receivers ({len(receivers)})",
    )
    for name, color, marker in (("train", "#0072B2", "o"), ("test", "#D55E00", "x")):
        rows = sources.loc[role.index[role.eq(name)]]
        ax.scatter(
            rows.source_x_m,
            rows.source_y_m,
            s=22,
            c=color,
            marker=marker,
            label=f"{'observed' if name == 'train' else 'test'} shots ({len(rows)})",
        )
    ax.set(
        xlabel="UTM easting (m)",
        ylabel="UTM northing (m)",
        title=(
            f"{mask_title}\n"
            f"{counts['observed_trace_count']:,} observed / {counts['test_trace_count']:,} "
            "test eligible traces; "
            f"missing {100 * counts['eligible_trace_missing_fraction']:.4f}%"
        ),
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_missing_gather(experiment, manifest, contract, source, support, output, clip):
    rows = manifest.loc[manifest.source_cell.eq(source)].sort_values("receiver_cell")
    target = np.load(experiment / "input/waveforms.npy", mmap_mode="r")[
        manifest.index.get_indexer(rows.index)
    ]
    methods = contract["methods"]
    fig, axes = plt.subplots(
        2, len(methods) + 1, figsize=(22, 8), sharex=True, sharey=True, constrained_layout=True
    )
    extent = [-0.5, len(rows) - 0.5, (len(target[0]) - 1) * contract["sample_interval_us"] / 1e6, 0]
    options = dict(
        cmap="seismic", vmin=-clip, vmax=clip, aspect="auto", extent=extent, interpolation="nearest"
    )
    view = axes[0, 0].imshow(target.T, **options)
    axes[0, 0].set_title("Original (hidden)")
    axes[1, 0].axis("off")
    for column, method in enumerate(methods, start=1):
        directory = experiment / "method_runs" / method
        record = json.loads((directory / "manifest.json").read_text())
        axes[0, column].set_title(method)
        axes[1, column].set_title("Prediction − original")
        if record["status"] == "completed":
            cells = np.load(directory / "cell_ids.npy")
            indices = pd.Index(cells).get_indexer(rows.cell_id)
            if (indices < 0).any():
                raise ValueError("whole-shot gather prediction is incomplete")
            predicted = np.load(directory / "prediction.npy", mmap_mode="r")[indices]
            axes[0, column].imshow(predicted.T, **options)
            axes[1, column].imshow((predicted - target).T, **options)
        else:
            for ax in axes[:, column]:
                ax.text(0.5, 0.5, "NA: failed", transform=ax.transAxes, ha="center")
    for ax in axes[:, 0]:
        ax.set_ylabel("Physical time (s)")
    for ax in axes[1, 1:]:
        ax.set_xlabel("Receiver index (cell order)")
    first = rows.iloc[0]
    support_label = support
    if contract.get("mask_family") == "random_shot":
        support_label = (
            "one-sided source-axis support"
            if support == "edge"
            else "two-sided source-axis support"
        )
    fig.suptitle(
        f"{support_label} missing shot: source_cell={source}, line={first.source_grid_line}, "
        f"point={first.source_grid_point}; {len(rows)} eligible traces; shared scale ±{clip:.4g}"
    )
    fig.colorbar(view, ax=axes.ravel().tolist(), label="Physical amplitude", shrink=0.7)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def export_forge_shot_results(experiment, output):
    """Export geometry immediately; export scores and gathers only after final scoring."""
    output.mkdir(parents=True, exist_ok=False)
    seal = json.loads((experiment / "experiment.json").read_text())
    verify_hashes(experiment, seal["artifacts"])
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    partition = pd.read_parquet(experiment / "input/partition.parquet")
    counts = json.loads((experiment / "input/mask_counts.json").read_text())
    chosen = selected_shot_gathers(partition, contract["spatial_shape"])
    if chosen != json.loads((experiment / "input/gather_selection.json").read_text()):
        raise ValueError("gather selection differs from the pre-training geometry selection")
    plot_shot_acquisition(manifest, partition, counts, output / "acquisition_mask.png")
    provenance = {
        "study_id": seal["study_id"],
        "run": str(experiment.relative_to(experiment.parents[2])),
        "experiment_hash": sha256_file(experiment / "experiment.json"),
        "render_source_sha256": sha256_file(Path(__file__)),
        "input_hashes": {
            name: digest for name, digest in seal["artifacts"].items() if name.startswith("input/")
        },
        "mask": counts,
        "selected_gathers": chosen,
        "selection_rule": "geometry only: central line, central missing point and missing edge",
        "fixed_settings": contract["fixed_settings"],
    }
    if (experiment / "outcomes.json").is_file():
        provenance["method_artifacts"] = {}
        for method in contract["methods"]:
            path = experiment / "method_runs" / method / "manifest.json"
            record = json.loads(path.read_text())
            provenance["method_artifacts"][method] = {
                "status": record["status"],
                "manifest_sha256": sha256_file(path),
                **{
                    name: record.get("artifacts", {}).get(name)
                    for name in ("prediction.npy", "cell_ids.npy", "config.resolved.yaml")
                },
            }
        for name in (
            "summary.csv",
            "per_shot.csv",
            "shot_support.csv",
            "shot_equal_weight.csv",
            "paired_shot_differences.csv",
            "outcomes.json",
        ):
            shutil.copyfile(experiment / name, output / name)
        observed = np.load(experiment / "input/observed_waveforms.npy", mmap_mode="r")
        clip = float(
            np.percentile(abs(observed), contract["visualization"]["clip_observed_percentile"])
        )
        provenance["display_clip"] = clip
        for support, source in chosen.items():
            plot_missing_gather(
                experiment,
                manifest,
                contract,
                source,
                support,
                output / f"gather_{support}.png",
                clip,
            )
    provenance["artifacts"] = {
        path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    write_json(output / "provenance.json", provenance)
    return output
