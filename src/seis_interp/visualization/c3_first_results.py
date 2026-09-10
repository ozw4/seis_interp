"""Fixed-geometry C3 pilot figures with shared reference-derived contrast."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs

_METHODS = ("pocs", "drr", "siren5d", "ccnet5d", "relational_trace_graph")
_AXES = ("source_line", "shot_in_line", "relative_receiver_x", "relative_receiver_y")
_COORDINATES = ("source_x_m", "source_y_m", "relative_receiver_x_m", "relative_receiver_y_m")


def plot_c3_first_results(
    *,
    inputs: C3VolumeRunInputs,
    interim_dir: Path,
    predictions: Mapping[str, np.ndarray],
    histories: Mapping[str, list],
    output_dir: Path,
) -> dict:
    """Read only fixed sections/ID examples at this visualization boundary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    volume = inputs.observed_volume
    source = np.load(Path(interim_dir) / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
    start, stop = inputs.volume_metadata["selection"]["time"]
    centers = tuple(size // 2 for size in volume.array_rows.shape)
    sections = []
    for axis in (3, 1, 0):
        selection = tuple(
            slice(None) if position == axis else center for position, center in enumerate(centers)
        )
        rows = volume.array_rows[selection]
        reference = np.asarray(source[rows, start:stop], dtype=np.float64).T
        sections.append((axis, selection, rows, reference))
    reference_abs = np.concatenate([np.abs(item[3]).ravel() for item in sections])
    clip = float(np.percentile(reference_abs, 99))
    display_clip = clip if clip > 0.0 else 1.0
    display_methods = (
        *_METHODS,
        *(method for method in predictions if method not in _METHODS),
    )
    records = []
    for axis, selection, rows, reference in sections:
        figure, axes = plt.subplots(
            1 + len(display_methods),
            2,
            figsize=(12, 3 * (1 + len(display_methods))),
            constrained_layout=True,
        )
        _image(axes[0, 0], reference, volume.time_s, display_clip, "Reference", _AXES[axis])
        _image(
            axes[0, 1],
            volume.values[(slice(None), *selection)],
            volume.time_s,
            display_clip,
            "Masked observed input",
            _AXES[axis],
        )
        for position, method in enumerate(display_methods, start=1):
            if method not in predictions:
                for subplot, role in zip(axes[position], ("prediction", "residual"), strict=True):
                    subplot.text(
                        0.5,
                        0.5,
                        f"{method}: no accepted full-volume {role}",
                        ha="center",
                        va="center",
                        transform=subplot.transAxes,
                    )
                    subplot.set_axis_off()
                continue
            prediction = np.asarray(
                predictions[method][(slice(None), *selection)], dtype=np.float64
            )
            _image(axes[position, 0], prediction, volume.time_s, display_clip, method, _AXES[axis])
            _image(
                axes[position, 1],
                reference - prediction,
                volume.time_s,
                display_clip,
                f"{method}: reference minus prediction",
                _AXES[axis],
            )
        figure.suptitle(
            f"Fixed central section: time–{_AXES[axis]} (local index); "
            f"shared amplitude clip ±{display_clip:.4g}"
        )
        filename = f"section_time_{_AXES[axis]}.png"
        figure.savefig(output / filename, dpi=110)
        plt.close(figure)
        geometry = inputs.index_table.set_index("array_row").loc[rows]
        target_count = int(volume.evaluation_target_trace_mask[selection].sum())
        records.append(
            {
                "file": filename,
                "varying_axis": _AXES[axis],
                "fixed_local_indices": {
                    name: centers[position]
                    for position, name in enumerate(_AXES)
                    if position != axis
                },
                "array_rows": rows.tolist(),
                "physical_coordinates_m": {name: geometry[name].tolist() for name in _COORDINATES},
                "target_trace_count": target_count,
                "target_coverage_note": "No target traces on this fixed section."
                if target_count == 0
                else "All targets retained; section fixed by geometry before amplitude inspection.",
            }
        )
    traces = _plot_traces(plt, inputs, source, predictions, output)
    curves = _plot_curves(plt, histories, output)
    return {
        "status": "created",
        "sections": records,
        "traces": traces,
        "learning_curves": curves,
        "clip_rule": (
            "99th percentile of absolute reference amplitudes pooled over three fixed sections; "
            "same symmetric clip for input, predictions and residuals"
        ),
        "clip_abs_amplitude": clip,
        "display_clip_abs_amplitude": display_clip,
        "time_s": volume.time_s.tolist(),
        "geometry_note": (
            "Local source-line and shot-in-line indices retain source stagger; "
            "shot-in-line is not an independent absolute source-y Cartesian axis. "
            "Physical source coordinates are recorded for every displayed trace."
        ),
    }


def _image(
    axis, values: np.ndarray, time_s: np.ndarray, clip: float, title: str, spatial_axis: str
) -> None:
    step = float(time_s[1] - time_s[0]) if len(time_s) > 1 else 0.008
    axis.imshow(
        values,
        cmap="gray",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=[
            -0.5,
            values.shape[1] - 0.5,
            float(time_s[-1] + step / 2),
            float(time_s[0] - step / 2),
        ],
    )
    axis.set(title=title, xlabel=f"Local {spatial_axis} index", ylabel="Time (s)")
    axis.set_xticks(
        np.unique(np.linspace(0, values.shape[1] - 1, min(values.shape[1], 6), dtype=int))
    )


def _plot_traces(
    plt,
    inputs: C3VolumeRunInputs,
    source: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    output: Path,
) -> dict:
    volume = inputs.observed_volume
    targets = np.sort(volume.array_rows[volume.evaluation_target_trace_mask])[:3]
    positions = {int(row): position for position, row in enumerate(volume.array_rows.ravel())}
    start, stop = inputs.volume_metadata["selection"]["time"]
    references = np.asarray(source[targets, start:stop], dtype=np.float64)
    reference_max = float(np.max(np.abs(references)))
    reference_limit = 1.05 * (reference_max if reference_max > 0 else 1.0)
    out_of_range = {method: {} for method in predictions}
    figure, axes = plt.subplots(
        len(targets), 2, figsize=(16, 3.5 * len(targets)), squeeze=False, constrained_layout=True
    )
    for position, row in enumerate(targets):
        spatial = np.unravel_index(positions[int(row)], volume.array_rows.shape)
        predicted_traces = {
            method: np.asarray(values[(slice(None), *spatial)], dtype=np.float64)
            for method, values in predictions.items()
        }
        outside = []
        for method, values in predicted_traces.items():
            count = int(np.count_nonzero(np.abs(values) > reference_limit))
            out_of_range[method][str(int(row))] = count
            if count:
                outside.append(f"{method}: {count}/{len(values)}")
        for column, axis in enumerate(axes[position]):
            axis.plot(
                volume.time_s,
                references[position],
                color="black",
                linewidth=1.8,
                label="reference",
            )
            for method, values in predicted_traces.items():
                displayed = (
                    np.ma.masked_outside(values, -reference_limit, reference_limit)
                    if column == 0
                    else values
                )
                axis.plot(volume.time_s, displayed, linewidth=0.8, label=method)
            view = "Reference-range view" if column == 0 else "Full-amplitude view"
            axis.set(
                title=f"Target array_row / trace_id {row}; {view}",
                xlabel="Time (s)",
                ylabel="Physical amplitude",
            )
            axis.legend(fontsize=7, ncol=3)
        axes[position, 0].set_ylim(-reference_limit, reference_limit)
        axes[position, 0].text(
            0.01,
            0.98,
            "Off-scale samples (hidden; no connecting lines): "
            + ("; ".join(outside) if outside else "none"),
            transform=axes[position, 0].transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    figure.suptitle("Fixed target waveforms; unchanged physical amplitudes in both views")
    figure.savefig(output / "target_traces.png", dpi=130)
    plt.close(figure)
    geometry = inputs.index_table.set_index("array_row").loc[targets]
    return {
        "file": "target_traces.png",
        "selection_rule": "first three target array_row IDs in ascending order",
        "target_ids": targets.tolist(),
        "physical_coordinates_m": {name: geometry[name].tolist() for name in _COORDINATES},
        "reference_range_view": {
            "ylim": [-reference_limit, reference_limit],
            "reference_max_abs_amplitude": reference_max,
            "margin_fraction": 0.05,
            "zero_reference_fallback_abs_amplitude": 1.0,
            "limit_rule": (
                "shared symmetric limit from the fixed target references: max absolute value "
                "with 5 percent margin; unit fallback when all zero"
            ),
            "line_rendering": "off-scale samples hidden without connecting across hidden samples",
            "out_of_range_sample_counts": out_of_range,
            "out_of_range_total_sample_counts": {
                method: sum(counts.values()) for method, counts in out_of_range.items()
            },
        },
        "full_amplitude_view": "linear autoscale of all methods and reference per target",
        "amplitude_transformation": (
            "none; reference view hides off-scale samples, full view retains every sample"
        ),
    }


def _plot_curves(plt, histories: Mapping[str, list], output: Path) -> dict:
    labels = {
        "siren5d": "Observed-volume normalized point MSE",
        "ccnet5d": "Normalized complete-patch MSE",
        "relational_trace_graph": "Cumulative normalized masked-query MSE",
    }
    if "nersi" in histories:
        labels["nersi"] = "Observed-volume normalized profile masked MSE"
    records = {}
    for method, label in labels.items():
        history = histories.get(method, [])
        samples = [
            (item["step"], item["train_loss"])
            for item in history
            if "step" in item and "train_loss" in item
        ]
        if not samples:
            records[method] = {
                "status": "unavailable",
                "reason": "No native training-history samples available.",
                "loss_meaning": label,
            }
            continue
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot([item[0] for item in samples], [item[1] for item in samples], marker=".")
        axis.set(xlabel="Optimizer step", ylabel=label, title=method)
        filename = f"learning_curve_{method}.png"
        figure.savefig(output / filename, dpi=130)
        plt.close(figure)
        records[method] = {
            "status": "created",
            "file": filename,
            "loss_meaning": label,
            "sample_count": len(samples),
        }
    return records
