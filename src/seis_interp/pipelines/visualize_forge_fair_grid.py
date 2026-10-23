"""Export paper figures from the completed FORGE fair-grid final evaluation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seis_interp.data.forge_fair_grid_gathers import (
    load_evaluated_gather_inputs,
    load_gather_predictions,
    read_gather_reference,
)
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.visualization.forge_publication_gathers import (
    DISPLAY_METHODS,
    LABELS,
    PANEL_TITLES,
    central_gather_cells,
    prepare_publication_gathers,
)
from seis_interp.visualization.publication_sections import create_publication_section


def visualize_forge_fair_grid(
    repo, evaluation, output, *, seed=7311, dpi=1200, font_family="DejaVu Sans"
):
    """Four central gathers in physical and shared-reference-RMS views, PNG/PDF."""
    start = utc_now()
    record, preparation, contract, index = load_evaluated_gather_inputs(repo, evaluation)
    sections = central_gather_cells(index, tuple(contract["spatial_shape"]))
    cells = {cell for section in sections for cell in section["cell_ids"]}
    index = index.loc[index.cell_id.isin(cells)].reset_index(drop=True)
    predictions, sources = load_gather_predictions(repo, record, index, seed=seed)
    reference = read_gather_reference(repo, preparation, contract, index)
    time = np.arange(contract["time_sample_count"]) * contract["sample_interval_us"] / 1e6
    lower, upper = contract["visualization"]["time_range_s"]
    keep = (time >= lower) & (time <= upper)
    output.mkdir(parents=True, exist_ok=False)
    index.to_parquet(output / "gather_index.parquet", index=False)
    np.savez_compressed(output / "gathers.npz", reference=reference, time_s=time, **predictions)
    clips = {}
    for mode in ("trace_rms", "physical"):
        prepared, clip = prepare_publication_gathers(
            index, reference, predictions, sections, normalization=mode
        )
        clips[mode] = clip
        for section in prepared:
            name = section["varying_axis"]
            figure = create_publication_section(
                [panel[keep] for panel in section["panels"]],
                time_s=time[keep],
                spatial_label=LABELS[name],
                clip=clip,
                panel_titles=PANEL_TITLES,
                layout="2-6-6",
                show_colorbar=False,
                font_family=font_family,
            )
            for extension in ("png", "pdf"):
                figure.savefig(
                    output / f"gather_{name}_{mode}.{extension}", dpi=dpi, facecolor="white"
                )
            plt.close(figure)
        sections = [{k: v for k, v in section.items() if k != "panels"} for section in prepared]
    for module in (
        "data/forge_fair_grid_gathers.py",
        "visualization/forge_publication_gathers.py",
        "visualization/publication_sections.py",
        "pipelines/visualize_forge_fair_grid.py",
    ):
        path = Path("src/seis_interp") / module
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / path).read_bytes())
    write_json(
        output / "manifest.json",
        {
            **execution_identity(repo),
            "status": "completed",
            "start_time": start,
            "end_time": utc_now(),
            "evaluation": str(evaluation.relative_to(repo)),
            "evaluation_manifest_sha256": sha256_file(evaluation / "manifest.json"),
            "preparation": record["preparation"],
            "preparation_hash": record["preparation_hash"],
            "display_seed": seed,
            "selection_basis": "same fixed seed; geometry-only central sections",
            "visualization_only": True,
            "new_inference": False,
            "new_test_scoring": False,
            "reference_traces_read_for_display": len(index),
            "methods": sources,
            "panel_method_order": list(DISPLAY_METHODS),
            "panel_titles": list(PANEL_TITLES),
            "sections": sections,
            "layout": [2, 6, 6],
            "width_mm": 178,
            "height_mm": 142,
            "time_range_s": [lower, upper],
            "dpi": dpi,
            "font_family": font_family,
            "clip_abs_amplitude": clips,
            "clip_rule": "pooled full-duration reference 99th percentile over four sections",
            "trace_rms_normalization": (
                "all panels share each reference trace's full-duration RMS; zero RMS uses divisor 1"
            ),
            "residual_sign": "reference minus completed prediction; observed columns are zero",
            "cmap": "seismic",
            "interpolation": "None",
            "colorbar": False,
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return output
