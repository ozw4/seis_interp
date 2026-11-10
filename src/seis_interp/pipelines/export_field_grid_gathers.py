"""Export the common-grid gather comparison with the conditional field as the implicit panel.

Visualization only: no inference, no new test scoring and no selection. This is
study_067's figure with its neural-implicit panel taken from the study that
replaced the representative method, so each method's panel comes from its own
run. Every run must share the displayed seed's partition byte for byte, or the
panels would not be the same traces, and that is checked here rather than assumed.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.data.study067_analysis_inputs import read_yaml, require_output_directory
from seis_interp.pipelines.export_common_grid_gathers import (
    CODE_ROOT,
    FONT_FAMILY,
    NORMALIZATIONS,
    SNR_TOLERANCE_DB,
    grid_index,
    load_method_rows,
    load_section_rows,
    section_records,
)
from seis_interp.visualization.common_grid_gathers import (
    FIELD_DISPLAY_METHODS,
    FIELD_METHOD_TITLES,
    LABELS,
    SECTION_AXES,
    central_sections,
    completed_section_panels,
    panel_titles,
    pooled_reference_clip,
)
from seis_interp.visualization.publication_sections import create_publication_section

CODE = (
    "src/seis_interp/visualization/common_grid_gathers.py",
    "src/seis_interp/visualization/publication_sections.py",
    "src/seis_interp/pipelines/export_common_grid_gathers.py",
    "src/seis_interp/pipelines/export_field_grid_gathers.py",
    "src/seis_interp/data/study067_analysis_inputs.py",
    "src/seis_interp/evaluation/study067_trace_metrics.py",
    "scripts/export_field_grid_gathers.py",
)


def shared_partition(runs, seed):
    """One partition digest for every contributing run, or the disagreement."""
    digests = {
        str(run): sha256_file(run / "input" / f"seed_{seed}" / "partition.parquet")
        for run in dict.fromkeys(runs)
    }
    if len(set(digests.values())) != 1:
        raise ValueError(f"contributing runs disagree on the displayed partition: {digests}")
    return digests


def export_dataset(repo, experiments, dataset, seed, output, dpi):
    """One dataset's sections, with each method's panel loaded from its own run."""
    if tuple(experiments) != FIELD_DISPLAY_METHODS:
        raise ValueError(f"experiments must follow the display order {FIELD_DISPLAY_METHODS}")
    partitions = shared_partition(experiments.values(), seed)
    # Roles, reference traces and the masked input are the same in every run, so read
    # them once from the run that supplies the proposed method's panel.
    host = experiments["regsi_real"]
    contract = read_yaml(host / "study/config.yaml")
    index = grid_index(host, dataset, contract["spatial_shape"])
    sections = central_sections(index.cell_id, contract["spatial_shape"], SECTION_AXES[dataset])
    cells = sorted({cell for section in sections for cell in section["cell_ids"]})
    table, reference, observed = load_section_rows(repo, host, dataset, contract, seed, cells)
    predictions, sources, checks = {}, {}, []
    for method, experiment in experiments.items():
        predictions[method], sources[method], method_checks = load_method_rows(
            repo, experiment, method, seed, table, reference
        )
        checks.extend(method_checks)
    acceptance = pd.DataFrame(checks)
    failed = acceptance.max_abs_snr_db_difference.gt(SNR_TOLERANCE_DB)
    if failed.any():
        raise ValueError(f"display arrays disagree with saved scores:\n{acceptance[failed]}")
    time = np.arange(contract["time_sample_count"]) * contract["sample_interval_us"] / 1e6
    lower, upper = contract.get("visualization", {}).get("time_range_s", (time[0], time[-1]))
    keep = (time >= lower) & (time <= upper)
    output.mkdir()
    lookup = pd.Index(table.cell_id)
    clips, artifacts = {}, []
    for normalization in NORMALIZATIONS[dataset]:
        panel_sets = []
        for section in sections:
            rows = lookup.get_indexer(section["cell_ids"])
            panel_sets.append(
                completed_section_panels(
                    reference[rows],
                    observed[rows],
                    {method: predictions[method][rows] for method in FIELD_DISPLAY_METHODS},
                    normalization=normalization,
                    display_methods=FIELD_DISPLAY_METHODS,
                )
            )
        clips[normalization] = pooled_reference_clip(panel_sets)
        for section, panels in zip(sections, panel_sets, strict=True):
            figure = create_publication_section(
                [panel[keep] for panel in panels],
                time_s=time[keep],
                spatial_label=LABELS[section["varying_axis"]],
                clip=clips[normalization],
                panel_titles=panel_titles(FIELD_METHOD_TITLES),
                layout="2-5-5",
                show_colorbar=False,
                font_family=FONT_FAMILY,
            )
            for extension in ("png", "pdf"):
                path = output / f"gather_{section['varying_axis']}_{normalization}.{extension}"
                figure.savefig(path, dpi=dpi, facecolor="white")
                artifacts.append(path)
            plt.close(figure)
    rows, summaries = section_records(sections, table)
    rows.to_csv(output / "section_traces.csv", index=False)
    acceptance.to_csv(output / "acceptance.csv", index=False)
    return {
        "host_experiment": str(host.relative_to(repo)),
        "contract_sha256": sha256_file(host / "study/config.yaml"),
        "partition_sha256": partitions[str(host)],
        "contributing_run_partition_sha256": partitions,
        "spatial_shape": list(contract["spatial_shape"]),
        "time_range_s": [float(lower), float(upper)],
        "sample_interval_s": contract["sample_interval_us"] / 1e6,
        "reference_traces_read_for_display": len(table),
        "sections": summaries,
        "methods": sources,
        "clip": clips,
        "artifacts": {
            str(path.relative_to(output.parent)): sha256_file(path) for path in artifacts
        },
    }


def export_field_grid_gathers(repo, experiments, output, *, seed=7321, dpi=1200):
    """Write both datasets' figures and one manifest describing every displayed rule."""
    repo = Path(repo).resolve()
    if set(experiments) != set(NORMALIZATIONS):
        raise ValueError(f"experiments must cover exactly {sorted(NORMALIZATIONS)}")
    created = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True)
    changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True)
    output = require_output_directory(repo, output)
    manifest = {
        "created_at": created,
        "git_commit": commit.strip(),
        "worktree_changes": changes.splitlines(),
        "display_seed": seed,
        "selection_basis": "registered selected cell, seed 7321, geometry-only central sections",
        "visualization_only": True,
        "new_inference": False,
        "new_test_scoring": False,
        "panel_method_order": list(FIELD_DISPLAY_METHODS),
        "panel_titles": list(panel_titles(FIELD_METHOD_TITLES)),
        "replaces": {
            "figure": "results/study_067_forge_c3_common_grid/"
            "20261104T011853Z_35d86916bcc7_gathers",
            "panel": "nersi_profile replaced by cinr_real",
        },
        "residual_rule": "reference minus completed prediction; observed traces are copied "
        "from the reference, so their residual is zero",
        "masked_input_rule": "the seed's stored observed waveforms, checked against the reference",
        "amplitude_rule": "saved physical predictions; FORGE carries the registered per-trace "
        "oracle RMS",
        "trace_rms_normalization": "every panel divided by the full-duration reference RMS "
        "per trace, dividing by one where that RMS is zero",
        "clip_rule": "pooled full-duration reference 99th percentile over the dataset's sections",
        "acceptance_rule": f"displayed arrays reproduce each run's saved per-trace SNR within "
        f"{SNR_TOLERANCE_DB} dB",
        "layout": [2, 5, 5],
        "width_mm": 178,
        "height_mm": 142,
        "dpi": dpi,
        "font_family": FONT_FAMILY,
        "cmap": "seismic",
        "interpolation": None,
        "colorbar": False,
        "datasets": {},
        "code_sha256": {name: sha256_file(CODE_ROOT / name) for name in CODE},
    }
    for dataset in sorted(experiments):
        manifest["datasets"][dataset] = export_dataset(
            repo, experiments[dataset], dataset, seed, output / dataset, dpi
        )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    return output
