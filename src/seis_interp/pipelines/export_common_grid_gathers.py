"""Export central gathers of the common-grid comparison from saved seed predictions.

Visualization only: no inference, no new test scoring and no selection. The
displayed cells come from the grid shape, the methods' cells from the
registered selection files, and every displayed hidden trace is checked
against the per-trace scores saved by its run.
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
from seis_interp.data.study067_analysis_inputs import (
    partition,
    read_json,
    read_yaml,
    require_output_directory,
    saved_traces,
    target_source,
)
from seis_interp.evaluation.study067_trace_metrics import waveform_columns
from seis_interp.visualization.common_grid_gathers import (
    AXES,
    DISPLAY_METHODS,
    LABELS,
    PANEL_TITLES,
    SECTION_AXES,
    central_sections,
    completed_section_panels,
    pooled_reference_clip,
)
from seis_interp.visualization.publication_sections import create_publication_section

NORMALIZATIONS = {"c3": ("physical",), "forge": ("physical", "trace_rms")}
FONT_FAMILY = "DejaVu Sans"
SNR_TOLERANCE_DB = 1e-6
# Provenance comes from the checkout that runs this code; data paths use `repo`.
CODE_ROOT = Path(__file__).resolve().parents[3]
CODE = (
    "src/seis_interp/visualization/common_grid_gathers.py",
    "src/seis_interp/visualization/publication_sections.py",
    "src/seis_interp/pipelines/export_common_grid_gathers.py",
    "src/seis_interp/data/study067_analysis_inputs.py",
    "src/seis_interp/evaluation/study067_trace_metrics.py",
    "scripts/export_study067_gathers.py",
)


def selected_id(experiment, method):
    record = read_json(experiment / "workers" / f"{method}_selection.json")
    if record["test_used_for_selection"]:
        raise ValueError(f"{method}: selection used test data")
    return record["winner"]["id"]


def grid_index(experiment, dataset, spatial_shape):
    """Grid indices per cell, checked against row-major raveling of the grid axes."""
    name = "geometry.parquet" if dataset == "c3" else "manifest.parquet"
    table = pd.read_parquet(experiment / "input" / name, columns=["cell_id", *AXES])
    expected = np.ravel_multi_index(
        tuple(table[axis].to_numpy() for axis in AXES), tuple(spatial_shape)
    )
    if not np.array_equal(expected, table.cell_id.to_numpy()):
        raise ValueError("cell_id is not the row-major index of the grid axes")
    return table


def load_section_rows(repo, experiment, dataset, contract, seed, cells):
    """Reference traces and seed roles for the displayed cells only."""
    assignment = partition(experiment, seed)
    role = "trace_role" if "trace_role" in assignment else "role"
    rows = assignment.set_index("cell_id", verify_integrity=True).loc[cells]
    values, positions, samples = target_source(repo, experiment, dataset, contract)
    reference = np.asarray(values[positions.loc[cells].to_numpy(), :samples], dtype=np.float32)
    table = pd.DataFrame(
        {
            "cell_id": cells,
            "trace_id": rows.trace_id.to_numpy(),
            "role": rows[role].to_numpy(),
            "clean_target": rows.clean_target.to_numpy(dtype=bool),
        }
    )
    observed = table.role.eq("train").to_numpy()
    # The masked input must be the seed's actual model input, not a re-derivation.
    train = pd.Index(assignment.loc[assignment[role].eq("train"), "cell_id"])
    stored = np.load(
        experiment / "input" / f"seed_{seed}" / "observed_waveforms.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if len(stored) != len(train):
        raise ValueError("observed waveforms do not match the seed's observed traces")
    inputs = np.asarray(stored[train.get_indexer(table.cell_id[observed])])
    if not np.array_equal(inputs, reference[observed]):
        raise ValueError("observed input waveforms differ from the reference traces")
    return table, reference, observed


def load_method_rows(repo, experiment, method, seed, table, reference):
    """Saved predictions for hidden display traces, checked against the run's scores."""
    winner = selected_id(experiment, method)
    run = experiment / "method_runs" / method / "evaluation" / f"seed_{seed}"
    manifest = read_json(run / "manifest.json")
    if (manifest["id"], manifest["trace_seed"], manifest["status"]) != (winner, seed, "completed"):
        raise ValueError(f"{method}: seed {seed} run is not the completed selected cell")
    digest = sha256_file(run / "prediction.npy")
    if digest != manifest["artifacts"]["prediction.npy"]:
        raise ValueError(f"{method}: prediction.npy hash differs from its manifest")
    prediction = np.load(run / "prediction.npy", mmap_mode="r", allow_pickle=False)
    rows = pd.Index(np.load(run / "cell_ids.npy", allow_pickle=False)).get_indexer(table.cell_id)
    hidden = table.role.ne("train").to_numpy()
    if not np.array_equal(rows >= 0, hidden):
        raise ValueError(f"{method}: predicted cells differ from the hidden display traces")
    values = np.full(reference.shape, np.nan, dtype=np.float32)
    values[hidden] = prediction[rows[hidden]]
    checks = []
    for domain in ("validation", "test"):
        selected = table.role.eq(domain).to_numpy()
        if not selected.any():
            continue
        saved = saved_traces(run, domain).set_index("cell_id").loc[table.cell_id[selected]]
        columns = waveform_columns(values[selected], reference[selected])
        checks.append(
            {
                "method": method,
                "candidate_id": winner,
                "domain": domain,
                "trace_count": int(selected.sum()),
                "max_abs_snr_db_difference": float(
                    np.max(np.abs(columns["snr_db"] - saved.snr_db.to_numpy()))
                ),
                "max_relative_error_energy_difference": float(
                    np.max(np.abs(columns["error_energy"] / saved.error_energy.to_numpy() - 1))
                ),
            }
        )
    source = {
        "candidate_id": winner,
        "run": str(run.relative_to(repo)),
        "prediction_sha256": digest,
        "manifest_sha256": sha256_file(run / "manifest.json"),
    }
    return values, source, checks


def section_records(sections, table):
    lookup = table.set_index("cell_id")
    rows, summaries = [], []
    for section in sections:
        roles = lookup.loc[section["cell_ids"]]
        rows.extend(
            {"varying_axis": section["varying_axis"], "position": position, "cell_id": cell}
            | lookup.loc[cell].to_dict()
            for position, cell in enumerate(section["cell_ids"])
        )
        summaries.append(
            {
                **section,
                "role_counts": {
                    str(name): int(count) for name, count in roles.role.value_counts().items()
                },
                "non_clean_target_count": int((~roles.clean_target).sum()),
            }
        )
    return pd.DataFrame(rows), summaries


def export_dataset(repo, experiment, dataset, seed, output, dpi):
    contract = read_yaml(experiment / "study/config.yaml")
    index = grid_index(experiment, dataset, contract["spatial_shape"])
    sections = central_sections(index.cell_id, contract["spatial_shape"], SECTION_AXES[dataset])
    cells = sorted({cell for section in sections for cell in section["cell_ids"]})
    table, reference, observed = load_section_rows(repo, experiment, dataset, contract, seed, cells)
    predictions, sources, checks = {}, {}, []
    for method in DISPLAY_METHODS:
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
                    {method: predictions[method][rows] for method in DISPLAY_METHODS},
                    normalization=normalization,
                )
            )
        clips[normalization] = pooled_reference_clip(panel_sets)
        for section, panels in zip(sections, panel_sets, strict=True):
            figure = create_publication_section(
                [panel[keep] for panel in panels],
                time_s=time[keep],
                spatial_label=LABELS[section["varying_axis"]],
                clip=clips[normalization],
                panel_titles=PANEL_TITLES,
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
        "experiment": str(experiment.relative_to(repo)),
        "contract_sha256": sha256_file(experiment / "study/config.yaml"),
        "partition_sha256": sha256_file(
            experiment / "input" / f"seed_{seed}" / "partition.parquet"
        ),
        "spatial_shape": list(contract["spatial_shape"]),
        "time_range_s": [float(lower), float(upper)],
        "sample_interval_s": contract["sample_interval_us"] / 1e6,
        "reference_traces_read_for_display": len(table),
        "methods": sources,
        "sections": summaries,
        "normalizations": list(NORMALIZATIONS[dataset]),
        "clip_abs_amplitude": clips,
        "acceptance_max_abs_snr_db_difference": float(acceptance.max_abs_snr_db_difference.max()),
        "figures": [path.name for path in artifacts],
    }


def export_common_grid_gathers(repo, experiments, output, *, seed=7321, dpi=1200):
    """Write C3 and FORGE central gathers with a manifest under a new output directory."""
    created = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True)
    changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True)
    output = require_output_directory(repo, output)
    datasets = {
        dataset: export_dataset(repo, experiment, dataset, seed, output / dataset, dpi)
        for dataset, experiment in experiments.items()
    }
    write_json(
        output / "manifest.json",
        {
            "created_at": created,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit.strip(),
            "worktree_changes": changes.splitlines(),
            "display_seed": seed,
            "selection_basis": (
                f"registered selected cell, seed {seed}, geometry-only central sections"
            ),
            "visualization_only": True,
            "new_inference": False,
            "new_test_scoring": False,
            "panel_method_order": list(DISPLAY_METHODS),
            "panel_titles": list(PANEL_TITLES),
            "residual_rule": (
                "reference minus completed prediction; observed traces are copied from the "
                "reference, so their residual is zero"
            ),
            "masked_input_rule": (
                "the seed's observed input traces (checked equal to the reference)"
            ),
            "amplitude_rule": (
                "saved physical predictions; FORGE predictions already carry the per-trace "
                "oracle RMS used by the registered score"
            ),
            "trace_rms_normalization": (
                "every panel divided by the reference trace's full-duration RMS; zero RMS uses 1"
            ),
            "clip_rule": (
                "pooled full-duration reference 99th percentile over the dataset's sections"
            ),
            "acceptance_rule": (
                f"per-trace SNR of displayed hidden traces equals the run's saved scores within "
                f"{SNR_TOLERANCE_DB} dB"
            ),
            "layout": [2, 5, 5],
            "width_mm": 178,
            "height_mm": 142,
            "dpi": dpi,
            "font_family": FONT_FAMILY,
            "cmap": "seismic",
            "interpolation": "None",
            "colorbar": False,
            "datasets": datasets,
            "code_sha256": {path: sha256_file(CODE_ROOT / path) for path in CODE},
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return output
