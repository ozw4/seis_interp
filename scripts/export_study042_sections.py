"""Export Study 042's currently adopted comparison in the publication layout."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_v3_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_v3_comparison import validate_v3_run_artifacts
from seis_interp.visualization.c3_central_sections import prepare_central_sections
from seis_interp.visualization.publication_sections import create_publication_section

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/study_042_c3_v3_five_method_comparison"
FORMAL = "runs/study_042_c3_v3_five_method_comparison/20260914T000133Z_03c2fc83df18_formal"
METHODS = ("pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph")
TITLES = [
    "Reference",
    "Masked\ninput",
    "POCS",
    "DRR",
    "NeRSI",
    "CCNet-5D",
    "Proposed\nGNN",
    "Residual\n(POCS)",
    "Residual\n(DRR)",
    "Residual\n(NeRSI)",
    "Residual\n(CCNet-5D)",
    "Residual\n(GNN)",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.findfont("Arial", fallback_to_default=False)
    paths = {
        name: ROOT / value
        for name, value in yaml.safe_load((STUDY / "inputs.yaml").read_text()).items()
    }
    config = load_resolved_config(STUDY / "pocs.yaml")
    inputs = load_c3_random80_v3_inputs(config=config, **paths)
    adopted = {
        "nersi": STUDY / "nersi_v3_result.lock.json",
        "relational_trace_graph": ROOT
        / "studies/study_044_c3_v3_gnn_target_tuning/gnn_v3_result.lock.json",
    }
    runs = {method: ROOT / FORMAL / method for method in METHODS}
    adoption_records = {}
    adopted_locks = {}
    for method, path in adopted.items():
        lock = json.loads(path.read_text())
        runs[method] = ROOT / lock["run_directory"]
        adopted_locks[method] = lock
        for name, digest in lock["run_file_sha256"].items():
            if file_sha256(runs[method] / name) != digest:
                raise ValueError(f"Adopted {method} artifact hash mismatch: {name}")
        adoption_records[method] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
        }
    predictions, metrics = {}, {}
    for method, path in runs.items():
        if method in adopted_locks:
            # Adoption locks pin every artifact, including the GNN EMA checkpoint.
            lock = adopted_locks[method]
            if json.loads((path / "inputs.lock.json").read_text()) != inputs.inputs_lock:
                raise ValueError(f"Adopted {method} input lock differs from verified v3 inputs")
            evaluation = json.loads((path / "metrics.json").read_text())["evaluation_target"]
            if evaluation != lock["evaluation_target"]:
                raise ValueError(f"Adopted {method} metrics differ from the adoption lock")
            if evaluation["covered_target_trace_count"] != inputs.inputs_lock["target_trace_count"]:
                raise ValueError(f"Adopted {method} lacks full target coverage")
            row = {**evaluation, "prediction_sha256": lock["run_file_sha256"]["prediction.npy"]}
        else:
            row = validate_v3_run_artifacts(method, path, inputs.inputs_lock)
        metrics[method] = {
            "run": str(path.relative_to(ROOT)),
            "mean_trace_snr_db": row["mean_trace_snr_db"],
            "prediction_sha256": row["prediction_sha256"],
        }
        predictions[method] = np.load(path / "prediction.npy", mmap_mode="r", allow_pickle=False)
        print(f"{method}: {row['mean_trace_snr_db']:.4f} dB", flush=True)
    volume = inputs.observed_volume
    source = np.load(paths["interim_dir"] / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
    sections, clip = prepare_central_sections(
        reference_amplitudes=source,
        array_rows=volume.array_rows,
        observed_values=volume.values,
        predictions=predictions,
        time_selection=tuple(inputs.volume_metadata["selection"]["time"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    geometry = inputs.index_table.set_index("array_row")
    for section in sections:
        name = section["varying_axis"]
        figure = create_publication_section(
            section.pop("panels"),
            time_s=volume.time_s,
            spatial_label={
                "relative_receiver_y": "Receiver y\nindex",
                "shot_in_line": "Local shot\nin line\nindex",
                "source_line": "Local source\nline index",
            }[name],
            clip=clip,
            panel_titles=TITLES,
            layout="2-5-5",
            show_colorbar=False,
        )
        for extension in ("png", "pdf"):
            path = args.output_dir / f"section_time_{name}.{extension}"
            figure.savefig(path, dpi=1200, facecolor="white")
            artifacts[path.name] = file_sha256(path)
        plt.close(figure)
        section["physical_coordinates_m"] = {
            key: geometry.loc[section["array_rows"], key].tolist()
            for key in (
                "source_x_m",
                "source_y_m",
                "relative_receiver_x_m",
                "relative_receiver_y_m",
            )
        }
    manifest = {
        "study": STUDY.name,
        "selection_basis": "current documented adopted runs; not automatic SNR maximization",
        "independent_evaluation": False,
        "methods": metrics,
        "adoption_locks": adoption_records,
        "inputs_lock": inputs.inputs_lock,
        "sections": sections,
        "time_s": volume.time_s.tolist(),
        "clip_abs_amplitude": clip,
        "clip_rule": "pooled reference 99th percentile across three fixed central sections",
        "panel_titles": TITLES,
        "layout": [2, 5, 5],
        "width_mm": 178,
        "height_mm": 142,
        "dpi": 1200,
        "font": "Arial",
        "cmap": "seismic",
        "interpolation": "None",
        "colorbar": False,
        "artifact_sha256": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created three PNG/PDF pairs: {args.output_dir}")


if __name__ == "__main__":
    main()
