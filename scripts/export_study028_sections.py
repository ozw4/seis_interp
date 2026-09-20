"""Re-render the adopted Study 028 sections from their recorded native arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from seis_interp.data.file_checksums import file_sha256
from seis_interp.visualization.publication_sections import create_publication_section

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize"
AXES = ("source_line", "shot_in_line", "relative_receiver_x", "relative_receiver_y")
METHODS = ("pocs", "drr", "siren5d", "ccnet5d", "relational_trace_graph")
PANEL_TITLES = [
    "Reference",
    "Masked observed input",
    "POCS",
    "Reference − POCS",
    "DRR",
    "Reference − DRR",
    "SIREN",
    "Reference − SIREN",
    "CCNet-5D",
    "Reference − CCNet-5D",
    "Relational trace graph",
    "Reference − Relational trace graph",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, help="Optional local Arial TTF file")
    parser.add_argument("--display", choices=("image", "wiggle"), default="image")
    parser.add_argument("--columns", type=int, choices=(2, 6), default=2)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--layout", choices=("grid", "2-5-5"), default="grid")
    args = parser.parse_args()
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if args.font_file:
        font_manager.fontManager.addfont(str(args.font_file))
    font_path = font_manager.findfont("Arial")
    font_family = font_manager.FontProperties(fname=font_path).get_name()
    summary = json.loads((SOURCE / "first_results.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    suite_path = ROOT / summary["suite_path"]
    suite = json.loads(suite_path.read_text())
    reference_path = (suite_path.parent / suite["interim"] / "amplitudes.npy").resolve()
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    native_paths = {
        method: ROOT / path / "artifacts/prediction.npy"
        for method, path in manifest["native_run_paths"].items()
    }
    predictions = {
        method: np.load(path, mmap_mode="r", allow_pickle=False)
        for method, path in native_paths.items()
    }
    for row in summary["rows"]:
        method = row["method"]
        if method in native_paths and file_sha256(native_paths[method]) != row["prediction_sha256"]:
            raise ValueError(f"Recorded prediction hash mismatch: {method}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start, stop = summary["binding"]["selection"]["time"]
    clip = summary["figures"]["display_clip_abs_amplitude"]
    artifacts = {}
    order = list(range(12)) if args.columns == 2 else [*range(0, 12, 2), *range(1, 12, 2)]
    if args.layout == "2-5-5":
        order = [0, 1, *range(2, 12, 2), *range(3, 12, 2)]
    titles = PANEL_TITLES
    if args.columns == 6 or args.layout == "2-5-5":
        titles = [
            "Reference",
            "Masked\ninput",
            "POCS",
            "Residual\n(POCS)",
            "DRR",
            "Residual\n(DRR)",
            "SIREN",
            "Residual\n(SIREN)",
            "CCNet-5D",
            "Residual\n(CCNet-5D)",
            "Relational\ntrace graph",
            "Residual\n(trace graph)",
        ]
    titles = [titles[index] for index in order]
    for section in summary["figures"]["sections"]:
        selection = (
            slice(None),
            *(
                slice(None)
                if name == section["varying_axis"]
                else section["fixed_local_indices"][name]
                for name in AXES
            ),
        )
        truth = np.asarray(reference[section["array_rows"], start:stop], dtype=np.float64).T
        panels = [truth, predictions["zero_fill"][selection]]
        for method in METHODS:
            prediction = np.asarray(predictions[method][selection], dtype=np.float64)
            panels.extend((prediction, truth - prediction))
        spatial_label = f"Local {section['varying_axis'].replace('_', ' ')} index"
        if section["varying_axis"] == "relative_receiver_y":
            spatial_label = "Receiver y index"
        if args.columns == 6 or args.layout == "2-5-5":
            spatial_label = {
                "relative_receiver_y": "Receiver y\nindex",
                "shot_in_line": "Local shot\nin line\nindex",
                "source_line": "Local source\nline index",
            }[section["varying_axis"]]
        figure = create_publication_section(
            [panels[index] for index in order],
            time_s=np.asarray(summary["figures"]["time_s"]),
            spatial_label=spatial_label,
            clip=clip,
            font_family=font_family,
            panel_titles=titles,
            display=args.display,
            columns=args.columns,
            show_colorbar=not args.no_colorbar,
            layout=args.layout,
        )
        for extension in ("png", "pdf"):
            path = args.output_dir / Path(section["file"]).with_suffix(f".{extension}")
            figure.savefig(path, dpi=1200, facecolor="white")
            artifacts[path.name] = file_sha256(path)
        plt.close(figure)
    record = {
        "source_results": str(SOURCE.relative_to(ROOT)),
        "source_summary_sha256": file_sha256(SOURCE / "first_results.json"),
        "source_summary_run": manifest["source_summary_run"],
        "native_run_paths": manifest["native_run_paths"],
        "sections": summary["figures"]["sections"],
        "width_mm": 178,
        "dpi": 1200,
        "display": args.display,
        "colorbar": args.display == "image" and not args.no_colorbar,
        "interpolation": "None" if args.display == "image" else "linear trace segments",
        "resolved_interpolation": "none" if args.display == "image" else "linear trace segments",
        "axis_labels": {"y": "once per row", "x": "once per column"},
        "axis_ticks": False,
        "axis_tick_labels": True,
        "panel_titles": titles,
        "layout": {"rows": 12 // args.columns, "columns": args.columns},
        "panel_letter_position": "outside upper-left corner",
        "panel_letter_style": {"fontsize": 10, "fontweight": "bold"},
        "font_requested": "Arial",
        "font_used": font_family,
        "cmap": "seismic" if args.display == "image" else None,
        "display_clip_abs_amplitude": clip,
        "clip_rule": summary["figures"]["clip_rule"],
        "panels_row_major": [
            "Reference",
            "Masked observed input",
            *(label for method in METHODS for label in (method, f"Reference minus {method}")),
        ],
        "artifact_sha256": artifacts,
    }
    record["panels_row_major"] = [record["panels_row_major"][index] for index in order]
    if args.layout == "2-5-5":
        record["layout"] = {"rows": 3, "columns_per_row": [2, 5, 5]}
        record["axis_labels"]["x"] = "bottom method columns only; top panels retain tick values"
    if args.display == "wiggle":
        record["wiggle"] = {
            "trace_spacing": 1,
            "maximum_excursion_trace_spacings": 0.45,
            "gain": "shared across all traces and panels; no per-trace normalization",
            "positive_lobes": "black fill with linearly located zero crossings",
            "linewidth_pt": 0.35,
            "clipping": "display only; original amplitude arrays unchanged",
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Created three PNG/PDF pairs in {args.output_dir}; font: {font_family}")


if __name__ == "__main__":
    main()
