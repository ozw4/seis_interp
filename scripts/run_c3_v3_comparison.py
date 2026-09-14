"""Execute Study 042 once from the repository root."""

import argparse
from pathlib import Path

import yaml

from seis_interp.pipelines.run_c3_v3_comparison import run_c3_v3_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = Path("studies/study_042_c3_v3_five_method_comparison")
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    configs = {
        method: study / f"{name}.yaml"
        for method, name in (
            ("pocs", "pocs"),
            ("drr", "drr"),
            ("nersi", "nersi_no_time_shear"),
            ("ccnet5d", "ccnet5d"),
            ("relational_trace_graph", "gnn"),
        )
    }
    summary = run_c3_v3_comparison(
        input_paths={key: Path(value) for key, value in inputs.items()},
        configs=configs,
        output_dir=args.output,
    )
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
