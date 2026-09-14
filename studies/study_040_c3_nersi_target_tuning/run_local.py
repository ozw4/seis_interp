"""Run one explicitly configured local NeRSI experiment on the fixed inputs."""

import argparse
import json
import tarfile
from pathlib import Path

import yaml

from seis_interp.pipelines.interpolate_local_nersi import interpolate_local_nersi_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".source.tar").exists():
        raise FileExistsError("local output or source snapshot already exists")
    study = Path(__file__).resolve().parent
    paths = {
        k + "_dir": (study / v).resolve()
        for k, v in yaml.safe_load((study / "inputs.yaml").read_text())["paths"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output.with_suffix(".source.tar"), "x") as archive:
        for path in sorted(Path("src/seis_interp").rglob("*.py")):
            archive.add(path)
        for path in sorted(study.glob("*.yaml")):
            archive.add(path)
        archive.add(Path(__file__))
    frozen = json.loads(
        (
            study.parent / "study_037_c3_neural_mse_loss_ablation/stage_1b_results.lock.json"
        ).read_text()
    )["inputs_lock"]
    interpolate_local_nersi_run(
        config_path=args.config,
        output_dir=args.output,
        input_paths=paths,
        expected_inputs_lock=frozen,
    )


if __name__ == "__main__":
    main()
