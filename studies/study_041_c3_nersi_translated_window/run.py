"""Prepare and run one translated window, then verify checkpoint restoration."""

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_artifacts import write_benchmark_json
from seis_interp.data.c3_poc_inputs import load_c3_random80_window_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.qc_c3 import qc_c3_crop
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.c3_volume_nersi_data import build_c3_volume_nersi_data
from seis_interp.training.c3_volume_nersi_prediction import predict_c3_volume_nersi
from seis_interp.training.nersi_checkpoints import (
    load_fixed_step_nersi_checkpoint,
    validate_fixed_step_nersi_checkpoint_input_binding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.data_output,
        args.output,
        args.output.with_suffix(".source.tar"),
        args.output.with_suffix(".log"),
        args.output.with_suffix(".audit.json"),
    ):
        if path.exists():
            raise FileExistsError(path)
    config = load_resolved_config(args.config)
    selection = config["benchmark_volume"]["selection"]
    interim = Path("data/interim/c3_na/all_ffids")
    qc = Path("data/processed/c3_na/study_029_c3_amplitude_qc")
    paths = dict(
        interim_dir=interim,
        processed_dir=qc / "partition",
        mask_dir=qc / "masks/c3_benchmark_test_random_trace_80_seed42",
        case_dir=args.data_output / "case",
        volume_dir=args.data_output / "volume",
    )
    qc_c3_crop(
        interim,
        args.data_output / "qc",
        source_line_range=(25, 41),
        time_range=(0, 384),
        spatial_lengths=(32, 8, 32),
        explicit_ranges=selection,
    )
    prepare_benchmark_case(
        interim,
        paths["processed_dir"],
        paths["mask_dir"],
        paths["case_dir"],
        case_id=config["benchmark_case"]["id"],
    )
    prepare_c3_volume_index(
        **{key: value for key, value in paths.items() if key != "volume_dir"},
        output_dir=paths["volume_dir"],
        volume_id=config["benchmark_volume"]["id"],
        **{f"{axis}_range": tuple(selection[axis]) for axis in VOLUME_AXIS_ORDER},
    )
    inputs = load_c3_random80_window_inputs(config=config, **paths)
    frozen = json.loads(
        Path("studies/study_037_c3_neural_mse_loss_ablation/stage_1b_results.lock.json").read_text()
    )["inputs_lock"]
    if inputs.inputs_lock["mask"] != frozen["mask"]:
        raise ValueError("outer mask differs from the frozen reference")
    write_benchmark_json(args.data_output / "inputs.lock.json", inputs.inputs_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output.with_suffix(".source.tar"), "x") as archive:
        for path in sorted(Path("src/seis_interp").rglob("*.py")):
            archive.add(path)
        archive.add(args.config)
        archive.add(Path(__file__))
    command = [
        sys.executable,
        "-m",
        "seis_interp.cli",
        "interpolate",
        "nersi",
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--json",
    ]
    for key, path in paths.items():
        command.extend(["--" + key.removesuffix("_dir"), str(path)])
    print(f"Training: {args.output}", flush=True)
    with args.output.with_suffix(".log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = json.loads((args.output / "metadata.json").read_text())
    metrics = json.loads((args.output / "metrics.json").read_text())
    if metadata["status"] != "success":
        raise ValueError("run did not succeed")
    if json.loads((args.output / "inputs.lock.json").read_text()) != inputs.inputs_lock:
        raise ValueError("run used different verified inputs")
    for artifact in metadata["artifacts"].values():
        if file_sha256(args.output / artifact["path"]) != artifact["sha256"]:
            raise ValueError("artifact hash mismatch")
    loaded = load_fixed_step_nersi_checkpoint(args.output / "final.pt", device="cpu")
    observed = inputs.observed_volume
    scale = compute_observed_global_rms(observed.values, observed.observed_trace_mask)
    data = build_c3_volume_nersi_data(
        observed, amplitude_scale=scale, time_alignment=config.get("time_alignment")
    )
    validate_fixed_step_nersi_checkpoint_input_binding(loaded, inputs.inputs_lock, data)
    resources = metadata["resource_usage"]
    torch.backends.cudnn.benchmark = resources["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = resources["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = resources["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(resources["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = resources["cuda_matmul_allow_tf32"]
    restored = predict_c3_volume_nersi(
        loaded.model,
        data,
        observed,
        batch_size=config["prediction"]["batch_size"],
        device=config["training"]["device"],
    )
    saved = np.load(args.output / "prediction.npy", mmap_mode="r")
    maximum = float(np.max(np.abs(restored.values - saved)))
    np.testing.assert_allclose(restored.values, saved, rtol=1e-6, atol=1e-6)
    rescored = evaluate_c3_volume_prediction(
        restored.values,
        observed,
        interim_dir=interim,
        volume_metadata=inputs.volume_metadata,
        target_coverage_mask=observed.evaluation_target_trace_mask,
        include_trace_snr=True,
    )
    if rescored != metrics:
        raise ValueError("restored evaluation differs from saved metrics")
    write_benchmark_json(
        args.output.with_suffix(".audit.json"),
        {
            "status": "success",
            "run_directory": str(args.output),
            "max_abs_restoration_error": maximum,
            "metrics": rescored,
            "source_archive_sha256": file_sha256(args.output.with_suffix(".source.tar")),
        },
    )
    print(json.dumps(rescored, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
