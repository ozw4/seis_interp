"""Reverify NeRSI tuning runs and append them to the recorded comparison."""

import argparse
from pathlib import Path

from seis_interp.configuration import load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import (
    read_poc_json,
    validate_poc_run_artifacts,
    write_poc_comparison,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output must not already exist")
    study = Path(__file__).resolve().parent
    previous = read_poc_json(args.previous_summary)
    baseline, lock = validate_poc_run_artifacts(
        "nersi", Path(previous["baseline"]["output_directory"])
    )
    frozen = read_poc_json(
        study.parent / "study_037_c3_neural_mse_loss_ablation/stage_1b_results.lock.json"
    )
    if lock != previous["inputs_lock"] or lock != frozen["inputs_lock"]:
        raise ValueError("baseline, frozen inputs and previous comparison differ")
    paths = [Path(row["output_directory"]) for row in previous["runs"]] + args.run
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("comparison contains duplicate run directories")
    archives = dict(previous["source_archives"])
    for path, expected in archives.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"source archive changed: {path}")
    rows = []
    for path in paths:
        row, current_lock = validate_poc_run_artifacts("nersi", path)
        if current_lock != lock:
            raise ValueError(f"different full input lock: {path}")
        config = load_resolved_config(path / "config.resolved.yaml")
        if config != load_resolved_config(study / f"{path.name}.yaml"):
            raise ValueError(f"candidate configuration differs: {path}")
        metadata = read_poc_json(path / "metadata.json")
        metrics = read_poc_json(path / "metrics.json")
        recorded_optimization = metadata["method_details"].get("optimization")
        expected_optimization = config.get("optimization")
        if expected_optimization is not None:
            expected_optimization = {
                **expected_optimization,
                "prediction_weights": "final_ema"
                if expected_optimization["ema_decay"] is not None
                else "final_raw",
                "ema_initialization": "first_post_update_weights",
            }
        model = metadata["method_details"].get("model", {})
        if (
            recorded_optimization != expected_optimization
            or ("coordinate_mapping" in model)
            != (config.get("profile_coordinates") == "cartesian_cmp_half_offset")
            or ("axis_frequency_limits" in model) != ("fourier_bandlimit" in config)
        ):
            raise ValueError(f"optimization or coordinate encoding differs: {path}")
        if (
            config["training"]["loss"] != "masked_trace_mse"
            or metadata["loss_or_native_objective"] != "masked_trace_mse"
            or metadata["training_or_reconstruction"]["loss"] != "masked_trace_mse"
            or row["optimizer_updates"] != config["training"]["max_steps"]
            or metrics["observed_max_abs_error"] != 0
            or metadata["method_details"].get("time_alignment") != config.get("time_alignment")
            or metadata["training_or_reconstruction"].get("augmentation")
            != config.get("augmentation")
        ):
            raise ValueError(f"loss, budget, observations or preprocessing differs: {path}")
        row.update(
            candidate=path.name,
            config=config,
            config_sha256=file_sha256(path / "config.resolved.yaml"),
            delta_snr_db=row["snr_db"] - baseline["snr_db"],
        )
        rows.append(row)
        archive = path.parent / "nersi_source.tar"
        if archive.exists():
            archives[str(archive)] = file_sha256(archive)
    best = max(rows, key=lambda row: row["snr_db"])
    summary = {
        **previous,
        "baseline": baseline,
        "target_snr_db": 15.0,
        "target_comparison": "strictly_greater",
        "target_reached": best["snr_db"] > 15.0,
        "best_candidate": best["candidate"],
        "best_output_directory": best["output_directory"],
        "runs": rows,
        "source_archives": archives,
        "total_optimizer_updates": sum(row["optimizer_updates"] for row in rows),
        "total_training_seconds": sum(row["training_or_reconstruction_seconds"] for row in rows),
        "previous_comparison": {
            "path": str(args.previous_summary),
            "sha256": file_sha256(args.previous_summary),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_poc_comparison(args.output, summary)
    print(f"{len(rows)} verified runs; best {best['candidate']}: {best['snr_db']:.10f} dB")


if __name__ == "__main__":
    main()
