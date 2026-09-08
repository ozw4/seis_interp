"""Prepare a fixed C3 benchmark through existing partition, mask, case, and volume APIs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp import run_records
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_artifacts import (
    benchmark_file_record,
    benchmark_relative_path,
    load_c3_geometry_inputs,
    write_benchmark_json,
    write_benchmark_yaml,
)
from seis_interp.data.c3_benchmark_suite import INFERENCE_CONTRACT, write_c3_benchmark_suite
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.evaluation.c3_volume_metrics import validate_c3_volume_evaluation_config
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.pipelines.qc_c3 import qc_c3_crop, qc_c3_geometry
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.processing.c3_benchmark_masks import (
    benchmark_case_config,
    effective_c3_crop_mask,
    validated_benchmark_cases,
)
from seis_interp.processing.c3_benchmark_partition import (
    audit_c3_benchmark_partition,
    c3_benchmark_partition_ranges,
)
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER


def prepare_c3_benchmark(
    config_path: Path,
    inputs_path: Path,
    *,
    output_dir: Path | None = None,
    execute: bool = False,
    case_ids: list[str] | None = None,
    plots: bool = False,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict:
    """Default to a read-only plan; only explicit execute writes new C3 artifacts."""
    config = load_resolved_config(config_path)
    inputs = load_resolved_config(inputs_path)
    _validate_preparation_contract(config, inputs, MAIN_C3_DIMENSIONS)
    interim = (inputs_path.parent / inputs["source"]["interim"]).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (inputs_path.parent / inputs["outputs"]["root"]).resolve()
    )
    missing = [
        str(interim / name)
        for name in ("traces.parquet", "time_s.npy", "amplitudes.npy", "dataset.json")
        if not (interim / name).is_file()
    ]
    cases = _selected_cases(validated_benchmark_cases(inputs["cases"]), case_ids)
    plan = {
        "status": "dry_run",
        "ready": not missing and not output.exists(),
        "input": str(interim),
        "output": str(output),
        "missing_inputs": missing,
        "output_exists": output.exists(),
        "fixed_shape": list(MAIN_C3_DIMENSIONS.shape),
        "time_samples": [0, 384],
        "sail_lines_inclusive": [25, 40],
        "start_rule": config["c3_benchmark"]["start_rule"],
        "cases": cases,
        "complete_case_count": len(inputs["cases"]),
        "selected_case_count": len(cases),
    }
    if not execute:
        return plan
    if missing:
        raise FileNotFoundError(f"benchmark interim inputs are missing: {missing}")
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    sources = [inputs_path]
    path = Path(config_path).resolve()
    while path not in sources:
        sources.append(path)
        source_config = yaml.safe_load(path.read_text())
        if "extends" not in source_config:
            break
        path = (path.parent / source_config["extends"]).resolve()
    manifest = inputs.get("source", {}).get("manifest")
    if manifest is not None:
        sources.append((inputs_path.parent / manifest).resolve())
    result = prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=config,
        inputs=inputs,
        dimensions=MAIN_C3_DIMENSIONS,
        case_ids=case_ids,
        plots=plots,
        source_paths=sources,
        progress_reporter=progress_reporter,
    )
    if result["status"] != "locked":
        return {**result, "output": str(output)}
    record = benchmark_file_record(output / "benchmark_suite.json", output)
    return {
        "status": "locked",
        "output": str(output),
        "manifest": record,
        "case_count": len(result["cases"]),
        "shape": result["contract"]["shape"],
        "partition_summary": result["partition_summary"],
    }


def prepare_c3_benchmark_artifacts(
    interim_dir: Path,
    output_dir: Path,
    *,
    config: dict,
    inputs: dict,
    dimensions: C3BenchmarkDimensions,
    case_ids: list[str] | None = None,
    plots: bool = False,
    source_paths: list[Path] | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict:
    """Run preparation with explicit expectations, also usable on tiny synthetic data.

    The public CLI always supplies MAIN_C3_DIMENSIONS. SEG C3 NA inputs cannot use
    the smaller expected dimensions used by independent synthetic fixtures.
    """
    _validate_preparation_contract(config, inputs, dimensions)
    recipes = validated_benchmark_cases(inputs["cases"])
    selected = _selected_cases(recipes, case_ids)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    config = deepcopy(config)
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    report = progress_reporter or (lambda message: None)
    generation = {
        **run_records.current_git_metadata(),
        "created_at_utc": run_records.utc_timestamp(),
    }
    sources = []
    try:
        if source_paths:
            (output / "sources").mkdir()
            for number, path in enumerate(source_paths):
                destination = output / "sources" / f"{number}_{path.name}"
                shutil.copyfile(path, destination)
                record = benchmark_file_record(destination, output)
                sources.append(
                    {
                        "file": record["path"],
                        "sha256": record["sha256"],
                        "source_path": benchmark_relative_path(path, output),
                    }
                )
        crop_summaries, ranges = _prepare_crop_artifacts(
            interim_dir, output, config, inputs, dimensions, plots, report, completed
        )
        write_benchmark_yaml(output / "config.resolved.yaml", config)
        write_benchmark_yaml(output / "inputs.resolved.yaml", deepcopy(inputs))
        partition_config = {
            key: deepcopy(config[key]) for key in ("project", "data", "normalization", "sampling")
        }
        write_benchmark_yaml(output / "partition.config.yaml", partition_config)
        sources.append(
            {
                "file": "partition.config.yaml",
                "source_path": "config.resolved.yaml",
                "sha256": benchmark_file_record(output / "partition.config.yaml", output)["sha256"],
            }
        )
        report("Preparing the unfiltered source-line partition once for all cases.")
        preparation = prepare_baseline_dataset(
            interim_dir,
            output / "partition",
            holdout_fraction=None,
            validation_fraction_of_holdout=None,
            random_seed=config["project"]["random_seed"],
            split_scope="c3_source_line_blocks",
            source_line_ranges=ranges,
            coordinate_normalization=config["normalization"]["coordinates"],
            amplitude_normalization=config["normalization"]["amplitude"],
        )
        table, _ = load_c3_geometry_inputs(interim_dir)
        crops = {
            key: pd.read_parquet(output / "qc" / key / "crop_index.parquet")
            for key in crop_summaries
        }
        pool, partition_summary = audit_c3_benchmark_partition(
            table,
            pd.read_parquet(output / "partition" / "trace_split.parquet"),
            preparation,
            crops,
            time_range=dimensions.time_range,
        )
        np.save(output / "train_pool.npy", pool, allow_pickle=False)
        write_benchmark_json(output / "partition_summary.json", partition_summary)
        completed.append("partition")
        (output / "case_configs").mkdir()
        entries = []
        for recipe in selected:
            case_id = recipe["case_id"]
            report(f"Preparing case {len(entries) + 1}/{len(selected)}: {case_id}")
            entries.append(
                _prepare_case_artifacts(interim_dir, output, config, recipe, crop_summaries, crops)
            )
            completed.append(case_id)
        if len(entries) != len(recipes):
            partial = {
                "status": "partial",
                "completed": completed,
                "missing_cases": [
                    recipe["case_id"] for recipe in recipes if recipe not in selected
                ],
            }
            write_benchmark_json(output / "preparation_report.json", partial)
            return partial
        suite = {
            "status": "locked",
            "path_base": "manifest_directory",
            "generation": generation,
            "contract": config["c3_benchmark"],
            "inference": config["c3_benchmark"]["inference"],
            "evaluation": config["evaluation"],
            "interim": benchmark_relative_path(interim_dir, output),
            "processed": "partition",
            "geometry": "qc/geometry",
            "crops": {"test": "qc/test", "validation": "qc/validation"},
            "partition_summary": partition_summary,
            "train_pool": {
                "file": "train_pool.npy",
                "trace_count": len(pool),
                "time_samples": list(dimensions.time_range),
            },
            "cases": entries,
            "source_records": sources,
        }
        report("Recomputing file hashes and checking every case before writing the suite manifest.")
        locked = write_c3_benchmark_suite(output, suite, dimensions=dimensions)
        completed.append("benchmark_suite")
        write_benchmark_json(
            output / "preparation_report.json", {"status": "locked", "completed": completed}
        )
        return locked
    except Exception as error:
        write_benchmark_json(
            output / "preparation_report.json",
            {"status": "failed", "completed": completed, "error": str(error)},
        )
        raise


def _validate_preparation_contract(
    config: dict, inputs: dict, dimensions: C3BenchmarkDimensions
) -> None:
    dimensions.validate(config)
    validate_c3_volume_evaluation_config(config)
    if config["c3_benchmark"]["inference"] != INFERENCE_CONTRACT:
        raise ValueError("benchmark inference must use only same-crop observed amplitudes")
    if (
        config["sampling"]["split_scope"] != "c3_source_line_blocks"
        or "trace_amplitude_filter" in config["sampling"]
    ):
        raise ValueError("benchmark requires unfiltered c3_source_line_blocks partitions")
    validated_benchmark_cases(inputs["cases"])


def _selected_cases(recipes: list[dict], case_ids: list[str] | None) -> list[dict]:
    if case_ids is None:
        return recipes
    unknown = set(case_ids) - {recipe["case_id"] for recipe in recipes}
    if unknown or not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError(
            f"requested case IDs must be unique configured cases; unknown={sorted(unknown)}"
        )
    return [recipe for recipe in recipes if recipe["case_id"] in case_ids]


def _prepare_crop_artifacts(
    interim_dir, output, config, inputs, dimensions, plots, report, completed
):
    """Resolve and record geometry/crop choices before preparing any mask."""
    report("Checking the fixed sail-line mapping and sample grid.")
    geometry = qc_c3_geometry(
        interim_dir,
        output / "qc" / "geometry",
        requested_inclusive=dimensions.sail_line_numbers,
        time_range=dimensions.time_range,
        original_number_column=inputs.get("sail_line_numbering", {}).get("original_number_column"),
    )
    completed.append("geometry")
    config["c3_benchmark"]["sail_lines"] = geometry["sail_lines"]
    line_range = tuple(geometry["sail_lines"]["index_range"])
    config["benchmark_volume"]["selection"]["source_line"] = list(line_range)
    dimensions.validate(config)
    ranges = c3_benchmark_partition_ranges(line_range, geometry["sail_lines"]["source_line_count"])
    config["sampling"]["source_line_ranges"] = {key: list(value) for key, value in ranges.items()}
    report("Resolving the test crop by geometry and checking its fixed time samples.")
    test_crop = qc_c3_crop(
        interim_dir,
        output / "qc" / "test",
        source_line_range=line_range,
        time_range=dimensions.time_range,
        spatial_lengths=dimensions.shape[2:],
        explicit_ranges=config["benchmark_volume"]["selection"],
        plots=plots,
    )
    config["benchmark_volume"]["selection"] = test_crop["selection"]
    validation_range = ranges["validation"]
    count = min(dimensions.shape[1], validation_range[1] - validation_range[0])
    start = validation_range[0] + (validation_range[1] - validation_range[0] - count) // 2
    report("Resolving the validation crop inside its separate source-line block.")
    validation_crop = qc_c3_crop(
        interim_dir,
        output / "qc" / "validation",
        source_line_range=(start, start + count),
        time_range=dimensions.time_range,
        spatial_lengths=dimensions.shape[2:],
        plots=plots,
    )
    crop_summaries = {"test": test_crop, "validation": validation_crop}
    completed.extend(["test_crop", "validation_crop"])
    return crop_summaries, ranges


def _prepare_case_artifacts(interim_dir, output, config, recipe, crop_summaries, crops):
    """Bind one configured mask, case, and volume through the existing public APIs."""
    case_id = recipe["case_id"]
    mask_dir = output / "masks" / case_id
    case_dir = output / "cases" / case_id
    volume_dir = output / "volumes" / case_id
    case_config = benchmark_case_config(
        config, recipe, crop_summaries[recipe["partition"]]["selection"]
    )
    config_file = output / "case_configs" / f"{case_id}.yaml"
    write_benchmark_yaml(config_file, case_config)
    prepare_interpolation_mask(
        interim_dir,
        output / "partition",
        mask_dir,
        **{key: recipe[key] for key in ("partition", "kind", "missing_fraction", "random_seed")},
    )
    prepare_benchmark_case(interim_dir, output / "partition", mask_dir, case_dir, case_id=case_id)
    selection = case_config["benchmark_volume"]["selection"]
    prepare_c3_volume_index(
        interim_dir,
        output / "partition",
        mask_dir,
        case_dir,
        volume_dir,
        volume_id=case_config["benchmark_volume"]["id"],
        **{f"{axis}_range": tuple(selection[axis]) for axis in VOLUME_AXIS_ORDER},
    )
    mask, mask_metadata = load_interpolation_mask(mask_dir)
    return {
        **recipe,
        "mask_dir": benchmark_relative_path(mask_dir, output),
        "case_dir": benchmark_relative_path(case_dir, output),
        "volume_dir": benchmark_relative_path(volume_dir, output),
        "config_file": benchmark_relative_path(config_file, output),
        "partition_counts": mask_metadata["counts"],
        "effective_mask": effective_c3_crop_mask(
            crops[recipe["partition"]], mask, kind=recipe["kind"]
        ),
    }
