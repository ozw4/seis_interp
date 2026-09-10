"""Bind one frozen validation case to the existing native model input contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.data.c3_benchmark_artifacts import read_benchmark_json
from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_supervised_source,
    load_c3_benchmark_training_graph_domain,
)
from seis_interp.data.c3_benchmark_suite import (
    SUITE_FILE_NAME,
    VerifiedC3BenchmarkSuite,
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.processing.training_coordinates import coordinate_order_for_features
from seis_interp.training.ccnet5d_source_binding import (
    QC_TRAINING_DATASET,
    load_ccnet5d_supervised_source,
)


@dataclass(frozen=True)
class C3FirstResultInputs:
    """Verified paths and metadata; no copied waveforms or generated data artifacts."""

    suite_dir: Path
    manifest: dict
    entry: dict
    paths: dict[str, Path]
    volume: dict
    partition_seed: int
    dataset_id: str
    input_hashes: dict


def read_first_results_yaml(path: Path) -> dict:
    """Read an explicit experiment mapping without inheriting an unrelated study."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "extends" in value:
        raise ConfigurationError(f"{path} must contain an explicit mapping without extends")
    return value


def resolve_c3_first_result_inputs(
    inputs_path: Path,
    *,
    case_id: str | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> C3FirstResultInputs:
    """Verify the expected manifest hash before resolving its case's exact native paths."""
    inputs_path = Path(inputs_path).resolve()
    contract = read_first_results_yaml(inputs_path)
    frozen = contract["frozen_suite"]
    manifest_path = (inputs_path.parent / frozen["manifest"]).resolve()
    if manifest_path.name != SUITE_FILE_NAME:
        raise ConfigurationError(f"frozen_suite.manifest must name {SUITE_FILE_NAME}")
    digest = file_sha256(manifest_path)
    if digest != frozen["expected_sha256"]:
        raise ValueError("frozen suite SHA-256 differs from expected_sha256")
    required = contract["required_cases"]
    if not isinstance(required, list) or len(required) != 1:
        raise ConfigurationError("first results require exactly one fixed validation case")
    case_id = required[0] if case_id is None else case_id
    if case_id != required[0]:
        raise ConfigurationError("case_id differs from the fixed required validation case")
    directory = manifest_path.parent
    suite = load_c3_benchmark_input_manifest(
        directory, case_id=case_id, dimensions=dimensions, verified_suite=verified_suite
    )
    entry = c3_suite_case(suite, case_id)
    if entry["partition"] != "validation":
        raise ConfigurationError("first results only accept a validation case")
    paths = {
        "interim_dir": suite_path(directory, suite["interim"]),
        "processed_dir": suite_path(directory, suite["processed"]),
        **{
            key: suite_path(directory, entry[key]) for key in ("mask_dir", "case_dir", "volume_dir")
        },
    }
    volume = read_benchmark_json(paths["volume_dir"] / "volume.json")
    expected = contract["validation_contract"]
    for key in ("partition", "shape", "selection"):
        if volume[key] != expected[key]:
            raise ConfigurationError(f"validation {key} differs from the frozen input contract")
    if volume["selection"]["time"] != list(dimensions.time_range):
        raise ConfigurationError("validation time differs from the authorized training time")
    if suite["train_pool"]["time_samples"] != list(dimensions.time_range):
        raise ConfigurationError("training pool time differs from the fixed time contract")
    regions = contract.get("ccnet_regions", {})
    if "time_samples" in regions and regions["time_samples"] != list(dimensions.time_range):
        raise ConfigurationError("CCNet region time differs from the authorized training time")
    preparation = read_benchmark_json(paths["processed_dir"] / "preparation.json")
    dataset = read_benchmark_json(paths["interim_dir"] / "dataset.json")
    return C3FirstResultInputs(
        suite_dir=directory,
        manifest=suite,
        entry=entry,
        paths=paths,
        volume=volume,
        partition_seed=preparation["random_seed"],
        dataset_id=dataset["dataset_id"],
        input_hashes={
            "suite": digest,
            "case": file_sha256(paths["case_dir"] / "benchmark_case.json"),
            "volume": {
                name: file_sha256(paths["volume_dir"] / name)
                for name in ("volume.json", "volume_index.parquet")
            },
            "train_pool": file_sha256(suite_path(directory, suite["train_pool"]["file"])),
        },
    )


def build_c3_first_result_native_config(
    action: str,
    *,
    fragment: Mapping,
    binding: C3FirstResultInputs,
    seeds: Mapping,
    ccnet_regions: Mapping | None = None,
) -> dict:
    """Construct only native sections, retaining each API's distinct seed semantics."""
    if seeds["partition"] != binding.partition_seed:
        raise ConfigurationError("experiment partition seed differs from prepared partition")
    sections = _method_sections(action)
    if set(fragment) != sections:
        raise ConfigurationError(f"{action} fragment must contain exactly {sorted(sections)}")
    config = deepcopy(dict(fragment))
    uses_partition = action in ("ccnet-train", "gnn-train", "gnn-preflight", "gnn-predict")
    config["project"] = {
        "random_seed": binding.partition_seed if uses_partition else binding.entry["random_seed"]
    }
    config["data"] = {"dataset_id": binding.dataset_id}
    if action not in ("ccnet-train", "gnn-train", "gnn-preflight"):
        config["benchmark_case"] = {"id": binding.entry["case_id"]}
        config["benchmark_volume"] = {
            "id": binding.volume["volume_id"],
            "selection": deepcopy(binding.volume["selection"]),
        }
        config["interpolation_mask"] = {
            key: binding.entry[key] for key in ("partition", "kind", "missing_fraction")
        }
    if "evaluation" in config:
        evaluation = config["evaluation"]
        expected = {
            key: binding.manifest["evaluation"][key] for key in ("primary_metric", "domain")
        }
        if action in ("gnn-train", "gnn-preflight"):
            expected["query_batch_size"] = evaluation.get("query_batch_size")
        if evaluation != expected:
            raise ConfigurationError("method evaluation differs from the frozen suite contract")
    if "training" in config and config["training"]["random_seed"] != seeds["training"]:
        raise ConfigurationError("training.random_seed differs from the experiment training seed")
    if action == "siren":
        if seeds["siren_sampler"] != seeds["training"]:
            raise ConfigurationError("native SIREN uses the training seed for its sampler")
        features = config["model"]["coordinate_features"]
        if config["model"]["input_features"] != len(coordinate_order_for_features(features)):
            if features == "cmp_offset_azimuth":
                raise ConfigurationError("SIREN cmp_offset_azimuth requires six input features")
            raise ConfigurationError("SIREN input_features must match coordinate_features")
    if action in ("gnn-train", "gnn-preflight"):
        training_data = config["training_data"]
        expected_data = {
            "pool": "all_train_traces",
            "time_samples": binding.manifest["train_pool"]["time_samples"],
        }
        if (
            not isinstance(training_data, Mapping)
            or set(training_data) - (set(expected_data) | {"max_abs_amplitude"})
            or any(training_data.get(key) != value for key, value in expected_data.items())
        ):
            raise ConfigurationError("GNN training must use the exact all_train_traces pool/time")
        if "max_abs_amplitude" in training_data:
            config_values.positive_float(
                training_data["max_abs_amplitude"], "training_data.max_abs_amplitude"
            )
        if seeds["gnn_episodes"] != seeds["training"]:
            raise ConfigurationError("native GNN uses the training seed for its episodes")
    if action == "ccnet-train":
        if config["patches"]["random_seed"] != seeds["ccnet_patches"]:
            raise ConfigurationError("patches.random_seed differs from ccnet_patches seed")
        full_training_dataset = config["supervision"].get("sampling_domain") == QC_TRAINING_DATASET
        required_regions = ("selection",) if full_training_dataset else ("fit", "selection")
        if ccnet_regions is None or not all(ccnet_regions.get(key) for key in required_regions):
            names = "/".join(required_regions)
            raise ConfigurationError(f"CCNet {names} regions must be explicitly resolved")
        if not full_training_dataset:
            config["supervision"]["fit_region"] = deepcopy(ccnet_regions["fit"])
        config["supervision"]["selection_region"] = deepcopy(ccnet_regions["selection"])
    return config


def validate_c3_first_result_training_inputs(
    action: str,
    *,
    binding: C3FirstResultInputs,
    config: Mapping,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
    normalization_checkpoint_path: Path | None = None,
) -> dict:
    """Check teacher regions or exact graph rows against the suite's canonical pool."""
    if action == "ccnet-train":
        if config["supervision"].get("sampling_domain") == QC_TRAINING_DATASET:
            source = load_ccnet5d_supervised_source(
                config,
                interim_dir=binding.paths["interim_dir"],
                processed_dir=binding.paths["processed_dir"],
                suite_dir=binding.suite_dir,
                normalization_checkpoint_path=normalization_checkpoint_path,
                dimensions=dimensions,
            )
            fit_trace_count = source.inputs_lock["training_dataset"]["authorized_trace_count"]
        else:
            source = load_c3_benchmark_supervised_source(
                binding.suite_dir,
                fit_region=config["supervision"]["fit_region"],
                selection_region=config["supervision"]["selection_region"],
                dimensions=dimensions,
                verified_suite=verified_suite,
            )
            fit_trace_count = int(source.fit.array_rows.size)
        return {
            "fit_trace_count": fit_trace_count,
            "selection_trace_count": int(source.selection.array_rows.size),
            "inputs_lock": source.inputs_lock,
        }
    if action in ("gnn-train", "gnn-preflight"):
        domain = load_c3_benchmark_training_graph_domain(
            binding.suite_dir, dimensions=dimensions, verified_suite=verified_suite
        )
        return {"pool": "all_train_traces", "trace_count": len(domain.array_rows)}
    return {}


def _method_sections(action: str) -> set[str]:
    if action in ("pocs", "drr"):
        return {action, "evaluation"}
    if action in ("siren", "nersi"):
        return {"model", "training", "prediction", "evaluation"}
    if action == "ccnet-train":
        return {"model", "supervision", "patches", "training", "selection"}
    if action in ("ccnet-predict", "gnn-predict"):
        return {"prediction", "evaluation"}
    if action in ("gnn-train", "gnn-preflight"):
        return {
            "model",
            "graph",
            "geometry_features",
            "training_data",
            "training_mask",
            "training",
            "evaluation",
        }
    raise ConfigurationError(f"unsupported native action: {action}")
