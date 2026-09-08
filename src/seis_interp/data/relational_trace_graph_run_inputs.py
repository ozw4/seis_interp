"""Bind trace graph run declarations and checkpoint provenance to verified domains."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from seis_interp.configuration import ConfigurationError
from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.prepared_partition import PREPARATION_FILE_NAME
from seis_interp.data.trace_graph_domain import TraceGraphDomain


def trace_graph_run_input_metadata(
    domain: TraceGraphDomain,
    *,
    config: Mapping[str, object],
    processed_dir: Path,
    case_dir: Path | None = None,
    expected_partition: str | None = None,
) -> dict[str, object]:
    """Check declarations after domain hash verification, without reading amplitudes."""
    preparation = json.loads((Path(processed_dir) / PREPARATION_FILE_NAME).read_text("utf-8"))
    if config["project"]["random_seed"] != preparation["random_seed"]:
        raise ConfigurationError("project.random_seed does not match the prepared partition seed")
    if config["data"]["dataset_id"] != preparation["dataset_id"]:
        raise ConfigurationError("data.dataset_id does not match the prepared dataset")
    partition = domain.inputs_lock["partition"]
    if expected_partition is not None and partition != expected_partition:
        raise ValueError(f"expected {expected_partition} partition, got {partition}")
    metadata = {
        "dataset_id": preparation["dataset_id"],
        "partition": partition,
        "partition_random_seed": preparation["random_seed"],
        "time_samples": list(domain.time_samples),
        "time_s": domain.time_s.tolist(),
        "trace_count": len(domain.trace_ids),
        "observed_trace_count": int(domain.observed_mask.sum()),
        "query_count": int(domain.query_mask.sum()),
    }
    if case_dir is not None:
        case = load_benchmark_case(case_dir)
        if "benchmark_case" in config and config["benchmark_case"]["id"] != case["case_id"]:
            raise ConfigurationError("benchmark_case.id does not match the verified case")
        if "interpolation_mask" in config:
            actual = {"partition": case["partition"], **case["mask"]}
            for key, value in config["interpolation_mask"].items():
                if value != actual[key]:
                    raise ConfigurationError(f"interpolation_mask.{key} does not match the case")
        metadata.update(case_id=case["case_id"], mask=deepcopy(case["mask"]))
    volume = domain.inputs_lock.get("benchmark_volume")
    if "benchmark_volume" in config:
        declared = config["benchmark_volume"]
        if (
            volume is None
            or declared["id"] != volume["volume_id"]
            or declared["selection"] != volume["selection"]
        ):
            raise ConfigurationError(
                "benchmark_volume declaration does not match the selected volume"
            )
    if volume is not None:
        metadata["benchmark_volume"] = deepcopy(volume)
    return metadata


def validate_trace_graph_checkpoint_provenance(
    training_provenance: Mapping[str, object], domain: TraceGraphDomain
) -> None:
    """Require frozen benchmark data to share the checkpoint's verified dataset/split."""
    source = training_provenance.get("source_inputs_lock")
    if not isinstance(source, Mapping) or source.get("partition") != "train":
        raise ValueError("benchmark checkpoint requires verified train source_inputs_lock")
    if domain.inputs_lock.get("partition") not in ("validation", "test"):
        raise ValueError("frozen benchmark requires a validation or test partition")
    source_files = source.get("input_files")
    if source_files is None:
        source_files = source.get("benchmark_case", {}).get("input_files")
    benchmark_files = domain.inputs_lock.get("benchmark_case", {}).get("input_files")
    if not isinstance(source_files, Mapping) or not isinstance(benchmark_files, Mapping):
        raise ValueError("checkpoint and benchmark require verified input_files")
    for group in ("interim", "processed"):
        if group not in source_files or source_files[group] != benchmark_files.get(group):
            raise ValueError(f"checkpoint {group} hashes do not match the verified benchmark")
