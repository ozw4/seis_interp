"""Bind supervised CCNet provenance to a disjoint, verified C3 benchmark."""

from __future__ import annotations

from collections.abc import Mapping

from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER, validated_index_range


def validate_ccnet5d_benchmark_provenance(
    training_provenance: Mapping[str, object], inputs: C3VolumeRunInputs
) -> None:
    """Require the same dataset/split and no spatial teacher/evaluation overlap."""
    source = training_provenance["source_inputs_lock"]
    case = inputs.case
    if source["partition"] != "train" or case["partition"] not in ("validation", "test"):
        raise ValueError("CCNet teachers must use train and benchmark must use validation or test")
    if source["dataset_id"] != case["dataset_id"]:
        raise ValueError("checkpoint dataset_id does not match the benchmark")
    for group in ("interim", "processed"):
        if source[group] != case["input_files"][group]:
            raise ValueError(f"checkpoint {group} hashes do not match the verified benchmark")
    benchmark_selection = inputs.volume_metadata["selection"]
    for name in ("fit", "selection"):
        teacher = source["regions"][name]["selection"]
        overlaps = []
        for axis in VOLUME_AXIS_ORDER[1:]:
            lower, upper = validated_index_range(teacher[axis], name=f"{name}.{axis}")
            benchmark_lower, benchmark_upper = benchmark_selection[axis]
            overlaps.append(lower < benchmark_upper and benchmark_lower < upper)
        if all(overlaps):
            raise ValueError(f"checkpoint {name} region overlaps benchmark trace space")
