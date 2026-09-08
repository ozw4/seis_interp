"""Validate finite C3 case recipes and measure effective crop visibility."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from seis_interp.config_values import nonnegative_integer, positive_float
from seis_interp.data.benchmark_case_store import validated_case_id
from seis_interp.processing.interpolation_masks import MASK_KINDS, validate_interpolation_mask


def validated_benchmark_cases(value: object) -> list[dict]:
    """Keep configured case order and mask seeds, without reading amplitudes."""
    if not isinstance(value, list) or not value:
        raise ValueError("cases must be a nonempty finite list")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "partition",
            "kind",
            "missing_fraction",
            "random_seed",
        }:
            raise ValueError(
                "each case must contain case_id, partition, kind, missing_fraction, random_seed"
            )
        validated_case_id(item["case_id"])
        if item["partition"] not in ("test", "validation") or item["kind"] not in MASK_KINDS:
            raise ValueError("case must use a test/validation partition and an existing mask kind")
        fraction = positive_float(item["missing_fraction"], "missing_fraction")
        if fraction >= 1:
            raise ValueError("missing_fraction must be less than 1")
        nonnegative_integer(item["random_seed"], "random_seed")
        result.append(deepcopy(item))
    if len({case["case_id"] for case in result}) != len(result):
        raise ValueError("case IDs must be unique")
    if {case["partition"] for case in result} != {"test", "validation"}:
        raise ValueError("case list must include test and validation")
    return result


def benchmark_case_config(config: dict, recipe: dict, selection: dict) -> dict:
    """Translate one inputs.yaml case into a CLI config, preserving suite settings."""
    return {
        "project": {"random_seed": recipe["random_seed"]},
        "data": deepcopy(config["data"]),
        "interpolation_mask": {
            key: recipe[key] for key in ("partition", "kind", "missing_fraction")
        },
        "benchmark_case": {"id": recipe["case_id"]},
        "benchmark_volume": {"id": recipe["case_id"] + "_volume", "selection": deepcopy(selection)},
        "evaluation": deepcopy(config["evaluation"]),
    }


def effective_c3_crop_mask(index: pd.DataFrame, mask: pd.DataFrame, *, kind: str) -> dict:
    """Join trace roles by canonical array_row, independent of waveform values."""
    if kind not in MASK_KINDS:
        raise ValueError("unsupported mask kind")
    validate_interpolation_mask(
        mask, expected_array_rows=mask["array_row"].to_numpy(dtype=np.int64)
    )
    selected = index[["array_row", "ffid"]].merge(
        mask, on="array_row", how="left", validate="one_to_one", sort=False
    )
    if selected["observation_role"].isna().any():
        raise ValueError("crop contains rows outside the partition mask")
    observed = int(selected["observation_role"].eq("observed").sum())
    target = int(selected["observation_role"].eq("evaluation_target").sum())
    if not observed or not target:
        raise ValueError("crop must contain both observed and evaluation target traces")
    groups = selected.groupby("ffid")["observation_role"]
    whole = groups.apply(lambda values: bool(values.eq("evaluation_target").all()))
    if kind == "random_whole_ffid" and groups.nunique().gt(1).any():
        raise ValueError("whole-FFID mask must hide every candidate trace of each selected FFID")
    return {
        "trace_count": len(selected),
        "observed_trace_count": observed,
        "target_trace_count": target,
        "trace_missing_fraction": target / len(selected),
        "ffid_count": len(whole),
        "fully_missing_ffid_count": int(whole.sum()),
        "fully_missing_ffid_fraction": float(whole.mean()),
    }


def validate_partition_mask_ffids(
    canonical_partition: pd.DataFrame, mask: pd.DataFrame, *, kind: str
) -> None:
    """Check whole-FFID atomicity over the entire candidate domain before cropping."""
    validate_interpolation_mask(
        mask, expected_array_rows=canonical_partition["array_row"].to_numpy(dtype=np.int64)
    )
    if kind == "random_whole_ffid":
        joined = canonical_partition[["array_row", "ffid"]].merge(
            mask, on="array_row", validate="one_to_one"
        )
        if joined.groupby("ffid")["observation_role"].nunique().gt(1).any():
            raise ValueError("whole-FFID mask splits a candidate FFID")
