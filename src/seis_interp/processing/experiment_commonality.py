"""Validate one shared grid and explicitly justified dataset differences."""

import hashlib
from pathlib import Path

import yaml


def flatten_settings(value, prefix=""):
    """Keep list indices explicit so a time length exception cannot hide spatial changes."""
    if isinstance(value, (dict, list)) and value:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return {
            key: leaf
            for name, child in items
            for key, leaf in flatten_settings(child, f"{prefix}.{name}".lstrip(".")).items()
        }
    return {prefix: value}


def check_fixed_settings(left, right, allowed, label):
    if any(not isinstance(reason, str) or not reason.strip() for reason in allowed.values()):
        raise ValueError("every dataset difference requires a reason")
    first, second = flatten_settings(left), flatten_settings(right)
    missing = object()
    differences = {
        key
        for key in first.keys() | second.keys()
        if first.get(key, missing) != second.get(key, missing)
    }
    undeclared = differences - allowed.keys()
    if undeclared:
        raise ValueError(f"{label}: undeclared fixed differences: {sorted(undeclared)}")


def load_experiment_grid(study: Path, contract: dict) -> dict:
    """Existing studies keep their local grid; common studies pin the shared bytes."""
    binding = contract.get("common_grid")
    if binding is None:
        return yaml.safe_load((study / "grid.yaml").read_text())
    shared = (study / binding["path"]).resolve()
    digest = hashlib.sha256(shared.read_bytes()).hexdigest()
    if digest != binding["sha256"]:
        raise ValueError("common grid hash differs from the declared grid")
    local = study / "grid.yaml"
    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() != digest:
        raise ValueError("dataset grid hash differs from the common grid")
    root = shared.parent
    differences = yaml.safe_load((root / "dataset_differences.yaml").read_text())
    contracts = [
        yaml.safe_load((root / dataset / "config.yaml").read_text()) for dataset in ("c3", "forge")
    ]
    if any(item.get("common_grid", {}).get("sha256") != digest for item in contracts):
        raise ValueError("both datasets must declare the same grid hash")
    for dataset, item in zip(("c3", "forge"), contracts, strict=True):
        folder = root / dataset
        if (folder / item["common_grid"]["path"]).resolve() != shared:
            raise ValueError("both datasets must use the same shared grid file")
        local_grid = folder / "grid.yaml"
        if local_grid.exists() and hashlib.sha256(local_grid.read_bytes()).hexdigest() != digest:
            raise ValueError("dataset grid hash differs from the common grid")
    grid = yaml.safe_load(shared.read_text())
    for method in grid["methods"]:
        configs = [
            yaml.safe_load((root / dataset / f"{method}.yaml").read_text())
            for dataset in ("c3", "forge")
        ]
        check_fixed_settings(*configs, differences["methods"].get(method, {}), method)
    execution_keys = {
        "study_id",
        "common_grid",
        "import_from",
        "cuda_execution",
        "method_execution",
        "reference_experiment",
        "visualization",
    }
    check_fixed_settings(
        *[
            {key: value for key, value in item.items() if key not in execution_keys}
            for item in contracts
        ],
        differences["inputs"],
        "inputs",
    )
    return grid
