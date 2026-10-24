"""Read-only seed inventory, pruning blinded experiments before traversal."""

import json
import os
import re
from pathlib import Path

import yaml


def seed_occurrences(value, seeds, prefix=()):
    """Only return seed-related scalar fields/partition paths, never metric values."""
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(seed_occurrences(item, seeds, (*prefix, str(key))))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            rows.extend(seed_occurrences(item, seeds, (*prefix, str(i))))
    else:
        field = ".".join(prefix)
        for seed in seeds:
            scalar = str(value) == str(seed)
            path = isinstance(value, str) and re.search(
                rf"(?:seed|partition|mask)[_/=-]{seed}(?!\d)", value
            )
            key_path = re.search(rf"(?:seed|partition|mask)[_/=-]{seed}(?!\d)", field) or re.search(
                rf"(?:partitions|split_seeds)[./]{seed}(?:[./]|$)", field
            )
            if not (
                (scalar and re.search("seed|partition|split|mask|sampling", field, re.I))
                or path
                or key_path
            ):
                continue
            if "evaluation_seeds" in field:
                kind = "evaluation_seed_declaration"
            elif re.search("model_seed|training.seed|bootstrap|rng", field, re.I):
                kind = "non_partition_seed"
            else:
                kind = "partition_seed_use"
            rows.append({"seed": seed, "field": field, "kind": kind})
    return rows


def audit_seed_namespaces(repo, excluded_experiments, seeds=(7321, 7322, 7323)):
    """Search all experiment/manifest JSON and YAML/config JSON under runs/studies.

    Excluded roots are removed from os.walk's directory list before descending;
    no seed output, summary, or directory within them is enumerated or read.
    Frozen source snapshots are included in the inventory.
    """
    excluded = {Path(p).resolve() for p in excluded_experiments}
    rows, errors, files = [], [], []
    for tree in (repo / "runs", repo / "studies"):
        for parent, directories, names in os.walk(tree, topdown=True, followlinks=False):
            base = Path(parent)
            directories[:] = sorted(
                d
                for d in directories
                if (base / d).resolve() not in excluded and not (base / d).is_symlink()
            )
            for name in sorted(names):
                path = base / name
                lower = name.lower()
                selected = path.suffix in (".yaml", ".yml") or (
                    path.suffix == ".json"
                    and (name == "experiment.json" or "manifest" in lower or "config" in lower)
                )
                if not selected or path.is_symlink():
                    continue
                files.append(str(path.relative_to(repo)))
                try:
                    contents = path.read_text()
                    # Avoid parsing thousands of irrelevant frozen method YAMLs.
                    if not any(re.search(rf"(?<!\d){seed}(?!\d)", contents) for seed in seeds):
                        continue
                    value = (
                        json.loads(contents) if path.suffix == ".json" else yaml.safe_load(contents)
                    )
                    rows.extend(
                        {"path": str(path.relative_to(repo)), **row}
                        for row in seed_occurrences(value, seeds)
                    )
                except (OSError, ValueError, yaml.YAMLError) as error:
                    errors.append(
                        {"path": str(path.relative_to(repo)), "error_type": type(error).__name__}
                    )
    uses = [row for row in rows if row["kind"] == "partition_seed_use"]
    return {
        "seeds": list(seeds),
        "excluded_experiments": [
            str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p) for p in sorted(excluded)
        ],
        "scanned_file_count": len(files),
        "scanned_files": files,
        "occurrences": rows,
        "partition_seed_uses": uses,
        "errors": errors,
        "passed": not uses and not errors,
        "method": (
            "top-down os.walk; prune blinded roots before descent; inspect all YAML plus "
            "experiment/manifest/config JSON, including frozen snapshots; classify seed "
            "scalars and seed/partition/mask paths"
        ),
    }
