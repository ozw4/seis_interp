"""Re-seal a completed FORGE preparation for a study that adds methods on identical inputs.

The parent's sealed artifacts are verified and copied byte for byte; no waveform
is read. Only three things change: the input contract's ``study_id``, one
resolved config per added method (bound to the parent's shared mask, amplitude
restoration, alignment and global RMS exactly as ``prepare_forge_mvp`` binds its
own methods) and the seal, which records the parent seal's sha256. The new
study's ``config.yaml`` must equal the parent's contract apart from ``study_id``,
so the shared conditions cannot drift. The parent's code identity
(``git_commit``, ``dirty_worktree``, ``implementation_hashes``) moves under
``derived_from``; the copied ``preparation_gates.json`` and
``tables/run_matrix.csv`` remain the parent's records of its own six methods.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.forge_mvp_config import validate_added_method_config, validate_parameter_counts
from seis_interp.processing.forge_mvp_contract import validation_contract
from seis_interp.training.forge_mvp_methods import model_parameter_count

SEAL = "preparation_manifest.json"
CONTRACT = "input/input_contract.json"
# Keys ``prepare_forge_mvp`` writes into every resolved method config, in its order.
BINDING_KEYS = (
    "amplitude_scaling",
    "trace_scale_hash",
    "idw",
    "alignment",
    "mask_hash",
    "mask_id",
    "mask_seed",
    "global_rms",
)


def derive_forge_preparation(repo: Path, parent: Path, study: Path, methods: list[str]) -> Path:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("list each added method once")
    parent_relative = str(parent.resolve().relative_to(repo.resolve()))
    parent_seal = json.loads((parent / SEAL).read_text())
    verify_hashes(parent, parent_seal["artifacts"])
    if parent_seal["status"] != "prepared" or "derived_from" in parent_seal:
        raise ValueError("the parent must be a completed, directly prepared preparation")
    config = yaml.safe_load((study / "config.yaml").read_text())
    validation = parent_seal.get("selection_domain") == "validation_holdout"
    contract = validation_contract(config) if validation else config
    parent_contract = json.loads((parent / CONTRACT).read_text())
    if {**parent_contract, "study_id": config["study_id"]} != contract:
        raise ValueError("the study contract differs from the parent's apart from study_id")
    binding = _shared_binding(parent)
    canvas = parent_seal.get("canvas_sample_count", parent_seal["time_sample_count"])
    resolved, counts = {}, {}
    for method in methods:
        method_config = yaml.safe_load((study / f"{method}.yaml").read_text())
        if method_config["method_id"] != method:
            raise ValueError(f"{method}.yaml declares another method")
        if (parent / "configs" / f"{method}.yaml").exists():
            raise ValueError(f"{method} is already a method of the parent preparation")
        validate_added_method_config(method_config, contract)
        counts[method] = model_parameter_count(method_config, canvas)
        resolved[method] = {**method_config, **binding}
    if "matched_training" in contract:
        validate_parameter_counts(counts, contract["matched_training"])
    identity = execution_identity(repo)
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + identity["git_commit"][:12]
    run = repo / "runs" / config["study_id"] / (run_id + ("_validation" if validation else ""))
    run.mkdir(parents=True, exist_ok=False)
    for relative in parent_seal["artifacts"]:
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative != CONTRACT:
            shutil.copyfile(parent / relative, target)
    write_json(run / CONTRACT, contract)
    for method, method_config in resolved.items():
        path = run / "configs" / f"{method}.yaml"
        path.write_text(yaml.safe_dump(method_config, sort_keys=False))
    for source in sorted(study.glob("*.yaml")):
        target = run / "source" / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    parent_identity = ("git_commit", "dirty_worktree", "implementation_hashes", "artifacts")
    seal = {
        **{key: value for key, value in parent_seal.items() if key not in parent_identity},
        "study_id": config["study_id"],
        "git_commit": identity["git_commit"],
        "dirty_worktree": identity["dirty_worktree"],
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "parameter_counts": {**parent_seal.get("parameter_counts", {}), **counts},
        "derived_from": {
            "preparation": parent_relative,
            "manifest_sha256": sha256_file(parent / SEAL),
            "added_methods": list(methods),
            "derivation_source_sha256": sha256_file(Path(__file__)),
            **{
                f"parent_{key}": parent_seal[key]
                for key in parent_identity[:3]
                if key in parent_seal
            },
        },
        "artifacts": {
            str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()
        },
    }
    write_json(run / SEAL, seal)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return run


def _shared_binding(parent: Path) -> dict:
    """The binding every resolved parent config carries; all methods must agree."""
    configs = [yaml.safe_load(p.read_text()) for p in sorted((parent / "configs").glob("*.yaml"))]
    if not configs:
        raise ValueError("the parent preparation has no method configs")
    binding = {key: configs[0][key] for key in BINDING_KEYS if key in configs[0]}
    if any({key: c.get(key) for key in binding} != binding for c in configs):
        raise ValueError("parent method configs disagree on the shared binding")
    if any(key in c for c in configs for key in set(BINDING_KEYS) - set(binding)):
        raise ValueError("parent method configs disagree on the shared binding keys")
    return binding
