"""Bind replacement methods to continuing C3 workers without changing frozen runs."""

import json
from pathlib import Path

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json


def bind_c3_continuing_experiment(repo: Path, experiment: Path, relative: str, methods):
    """Require identical frozen inputs before referring to the remaining method workers."""
    source = (repo / relative).resolve()
    if not source.is_relative_to(repo.resolve()) or source == experiment.resolve():
        raise ValueError("continuing experiment must be another run inside the repository")
    seal = json.loads((source / "experiment.json").read_text())
    hashes = {name: value for name, value in seal["artifacts"].items() if name.startswith("input/")}
    verify_hashes(source, hashes)
    verify_hashes(experiment, hashes)
    for method in methods:
        path = source / "workers" / f"{method}.json"
        state = json.loads(path.read_text())
        if state["status"] != "stopped":
            raise ValueError(f"replacement requires a stopped original worker: {method}")
    return {
        "experiment": str(source.relative_to(repo.resolve())),
        "experiment_sha256": sha256_file(source / "experiment.json"),
        "replaced_methods": list(methods),
    }


def c3_method_experiments(experiment: Path) -> dict[str, Path]:
    """Resolve each method's current source from the immutable replacement binding."""
    seal = json.loads((experiment / "experiment.json").read_text())
    binding = seal.get("continuing_experiment")
    if binding is None:
        return {method: experiment for method in seal.get("methods", C3_FAIR_GRID_METHODS)}
    repo = experiment.parents[2]
    source = (repo / binding["experiment"]).resolve()
    verify_hashes(source, {"experiment.json": binding["experiment_sha256"]})
    return {
        method: experiment if method in binding["replaced_methods"] else source
        for method in C3_FAIR_GRID_METHODS
    }


def publish_c3_replacement_readiness(experiment: Path) -> None:
    """Expose replacement S1 readiness to the original workers' existing barrier."""
    seal = json.loads((experiment / "experiment.json").read_text())
    binding = seal["continuing_experiment"]
    source = experiment.parents[2] / binding["experiment"]
    for method in binding["replaced_methods"]:
        current_path = experiment / "workers" / f"{method}.json"
        if not current_path.is_file():
            continue
        current = json.loads(current_path.read_text())
        original_path = source / "workers" / f"{method}.json"
        original = json.loads(original_path.read_text())
        if original["status"] not in ("stopped", "replaced"):
            raise ValueError(f"cannot replace an active original worker: {method}")
        if current.get("s1_completed"):
            if current["completed_s1"] != 25 or not current.get("selected_configuration"):
                raise ValueError("replacement readiness requires all 25 outcomes and a selection")
            original.update(
                status="replaced",
                phase="replaced",
                original_s1_completed=False,
                s1_completed=True,
                replacement_experiment=str(experiment.relative_to(experiment.parents[2])),
                replacement_worker_pid=current["pid"],
                replacement_completed_s1=current["completed_s1"],
                replacement_configuration=current["selected_configuration"],
            )
            temporary = original_path.with_suffix(".replacement.tmp")
            write_json(temporary, original)
            temporary.replace(original_path)
