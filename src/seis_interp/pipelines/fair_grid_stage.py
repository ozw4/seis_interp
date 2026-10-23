"""Generate, track and close the stages of the pre-registered fair grid search.

Layout under ``runs/<study>/fair_grid/<method>/``::

    <stage>/stage.json, <stage>/<run>.yaml   one matched-tuning search yaml per run
    final/stage.json, final/<run>.yaml        the selection record and its final runs

A planned run is complete when exactly one completed matched-tuning run under
``runs/<study>/`` holds a byte-identical copy of its yaml and used the stage's
preparation. A run whose tuning failed with a reason matching the grid's
``selection.diverged_runs.failure_reason_pattern`` diverged: it is terminal
(scored +inf, see ``fair_grid_rules``) and refuses any re-run of its yaml; any
other failure (e.g. out of memory) leaves the run open for a re-run.
A stage is generated only from finished earlier stages, and every
generated run is checked before it is written: the matched budget and parameter
band (``matched_candidate_config`` on the aligned canvas) at the validation and
final observed counts, a multiple of the loss traces per update hidden per ReGSI
episode, and at least that many hidden traces in every CCNet5D patch placement.
Each ``stage.json`` records the grid.yaml sha256, the rule
inputs (scores and their tuning runs) and the decision, so the choices replay.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, write_json
from seis_interp.data.forge_mvp_run_records import read_search_results
from seis_interp.forge_mvp_config import TUNABLE_NEURAL_METHODS
from seis_interp.pipelines.tune_forge_matched import TUNABLE_SECTIONS, matched_candidate_config
from seis_interp.processing.fair_grid_rules import (
    STAGES,
    PlannedRun,
    RunRecord,
    apply_overrides,
    compose_stage,
    decay_key,
    final_overrides,
    hidden_trace_count,
    is_classical,
    is_divergence,
    select_final,
)
from seis_interp.training.ccnet5d_observed_patches import placement_pseudo_target_counts

MASK_PATH = "masks/m1_random80_mvp_v1.parquet"
TUNER = ".venv/bin/python scripts/forge_matched_tuning.py"
REGSI_ENVIRONMENT = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
DOMAINS = ("validation", "final")


@dataclass(frozen=True)
class FairGridContext:
    """Both sealed preparations, reduced to what generation and validation need.

    ``preparations[domain]`` is ``{"path": <repo-relative>, "manifest_sha256": ...}``;
    ``base_configs[domain][method_id]`` the preparation's method config;
    ``observed_masks[domain]`` the observed cells on the spatial grid.
    """

    preparations: dict
    base_configs: dict
    observed_masks: dict
    budget: dict
    canvas_sample_count: int


def load_grid(grid_path: Path) -> tuple[dict, str]:
    return yaml.safe_load(grid_path.read_text()), sha256_file(grid_path)


def load_fair_grid_context(
    repo: Path, grid: dict, validation_preparation: Path, final_preparation: Path
) -> FairGridContext:
    """Verify both preparations against the pre-registered masks and load them."""
    paths = {"validation": validation_preparation.resolve(), "final": final_preparation.resolve()}
    seals = {d: json.loads((p / "preparation_manifest.json").read_text()) for d, p in paths.items()}
    expected = grid["preparations"]
    validation, final = seals["validation"], seals["final"]
    if (
        validation.get("selection_domain") != "validation_holdout"
        or validation["mask_hash"] != expected["validation_mask_hash"]
        or validation["parent_mask_hash"] != expected["final_mask_hash"]
        or final.get("selection_domain", "final_test") != "final_test"
        or final["mask_hash"] != expected["final_mask_hash"]
    ):
        raise ValueError("preparations do not carry the pre-registered masks")
    for key in ("study_id", "mapping_hash", "canvas_sample_count", "alignment"):
        if validation[key] != final[key]:
            raise ValueError(f"validation and final preparations differ in {key}")
    if validation["study_id"] != grid["study_id"]:
        raise ValueError("preparations belong to another study")
    mask = pd.read_parquet(paths["validation"] / MASK_PATH)
    cells = np.sort(mask.loc[mask.split.eq("test"), "cell_id"].to_numpy()).astype(np.int64)
    if _sha256_bytes(cells.tobytes()) != expected["validation_cells_sha256"]:
        raise ValueError("validation cells differ from the pre-registered holdout")
    contracts = {
        d: json.loads((p / "input/input_contract.json").read_text()) for d, p in paths.items()
    }
    if contracts["validation"]["matched_training"] != contracts["final"]["matched_training"]:
        raise ValueError("preparations differ in the matched budget")
    method_ids = sorted({m for spec in grid["methods"].values() for m in _method_ids(spec)})
    base_configs, observed_masks = {}, {}
    for domain, path in paths.items():
        base_configs[domain] = {}
        for method_id in method_ids:
            inputs, config = load_model_inputs(path, method_id)
            base_configs[domain][method_id] = config
        observed = np.zeros(int(np.prod(inputs.spatial_shape)), dtype=bool)
        observed[inputs.observed_cells] = True
        observed_masks[domain] = observed.reshape(inputs.spatial_shape)
    context = FairGridContext(
        preparations={
            d: {
                "path": str(p.relative_to(repo.resolve())),
                "manifest_sha256": sha256_file(p / "preparation_manifest.json"),
            }
            for d, p in paths.items()
        },
        base_configs=base_configs,
        observed_masks=observed_masks,
        budget=contracts["validation"]["matched_training"],
        canvas_sample_count=validation["canvas_sample_count"],
    )
    _verify_context(grid, context)
    return context


def plan_fair_grid_stage(
    repo: Path,
    grid_path: Path,
    method: str,
    stage: str,
    context: FairGridContext,
    *,
    devices: tuple[str, ...] | None = None,
) -> Path:
    """Write one stage's search yamls from the grid and the completed earlier stages.

    ``devices`` assigns the runs round-robin to execution devices instead of the grid's
    method device. It changes where a run executes, never the grid or the candidates.
    """
    grid, grid_hash = load_grid(grid_path)
    spec = _method_spec(grid, method)
    _validate_devices(spec, devices)
    stages = _stages(spec)
    if stage not in stages:
        raise ValueError(f"{method}: stage must be one of {stages}")
    _verify_context(grid, context)
    directory = _method_directory(repo, grid, method) / stage
    if directory.exists():
        raise FileExistsError(f"{directory} is already planned")
    history = _history(repo, grid, grid_hash, method, stages[: stages.index(stage)], context)
    runs, decision = compose_stage(stage, grid, spec, history.planned, history.records)
    checked = [_checked_run(context, spec, run) for run in runs]
    _write_plan(
        repo,
        directory,
        grid,
        grid_hash,
        method,
        stage,
        spec,
        context,
        "validation",
        checked,
        {"rule_inputs": history.inputs, "decision": decision},
        devices,
    )
    return directory


def select_fair_grid(
    repo: Path,
    grid_path: Path,
    method: str,
    context: FairGridContext,
    *,
    devices: tuple[str, ...] | None = None,
) -> Path:
    """Write the selection record (``final/stage.json``) and the final-retraining yamls."""
    grid, grid_hash = load_grid(grid_path)
    spec = _method_spec(grid, method)
    _validate_devices(spec, devices)
    _verify_context(grid, context)
    directory = _method_directory(repo, grid, method) / "final"
    if directory.exists():
        raise FileExistsError(f"{directory} is already selected")
    stages = _stages(spec)
    history = _history(repo, grid, grid_hash, method, stages, context)
    reseed = None if is_classical(spec) else history.documents["R"]["decision"]
    choice, runs = select_final(grid, spec, history.planned, history.records, reseed)
    checked = [_checked_run(context, spec, run, final=True) for run in runs]
    _write_plan(
        repo,
        directory,
        grid,
        grid_hash,
        method,
        "final",
        spec,
        context,
        "final",
        checked,
        {"rule_inputs": history.inputs, "decision": choice},
        devices,
    )
    return directory


def fair_grid_status(repo: Path, grid_path: Path) -> pd.DataFrame:
    """One row per planned run: its tuning run, status and validation scores."""
    grid, _ = load_grid(grid_path)
    index = _tuning_index(repo, grid)
    pattern = _divergence_pattern(grid)
    decays = [decay_key(decay) for decay in grid["ema_eval_decays"]]
    rows = []
    for method, spec in grid["methods"].items():
        directory = _method_directory(repo, grid, method)
        for stage in (*_stages(spec), "final"):
            document = _stage_document(directory / stage)
            if document is None:
                continue
            domain = "final" if stage == "final" else "validation"
            preparation = document["preparations"][domain]["path"]
            for run in document["runs"]:
                status, tuning, results = _run_status(index, run, preparation, pattern)
                row = {
                    "method": method,
                    "stage": stage,
                    "run": run["name"],
                    "method_id": run["method_id"],
                    "status": status,
                    "tuning_run": None if tuning is None else str(tuning.relative_to(repo)),
                }
                if results is not None and "primary_relative_mse" in results:
                    scores = dict(zip(results.ema_decay, results.primary_relative_mse, strict=True))
                    row.update({f"v_{key}": scores.get(key) for key in (*decays, "none")})
                    row["best_score"] = results.primary_relative_mse.min()
                rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class _History:
    planned: dict
    records: dict
    documents: dict
    inputs: list


def _history(repo, grid, grid_hash, method, stages, context) -> _History:
    """Planned runs and finished (completed or diverged) records of the given stages."""
    directory = _method_directory(repo, grid, method)
    index = _tuning_index(repo, grid)
    pattern = _divergence_pattern(grid)
    history = _History({}, {}, {}, [])
    decays = [decay_key(decay) for decay in grid["ema_eval_decays"]]
    for stage in stages:
        document = _stage_document(directory / stage)
        if document is None:
            raise ValueError(f"{method}: stage {stage} has not been planned")
        if document["grid_sha256"] != grid_hash:
            raise ValueError(f"{method}: grid.yaml changed after stage {stage} was planned")
        if document["preparations"] != context.preparations:
            raise ValueError(f"{method}: stage {stage} used other preparations")
        history.documents[stage] = document
        preparation = context.preparations["validation"]["path"]
        for run in document["runs"]:
            status, tuning, results = _run_status(index, run, preparation, pattern)
            if status not in ("completed", "diverged"):
                raise ValueError(f"{method}: stage {stage} is incomplete ({run['name']}: {status})")
            record = (
                _run_record(run, stage, results, decays, grid["selection"])
                if status == "completed"
                else RunRecord.diverged_run(run["name"], stage, _score_keys(run, decays))
            )
            history.planned[run["name"]] = PlannedRun(
                run["name"], run["method_id"], run["overrides"], run.get("axes")
            )
            history.records[run["name"]] = record
            history.inputs.append(
                {
                    "run": run["name"],
                    "stage": stage,
                    "method_id": run["method_id"],
                    "tuning_run": str(tuning.relative_to(repo)),
                    "tuning_manifest_sha256": sha256_file(tuning / "manifest.json"),
                    "status": status,
                    "scores": None if record.diverged else record.scores,
                    "train_loss": record.train_loss,
                }
            )
    return history


def _score_keys(run, decays):
    return ["none"] if run["method_id"] not in TUNABLE_NEURAL_METHODS else decays


def _run_record(run, stage, results, decays, selection) -> RunRecord:
    keys = _score_keys(run, decays)
    scores = dict(zip(results.ema_decay, results.primary_relative_mse, strict=True))
    if sorted(scores) != sorted(keys) or len(results) != len(keys):
        raise ValueError(f"{run['name']}: scored EMA shadows differ from the grid")
    column = selection["train_loss"]
    loss = float(results[column].iloc[0]) if column in results else None
    return RunRecord(run["name"], stage, {key: float(scores[key]) for key in keys}, loss)


def _checked_run(context, spec, run: PlannedRun, final=False) -> dict:
    """Search yaml candidate of one run after every feasibility check passes."""
    budget = context.budget
    traces = budget["loss_traces_per_update"]
    overrides = {"validation": run.overrides, "final": final_overrides(spec, run.overrides)}
    feasibility, candidates = {}, {}
    for domain in ("final",) if final else DOMAINS:
        base = context.base_configs[domain][run.method_id]
        candidate, resolved = _candidate(base, run.method_id, run.name, overrides[domain])
        observed = context.observed_masks[domain]
        count = int(observed.sum())
        config, parameters = matched_candidate_config(
            base, candidate, spec["device"], budget, context.canvas_sample_count, count
        )
        if any(config[s] != resolved[s] for s in TUNABLE_SECTIONS[run.method_id]):
            raise ValueError(f"{run.name}: tuner config differs from the generated overrides")
        entry = {"observed_trace_count": count, "parameter_count": parameters}
        if run.method_id.startswith("regsi"):
            fraction = resolved["training"]["inner_mask_fraction"]
            hidden = hidden_trace_count(count, fraction)
            if hidden % traces:
                raise ValueError(
                    f"{run.name}: inner mask {fraction} hides {hidden} of {count} traces, "
                    f"not a multiple of {traces}"
                )
            entry.update(inner_mask_fraction=fraction, hidden_traces_per_episode=hidden)
        if run.method_id == "ccnet5d_grid":
            counts = placement_pseudo_target_counts(
                observed,
                patch_shape=tuple(resolved["patches"]["shape"][1:]),
                inner_mask_fraction=resolved["patches"]["inner_mask_fraction"],
            )
            minimum = int(counts.min()) if len(counts) else 0
            if minimum < traces:
                raise ValueError(
                    f"{run.name}: a patch placement hides {minimum} < {traces} traces at N={count}"
                )
            entry.update(placements=len(counts), min_hidden_traces_per_patch=minimum)
        feasibility[domain], candidates[domain] = entry, candidate
    return {
        **asdict(run),
        **({"final_overrides": overrides["final"]} if final else {}),
        "candidate": candidates["final" if final else "validation"],
        "feasibility": feasibility,
    }


def _candidate(base, method_id, name, overrides):
    """Tuner candidate: every tunable key whose resolved value differs from the base."""
    resolved = apply_overrides(base, overrides)
    if set(resolved) != set(base) or any(
        resolved[key] != base[key]
        for key in base
        if key not in (*TUNABLE_SECTIONS[method_id], "model_seed")
    ):
        raise ValueError(f"{name}: overrides may change tunable sections and model_seed only")
    candidate = {"name": name}
    if resolved.get("model_seed") != base.get("model_seed"):
        candidate["model_seed"] = resolved["model_seed"]
    for section in TUNABLE_SECTIONS[method_id]:
        changed = {
            key: value
            for key, value in resolved[section].items()
            if key not in base[section] or base[section][key] != value
        }
        if changed:
            candidate[section] = changed
    return candidate, resolved


def _write_plan(
    repo, directory, grid, grid_hash, method, stage, spec, context, domain, runs, record, devices
):
    final = domain == "final"
    preparation = context.preparations[domain]["path"]
    directory.mkdir(parents=True)
    written, commands = [], []
    for position, run in enumerate(runs):
        device = devices[position % len(devices)] if devices else spec["device"]
        document = {
            "method_id": run["method_id"],
            "purpose": "final_evaluation" if final else "selection",
            "evaluate": not final,
            **({} if final else {"target_relative_mse": 0.0, "stop_when_target_met": False}),
            "device": device,
            **spec.get("execution", {}),
            "candidates": [run.pop("candidate")],
        }
        path = directory / f"{run['name']}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        written.append({**run, "yaml": str(path.relative_to(repo)), "sha256": sha256_file(path)})
        prefix = REGSI_ENVIRONMENT if run["method_id"].startswith("regsi") else ""
        commands.append(
            f"{prefix}{TUNER} --preparation {preparation} --config {path.relative_to(repo)}"
        )
    write_json(
        directory / "stage.json",
        {
            "study_id": grid["study_id"],
            "method": method,
            "stage": stage,
            "grid_sha256": grid_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "preparations": context.preparations,
            **({"devices": list(devices)} if devices else {}),
            **record,
            "runs": written,
            "commands": commands,
        },
    )


def _run_status(index, run, preparation, pattern):
    """(status, tuning directory, this run's result rows) of one planned run."""
    matches, diverged = [], []
    for directory in index.get(run["sha256"], []):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["method_id"] == run["method_id"] and manifest["preparation"] == preparation:
            matches.append((directory, manifest["status"]))
            if manifest["status"] == "failed" and is_divergence(
                manifest.get("failure_reason"), pattern
            ):
                diverged.append(directory)
    completed = [directory for directory, status in matches if status == "completed"]
    if len(completed) > 1:
        raise ValueError(f"{run['name']}: several completed tuning runs share its yaml")
    if diverged:
        if len(diverged) > 1 or completed:
            raise ValueError(f"{run['name']}: diverged and was re-run; a divergence is terminal")
        return "diverged", diverged[0], None
    if not completed:
        if not matches:
            return "pending", None, None
        directory, status = sorted(matches)[-1]
        return status, directory, None
    directory = completed[0]
    results = read_search_results(directory / "search_results.csv")
    results = results.loc[results.name.eq(run["name"])].reset_index(drop=True)
    results["ema_decay"] = results.ema_decay.fillna("none")
    return "completed", directory, results


def _validate_devices(spec, devices):
    """Execution devices of the same kind as the grid's method device (GPU methods only)."""
    if devices is None:
        return
    if (
        not devices
        or not spec["device"].startswith("cuda:")
        or len(set(devices)) != len(devices)
        or not all(
            isinstance(d, str) and d.startswith("cuda:") and d[5:].isdigit() for d in devices
        )
    ):
        raise ValueError("devices must be distinct cuda:<index> values for a GPU method")


def _tuning_index(repo, grid) -> dict:
    """sha256 of each matched-tuning search.yaml -> tuning directories."""
    index = {}
    for path in sorted((repo / "runs" / grid["study_id"]).glob("*_matched_tuning/search.yaml")):
        index.setdefault(sha256_file(path), []).append(path.parent)
    return index


def _divergence_pattern(grid):
    return grid["selection"]["diverged_runs"]["failure_reason_pattern"]


def _verify_context(grid, context):
    counts = grid["preparations"]["observed_trace_counts"]
    for domain in DOMAINS:
        if int(context.observed_masks[domain].sum()) != counts[domain]:
            raise ValueError(f"{domain} observed count differs from the pre-registered count")


def _stage_document(directory):
    path = directory / "stage.json"
    return json.loads(path.read_text()) if path.exists() else None


def _method_spec(grid, method):
    if method not in grid["methods"]:
        raise ValueError(f"unknown fair-grid method {method}")
    return grid["methods"][method]


def _method_ids(spec):
    return [spec["method_id"], *([spec["pair_method_id"]] if "pair_method_id" in spec else [])]


def _stages(spec):
    return ("S1",) if is_classical(spec) else STAGES


def _method_directory(repo, grid, method):
    return repo / "runs" / grid["study_id"] / "fair_grid" / method


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
