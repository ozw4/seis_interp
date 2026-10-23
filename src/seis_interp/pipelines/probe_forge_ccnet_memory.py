"""Plan CCNet memory/timing probes and validation-only end-to-end checks."""

import gc
import json
import math
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import yaml

from seis_interp.processing.c3_region_search import c3_search_candidates, resolve_c3_candidate
from seis_interp.processing.experiment_commonality import load_experiment_grid


def forge_ccnet_probe_plan(
    study, *, mode="memory", candidate_ids=None, cudnn_benchmark="off", time_kernel_size=3
):
    """Use the current common grid, fixing time dilation at its largest value."""
    contract = yaml.safe_load((study / "config.yaml").read_text())
    grid = load_experiment_grid(study, contract)
    base = yaml.safe_load((study / "ccnet5d_grid.yaml").read_text())
    candidates = c3_search_candidates(base, grid["methods"]["ccnet5d_grid"])
    dilation = max(item["config"]["model"]["time_dilation"] for item in candidates)
    plan = [item for item in candidates if item["config"]["model"]["time_dilation"] == dilation]
    if dilation != 6 or [item["config"]["patches"]["shape"][1:] for item in plan] != [
        [4, 8, 4, 8],
        [4, 12, 4, 12],
        [6, 12, 6, 12],
        [8, 12, 8, 12],
        [8, 16, 8, 16],
    ]:
        raise ValueError("probe requires the five study_067 patches at time dilation 6")
    if mode not in ("memory", "timing", "end-to-end"):
        raise ValueError("unknown probe mode")
    if cudnn_benchmark not in ("off", "on", "both") or time_kernel_size not in (3, 9):
        raise ValueError("invalid benchmark or time kernel option")
    if candidate_ids is None and mode == "end-to-end":
        candidate_ids = ["a01_b02", "a05_b02"]
    if candidate_ids is not None:
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be distinct and nonempty")
        by_id = {item["id"]: item for item in candidates}
        if set(candidate_ids) - by_id.keys():
            raise ValueError("unknown probe candidate ID")
        plan = [by_id[name] for name in candidate_ids]
    for item in plan:
        config = item["config"]
        if (
            config["training"]["mixed_precision"] != "bf16"
            or config["prediction"].get("mixed_precision") != "bf16"
            or not config["execution"].get("full_trace_prediction")
            or config["patches"]["inner_mask_fraction"] != 0.25
            or config["training"]["supervised_traces_per_update"] != 64
            or contract["matched_training"]["loss_traces_per_update"] != 64
        ):
            raise ValueError("probe requires bf16, mask 0.25, full-time prediction and 64 traces")
        if time_kernel_size == 9:
            config["model"].update(time_kernel_size=9, hidden_channels=36, intermediate_channels=37)
    if cudnn_benchmark != "off":
        expanded = []
        for item in plan:
            trials = [(False, "off,on"), (True, "off,on"), (True, "on,off"), (False, "on,off")]
            for enabled, order in trials if cudnn_benchmark == "both" else [(True, None)]:
                trial = deepcopy(item)
                trial["config"].setdefault("execution", {})["cudnn_benchmark"] = enabled
                trial["candidate_id"] = item["id"]
                trial["id"] += "_benchmark_on" if enabled else "_benchmark_off"
                if order is not None:
                    trial["benchmark_order"] = order
                    trial["id"] += "_order1" if order == "off,on" else "_order2"
                expanded.append(trial)
        plan = expanded
    return plan


def run_forge_ccnet_memory_probe(
    repo,
    study,
    device,
    minimum_free_memory_gib,
    reserve_gib=8.0,
    *,
    mode="memory",
    updates=None,
    candidate_ids=None,
    cudnn_benchmark="off",
    time_kernel_size=3,
):
    """Write a fresh probe directory, never modifying existing experiment records."""
    if not math.isfinite(minimum_free_memory_gib) or minimum_free_memory_gib <= 0:
        raise ValueError("minimum free memory must be positive and finite")
    if not math.isfinite(reserve_gib) or reserve_gib <= 0:
        raise ValueError("reserve_gib must be positive and finite")
    updates = (
        updates
        if updates is not None
        else {"memory": 1, "timing": 300, "end-to-end": 1000}.get(mode)
    )
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    plan = (
        forge_ccnet_probe_plan(study)
        if (mode, candidate_ids, cudnn_benchmark, time_kernel_size) == ("memory", None, "off", 3)
        else forge_ccnet_probe_plan(
            study,
            mode=mode,
            candidate_ids=candidate_ids,
            cudnn_benchmark=cudnn_benchmark,
            time_kernel_size=time_kernel_size,
        )
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import pandas as pd
    import torch

    from seis_interp.data.forge_mvp_artifacts import write_json
    from seis_interp.data.forge_mvp_run_records import execution_identity
    from seis_interp.data.forge_trace_experiment import load_forge_trace_inputs
    from seis_interp.processing.forge_grid_partition import (
        partition_forge_traces,
        verify_forge_s1_partition,
    )
    from seis_interp.training.ccnet5d_memory_probe import (
        benchmark_comparisons,
        measure_ccnet5d_memory,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    contract = yaml.safe_load((study / "config.yaml").read_text())
    reference = repo / contract["reference_experiment"]
    original = yaml.safe_load((reference / "study/config.yaml").read_text())
    for key in (
        "candidate_id",
        "spatial_shape",
        "time_sample_count",
        "sample_interval_us",
        "preprocessing",
        "time_sampling",
    ):
        if contract[key] != original[key]:
            raise ValueError(f"FORGE reference preprocessing/input differs: {key}")
    inputs, _, partition = load_forge_trace_inputs(
        reference, "ccnet5d_grid", verify_target_waveforms=False
    )
    manifest = pd.read_parquet(reference / "input/manifest.parquet")
    verify_forge_s1_partition(
        partition_forge_traces(manifest, contract, contract["trace_split"]["search_seed"]),
        partition,
    )
    validation = partition.cell_id[partition.role.eq("validation")].to_numpy()
    inputs = replace(
        inputs,
        test_cells=validation,
        test_rms=inputs.test_rms[pd.Index(inputs.test_cells).get_indexer(validation)],
        excluded_cells=partition.cell_id[partition.role.eq("test")].to_numpy(),
    )
    seal = json.loads((reference / "experiment.json").read_text())
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs/study_067_forge_c3_common_grid"
        / f"{stamp}_{identity['git_commit'][:12]}_ccnet_memory_probe"
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        **identity,
        "status": "running",
        "scientific_run": False,
        "mode": mode,
        "optimizer_updates": updates,
        "timing_warmup_updates": 10 if mode == "timing" else 0,
        "diagnostic_only": True,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "input_binding": inputs.binding,
        "reference_experiment": str(reference),
        "trace_seed": contract["trace_split"]["search_seed"],
        "minimum_free_memory_gib": minimum_free_memory_gib,
        "reserve_gib": reserve_gib,
        "free_memory_guard": (
            "threshold after reserve; allocator cap preserves the reserve from current free memory"
        ),
        "admission_margin_gib": contract["cuda_execution"]["admission_margin_gib"],
        "validation_scoring_calls": 0,
        "test_scoring_calls": 0,
        "patches": [],
    }
    write_json(output / "memory_probe.json", report)
    for item in plan:
        config = resolve_c3_candidate(
            item["config"] | {"alignment": seal["alignment"], "device": device},
            len(inputs.observed_cells),
            contract["matched_training"]["loss_traces_per_update"],
        )
        config["training"]["max_steps"] = updates
        (output / f"{item['id']}.config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        cap_bytes = max(0.0, free - reserve_gib * 2**30)
        fraction = cap_bytes / total
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        record = {
            "id": item["id"],
            "candidate_id": item.get("candidate_id", item["id"]),
            "patch_shape": config["patches"]["shape"],
            "time_dilation": config["model"]["time_dilation"],
            "free_memory_before_bytes": free,
            "total_device_bytes": total,
            "reserve_gib": reserve_gib,
            "per_process_memory_fraction": fraction,
            "cap_bytes": cap_bytes,
        }
        if "benchmark_order" in item:
            record["benchmark_order"] = item["benchmark_order"]
        report["patches"].append(record)
        if free - reserve_gib * 2**30 < minimum_free_memory_gib * 2**30:
            record.update(status="skipped", reason="insufficient_free_memory")
        else:
            try:
                if mode == "end-to-end":
                    trial = output / item["id"]
                    trial.mkdir()
                    record.update(_end_to_end_probe(inputs, config, reference, trial))
                    report["validation_scoring_calls"] += 1
                elif mode == "timing":
                    record.update(
                        measure_ccnet5d_memory(
                            inputs, config, device, optimizer_updates=updates, warmup_updates=10
                        )
                    )
                elif updates == 1:
                    record.update(measure_ccnet5d_memory(inputs, config, device))
                else:
                    record.update(
                        measure_ccnet5d_memory(inputs, config, device, optimizer_updates=updates)
                    )
                measured_budget = record["measured_peak_reserved_gib_ceil"] + 2
                original_id = record["candidate_id"]
                current = contract["cuda_execution"]["minimum_free_memory_gib"]["ccnet5d_grid"][
                    original_id
                ]
                record["suggested_required_free_memory_gib"] = max(current, measured_budget)
            except torch.cuda.OutOfMemoryError as error:
                record.update(
                    status="exceeds_cap",
                    reason="cuda_out_of_memory",
                    failure_reason=str(error),
                    uncapped_peak_exceeds_cap=True,
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        write_json(output / "memory_probe.json", report)
    report["status"] = (
        "passed" if all(row["status"] == "passed" for row in report["patches"]) else "incomplete"
    )
    report["benchmark_comparisons"] = benchmark_comparisons(report["patches"])
    write_json(output / "memory_probe.json", report)
    print(output, flush=True)
    return output


def _end_to_end_probe(inputs, config, reference, output):
    """Finalize a physical prediction before reading validation; store shadow metrics only."""
    import numpy as np
    import pandas as pd
    import torch

    from seis_interp.data.forge_headers import sha256_file
    from seis_interp.data.forge_mvp_artifacts import write_json
    from seis_interp.data.forge_mvp_run_records import peak_memory
    from seis_interp.data.forge_trace_experiment import read_forge_trace_targets
    from seis_interp.evaluation.forge_grid_results import score_forge_grid
    from seis_interp.training.forge_c3_methods import run_forge_c3_method

    if config["device"].startswith("cuda:"):
        torch.cuda.reset_peak_memory_stats(config["device"])
    result = run_forge_c3_method(inputs, config, checkpoint_path=output / "final.pt")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    np.save(output / "prediction.npy", result.prediction, allow_pickle=False)
    np.save(output / "cell_ids.npy", result.cell_ids, allow_pickle=False)
    pd.DataFrame(result.history).to_csv(output / "history.csv", index=False)
    rows, target = read_forge_trace_targets(reference, "validation")
    indices = pd.Index(result.cell_ids).get_indexer(rows.cell_id)
    if (indices < 0).any():
        raise ValueError("probe prediction does not cover validation")
    metrics, traces = score_forge_grid(result.prediction[indices], target, rows)
    traces.to_parquet(output / "validation_per_trace.parquet", index=False)
    memory = peak_memory(config["device"])
    record = {
        "status": "passed",
        "diagnostic_only": True,
        "validation_scoring_calls": 1,
        "test_scoring_calls": 0,
        "metrics": {"validation": metrics},
        "diagnostics": result.diagnostics,
        "ema_shadow_metrics": {
            decay: score_forge_grid(values[indices], target, rows)[0]
            for decay, values in result.shadow_predictions.items()
        },
        "cudnn_benchmark": config.get("execution", {}).get("cudnn_benchmark", False),
        "measured_peak_reserved_gib_ceil": math.ceil(memory["peak_cuda_reserved_bytes"] / 2**30),
        **memory,
    }
    record["artifacts"] = {
        path.name: sha256_file(path) for path in output.iterdir() if path.is_file()
    }
    write_json(output / "manifest.json", record)
    return record
