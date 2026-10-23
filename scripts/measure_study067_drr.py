"""Measure one prepared D-2 cell on CPU; never train or score any predictions."""

import argparse
import json
import os
import resource
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", choices=("forge", "c3"), required=True)
    parser.add_argument("--candidate", choices=("a05_b04", "a05_b05"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address-space-cap-gib", type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.address_space_cap_gib < float("inf"):
        parser.error("--address-space-cap-gib must be positive and finite")
    cap = int(args.address_space_cap_gib * 2**30)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"

    import yaml

    from seis_interp.data.c3_region_inputs import load_c3_region_inputs
    from seis_interp.data.forge_grid_inputs import load_forge_grid_inputs
    from seis_interp.data.forge_mvp_artifacts import verify_hashes
    from seis_interp.evaluation.drr_timing import measure_drr_candidate
    from seis_interp.processing.c3_region_search import c3_search_candidates

    experiment = args.experiment.resolve()
    seal = json.loads((experiment / "experiment.json").read_text())
    verify_hashes(experiment, seal["artifacts"])
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    expected_dataset = "forge_2017" if args.dataset == "forge" else "seg_c3_na"
    if contract["dataset_id"] != expected_dataset:
        parser.error("dataset does not match the frozen experiment")
    seed = contract["trace_split"]["search_seed"]
    base = yaml.safe_load((experiment / "study/drr_grid.yaml").read_text())
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    candidate = next(
        item
        for item in c3_search_candidates(base, grid["methods"]["drr_grid"])
        if item["id"] == args.candidate
    )
    config = candidate["config"]
    if args.dataset == "forge":
        inputs, loaded, _ = load_forge_grid_inputs(experiment, "drr_grid", seed, ("validation",))
        config["alignment"] = loaded["alignment"]
    else:
        inputs, _ = load_c3_region_inputs(experiment, seed, ("validation",))
    args.output.mkdir(parents=True, exist_ok=False)
    context = {
        "experiment": str(experiment),
        "dataset": args.dataset,
        "candidate": args.candidate,
        "address_space_cap_bytes": cap,
        "pid": os.getpid(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    (args.output / "request.json").write_text(json.dumps(context, indent=2) + "\n")
    with (args.output / "blocks.jsonl").open("x") as log:
        try:
            result = measure_drr_candidate(inputs, config, log)
        except MemoryError:
            (args.output / "result.json").write_text(
                json.dumps(context | {"status": "address_space_cap_exceeded"}, indent=2) + "\n"
            )
            raise
    result.update(context)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
