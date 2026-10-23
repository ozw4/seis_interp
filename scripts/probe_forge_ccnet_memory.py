"""Measure CCNet memory/timing or check prediction through validation scoring."""

import argparse
import math
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--study", type=Path, default=Path("studies/study_067_forge_c3_common_grid/forge")
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--minimum-free-memory-gib", type=float, default=17.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--mode", choices=("memory", "timing", "end-to-end"), default="memory")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--candidate", action="append", dest="candidate_ids")
    parser.add_argument("--cudnn-benchmark", choices=("off", "on", "both"), default="off")
    parser.add_argument("--time-kernel-size", type=int, choices=(3, 9), default=3)
    parser.add_argument(
        "--plan-only", action="store_true", help="print the CPU-only plan without creating a run"
    )
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("--device must be an explicit CUDA device, e.g. cuda:1")
    if not math.isfinite(args.minimum_free_memory_gib) or args.minimum_free_memory_gib <= 0:
        parser.error("--minimum-free-memory-gib must be positive and finite")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0:
        parser.error("--reserve-gib must be positive and finite")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from seis_interp.pipelines.probe_forge_ccnet_memory import (
        forge_ccnet_probe_plan,
        run_forge_ccnet_memory_probe,
    )

    if args.plan_only:
        import json

        print(
            json.dumps(
                forge_ccnet_probe_plan(
                    args.study.resolve(),
                    mode=args.mode,
                    candidate_ids=args.candidate_ids,
                    cudnn_benchmark=args.cudnn_benchmark,
                    time_kernel_size=args.time_kernel_size,
                ),
                indent=2,
            )
        )
    else:
        run_forge_ccnet_memory_probe(
            args.repo.resolve(),
            args.study.resolve(),
            args.device,
            args.minimum_free_memory_gib,
            args.reserve_gib,
            mode=args.mode,
            updates=args.updates,
            candidate_ids=args.candidate_ids,
            cudnn_benchmark=args.cudnn_benchmark,
            time_kernel_size=args.time_kernel_size,
        )


if __name__ == "__main__":
    main()
