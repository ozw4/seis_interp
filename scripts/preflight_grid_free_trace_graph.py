"""Measure a fixed trace graph query sample; report blockers without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_interp.pipelines.preflight_relational_trace_graph import (
    preflight_relational_trace_graph_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "interim", "processed", "mask", "case"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("checkpoint", "volume", "train-mask", "train-case", "train-volume"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--evaluate-baselines", action="store_true")
    parser.add_argument("--max-graph-seconds", type=float)
    parser.add_argument("--max-process-rss-bytes", type=int)
    parser.add_argument("--max-cuda-allocated-bytes", type=int)
    args = parser.parse_args(argv)
    report = preflight_relational_trace_graph_run(
        config_path=args.config,
        interim_dir=args.interim,
        processed_dir=args.processed,
        mask_dir=args.mask,
        case_dir=args.case,
        checkpoint_path=args.checkpoint,
        volume_dir=args.volume,
        train_mask_dir=args.train_mask,
        train_case_dir=args.train_case,
        train_volume_dir=args.train_volume,
        query_count=args.query_count,
        query_limit=args.query_limit,
        device_override=args.device,
        evaluate_baselines=args.evaluate_baselines,
        max_graph_seconds=args.max_graph_seconds,
        max_process_rss_bytes=args.max_process_rss_bytes,
        max_cuda_allocated_bytes=args.max_cuda_allocated_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
