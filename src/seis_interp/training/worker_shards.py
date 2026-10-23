"""Disjoint S1 workers and completion-only collection of their frozen outcomes."""

import json
import time

from seis_interp.training.worker_processes import worker_failure


def parse_shard(value):
    try:
        index, count = (int(part) for part in value.split("/"))
    except (ValueError, AttributeError) as error:
        raise ValueError("shard must be i/n, with 0 <= i < n") from error
    if not 0 <= index < count:
        raise ValueError("shard must be i/n, with 0 <= i < n")
    return index, count


def shard_candidates(candidates, shard):
    index, count = parse_shard(shard)
    return [item for position, item in enumerate(candidates) if position % count == index]


def shard_worker_name(method, shard):
    index, count = parse_shard(shard)
    return method if index == 0 else f"{method}_shard_{index}_of_{count}"


def collect_shard_records(experiment, method, candidates, shard, *, poll=1.0):
    """Only shard zero selects; retries must finish before any S1 selection."""
    index, count = parse_shard(shard)
    if index:
        return None
    for other in range(1, count):
        path = experiment / "workers" / f"{shard_worker_name(method, f'{other}/{count}')}.json"
        while True:
            if path.exists():
                state = json.loads(path.read_text())
                failure = worker_failure(state)
                if failure:
                    raise RuntimeError(f"{method} shard {other} failed: {failure}")
                if state.get("shard_completed") and state.get("status") != "retry_pending":
                    break
            time.sleep(poll)
    return [
        json.loads(
            (experiment / "method_runs" / method / "S1" / item["id"] / "manifest.json").read_text()
        )
        for item in candidates
    ]
