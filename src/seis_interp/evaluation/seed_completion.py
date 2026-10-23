"""Validate final seed records and expose incomplete evaluation populations."""


def seed_completion(records, seed_column, expected_seeds=None):
    seeds = [record[seed_column] for record in records]
    if expected_seeds is None:
        if len(seeds) != 3:
            raise ValueError("aggregation requires all three seeds")
        expected_seeds = seeds
    if (
        len(set(seeds)) != len(seeds)
        or len(set(expected_seeds)) != 3
        or set(seeds) - set(expected_seeds)
    ):
        raise ValueError("aggregation requires all three distinct evaluation seeds")
    for record in records:
        if record["status"] not in ("completed", "diverged"):
            raise ValueError(
                "post-barrier aggregation requires completed or explicitly diverged seeds"
            )
        if record["status"] == "completed" and not {"validation", "test"} <= set(record["metrics"]):
            raise ValueError("completed seeds require validation and test metrics")
    completed = [record for record in records if record["status"] == "completed"]
    details = {
        "missing_seeds": ",".join(str(seed) for seed in expected_seeds if seed not in seeds),
        "diverged_seeds": ",".join(
            str(row[seed_column]) for row in records if row["status"] == "diverged"
        ),
    }
    return completed, details
