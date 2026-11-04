"""Compute the registered S-a/S-b sensitivities from immutable saved CPU inputs."""

import argparse
import shlex
from pathlib import Path

from seis_interp.evaluation.study067_sensitivity_results import evaluate_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("c3", "forge", "study064", "primary-analysis", "sensitivity-env", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    experiments = {
        key: (repo / getattr(args, key)).resolve() for key in ("c3", "forge", "study064")
    }
    exports = {}
    for line in args.sensitivity_env.read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if len(tokens) == 2 and tokens[0] == "export":
            key, value = tokens[1].split("=", 1)
            exports[key] = value
    for role, key in (("sa_c3", "SA_C3"), ("sa_forge", "SA_FG"), ("sb_forge", "SB_FG")):
        experiments[role] = (repo / exports[key]).resolve()
    output = evaluate_sensitivity(
        repo,
        experiments,
        (repo / args.primary_analysis).resolve(),
        (repo / args.output).resolve(),
        provenance_paths=(
            (repo / args.sensitivity_env).resolve(),
            repo / "reports/study_067_design_round2.md",
            repo / "studies/study_067_forge_c3_common_grid/decisions.md",
        ),
    )
    print((output / "summary.md").read_text())


if __name__ == "__main__":
    main()
