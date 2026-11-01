"""Run the externally fixed sensitivity cell using this experiment's frozen source."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    args = parser.parse_args()
    from seis_interp.pipelines import run_c3_fixed_cell_sensitivity as runner

    experiment = args.experiment.resolve()
    if not Path(runner.__file__).resolve().is_relative_to(experiment / "source/src"):
        raise ValueError("PYTHONPATH must select the prepared experiment's frozen source/src")
    runner.run_c3_fixed_cell_sensitivity(args.repo.resolve(), experiment, args.production.resolve())


if __name__ == "__main__":
    main()
