"""Prepare or execute the frozen FORGE M1 MVP."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.pipelines.preflight_forge_mvp import preflight_forge_method, preflight_forge_mvp
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_mvp import run_forge_method, run_forge_mvp
from seis_interp.processing.forge_mvp_contract import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "method", "check-method"))
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        if args.preparation or args.output or args.method or args.preflight:
            parser.error("prepare uses the fixed study configuration only")
        print(prepare_forge_mvp(repo, repo / "studies/study_052_forge_m1_mvp"))
        return
    if args.preparation is None:
        parser.error("--preparation is required")
    preparation = args.preparation.resolve()
    if args.action in ("method", "check-method"):
        if args.output is None or args.method is None:
            parser.error("method requires --output and --method")
        function = run_forge_method if args.action == "method" else preflight_forge_method
        function(repo, preparation, args.output.resolve(), args.method)
    else:
        if args.method:
            parser.error("run executes the fixed six-method matrix")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or repo / "runs/study_052_forge_m1_mvp" / (
            stamp + "_" + execution_identity(repo)["git_commit"][:12] + "_" + args.action
        )
        if args.action == "preflight":
            result = preflight_forge_mvp(repo, preparation, output.resolve())
        else:
            if args.preflight is None:
                parser.error("run requires --preflight from the passing resource checks")
            result = run_forge_mvp(repo, preparation, output.resolve(), args.preflight.resolve())
        print(output)
        if result["status"] not in ("accepted", "passed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
