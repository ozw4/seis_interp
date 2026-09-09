#!/usr/bin/env python3
"""Audit explicit completed CCNet native runs using fixed CPU tolerances."""

import argparse
import json
from pathlib import Path

from seis_interp.evaluation.ccnet5d_run_audit import audit_c3_ccnet5d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "training-run", "prediction-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--expected-suite-sha256", required=True)
    report = audit_c3_ccnet5d(**vars(parser.parse_args()))
    print(json.dumps({"status": report["status"]}))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
