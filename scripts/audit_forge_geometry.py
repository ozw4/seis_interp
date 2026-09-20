"""Run the FORGE geometry EDA using its checked-in study conditions."""

from pathlib import Path

from seis_interp.pipelines.audit_forge_geometry import run_forge_geometry_audit

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    run_forge_geometry_audit(repo, repo / "studies/study_049_forge_geometry_eda")
