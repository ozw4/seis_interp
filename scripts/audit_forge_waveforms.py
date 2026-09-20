"""Run the FORGE waveform audit from the checked-in study conditions."""

from pathlib import Path

from seis_interp.pipelines.audit_forge_waveforms import run_forge_waveform_audit

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    run_forge_waveform_audit(repo, repo / "studies/study_048_forge_waveform_qc")
