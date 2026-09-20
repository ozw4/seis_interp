"""Run the fixed-exclusion and waveform-review FORGE study."""

from pathlib import Path

from seis_interp.pipelines.review_forge_qc import run_forge_qc_review

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    run_forge_qc_review(repo, repo / "studies/study_050_forge_qc_review")
