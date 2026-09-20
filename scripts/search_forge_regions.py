"""Search FORGE regions using the checked-in geometry-first study contract."""

from pathlib import Path

from seis_interp.pipelines.search_forge_regions import run_forge_region_search

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    run_forge_region_search(repo, repo / "studies/study_051_forge_region_search")
