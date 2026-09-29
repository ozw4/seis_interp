"""Run POCS parameter diagnostics on a completed FORGE experiment."""

from pathlib import Path

from seis_interp.pipelines.tune_forge_classical import tune_forge_classical


def tune_forge_pocs(repo: Path, source_run: Path, search_config: Path):
    return tune_forge_classical(repo, source_run, search_config, method_id="pocs_grid")
