"""Hash-bound amplitude rows of a C3 fair-grid preparation.

The only waveform access of a C3 fair-grid preparation: ``prepare_c3_fair_grid``
reads its observed rows, and the matched tuner and the final evaluator read the
scored rows (validation holdout or test set) once through
``read_c3_scored_targets``. Rows are samples ``0:time_sample_count`` of the
interim trace array, addressed by ``array_row`` from ``input/c3_trace_manifest.parquet``;
``array_row`` never enters the model geometry.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.forge_mvp_artifacts import verify_hashes

C3_TRUTH = "input/c3_truth.json"
C3_TRACE_MANIFEST = "input/c3_trace_manifest.parquet"
ROW_CHUNK = 8192


def c3_truth_record(inputs: dict, contract: dict) -> dict:
    """``input/c3_truth.json``: where the scored rows are and how they are sampled."""
    return {
        "amplitudes": inputs["amplitudes"],
        "amplitudes_sha256": inputs["amplitudes_sha256"],
        "time_s": inputs["time_s"],
        "time_s_sha256": inputs["time_s_sha256"],
        "time_sample_count": contract["time_sample_count"],
        "sample_interval_us": contract["sample_interval_us"],
    }


def validate_c3_time_axis(time_s: np.ndarray, time_sample_count: int, interval_us: int) -> None:
    """The crop's samples are the first ``time_sample_count`` samples at the contract interval."""
    expected = np.arange(time_sample_count, dtype=np.float64) * interval_us / 1e6
    if (
        time_s.ndim != 1
        or len(time_s) < time_sample_count
        or not np.allclose(time_s[:time_sample_count], expected, rtol=0, atol=1e-9)
    ):
        raise ValueError("interim time axis does not match the contract sampling")


def read_c3_amplitude_rows(path: Path, array_rows, time_sample_count: int) -> np.ndarray:
    """``amplitudes[array_rows, :time_sample_count]`` as float32, read in row order."""
    amplitudes = np.load(path, mmap_mode="r", allow_pickle=False)
    rows = np.asarray(array_rows, dtype=np.int64)
    if (
        amplitudes.ndim != 2
        or amplitudes.dtype != np.float32
        or amplitudes.shape[1] < time_sample_count
    ):
        raise ValueError("interim amplitudes must be float32 [trace, sample] traces")
    if (
        rows.ndim != 1
        or len(np.unique(rows)) != len(rows)
        or (len(rows) and (rows.min() < 0 or rows.max() >= len(amplitudes)))
    ):
        raise ValueError("invalid array_row request")
    values = np.empty((len(rows), time_sample_count), dtype=np.float32)
    order = np.argsort(rows, kind="stable")
    for start in range(0, len(rows), ROW_CHUNK):
        positions = order[start : start + ROW_CHUNK]
        values[positions] = amplitudes[rows[positions], :time_sample_count]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite waveform")
    return values


def read_c3_scored_targets(repo: Path, preparation: Path, mask: pd.DataFrame) -> np.ndarray:
    """Scored (``split == test``) traces of a sealed C3 preparation, in mask order.

    The truth record and trace manifest are checked against the seal, and the
    interim amplitude and time files against the truth record, before any read.
    """
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(
        preparation, {name: seal["artifacts"][name] for name in (C3_TRUTH, C3_TRACE_MANIFEST)}
    )
    truth = json.loads((preparation / C3_TRUTH).read_text())
    verify_hashes(
        repo,
        {
            truth["amplitudes"]: truth["amplitudes_sha256"],
            truth["time_s"]: truth["time_s_sha256"],
        },
    )
    count = truth["time_sample_count"]
    validate_c3_time_axis(
        np.load(repo / truth["time_s"], allow_pickle=False), count, truth["sample_interval_us"]
    )
    manifest = pd.read_parquet(preparation / C3_TRACE_MANIFEST).set_index(
        "cell_id", verify_integrity=True
    )
    scored = mask.loc[mask.split.eq("test")]
    rows = manifest.loc[scored.cell_id.to_numpy()]
    if not np.array_equal(rows.trace_id.to_numpy(), scored.trace_id.to_numpy()):
        raise ValueError("trace manifest does not match the mask")
    return read_c3_amplitude_rows(repo / truth["amplitudes"], rows.array_row.to_numpy(), count)
