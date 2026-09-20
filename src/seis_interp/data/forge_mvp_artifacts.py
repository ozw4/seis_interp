"""Hash-bound FORGE inputs and compact observed-only runtime data."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
import yaml

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.forge_headers import sha256_file


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_hashes(root: Path, hashes: dict) -> None:
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"input hash mismatch: {relative}")


def mvp_implementation_files(repo: Path) -> list[Path]:
    files = sorted((repo / "src/seis_interp").rglob("*.py"))
    files.append(repo / "scripts/forge_m1_mvp.py")
    return files


def verify_mvp_implementation(repo: Path, hashes: dict) -> None:
    expected = {str(path.relative_to(repo)) for path in mvp_implementation_files(repo)}
    if difference := expected.symmetric_difference(hashes):
        raise ValueError(f"input hash mismatch: implementation file set: {sorted(difference)}")
    verify_hashes(repo, hashes)


def load_candidate_inputs(repo: Path, inputs: dict, candidate_id: str):
    metadata_path = repo / inputs["region_metadata"]
    verify_hashes(repo, {inputs["region_metadata"]: inputs["region_metadata_sha256"]})
    metadata = json.loads(metadata_path.read_text())
    grids_path = metadata_path.parent / "candidate_grids.json"
    relative = grids_path.relative_to(repo).as_posix()
    verify_hashes(repo, {relative: metadata["artifacts"][relative]})
    grids = json.loads(grids_path.read_text())[candidate_id]
    base = Path(grids["trace_mapping"]).parent
    paths = {
        name: base / f"{name}.parquet"
        for name in ("trace_mapping", "source_stations", "receiver_stations", "grid_cells")
    }
    hashes = {str(path): metadata["artifacts"][str(path)] for path in paths.values()}
    hashes.update(
        {
            relative: metadata["artifacts"][relative],
            inputs["region_metadata"]: inputs["region_metadata_sha256"],
        }
    )
    verify_hashes(repo, hashes)
    tables = {name: pd.read_parquet(repo / path) for name, path in paths.items()}
    return tables, grids, hashes


def read_selected_traces(root: Path, rows: pd.DataFrame, samples: int, dt_us: int) -> np.ndarray:
    """Read explicitly requested trace indices only; never load a full shot gather."""
    rows = rows.reset_index(drop=True)
    if rows.duplicated(["source_file", "trace_index"]).any():
        raise ValueError("duplicate waveform request")
    values = np.empty((len(rows), samples), dtype=np.float32)
    for name, group in rows.groupby("source_file", sort=True):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("waveform path escapes dataset root")
        with segyio.open(
            str(path), "r", strict=False, ignore_geometry=True, endian="little"
        ) as handle:
            if (
                handle.bin[segyio.BinField.Format] != 5
                or handle.bin[segyio.BinField.Interval] != dt_us
                or len(handle.samples) != samples
            ):
                raise ValueError("SEG-Y format/time contract mismatch")
            for position, trace in zip(group.index, group.trace_index, strict=True):
                values[position] = handle.trace.raw[int(trace)]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite waveform")
    return values


def observed_global_rms(values: np.ndarray) -> float:
    """Float64 accumulation over the compact observed array, in cell order."""
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or not values.size
        or not np.isfinite(values).all()
    ):
        raise ValueError("expected finite nonempty float32 observed traces")
    energy = sum(
        float(np.square(values[start : start + 128].astype(np.float64)).sum())
        for start in range(0, len(values), 128)
    )
    rms = float(np.sqrt(energy / values.size))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("observed global RMS must be positive")
    return rms


@dataclass(frozen=True)
class ForgeModelInputs:
    """No evaluation amplitudes, raw file paths, or raw waveform reader."""

    geometry: pd.DataFrame
    observed_cells: np.ndarray
    test_cells: np.ndarray
    observed_values: np.ndarray
    spatial_shape: tuple
    time_s: np.ndarray
    global_rms: float
    binding: dict

    def __post_init__(self):
        size = int(np.prod(self.spatial_shape))
        ids = self.geometry.cell_id.to_numpy()
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= size):
            raise ValueError("invalid geometry cell IDs")
        combined = np.concatenate((self.observed_cells, self.test_cells))
        if len(np.unique(combined)) != len(combined) or not np.array_equal(
            np.sort(combined), np.sort(ids)
        ):
            raise ValueError("observed/test must disjointly cover eligible geometry")
        if (
            self.observed_values.shape != (len(self.observed_cells), len(self.time_s))
            or self.observed_values.dtype != np.float32
            or not np.isfinite(self.observed_values).all()
        ):
            raise ValueError("observed-only amplitude shape/dtype mismatch")
        if not np.isfinite(self.global_rms) or self.global_rms <= 0:
            raise ValueError("invalid global RMS")

    def read_observed(self, cells: np.ndarray) -> np.ndarray:
        positions = pd.Index(self.observed_cells).get_indexer(cells)
        if np.any(positions < 0):
            raise ValueError("amplitude access outside outer observed set")
        return self.observed_values[positions].copy()

    def volume(self) -> ObservedC3Volume:
        """Reuse the time-first volume container; axes follow the FORGE contract."""
        size = int(np.prod(self.spatial_shape))
        values = np.zeros((len(self.time_s), size), dtype=np.float32)
        values[:, self.observed_cells] = self.observed_values.T
        observed, target = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
        observed[self.observed_cells], target[self.test_cells] = True, True
        return ObservedC3Volume(
            values.reshape((len(self.time_s), *self.spatial_shape)),
            self.time_s.copy(),
            np.arange(size).reshape(self.spatial_shape),
            observed.reshape(self.spatial_shape),
            target.reshape(self.spatial_shape),
        )


def load_model_inputs(preparation: Path, method_id: str) -> tuple[ForgeModelInputs, dict]:
    """Verify the preparation seal and load only this method's geometry + O."""
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    if seal["status"] != "prepared":
        raise ValueError("preparation is not complete")
    config = yaml.safe_load((preparation / "configs" / f"{method_id}.yaml").read_text())
    if config["method_id"] != method_id:
        raise ValueError("method identity mismatch")
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    geometry = pd.read_parquet(
        preparation / "features" / f"{config['geometry_mode']}_geometry.parquet"
    )
    observed = pd.read_parquet(preparation / "input/observed_index.parquet").cell_id.to_numpy()
    if not np.array_equal(observed, mask.loc[mask.split.eq("observed"), "cell_id"]):
        raise ValueError("observed array index does not match fixed mask")
    binding = {
        key: seal[key]
        for key in (
            "study_id",
            "candidate_id",
            "mask_id",
            "mask_seed",
            "mapping_hash",
            "mask_hash",
            "input_hashes",
            "global_rms",
        )
    }
    binding["preparation_hash"] = sha256_file(preparation / "preparation_manifest.json")
    return ForgeModelInputs(
        geometry=geometry,
        observed_cells=observed,
        test_cells=mask.loc[mask.split.eq("test"), "cell_id"].to_numpy(),
        observed_values=np.load(preparation / "input/observed_waveforms.npy", mmap_mode="r"),
        spatial_shape=tuple(contract["spatial_shape"]),
        time_s=np.arange(contract["time_sample_count"], dtype=np.float64)
        * contract["sample_interval_us"]
        / 1e6,
        global_rms=seal["global_rms"],
        binding=binding,
    ), config
