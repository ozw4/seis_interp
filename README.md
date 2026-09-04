# seis_interp

`seis_interp` is a proof-of-concept repository for multidimensional seismic interpolation, inspired by the implicit neural representation (INR) method described in *Robust unsupervised 5D seismic data reconstruction on regular and irregular grids*. It is not intended to reproduce every number or experiment in the paper.

## Project scope

The main target is the SEG C3 Narrow-Azimuth dataset: held-out traces of a marine 3-D survey are reconstructed from their physical coordinates and from neighboring traces. The repository currently trains and evaluates four model families:

- a coordinate-only SIREN,
- a train-only physical-neighbor temporal trace inpainter,
- a whole-shot gather inpainter over the fixed receiver grid,
- a trace-node graph gather interpolator.

Research questions, conditions, and recorded outcomes live in numbered studies; see [Studies and reports](#studies-and-reports).

## Development environment

The repository includes a GPU-enabled Dev Container based on NVIDIA NGC PyTorch. It installs both OpenAI Codex CLI and Anthropic Claude Code CLI.

Before opening the container:

```bash
cp .devcontainer/.env.example .devcontainer/.env
mkdir -p ~/.config/gh
```

Open the repository in VS Code and run **Dev Containers: Rebuild and Reopen in Container**. The container creates `/workspace/.venv` and installs the project with the development, SEG-Y, data, and visualization extras. The virtual environment is created with system site packages so it reuses the PyTorch bundled with the NGC image instead of installing a second copy. If an existing `.venv` cannot see the system PyTorch, the setup script recreates it automatically.

After pulling dependency changes into an already-running container, refresh the same environment without rebuilding:

```bash
./scripts/setup_dev_environment.sh
```

Inside the container:

```bash
python -m seis_interp.cli doctor
codex
claude
```

The repository scripts automatically prefer `/workspace/.venv/bin/python`. The first invocation of each AI CLI may require interactive sign-in. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` may also be supplied through `.devcontainer/.env`. Do not commit credentials.

Codex and Claude user state are stored in Docker named volumes. This keeps their local databases writable and preserves authentication across normal container rebuilds without sharing the host SQLite state files.

## SEG C3 NA data

The Dev Container uses the repository data tree as its data root:

```text
SEIS_INTERP_DATA_ROOT=/workspace/data
```

The four SEG-Y files are stored locally under `/workspace/data/external/seg_c3_na/`. The manifest and documentation are tracked; raw SEG-Y files, `download.lock.yaml`, intermediate arrays, and processed datasets are ignored by Git.

From the repository root:

```bash
./scripts/download_seg_c3_na.sh
./scripts/verify_seg_c3_na.sh
./scripts/inspect_seg_c3_na.sh
```

The inspection script checks SEG-Y structure, FFID coverage, source and receiver geometry, midpoint, offset, azimuth, delay time, and sampled-amplitude statistics. Interrupted downloads resume from their `.part` files; use `--force` to discard existing complete and partial files. See [`data/external/seg_c3_na/README.md`](data/external/seg_c3_na/README.md) for details.

## Data preparation

The `data` command group acquires and prepares datasets:

```text
data download           download an external dataset
data verify             verify an external dataset against its manifest
data inspect            inspect SEG-Y structure and content
data prepare-c3-shot    write one complete SEG C3 NA shot as an interim dataset
data prepare-c3-survey  write every manifest-declared FFID as one interim dataset
data prepare-baseline   create a dataset partition and train-only normalization metadata
data prepare-mask       create interpolation visibility within one dataset partition
```

Run `python -m seis_interp.cli data <command> --help` for the full argument list. A survey-to-mask flow is:

```bash
python -m pip install -e ".[dev,data,segy]"

python -m seis_interp.cli data prepare-c3-survey \
  --output data/interim/c3_na/all_ffids

python -m seis_interp.cli data prepare-baseline \
  --config studies/<study>/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/<partition-id>

python -m seis_interp.cli data prepare-mask \
  --config studies/<study>/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/<partition-id> \
  --output data/processed/c3_na/<partition-id>/masks/<mask-id>
```

Each interim dataset contains four files:

```text
traces.parquet   one row per selected trace, with an array_row column
amplitudes.npy   float32 array of shape (n_traces, n_samples)
time_s.npy       float64 zero-based time axis in seconds
dataset.json     dataset metadata, including the source SHA-256
```

Row `i` of `traces.parquet` corresponds to `amplitudes.npy[i]` through `array_row`. The coordinate rules are documented in [`docs/coordinate_conventions.md`](docs/coordinate_conventions.md).

`data prepare-baseline` requires `--config` because the dataset partition is a study condition. It writes `trace_split.parquet`, train-only `normalization.json`, and `preparation.json`. Configuration values are resolved in this order: the file named by `extends`, the selected study config, then explicit CLI overrides. The preparation metadata records the resolved partition values, supported normalization methods, and repository-relative config source. Study-specific seed values belong at `project.random_seed`; `study.random_seed` is rejected.

`data prepare-mask` separately assigns `observed` and `evaluation_target` roles within one `train`, `validation`, or `test` partition. Its model-independent artifact contains `observation_mask.parquet` and `interpolation_mask.json`, so multiple masks can share one unchanged dataset partition. The supported kinds are currently `random_trace` and `random_whole_ffid`; a whole-FFID mask requires a dataset partition prepared with whole-FFID splitting. Partition, kind, and missing fraction come from `interpolation_mask.*`, while the seed comes from `project.random_seed`; `prepare-mask` does not provide CLI overrides for these conditions.

Before selecting the requested partition, `prepare-mask` canonicalizes duplicate physical trace cells across all non-excluded partitions by keeping the lowest `array_row`. Candidate counts therefore describe the canonicalized rows. `interpolation_mask.json` records this policy and the number of rows removed, while the existing partition artifact remains unchanged.

```yaml
project:
  random_seed: 42

interpolation_mask:
  partition: test
  kind: random_trace
  missing_fraction: 0.8
```

SEG-Y inputs and everything under `data/interim/` and `data/processed/` are generated or externally obtained data and must not be committed to Git.

## Training commands

The `train` command group trains the four model families:

```text
train siren                  coordinate-only SIREN on prepared dataset partitions
train neighbor-inpainter     physical-neighbor temporal trace inpainter
train shot-gather-inpainter  joint whole-shot gather inpainter
train trace-graph            trace-node graph gather interpolator
```

Every train command takes required `--config`, `--interim`, `--processed`, and `--output` paths, plus optional `--device` and `--json`; run `python -m seis_interp.cli train <command> --help` for details. With `--json`, metrics go to stdout and training progress goes to stderr. For example:

```bash
python -m seis_interp.cli train siren \
  --config studies/<study>/config.yaml \
  --interim data/interim/c3_na/ffid_<id> \
  --processed data/processed/c3_na/<split-name> \
  --output runs/<study>/<run-id>
```

## Run outputs

A training run writes:

```text
config.resolved.yaml
inputs.lock.json
metrics.json
run.json
artifacts/best.pt
```

`run.json` records the Git commit, UTC start and finish times, success status, effective device, Python and PyTorch versions, and random seed. Formal study run directories are immutable: choose a new run ID for every invocation. Scratch workspaces labeled in the study index instead maintain an overwriteable current output. The run directory and checkpoint are generated outputs and must not be committed to Git.

## Studies and reports

Numbered studies under `studies/` are the authoritative record of research questions, conditions, and outcomes. Start from the study index at [`studies/README.md`](studies/README.md). Accepted figures and human-readable reports, when they exist, live under `results/` and `reports/`.

## Quality checks

```bash
ruff check .
ruff format --check .
pytest
python -m seis_interp.cli doctor
```

## Repository layout

The authoritative layout rules are in [`docs/repository_layout.md`](docs/repository_layout.md). In brief:

```text
src/       reusable implementation, including the CLI and its command modules
scripts/   thin CLI wrappers and study runners
studies/   research questions, conditions, and decision records
data/      external data and reproducible processing stages
runs/      machine-generated execution records
results/   accepted research outputs, added only when needed
reports/   human-readable reports, added only when needed
```

Large SEG-Y files, intermediate arrays, checkpoints, and full run outputs are not committed to Git.
