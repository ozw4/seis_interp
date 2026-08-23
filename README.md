# seis_interp

`seis_interp` is a proof-of-concept repository for coordinate-based seismic interpolation inspired by the implicit neural representation (INR) method described in *Robust unsupervised 5D seismic data reconstruction on regular and irregular grids*.

The first target is a controlled experiment on the SEG C3 Narrow-Azimuth dataset. The goal is to test whether a sinusoidal multilayer perceptron can reconstruct held-out seismic traces from their physical coordinates. This repository is not intended to reproduce every number or experiment in the paper.

## Development environment

The repository includes a GPU-enabled Dev Container based on NVIDIA NGC PyTorch. It installs both OpenAI Codex CLI and Anthropic Claude Code CLI.

Before opening the container:

```bash
cp .devcontainer/.env.example .devcontainer/.env
mkdir -p ~/.config/gh
```

Edit `.devcontainer/.env` and set `SEISMIC_DATA_ROOT` to the host directory that contains the SEG C3 NA files. Then open the repository in VS Code and run **Dev Containers: Reopen in Container**.

Inside the container:

```bash
python -m seis_interp.cli doctor
codex
claude
```

The first invocation of each AI CLI may require interactive sign-in. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` may also be supplied through `.devcontainer/.env`. Do not commit credentials.

Codex and Claude user state are stored in Docker named volumes. This keeps their local databases writable and preserves authentication across normal container rebuilds without sharing the host SQLite state files.

## Prepare a complete SEG C3 NA shot

Install the SEG-Y and table extras, then convert one shot of a SEG-Y file into an interim trace dataset:

```bash
python -m pip install -e ".[dev,data,segy]"

python -m seis_interp.cli prepare-c3-shot \
  --input "$SEIS_INTERP_DATA_ROOT/<file>.segy" \
  --output data/interim/c3_na/complete_shot

python -m seis_interp.cli prepare-c3-shot \
  --input "$SEIS_INTERP_DATA_ROOT/<file>.segy" \
  --output data/interim/c3_na/ffid_<id> \
  --ffid <id>
```

Without `--ffid` the command selects the numerically smallest FFID whose trace count equals `--expected-traces` (544 by default). Each run writes four files into the output directory:

```text
traces.parquet   one row per selected trace, with an array_row column
amplitudes.npy   float32 array of shape (n_traces, n_samples)
time_s.npy       float64 zero-based time axis in seconds
dataset.json     dataset metadata, including the source SHA-256
```

Row `i` of `traces.parquet` corresponds to `amplitudes.npy[i]` through `array_row`. The coordinate rules are documented in [`docs/coordinate_conventions.md`](docs/coordinate_conventions.md).

SEG-Y inputs and everything under `data/interim/` are generated or externally obtained data and must not be committed to Git.

## Quality checks

```bash
ruff check .
ruff format --check .
pytest
```

## Repository layout

The authoritative layout rules are in [`docs/repository_layout.md`](docs/repository_layout.md). In brief:

```text
src/       reusable implementation
studies/   research questions, conditions, and decision records
data/      external data manifests and reproducible processing stages
runs/      machine-generated execution records
results/   accepted research outputs, added only when needed
reports/   human-readable reports, added only when needed
```

Large SEG-Y files, intermediate arrays, checkpoints, and full run outputs are not committed to Git.

## Current study

The initial study is [`study_001_c3_na_baseline`](studies/study_001_c3_na_baseline/README.md). It will compare the INR reconstruction against simple trace-interpolation baselines using complete held-out traces as ground truth.
