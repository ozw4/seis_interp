# seis_interp

`seis_interp` is a proof-of-concept repository for coordinate-based seismic interpolation inspired by the implicit neural representation (INR) method described in *Robust unsupervised 5D seismic data reconstruction on regular and irregular grids*.

The first target is a controlled experiment on the SEG C3 Narrow-Azimuth dataset. The goal is to test whether a sinusoidal multilayer perceptron can reconstruct held-out seismic traces from their physical coordinates. This repository is not intended to reproduce every number or experiment in the paper.

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

The inspection script checks SEG-Y structure, FFID coverage, source and receiver geometry, midpoint, offset, azimuth, delay time, and sampled-amplitude statistics. It reads complete trace headers but samples only 32 evenly spaced trace amplitudes per file by default.

```bash
./scripts/inspect_seg_c3_na.sh --sample-traces 64
./scripts/inspect_seg_c3_na.sh --json > seg_c3_na_inspection.json
```

Interrupted downloads resume from their `.part` files. Use `./scripts/download_seg_c3_na.sh --force` to discard existing complete and partial files. See [`data/external/seg_c3_na/README.md`](data/external/seg_c3_na/README.md) for details.

## Development environment

The repository includes a GPU-enabled Dev Container based on NVIDIA NGC PyTorch. It installs both OpenAI Codex CLI and Anthropic Claude Code CLI.

Before opening the container:

```bash
cp .devcontainer/.env.example .devcontainer/.env
mkdir -p ~/.config/gh
```

Open the repository in VS Code and run **Dev Containers: Rebuild and Reopen in Container**. The container creates `/workspace/.venv` and installs the project with the development, SEG-Y, data, and visualization extras.

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
scripts/   thin CLI wrappers
studies/   research questions, conditions, and decision records
data/      external data and reproducible processing stages
runs/      machine-generated execution records
results/   accepted research outputs, added only when needed
reports/   human-readable reports, added only when needed
```

Large SEG-Y files, intermediate arrays, checkpoints, and full run outputs are not committed to Git.

## Current study

The initial study is [`study_001_c3_na_baseline`](studies/study_001_c3_na_baseline/README.md). It will compare the INR reconstruction against simple trace-interpolation baselines using complete held-out traces as ground truth.
