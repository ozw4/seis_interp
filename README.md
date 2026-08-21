# seis_interp

`seis_interp` is a proof-of-concept repository for coordinate-based seismic interpolation inspired by the implicit neural representation (INR) method described in *Robust unsupervised 5D seismic data reconstruction on regular and irregular grids*.

The first target is a controlled experiment on the SEG C3 Narrow-Azimuth dataset. The goal is to test whether a sinusoidal multilayer perceptron can reconstruct held-out seismic traces from their physical coordinates. This repository is not intended to reproduce every number or experiment in the paper.

## SEG C3 NA data

The four SEG-Y files are stored outside this repository under `${SEIS_INTERP_DATA_ROOT}/external/seg_c3_na/`. The tracked manifest contains the public source URLs; a generated `download.lock.yaml` records trust-on-first-use local byte counts and SHA-256 checksums without storing an absolute path.

Run the download on the host because the Dev Container mounts external data read-only:

```bash
python -m pip install -e .
export SEIS_INTERP_DATA_ROOT=/absolute/path/to/seis_interp_data
./scripts/download_seg_c3_na.sh
./scripts/verify_seg_c3_na.sh
```

Interrupted downloads resume from their `.part` files. Use `./scripts/download_seg_c3_na.sh --force` to discard existing complete and partial files. See [`data/external/seg_c3_na/README.md`](data/external/seg_c3_na/README.md) for the storage layout and direct CLI commands.

## Development environment

The repository includes a GPU-enabled Dev Container based on NVIDIA NGC PyTorch. It installs both OpenAI Codex CLI and Anthropic Claude Code CLI.

Before opening the container:

```bash
cp .devcontainer/.env.example .devcontainer/.env
mkdir -p ~/.config/gh ~/.codex ~/.claude
```

Edit `.devcontainer/.env` and set `SEISMIC_DATA_ROOT` to the host data root that contains `external/seg_c3_na/`. Then open the repository in VS Code and run **Dev Containers: Reopen in Container**.

Inside the container:

```bash
python -m seis_interp.cli doctor
codex
claude
```

The first invocation of each AI CLI may require interactive sign-in. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` may also be supplied through `.devcontainer/.env`. Do not commit credentials.

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
