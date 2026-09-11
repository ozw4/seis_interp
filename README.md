# seis_interp

`seis_interp` is a proof-of-concept repository for multidimensional seismic interpolation, inspired by the implicit neural representation (INR) method described in *Robust unsupervised 5D seismic data reconstruction on regular and irregular grids*. It is not intended to reproduce every number or experiment in the paper.

## Project scope

The main target is the SEG C3 Narrow-Azimuth dataset: held-out traces of a marine 3-D survey are reconstructed from their physical coordinates and from neighboring traces. The repository trains and evaluates these model families:

- a coordinate-only SIREN,
- a train-only physical-neighbor temporal trace inpainter,
- a whole-shot gather inpainter over the fixed receiver grid,
- a trace-node graph gather interpolator.

The [grid-free relational trace graph](docs/relational_trace_graph.md) additionally supports arbitrary source/receiver queries, masked training, frozen checkpoints, and GCN/untyped controls. Its [draft study](studies/study_026_grid_free_multi_relation_gnn/README.md) provides variant configs and a limited-query preflight runner. CPU tests cover analytic off-grid waveforms; C3 training and generalization performance have not been established for this model.

It also provides a CPU/NumPy Fourier POCS-5D baseline for verified dense C3 benchmark volumes.

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
data prepare-benchmark-case  bind an existing partition and mask by exact file hashes
data prepare-c3-volume-index bind a dense 5D crop and trace-to-cell mapping to one case
```

Run `python -m seis_interp.cli data <command> --help` for the full argument list. A survey-to-volume-index flow is:

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

python -m seis_interp.cli data prepare-benchmark-case \
  --config studies/<study>/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/<partition-id> \
  --mask data/processed/c3_na/<partition-id>/masks/<mask-id> \
  --output data/processed/c3_na/<partition-id>/cases/<case-id>

python -m seis_interp.cli data prepare-c3-volume-index \
  --config studies/<study>/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/<partition-id> \
  --mask data/processed/c3_na/<partition-id>/masks/<mask-id> \
  --case data/processed/c3_na/<partition-id>/cases/<case-id> \
  --output data/processed/c3_na/<partition-id>/volumes/<volume-id>
```

Each interim dataset contains four files:

```text
traces.parquet   one row per selected trace, with an array_row column
amplitudes.npy   float32 array of shape (n_traces, n_samples)
time_s.npy       float64 zero-based time axis in seconds
dataset.json     dataset metadata, including the source SHA-256
```

Row `i` of `traces.parquet` corresponds to `amplitudes.npy[i]` through `array_row`. The coordinate rules are documented in [`docs/coordinate_conventions.md`](docs/coordinate_conventions.md).

`data prepare-baseline` requires `--config` because the dataset partition is a study condition. It writes `trace_split.parquet`, train-only `normalization.json`, and `preparation.json`. Configuration values are resolved in this order: the file named by `extends`, the selected study config, then explicit CLI overrides. The preparation metadata records the resolved partition values, supported normalization methods, and repository-relative config source. `split_scope: c3_source_line_blocks` requires explicit zero-based half-open `source_line_ranges` for train, validation, and test, and assigns every shot and receiver trace on one source line to the same partition. Study-specific seed values belong at `project.random_seed`; `study.random_seed` is rejected.

`data prepare-mask` separately assigns `observed` and `evaluation_target` roles within one `train`, `validation`, or `test` partition. Its model-independent artifact contains `observation_mask.parquet` and `interpolation_mask.json`, so multiple masks can share one unchanged dataset partition. The supported kinds are currently `random_trace` and `random_whole_ffid`; a whole-FFID mask requires `split_scope: whole_ffid` or the whole-shot assignment provided by `c3_source_line_blocks`. Partition, kind, and missing fraction come from `interpolation_mask.*`, while the seed comes from `project.random_seed`; `prepare-mask` does not provide CLI overrides for these conditions.

Before selecting the requested partition, `prepare-mask` verifies that `preparation.json` split counts match `trace_split.parquet`. For whole-FFID assignment scopes, it also verifies the FFID counts and that every non-excluded FFID belongs to exactly one effective split. It then canonicalizes duplicate physical trace cells across all non-excluded partitions by keeping the lowest `array_row`. Candidate counts therefore describe the canonicalized rows. `interpolation_mask.json` records this policy and the number of rows removed, while the existing partition artifact remains unchanged.

`data prepare-benchmark-case` validates an existing interim dataset, prepared partition, and mask artifact, then binds their nine files by exact SHA-256 in a model-independent `benchmark_case.json`. It does not regenerate or copy those artifacts. The case fixes the `canonical_present_traces` role domain and keeps evaluation-target amplitudes for scoring only.

`data prepare-c3-volume-index` binds one benchmark case by file hash and records a dense 5D crop plus its trace-to-cell mapping. It writes only `volume_index.parquet` and `volume.json`; amplitudes remain in the interim `amplitudes.npy`. Selection ranges are zero-based and half-open. The output order is `(time, source_line, shot_in_line, relative_receiver_x, relative_receiver_y)`: source lines rank ascending `source_x_m`, shots rank ascending `source_y_m` within each source line, and receiver axes rank source-relative offsets. The selected crop must lie within one dataset partition. The adapter requires exactly one canonical trace in every selected spatial cell and rejects incomplete shots or gaps in the physical C3 source and receiver grid. This repository contract does not claim to recover an unpublished exact paper crop. POCS, DRR, SIREN, CNN, and GNN runs should use the same case and volume directories; model-specific normalization and patching remain run concerns.

The model-independent runtime contract for assembling partial or whole-shot target observations and neighboring context gathers from these verified artifacts is documented in [`docs/masked_gather_inputs.md`](docs/masked_gather_inputs.md).

```yaml
project:
  random_seed: 42

sampling:
  split_scope: c3_source_line_blocks
  source_line_ranges:
    train: [0, 25]
    validation: [25, 35]
    test: [35, 50]

interpolation_mask:
  partition: test
  kind: random_trace
  missing_fraction: 0.8

benchmark_case:
  id: c3_na_test_random_trace_80_seed42

benchmark_volume:
  id: c3_na_test_t0_384_sl35_50_sh27_59_rx0_8_ry18_50
  selection:
    time: [0, 384]
    source_line: [35, 50]
    shot_in_line: [27, 59]
    relative_receiver_x: [0, 8]
    relative_receiver_y: [18, 50]
```

SEG-Y inputs and everything under `data/interim/` and `data/processed/` are generated or externally obtained data and must not be committed to Git.

## Training commands

The `train` command group trains these model families:

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

## Interpolation commands

The `interpolate` command group runs methods on a verified benchmark volume:

```text
interpolate pocs     CPU/NumPy Fourier POCS-5D
interpolate drr      CPU/NumPy damped rank-reduction 5D
interpolate siren    per-volume observed-only SIREN internal learning
interpolate nersi    per-volume observed-only profile-wise NeRSI reimplementation
interpolate ccnet5d  per-volume observed-only CCNet5D training and inference
interpolate relational-trace-graph  per-volume observed-only GNN training and inference
```

These commands require an existing prepared partition, interpolation mask, benchmark case,
and dense C3 volume index, with the same seven input/output path arguments:

```bash
python -m seis_interp.cli interpolate pocs \
  --config studies/<study>/config.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/<partition-id> \
  --mask data/processed/c3_na/<partition-id>/masks/<mask-id> \
  --case data/processed/c3_na/<partition-id>/cases/<case-id> \
  --volume data/processed/c3_na/<partition-id>/volumes/<volume-id> \
  --output runs/<study>/<run-id> \
  --json
```

The immutable run contains the resolved configuration, verified input lock, target-only physical
amplitude metrics, run metadata, and `prediction.npy`. Method-specific conditions
belong in the study configuration.

`interpolate siren` fits a fresh model for each selected volume using only observed samples and a
fixed number of training steps. It also writes `artifacts/final.pt`; its final-step checkpoint
contract is separate from the survey-wide `train siren` contract. It accepts
`--device` to override the configured training device. Its progress and warnings always go to
stderr, while stdout contains only the final human-readable summary or strict JSON with `--json`.

`interpolate nersi` is an explicit repository reimplementation, not an official or complete
reproduction. It maps each local `(source_line, shot_in_line, relative_receiver_x)` profile key to
the `(time, relative_receiver_y)` profile, fits only the selected volume's observed traces, and
uses evaluation-target amplitudes only after prediction for scoring. The initial clean-data C3
contract has no nuclear-norm term. It writes `final.pt` and `prediction.npy` at the run root and accepts `--device`. Its checkpoint is bound to
the exact verified case and volume hashes and is not a reusable pretrained model.

`interpolate ccnet5d` fits a fresh model on pseudo-masked observed traces within the same fixed
PoC volume, then predicts all evaluation targets. It uses observed-only global RMS normalization,
trace-relative loss, and a fixed optimizer-step budget, and writes `final.pt`.
It accepts `--device`; progress goes to stderr and the final summary to stdout, with strict JSON
when `--json` is supplied.

`interpolate relational-trace-graph` trains on whole-trace pseudo-masks within the same volume's
observed set, using shared observed-only global RMS and trace-relative loss. Geometry uses the
full analysis domain's fixed midpoint bounds. `training.max_steps` fixes the AdamW update count;
`training.inner_mask_fraction` fixes the random pseudo-mask fraction. The final state is saved to
`final.pt` before predicting every target, including queries without observed neighbors.
The dense physical prediction is scored by the common C3 evaluator. The command requires
`--volume`, accepts `--device`, and does not accept an external checkpoint or use validation.

## Run outputs

The fixed random-80 PoC benchmark is `c3_sl25_40_random80_observed_only_v1`.
Its selection, benchmark ID, and 0.8 outer missing fraction are defined in
`src/seis_interp/data/c3_poc_inputs.py`. Production inputs must match all five
selection ranges before artifact loading; config/artifact agreement alone is insufficient.
Input locks and run metadata use this same benchmark ID.

The five random-80 PoC methods (POCS, DRR, NeRSI, CCNet5D, and relational trace graph) write:

```text
config.resolved.yaml
inputs.lock.json
metadata.json
metrics.json
prediction.npy
```

Neural methods also write `final.pt` at the run root. Classical methods have no checkpoint.
The prediction is a finite full-shape physical-amplitude array with exact observed reinsertion.
The input lock records the dataset, full selection, O/T counts, and nested volume-file hashes.
Comparisons must verify the full input lock, not just the case and volume IDs.
`metrics.json` is the unchanged common evaluator result. `metadata.json` separates common
identity, normalization, objective, training/reconstruction, coverage, timing, and resource usage
from nested `method_details`.

Neural PoC seeds are explicit nonnegative integers, independently configurable and recorded in
the checkpoint and metadata. `project.random_seed` remains the outer benchmark seed.

| Method | Native seed fields |
| --- | --- |
| NeRSI | `training.model_initialization_seed`, `training.sampling_seed` |
| CCNet5D | `training.model_initialization_seed`, `patches.placement_seed`, `patches.inner_mask_seed` |
| Relational trace graph | `training.model_initialization_seed`, `training.episode_seed` |

Check the fixed input selection, mask, counts, and observed-only RMS without creating a run:

```bash
python -m seis_interp.cli poc check --interim INTERIM --processed PROCESSED \
  --mask MASK --case CASE --volume VOLUME --json
```

Other model-selection training runs write:

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
