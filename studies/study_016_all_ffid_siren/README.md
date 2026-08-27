# study_016_all_ffid_siren

## Status

`draft`

## Research question

Can one SIREN trained across every FFID in SEG C3 Narrow-Azimuth reconstruct validation traces
that are completely held out within each FFID?

This is survey-wide trace interpolation with every FFID represented during training. It is not
shot interpolation: no complete FFID is held out, and FFID is not added as a model input.

## Motivation

Study 014 showed that a complete-trace batch covering all training traces of FFID 2348 can
escape the near-zero predictor with the ordinary training-global-RMS target. This study extends
that observed sample density to the survey without loading all survey points into one batch:
one optimizer update consumes one FFID, and one shared coordinate network is updated across the
full survey.

Study 016 follows the already completed Study 015 budget-extension study. It therefore uses the
next available identifier even though the implementation proposal that preceded that later work
called this experiment Study 015.

## Input scope

The input is all four files declared by the SEG C3 NA manifest, covering FFIDs 2 through 4781.
Sail-line-end FFIDs with fewer than the 544 traces of a complete shot remain part of the dataset.
All 625 samples of every selected training or validation trace are used; no time window is cut.
The measured source checksums are locked in `inputs.yaml`.

## Split and leakage contract

Whole traces are split independently inside every FFID with seed 42. A 0.20 holdout is divided
with 0.25 of the holdout assigned to validation and the remainder to test. The split is based on
sorted `array_row` identifiers and an FFID-specific random stream, so adding another FFID cannot
change an existing FFID's membership. Every FFID must have non-empty train, validation, and test
sets. Time samples from one trace never cross split boundaries.

Only training traces fit coordinate bounds and the global amplitude RMS. Validation traces select
the checkpoint; test traces do not enter optimization, early stopping, or study selection.

## Coordinate and amplitude normalization

Time, CMP x/y, and offset use training-min/max linear scaling. Azimuth is encoded as sine and
cosine, retaining the existing six model features. Amplitudes are divided by one RMS computed
over all training samples across all FFIDs with bounded-memory accumulation.

Per-trace RMS targets are deliberately excluded: their scale is known for a training trace but is
not available at a held-out coordinate. Correlation and Huber losses are also excluded from this
first interpolation condition.

## Model and training conditions

The shared SIREN has six inputs, width 256, four sine layers, and `omega_0` and `hidden_omega`
both equal to 30. Adam minimizes point-wise L2 at learning rate `1.0e-4` with seed 42.

One optimizer update is all training traces of one FFID multiplied by all time samples. Batch size
therefore varies for incomplete FFIDs. One epoch visits every training FFID exactly once in a
seeded shuffled order. The ten-epoch budget is about 47,800 updates, chosen only to approximate
Study 014's 50,000-update budget; it is not claimed to be optimal. Early-stopping patience is
three epochs.

## Validation and checkpoint selection

At each epoch end, every validation trace is evaluated. Coordinates, targets, and predictions are
materialized for at most one FFID at a time, while reference and error energies are accumulated
into the point-weighted global S/N. The best checkpoint is selected only by this streaming global
validation S/N. No per-FFID or test-set metric affects selection.
If prediction error is exactly zero, the checkpoint retains floating-point positive infinity and
the strict JSON metrics record writes that mathematically exact outcome as the string `"inf"`.

## Reproduction

Prepare the all-FFID interim dataset:

```bash
python -m seis_interp.cli data prepare-c3-survey \
  --manifest data/external/seg_c3_na/manifest.yaml \
  --data-root "$SEIS_INTERP_DATA_ROOT" \
  --output data/interim/c3_na/all_ffids
```

Create per-FFID trace splits and training-only normalization:

```bash
python -m seis_interp.cli data prepare-baseline \
  --config studies/study_016_all_ffid_siren/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_random_split
```

Train the shared model:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_all_ffids"

python -m seis_interp.cli train siren \
  --config studies/study_016_all_ffid_siren/config.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_per_ffid_random_split \
  --output "runs/study_016_all_ffid_siren/$RUN_ID" \
  --device cuda:0
```

## Expected generated outputs

Preparation produces `traces.parquet`, memory-mappable `amplitudes.npy`, `time_s.npy`, and
`dataset.json`, followed by `trace_split.parquet`, `normalization.json`, and `preparation.json`.
Training creates one immutable run directory containing `config.resolved.yaml`,
`inputs.lock.json`, `metrics.json`, `run.json`, and `artifacts/best.pt`. Generated arrays, raw
SEG-Y data, checkpoints, and run outputs are not committed.

## Acceptance criteria

- Preparation locks all four manifest sources, covers FFIDs 2-4781, and retains incomplete FFIDs.
- Every FFID has train, validation, and test traces without time-sample leakage.
- Every epoch visits each training FFID once, with one complete-FFID optimizer update per visit.
- Training writes immutable provenance records and a loadable best checkpoint.
- Selection uses only streaming global S/N over all validation traces; test rows are unused.
- A near-zero validation S/N is a valid negative scientific result, not an execution failure.

## Limitations

This one-seed, one-condition POC does not hold out complete shots, evaluate the final test split,
sweep capacity or learning rate, schedule the learning rate, or use multiple GPUs. Point-weighted
global S/N may hide FFID-level variation. Ten epochs provide a provisional update budget rather
than evidence of convergence.

## Current result

The study has not been run. No result, run directory, checkpoint, or interpretation is recorded.

Historical rationale belongs in [`decisions.md`](decisions.md).
