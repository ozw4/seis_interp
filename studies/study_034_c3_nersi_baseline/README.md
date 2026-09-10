# study_034_c3_nersi_baseline

## Status

`planned`

The repository reimplementation and its finite validation plan are specified, but no
Study 034 preflight, candidate run, or test-partition evaluation has been executed.

## Research question

On the fixed QC validation case, how well does a profile-wise
`NeRSI (repository reimplementation)` interpolate every evaluation-target trace, and how
does its per-volume internal-learning regime differ from the proposed relational GNN's
train-partition pretraining followed by frozen validation inference?

## Hypothesis

Generating complete `(time, relative_receiver_y)` profiles from local regular-grid
`(source_line, shot_in_line, relative_receiver_x)` coordinates can exploit structure that
is different from point-wise coordinate prediction. The study does not assume that this
will match a paper-reported value or outperform the proposed GNN.

## Inputs

The only evaluation input is the existing Study 029 fixed QC suite with SHA-256
`f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee` and case
`c3_benchmark_validation_random_trace_80_seed142`. The selected volume has shape
`[384, 9, 32, 8, 32]` in
`(time, source_line, shot_in_line, relative_receiver_x, relative_receiver_y)` order.
Study 034 reuses the suite's case, mask, and volume artifacts; it does not create a
NeRSI-specific dataset or visibility mask.

Evaluation covers every validation evaluation-target trace and all 384 selected time
samples. Evaluation-target amplitudes are available only to the existing evaluator after
training and prediction have finished. The test partition is not used.

## Method

The method is explicitly identified as `NeRSI (repository reimplementation)`, not an
official implementation or a complete paper reproduction. Each profile key generates one
`(384, 32)` profile, giving `9 * 32 * 8 = 2304` profile keys. The key indices are normalized
independently to `[0, 1]`; amplitudes use one global RMS computed from observed samples in
the selected volume. Training uses observed-sample-weighted masked MSE and exactly 5,000
Adam updates, with 16 profiles per update. This is 80,000 profile presentations, not a claim
of equal sample exposure or equal compute relative to another method.

The paper-aligned core is an exponential Fourier mapping with `K=40`, a two-layer fully
connected encoder, and three convolution/PixelShuffle/activation blocks that each upsample
both profile axes by two. The initial C3 baseline has no nuclear-norm term. Frequency base,
channel widths, GELU, kernel size 3, linear signed output, optimizer details, and training
budget are repository choices documented in
[`implementation_contract.md`](implementation_contract.md).

Four candidates are fixed before validation:

| Candidate | Encoder width | Latent channels | Decoder channels | Learning rate |
|---|---:|---:|---|---:|
| A | 256 | 64 | `[64, 32, 16]` | `1e-3` |
| B | 256 | 64 | `[64, 32, 16]` | `3e-4` |
| C | 384 | 96 | `[96, 48, 24]` | `1e-3` |
| D | 384 | 96 | `[96, 48, 24]` | `3e-4` |

All candidates use training seed `20260908`, independently of mask seed 142, and prediction
batch size 64. No candidate is added, retried, or changed in response to its validation
score within this study.

## Reproduction

Resolve Candidate A preflight without writing a run:

```bash
python -m seis_interp.pipelines.c3_first_results \
  --config studies/study_034_c3_nersi_baseline/config_preflight.yaml \
  --inputs studies/study_034_c3_nersi_baseline/inputs.yaml \
  --action nersi \
  --preflight
```

Execute the disposable ten-update resource preflight in a fresh process:

```bash
python -m seis_interp.pipelines.c3_first_results \
  --config studies/study_034_c3_nersi_baseline/config_preflight.yaml \
  --inputs studies/study_034_c3_nersi_baseline/inputs.yaml \
  --action nersi \
  --preflight \
  --execute
```

Resolve each full candidate by replacing `a` with `b`, `c`, or `d` as needed:

```bash
python -m seis_interp.pipelines.c3_first_results \
  --config studies/study_034_c3_nersi_baseline/config_candidate_a.yaml \
  --inputs studies/study_034_c3_nersi_baseline/inputs.yaml \
  --action nersi
```

Add `--execute` only when starting the corresponding immutable full validation run. Run all
four candidates once before selecting a primary candidate.

## Expected outputs

The preflight records ten disposable optimizer updates, one prediction batch or less,
timings, peak resources, and linear full-run estimates. It performs no target scoring or
full-volume prediction, and its model state is not reused.

Each completed candidate produces the native immutable run records, full physical-amplitude
prediction, and fixed-final checkpoint. Comparison metadata must include the 2,304 profile
keys, profile shape `[384, 32]`, observed trace and sample counts read from the verified run,
80,000 profile presentations, parameter count, memory, per-volume training time, prediction
time, and target-only physical S/N and RMSE. GNN reporting separates train-partition
pretraining from frozen validation prediction; NeRSI reporting combines its per-volume fit
and prediction without claiming equal training information or equal compute.

## Acceptance criteria

The primary metric is `physical_amplitude_global_snr_db` on all validation targets and all
384 samples. A primary candidate is selected only after all four declared candidates have
completed on the same fixed case. Target RMSE, observed fit before hard reinsertion,
non-finite or coverage failures, runtime, and memory are diagnostics. Only the final
5,000-update checkpoint is eligible; there is no best-checkpoint selection.

The paper's public S/N values, including 15.16 dB and 18.67 dB, are context rather than
acceptance thresholds. A failed candidate is recorded without automatic retry. No result is
accepted until the saved prediction can be independently re-scored against the exact input
hashes and the final checkpoint can reproduce it within the declared numerical tolerance.

## Limitations

This is a regular-grid, per-volume internal-learning comparison on one validation case and
one training seed. It excludes nuclear-norm regularization, off-grid prediction, NMO,
sliding windows, regional models, training resume, SIREN shear, per-trace RMS scaling, and
SIREN envelope loss. Candidate widths and learning rates are repository choices because the
C3 paper description does not specify them. The test partition remains unevaluated, so this
study makes no test-performance or broad generalization claim.
