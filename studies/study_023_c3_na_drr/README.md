# study_023_c3_na_drr

## Status

`draft`

## Research question and inputs

How well does damped rank-reduction (DRR) 5D interpolation reconstruct missing traces in the
same verified SEG C3 Narrow-Azimuth validation case and volumes as
[POCS Study 022](../study_022_c3_na_pocs/README.md)?

`inputs.yaml` fixes the same four SEG-Y checksums, all-FFID interim dataset, source-line split,
mask, case, and formal/smoke volume paths. The physical source-line split is train `[0, 25)`,
validation `[25, 35)`, test `[35, 50)`. Duplicate physical trace cells retain the lowest
`array_row`. The initial validation mask is `random_trace`, requested missing fraction 0.8,
seed 42; realized missing fractions are recorded separately for each crop. Both volumes bind
`c3_na_validation_random_trace_80_seed42`.

| Use | Time | Source line | Shot in line | Receiver x | Receiver y |
|---|---:|---:|---:|---:|---:|
| Formal candidate | `[0, 384)` | `[25, 35)` | `[27, 59)` | `[0, 8)` | `[18, 50)` |
| Smoke | `[64, 128)` | `[25, 29)` | `[27, 35)` | `[0, 8)` | `[18, 34)` |

These are explicit repository crops, not a claimed reproduction of an unpublished paper crop.
The formal candidate has 81,920 spatial traces; smoke has 4,096 and uses the same nonzero-signal
time interval as Study 022.

## Method and initial conditions

The method is `damped_rank_reduction_5d`, independently implemented from the numerical contract
and Chen et al. (2016), [“Simultaneous denoising and reconstruction of 5-D seismic data via
damped rank-reduction method”](https://doi.org/10.1093/gji/ggw230), *Geophysical Journal
International*, 206(3), 1695–1717. No external GPL implementation was copied or adapted.

This is the reconstruction-only, hard observed-data consistency variant with a fixed
iteration count, not the paper's simultaneous denoising/reconstruction mode with relaxed
observation weights and convergence stopping. The pipeline records
`method_variant: reconstruction_only_hard_consistency` in `metrics.json` and `run.json`,
and `drr.mode: reconstruction_only` in `run.json`. Local spatial-window and acquisition-grid
conditions are specified below and recorded separately.

Axes are `(time, source_line, shot_in_line, relative_receiver_x, relative_receiver_y)`.
An orthonormal time rFFT, padded to the smallest power of two at least as large as the time
length, supplies complex 4D spatial slices. This uses the repository's regular dense grid,
not a remapping to the paper's midpoint/offset coordinates.

Each spatial axis of length `X` uses Hankel embedding length `floor(X/2)+1`. At each selected
frequency, exact CPU/NumPy `linalg.svd(full_matrices=False)` retains rank `r`, damping each
retained singular value `s` to `s * (1 - (delta/s)**p)`, where `delta` is the rank+1 singular
value. Anti-diagonal averaging restores the spatial grid. Every iteration reinserts the
observed frequency coefficients exactly. After inverse FFT, truncation, and conversion back
to the input dtype, observed time samples are also reinserted exactly.

The initial formal candidate uses rank 4, damping power 3, 10 iterations, and frequencies
from 5 Hz through Nyquist (inclusive). Spatial windows `[5, 8, 4, 8]` have requested overlap
`[0, 0, 0, 0]`. Their embedding shape is `[3, 5, 3, 5]`, start shape `[3, 4, 2, 4]`, and
Hankel matrix shape `[225, 96]`, so `4 < min(225, 96)`.

Smoke uses rank 4, power 3, 3 iterations, the same frequency limits, and spatial windows
`[4, 4, 4, 8]` with zero requested overlap. Its Hankel matrix shape is `[135, 32]`, also
valid for rank 4. These are initial conditions, not tuned settings or exact paper parameters.

Time is never windowed. Spatial windows are processed sequentially and their predictions
blended using the product of positive interior-Hann weights, `hanning(length+2)[1:-1]`,
without an analysis taper or spatial padding. The final window aligns to each volume edge;
that alignment can introduce overlap even when the requested overlap is zero. Empty windows
are skipped and do not dilute neighboring predictions. Traces uncovered by every nonempty
window stay zero and remain in the evaluation, with a warning.

Unselected frequency bins are zero for the missing prediction before time truncation.
DC and Nyquist are made real. There is no amplitude or coordinate normalization, NMO, AGC,
demeaning, denoising schedule, adaptive rank, early stopping, randomized SVD, GPU backend,
or block parallelism. This is not optimally damped DRR or robust DRR.

The independent study config does not inherit POCS or SIREN defaults. Its `normalization`
section exists only for generation of the shared prepare-baseline artifact; neither its
coordinate nor amplitude transform is applied by DRR. Run metadata records normalization
as `none`.

## Evaluation and selection protocol

The primary metric is physical-amplitude global S/N over all samples of the selected
`evaluation_target` traces only. RMSE, relative L2, zero-fill, and observed maximum absolute
error are supporting diagnostics. Zero-fill uses the same target domain. Neither observed
traces, per-trace dB averages, nor oracle amplitude scaling enter the primary metric.
The pipeline requires `evaluation.primary_metric: physical_amplitude_global_snr_db` and
`evaluation.domain: evaluation_target`, rejecting contradictory declarations before output
creation.

Rank, frequency range, iteration count, and windows may be selected using validation scores
and must be frozen before viewing a test case. Target truth never enters reconstruction,
rank estimation, or per-input frequency selection; it is first materialized for evaluation.
POCS, DRR, and subsequent models must use the same case, volume, and evaluation domain for
direct comparisons. `random_trace` and `random_whole_ffid` are separate conditions, each
requiring its own mask-bound case.

## Reproduction

Reuse the existing Study 022 artifacts; do not regenerate or overwrite valid shared inputs.
If absent, use its documented prepare-baseline, prepare-mask, prepare-benchmark-case, and
prepare-c3-volume-index commands. Study 023 configs declare the same preparation contract.
No formal-volume reconstruction or parameter sweep is launched by preparation.

For a reproducible smoke, use a clean checkout whose `src` is the imported package. From
the repository root, with the local data artifacts available at the following paths:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_smoke"

PYTHONPATH="$PWD/src" python -m seis_interp.cli interpolate drr \
  --config studies/study_023_c3_na_drr/config_smoke.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --volume data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34 \
  --output "runs/study_023_c3_na_drr/$RUN_ID" \
  --json
```

Each immutable run contains only `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
`run.json`, and `artifacts/prediction.npy`. Records include verified input hashes, Git SHA and
worktree state, actual FFT/bin/Hankel/window conditions, prediction layout, timings, process
maximum RSS, and uncovered counts. Generated data and run artifacts are not committed.

## Acceptance and limitations

Synthetic numerical tests must cover the Hankel layout and averaging with independent
references, damping, complex phase, frequency limits, odd/even time lengths, input dtype,
exact observations, and improvement over zero-fill on small low-rank fixtures. Window tests
cover all spatial axes, edges, empty windows, and trace-level coverage. Hash-bound synthetic
integration tests and a real CLI invocation must verify target-only evaluation and strict
JSON without Torch. Correctly rebinding changed target amplitudes must change metrics but
not prediction.

A clean-commit C3 smoke checks the execution contract; no minimum smoke S/N is required.
Promotion to formal results requires `git_worktree_dirty: false`; dirty development runs
remain allowed. SVD runtime and memory must be measured before considering optimization.
The small smoke is not evidence of DRR versus POCS performance on the formal volume. Test
partition evaluation, hyperparameter sweeps, figures, and result promotion are outside the
current task.

## Current result

Run `20260907T035913Z_ca7bc8b_smoke` completed from clean commit
`ca7bc8bd7bcac694703093376fe274b23179b0af`, recording `git_worktree_dirty: false`.
It used the smoke crop in the table above, the seed-42 validation `random_trace` mask with
requested missing fraction 0.8, and the unchanged `config_smoke.yaml` condition: rank 4,
damping power 3, 3 iterations, and nonoverlapping `[4, 4, 4, 8]` spatial windows.

The run evaluated 3,275 target traces (209,600 samples). Physical-amplitude target-only global
S/N was 2.9109 dB, RMSE 2.8662, and relative L2 0.7152. Zero-fill had S/N 0.0000 dB and
RMSE 4.0074, so the smoke improved target S/N by 2.9109 dB. Observed maximum absolute error
was exactly 0.0. No traces or samples were uncovered, and there were no warnings.

The float32 prediction has shape `[64, 4, 8, 8, 16]`. All eight spatial blocks were nonempty;
each used a `[135, 32]` Hankel matrix. The 64-point FFT processed 30 bins, from 5.8594 to
62.5000 Hz. Loading and hash verification took 3.9607 s, reconstruction 0.5783 s, and
evaluation 0.0051 s; process maximum RSS was 256.8164 MiB (whole-process scope).

Saved records passed strict-JSON, configuration, input-lock, finite-prediction, exact
observed-sample, and independently recomputed metric checks. The input lock is identical
to the clean Study 022 smoke's lock. Generated run artifacts remain the authoritative
full-precision record.

This is a small execution-contract smoke, not a formal-volume result, tuned validation
result, or a DRR-versus-POCS performance conclusion. The formal candidate and test partition
have not been run, no parameters have been selected from this result, and the study remains
`draft`.
