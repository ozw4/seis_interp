# study_024_c3_na_siren_volume

## Status

`draft`

## Research question

On the same SEG C3 Narrow-Azimuth validation volume and seed-42 `random_trace` 80% benchmark
used by Studies 022 and 023, how well does a SIREN fitted only to observed coordinate-amplitude
samples reconstruct the `evaluation_target` traces after a fixed optimizer-step budget?

## Inputs and common benchmark

`inputs.yaml` binds the same four SEG-Y checksums, all-FFID interim dataset, source-line split,
mask, case, and formal/smoke volume paths as
[POCS Study 022](../study_022_c3_na_pocs/README.md) and
[DRR Study 023](../study_023_c3_na_drr/README.md). The physical source-line split is train
`[0, 25)`, validation `[25, 35)`, and test `[35, 50)`. Duplicate physical trace cells retain
the lowest `array_row`.

The study uses the validation `random_trace` mask with requested missing fraction 0.8 and seed 42.
Both volumes bind benchmark case `c3_na_validation_random_trace_80_seed42`:

| Use | Time | Source line | Shot in line | Receiver x | Receiver y |
|---|---:|---:|---:|---:|---:|
| Formal candidate | `[0, 384)` | `[25, 35)` | `[27, 59)` | `[0, 8)` | `[18, 50)` |
| Smoke | `[64, 128)` | `[25, 29)` | `[27, 35)` | `[0, 8)` | `[18, 34)` |

The formal candidate contains 81,920 spatial traces. The smoke crop contains 4,096 spatial
traces, keeps every spatial axis non-singleton, and uses the same nonzero-signal interval as the
other common-benchmark studies. These are explicit repository crops, not published reference
crops. Requested and realized missing fractions remain separate recorded quantities.

`project.random_seed: 42` belongs to this benchmark identity: it must agree with the seed used to
construct the bound mask and case. The separate `training.random_seed: 42` fixes SIREN parameter
initialization and observed-point sampler draws. They deliberately begin with the same value, but
their data-binding and optimization roles are independent.

## Method and positioning

The method is `siren_5d` with
`method_variant: per_volume_internal_learning_fixed_steps`. One new coordinate network is fitted
for each selected volume. Its six features are:

```text
time_s
cmp_x_m
cmp_y_m
offset_m
azimuth_sin
azimuth_cos
```

Time and geometry bounds are fitted over the known time axis and all known selected-volume
coordinates. Amplitude scale is one global RMS fitted only from samples on observed traces.
Training divides observed amplitudes by that RMS; prediction multiplies by it to return to the
physical-amplitude domain.

The formal candidate uses a width-256 SIREN with four sine layers, one output, and both
`omega_0` and `hidden_omega` equal to 30. Adam minimizes point-wise L2 at learning rate
`1.0e-4`. Each update samples 65,536 observed points uniformly with replacement, and training
stops after exactly 20,000 optimizer steps. There is no scheduler, early stopping, target
validation, best-checkpoint selection, mixed precision, or gradient accumulation. The saved
`artifacts/final.pt` is the final fixed-step model, not a selected `best.pt`.

The smoke condition retains the same model, omega values, and learning rate while reducing the
training batch to 4,096 points, the budget to 100 steps, the reporting interval to 20, and the
prediction batch to 32,768 points on CPU. It checks the execution contract and is not a
performance-comparison condition.

This internal-learning setup follows the observed-coordinate fitting and missing-coordinate query
problem framing of [Liu et al. (2024)](https://doi.org/10.1109/TGRS.2024.3431439), while using the
repository's existing [SIREN](https://arxiv.org/abs/2006.09661) implementation. It is a SIREN
baseline, not a complete reproduction of ISR. It also differs from
[Study 016](../study_016_all_ffid_siren/README.md), which trains one survey-wide model with
within-FFID holdouts. This study trains one model per benchmark volume with the same input
visibility available to POCS and DRR.

## Leakage contract

- Only `observed_trace_mask == true` coordinate-amplitude samples enter the sampler and loss.
- The amplitude RMS is fitted only from observed trace samples.
- All selected coordinates may fit geometry bounds because acquisition geometry is known.
- Evaluation-target amplitudes are first materialized at the common evaluation boundary after
  training and full-volume prediction.
- Target metrics do not select a checkpoint, stop training, or alter an individual run.
- Model and training conditions must be fixed on validation before any future test-case result is
  inspected.

The full volume is queried in bounded coordinate chunks, converted back to physical amplitude,
and then given hard observed-data consistency by reinserting the input observed traces exactly.
The model's observed fit error before reinsertion remains a diagnostic.

## Evaluation and selection protocol

The primary metric is physical-amplitude global S/N over all samples of only the
`evaluation_target` traces. Target-only RMSE, relative L2, and zero-fill metrics support that
score. Observed model RMSE and maximum absolute error before reinsertion diagnose the learned
function; the shared evaluator verifies zero observed error after reinsertion.

The run requires `evaluation.primary_metric: physical_amplitude_global_snr_db` and
`evaluation.domain: evaluation_target`. A contradictory declaration is rejected before output
creation. Validation results may inform a later study-level choice, but no target metric may
change the model within the run. No minimum S/N is an implementation acceptance criterion.

## Reproduction

Reuse the existing Study 022 benchmark artifacts; do not regenerate or overwrite valid shared
inputs. If they are absent, follow Study 022's prepare-baseline, mask, benchmark-case, and volume
index commands.

For a reproducible CPU smoke, use a clean checkout whose `src` is the imported package. From the
repository root, with the shared local artifacts available:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_smoke"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$PWD/src" \
  python -m seis_interp.cli interpolate siren \
  --config studies/study_024_c3_na_siren_volume/config_smoke.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --volume data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34 \
  --output "runs/study_024_c3_na_siren_volume/$RUN_ID" \
  --device cpu \
  --json
```

The two thread environment variables bound CPU parallelism for this smoke invocation without
changing the study's numerical configuration. `run.json` records the effective Torch thread
count as `resources.torch_num_threads` so the intended limit can be checked.

To run the same smoke condition on the first CUDA device, keep `config_smoke.yaml` unchanged and
override only its effective device:

```bash
GPU_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_gpu_smoke"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$PWD/src" \
  python -m seis_interp.cli interpolate siren \
  --config studies/study_024_c3_na_siren_volume/config_smoke.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --volume data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34 \
  --output "runs/study_024_c3_na_siren_volume/$GPU_RUN_ID" \
  --device cuda:0 \
  --json
```

The resolved run configuration records `cuda:0`; model, optimizer, budget, prediction batch, and
the inherited `training.random_seed` remain the smoke condition above.

Do not launch the formal candidate or a parameter sweep as part of implementation or smoke
preparation.

## Expected generated outputs

Each successful run creates exactly:

```text
config.resolved.yaml
inputs.lock.json
metrics.json
run.json
artifacts/final.pt
artifacts/prediction.npy
```

The fixed-final checkpoint stores CPU weights, all SIREN constructor values, volume-local
normalization, coordinate metadata, completed steps, and final batch loss. The prediction is a
finite physical-amplitude array with the selected five-dimensional shape and exact observed-trace
reinsertion. Generated data, checkpoints, predictions, and run directories are not committed.

## Compute reporting

`run.json` records parameter count, optimizer steps, point batch size, prediction batch size,
training and prediction seconds, process maximum RSS, and effective Torch threads. CUDA runs also
record peak allocated and reserved memory, GPU name, compute capability, total device memory,
PyTorch CUDA build, cuDNN version, float32 matmul precision, and CUDA matmul/cuDNN TF32 flags.
The model parameter dtype and training input/target tensor dtypes are recorded separately from
the stored data dtype. These fields describe the effective numerical conditions; they do not
promise bitwise agreement between different GPU models or software builds.

The record also binds the Git commit and worktree state, verified inputs, device, versions,
coordinate and amplitude scale sources, and artifact paths. Top-level `random_seed` and
`input.mask.random_seed` denote the benchmark seed; `training.random_seed` denotes the independent
model-initialization and point-sampling seed.

## Acceptance criteria

- Training and normalization depend on observed amplitudes only.
- Changing only correctly rebound target truth can change metrics but not checkpoint weights or
  prediction.
- Both common mask kinds work in synthetic integration tests, and a same-seed CPU rerun is
  numerically repeatable.
- Changing only the training seed can change model weights while preserving the input lock,
  case, mask, evaluation-target counts, and observed-only amplitude RMS.
- Chunked prediction remains finite, restores physical amplitude, and preserves observed traces
  exactly.
- The final checkpoint reloads the same model constructor and scaling contract.
- Strict run records contain the actual model, training, prediction, resource, and provenance
  conditions.
- Promotion to a formal result requires a clean worktree; dirty development runs remain allowed
  and are explicitly recorded.

## Limitations

This initial one-seed, one-architecture condition does not sweep capacity, frequencies, learning
rate, batch size, or optimizer budget. It uses physical coordinates only, trains one model per
volume, and has no scheduler, early stopping, AMP, distributed training, pretrained model, or
survey-wide training. Large target gaps may be difficult for a physical-coordinate SIREN. The
smoke crop is execution evidence only and cannot support a formal performance conclusion.

## Current result

The clean CPU smoke run ID is `20260907T054518Z_9c07ccf_smoke`; its immutable local directory is
`runs/study_024_c3_na_siren_volume/20260907T054518Z_9c07ccf_smoke/`. It records implementation
commit `9c07ccfa0236e7c40cb3412dd1348e803dd77a4f`,
`git_worktree_dirty: false`, and `status: success`.

This historical run predates the separate `training.random_seed` field. At its recorded commit,
`project.random_seed: 42` supplied both model initialization and sampler draws. Its six generated
files remain unchanged; the current study configuration makes the same training seed explicit for
subsequent runs without rewriting this immutable evidence.

The run uses the smoke crop above, shape `[64, 4, 8, 8, 16]` in `float32`, with the shared
validation random-trace 80% mask and seed 42. The realized missing fraction is 0.7996.
It fits the width-256, four-sine-layer model with both omega values 30, no layer schedule or
skip connections, and 199,425 parameters. Adam/L2 performs exactly 100 updates at learning rate
`1.0e-4`, with 4,096 sampled points per update. Prediction uses batches of at most 32,768 points.
The effective Torch CPU thread count is 8.

| Quantity | Smoke measurement |
|---|---:|
| Observed training traces / samples | 821 / 52,544 |
| Observed-only amplitude RMS | 4.1126 |
| Final batch loss (before the last update) | 0.5018 |
| Evaluation-target traces / samples | 3,275 / 209,600 |
| Physical-amplitude target global S/N | 1.2691 dB |
| Target RMSE / relative L2 | 3.4626 / 0.8641 |
| Zero-fill target S/N / RMSE | 0.0000 dB / 4.0074 |
| Observed model RMSE before reinsertion | 3.1632 |
| Observed model maximum absolute error before reinsertion | 69.4093 |
| Observed maximum absolute error after reinsertion | exactly 0.0 |
| Uncovered traces / samples | 0 / 0 |
| Training / prediction time | 3.8628 s / 1.2181 s |
| Whole-process peak CPU RSS | 765.5781 MiB |

The run contains only the six expected files and has no warnings. Reloading the checkpoint and
its volume-local transforms reproduces all 262,144 saved prediction points bitwise on the same
CPU/thread setup. Predictions are finite and observed traces are bitwise preserved. Its input
lock is identical to the clean POCS and DRR smoke input locks.
Full-precision records in the run are authoritative; displayed measurements are rounded.

This is smoke execution-contract evidence, not a formal-volume result, a tuned result, or a
performance conclusion against POCS/DRR. No test-partition result has been inspected. Later
validation screening may examine step budget, learning rate, omega values, exponential layer
schedules, dense skip connections, or coordinate representation, without changing this immutable
run or using target metrics for within-run checkpoint selection.
