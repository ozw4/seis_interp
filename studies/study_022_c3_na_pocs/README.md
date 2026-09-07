# study_022_c3_na_pocs

## Status

`draft`

## Research question

For one fixed SEG C3 Narrow-Azimuth benchmark case and dense validation volume, how well does
Fourier POCS-5D reconstruct traces assigned the `evaluation_target` role?

This study evaluates a conventional, explicitly specified baseline. It does not implement
POCS-Net, denoising, or a claimed reproduction of an unpublished crop or a paper's complete
parameterization.

## Inputs and split contract

The input is the all-FFID interim dataset bound to the four SEG-Y checksums in `inputs.yaml`.
The current artifact has 50 physical source lines, so the contiguous source-line partition is:

```text
train       [0, 25)
validation  [25, 35)
test        [35, 50)
```

All shots and receiver traces from one source line remain in one partition. Duplicate physical
trace cells are canonicalized by retaining the lowest `array_row`. The study begins on the
validation partition with a seed-42 `random_trace` mask whose requested missing fraction is 0.8.
The requested mask fraction and the realized missing fraction inside a selected crop are recorded
separately.

The benchmark case fixes the partition, visibility roles, and source-file hashes. Both study
volumes reuse that case:

| Use | Time | Source line | Shot in line | Receiver x | Receiver y |
|---|---:|---:|---:|---:|---:|
| Formal candidate | `[0, 384)` | `[25, 35)` | `[27, 59)` | `[0, 8)` | `[18, 50)` |
| Smoke | `[64, 128)` | `[25, 29)` | `[27, 35)` | `[0, 8)` | `[18, 34)` |

The formal candidate contains 81,920 spatial traces. The smoke crop contains 4,096 spatial traces,
keeps all four spatial axes non-singleton, and uses a raw-data interval with non-zero signal; the
first 64 samples of this crop were all zero in the inspected interim artifact. These are explicit
initial C3 crops, not published reference crops.

## Method

The method is **Fourier POCS-5D, hard threshold, geometric schedule**. Input axes are
`(time, source_line, shot_in_line, relative_receiver_x, relative_receiver_y)`. It applies an
orthonormal real FFT along time, then an orthonormal four-dimensional spatial FFT independently at
every frequency. Complex coefficients below the scheduled magnitude threshold are removed without
changing retained phase. Observed samples are reinserted after every iteration and again in the
final physical-amplitude output.

The initial zero-filled observed spectrum supplies one fixed maximum-magnitude threshold reference
per frequency and block. The formal candidate uses 100 iterations, ratios 1.0 through 0.01 on a
geometric schedule, and windows of shape `[128, 8, 16, 8, 32]` with overlap
`[64, 4, 8, 4, 16]`. Overlapping predictions use a positive interior-Hann, synthesis-only blend.
The smoke configuration uses one block and 20 iterations.

The implementation is CPU/NumPy only. FFT work uses NumPy's float64/complex128 behavior and the
saved prediction returns to the input dtype. It uses every rFFT bin and does not apply amplitude
normalization, NMO, AGC, demeaning, padding, a frequency cutoff, early stopping, or an oracle
amplitude correction. The preparation artifact's train-only normalization metadata remains part of
the verified case binding but is not applied to POCS amplitudes.

The study configuration is independent of the SIREN training defaults. Its `normalization`
section declares only the settings required to generate the prepared baseline artifact;
POCS does not apply either coordinate or amplitude normalization. `run.json` records the
POCS amplitude normalization as `none`.

## Evaluation and comparison protocol

The primary metric is physical-amplitude global S/N over every sample of only the
`evaluation_target` traces in the selected volume. RMSE and relative L2 on that same domain and the
maximum observed-sample error are supporting diagnostics. Zero-fill is evaluated on exactly the
same target domain. Observed traces are never mixed into target S/N, and per-trace dB averages or
oracle-scaled values are not substitutes for the primary metric.

The run requires `evaluation.primary_metric: physical_amplitude_global_snr_db` and
`evaluation.domain: evaluation_target`; a contradictory declaration is rejected before
creating the output directory.

Hyperparameters are selected using validation cases. Once selected, iteration count, thresholds,
and window settings are fixed before evaluating a test case. A test target amplitude must not
influence threshold or window selection. `random_trace` and `random_whole_ffid` are distinct missing
data conditions; a future whole-FFID condition requires its own mask-bound case and must state
whether its settings were selected separately. Later methods must reuse the same case, volume, and
evaluation domain for a direct comparison.

A low score can indicate an inadequate provisional iteration or window condition and is not, by
itself, evidence of a general POCS limit. Conversely, matching or exceeding a published score is
not an implementation acceptance criterion. Samples uncovered by every non-empty window remain in
the target evaluation and their count must be reported.

## Reproduction

Prepare the common source-line partition and validation mask:

```bash
python -m seis_interp.cli data prepare-baseline \
  --config studies/study_022_c3_na_pocs/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/c3_source_line_blocks_seed42

python -m seis_interp.cli data prepare-mask \
  --config studies/study_022_c3_na_pocs/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --output data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42
```

Bind the case and create both volume indices:

```bash
python -m seis_interp.cli data prepare-benchmark-case \
  --config studies/study_022_c3_na_pocs/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --output data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42

python -m seis_interp.cli data prepare-c3-volume-index \
  --config studies/study_022_c3_na_pocs/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --output data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_t0_384_sl25_35_sh27_59_rx0_8_ry18_50

python -m seis_interp.cli data prepare-c3-volume-index \
  --config studies/study_022_c3_na_pocs/config_smoke.yaml \
  --input data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --output data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34
```

Run the smoke condition with a new immutable run ID:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_smoke"

python -m seis_interp.cli interpolate pocs \
  --config studies/study_022_c3_na_pocs/config_smoke.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --volume data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34 \
  --output "runs/study_022_c3_na_pocs/$RUN_ID" \
  --json
```

After the smoke run succeeds and validation settings are fixed, run the formal candidate by using
`config.yaml`, the formal volume directory, and a new run ID. Do not automatically download data,
launch the formal run, or start a parameter sweep as part of preparation.

## Expected generated outputs

Each POCS run writes `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`, `run.json`, and
`artifacts/prediction.npy`. The prediction has the selected volume's five-dimensional shape and
physical-amplitude dtype. The lock records the case and volume hashes; run metadata records method,
FFT/window conditions, environment, timings, and prediction layout. Generated data and runs are not
committed.

## Acceptance criteria

- Synthetic numerical tests preserve complex phase, all four spatial FFT axes, odd and even time
  lengths, input dtype, and exact observed samples while improving the fixed Fourier fixture over
  zero-fill.
- Both `random_trace` and `random_whole_ffid` contracts work on hash-bound synthetic cases.
- Window tests cover overlap, edges, empty blocks, and uncovered sample accounting.
- Target amplitudes are first read at the evaluation boundary; changing only a correctly rebound
  target changes metrics but not the prediction.
- The real CLI can take a small synthetic case through POCS, evaluation, strict-JSON run records,
  and prediction saving without Torch.
- A C3 smoke run, when its local generated inputs exist, verifies hashes, exact observed-sample
  preservation, finite saved predictions, target-only metrics, and uncovered sample reporting.

No minimum C3 S/N or comparison with Studies 017-021 is asserted before a conforming validation
run exists.

## Limitations

The initial thresholds, iteration count, and windows are starting values rather than tuned or
paper-reproduction values. The method assumes a regular dense crop and one trace mask shared by all
time samples. It has no NMO, anti-aliasing extension, NUFFT, checkpoint/resume path, GPU backend, or
automatic search. Window choice changes the method as well as its memory use. Results from this new
case/evaluation contract are not directly interchangeable with older studies' differently scoped
metrics.

## Current result

A local C3 smoke run completed on the `[64, 128)` validation crop using the unmodified starting
condition in `config_smoke.yaml`; its generated run ID is
`20260907T021659Z_56697de_smoke`. It evaluated 3,275 target traces (209,600 samples) and produced a
physical-amplitude target-only global S/N of 22.8842 dB, RMSE 0.2875, and relative L2 0.0717. The
zero-fill RMSE was 4.0074. The saved prediction had exact observed-sample consistency, zero
uncovered samples, and no warnings.

This smoke result checks the execution and evaluation contracts on a small initial crop; it is not
a formal-volume result, a tuned validation result, or evidence for a general performance claim. It
was generated from the development worktree before the POCS implementation was committed, so its
recorded base Git SHA alone is not a reproducible implementation reference; it must be rerun with a
new ID after commit before promotion. The formal candidate has not been run, and no threshold or
window selection has been made from test data. The study therefore remains `draft`.
