# Fixed C3 benchmark preparation

The C3 benchmark binds a new source-line partition, canonical training pool,
partition-wide masks, cases, and dense volume indices in `benchmark_suite.json`.
It uses the existing artifact formats and SHA-256 implementation. It does not
train models or run the full interpolation experiments.

The production contract fixes time samples `[0,384)`, sail lines 25–40 inclusive,
and shape `[384,16,32,8,32]` in `VOLUME_AXIS_ORDER`. Original sail-line numbers,
when available, are mapped once to global source-line indices. Otherwise the
global indices 25–40 are used. Physical time is copied from the selected input
`time_s` values. The preparation CLI has no switch that disables these checks.
Low-level preparation APIs accept explicit `C3BenchmarkDimensions` for small
synthetic fixtures; these cannot replace the main dimensions for `seg_c3_na`.

`data qc-c3-geometry` writes a line/FFID correspondence table and reports the
measured sampling grid, stagger, missing shots, duplicate physical cells, and
receiver coverage. It never opens the amplitude array. `data qc-c3-crop` resolves
the unspecified shot/receiver starts and checks selected signal samples in
chunks. Geometry validation uses the existing canonicalization and dense-grid
builder. The initial window is centered in the common available ranges of the
selected lines. Only unspecified starts can move after a failed grid check;
candidate order is Manhattan distance followed by lexicographic index order.
Explicit starts are respected. Time and source lines do not move.

Crop QC records its integer selection, measured time range, trace-to-cell table,
initial/resolved starts, and any adjustment reason. Valid zero amplitudes remain
valid samples. Nonfinite samples prevent locking. Optional figures use the
first local source line, shot, and receiver-x column, with a fixed 99th-percentile
absolute clip. Missing matplotlib does not prevent numeric QC.

The preparation entry point defaults to a dry-run:

```bash
python -m seis_interp.cli data prepare-c3-benchmark \
  --config studies/study_027_c3_na_benchmark/config.yaml \
  --inputs studies/study_027_c3_na_benchmark/inputs.yaml \
  --json
```

Add `--execute --plots` to create artifacts. `--output` selects a new directory;
an existing directory is always rejected. `--case-id` can select a subset for
diagnosis, but that produces a partial report, never a locked suite. Failures
preserve completed artifacts and write `preparation_report.json`. There is no
overwrite, automatic repair, download, or training step. JSON output is strict;
progress and errors go to stderr.

The test block contains the fixed sail lines. Train is before it and validation
is after it, using the measured total source-line count. Each role must be
nonempty, every source line must be covered, and FFIDs and canonical physical
pairs must not cross roles. Validation uses at most 16 available lines, the same
time range and shot/receiver lengths, and the same start-selection rule.

The finite `inputs.yaml:cases` list is the sole source of suite mask conditions
and seeds. `config.yaml` contains the crop and training contract; it has no
`c3_benchmark.mask_recipes` copy. Each case config is generated from its entry.
Each mask is sampled from its entire
canonical partition with the existing RNG, sorting and rounding rules. The
same premask crop is reused for all cases of a partition. Both nominal and
realized crop missing fractions are recorded, including fully missing FFIDs.
Verification checks whole-FFID atomicity over the full candidate population,
and reproduces the configured mask from its seed. A deficient role does not
trigger a seed redraw or a different crop.

`train_pool.npy` contains the canonical allowed train rows; the manifest binds
those rows together with training time `[0,384)`. The generic prepared
normalization remains a separate artifact. It is not the model scale contract:
CCNet's existing scale is fitted on its chosen fit-region subset of the allowed
train pool, and the graph scale is fitted on its allowed unmasked training pool.
The suite readers check the selected training time explicitly.

An explicit `sampling.trace_amplitude_filter` can apply physical trace QC before
partition normalization and mask generation. It uses the existing
`exclude_all_zero` and `max_abs_amplitude` settings and scans every original
sample of each trace. A sample strictly beyond the bound excludes its whole
trace; an exact boundary value remains eligible. Study 029 uses a bound of
10000 and retains zero traces. This source-integrity check is separate from
model normalization on the authorized training time range.

The partition keeps excluded rows and records the policy and exclusion counts.
Full verification recomputes those exclusions from the original amplitudes and
checks every declared count and row. QC cannot silently promote a duplicate
physical-cell alias or remove a member of a fixed evaluation crop. Such cases
fail preparation. Without an explicit filter the strict unfiltered contract
remains in force. A QC revision requires a new output directory and suite;
existing frozen inputs and experiment results remain unchanged.

The public functions in `data/c3_benchmark_inputs.py` check the common input
hashes, partition/time contract, and files needed by the requested reader.
A case read also checks its recipe, config, binding, and fixed crop mapping.
They do not recompute signal QC or regenerate any masks. The existing lower-level
readers retain their own hash, binding, row, geometry and observation checks.
Only preparation and full verification inspect all cases.

| Function | Consumer and boundary |
|---|---|
| `load_c3_benchmark_volume_inputs` | The shared POCS, DRR, SIREN-5D, CCNet-5D volume reader; only crop observations are materialized. |
| `load_c3_benchmark_graph_domain` | GNN support and queries; the matching `volume_dir` is required. |
| `load_c3_benchmark_training_graph_domain` | Exact canonical train rows and training time, before episode masking. |
| `load_c3_benchmark_supervised_source` | CCNet teacher regions and RMS, constrained to allowed train rows/time. |

For an explicit full verification before multiple reads in one process, construct
`VerifiedC3BenchmarkSuite` once and pass it to any of these four functions:

```python
from pathlib import Path
from seis_interp.data.c3_benchmark_suite import VerifiedC3BenchmarkSuite
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs

suite_dir = Path("data/processed/c3_na/study_027_c3_na_benchmark")
verified = VerifiedC3BenchmarkSuite(suite_dir)  # One complete verification.
for case_id in (
    "c3_benchmark_test_random_trace_50_seed42",
    "c3_benchmark_test_random_trace_50_seed43",
):
    inputs = load_c3_benchmark_volume_inputs(suite_dir, case_id, verified_suite=verified)
```

This object pins the manifest, directory and expected dimensions for the run;
it is not a persistent cache. A changed manifest or required file is rejected.
Common/selected file hashes and lower-level reader checks still run on each
read, including hashing the amplitudes. Unrelated case files are inspected by
full verification, not by an individual case or train-pool read. Omitting
`verified_suite` checks the current manifest and requested inputs directly,
without declaring that the unrelated cases have been reverified.

Case configs translate mask seeds into the existing data CLI's
`project.random_seed`; they are data-preparation configs, not training configs.
Training seeds live separately in `c3_benchmark.training.random_seeds` and do not
regenerate the partition or masks. The arbitrary-coordinate GNN API is unchanged.

SIREN-5D is the independent existing implementation: it uses known crop geometry
and fits amplitude normalization and training samples from crop observations
only. Target coordinates are available to all methods, while target waveforms
are excluded from normalization, model inputs, neighborhood construction, and
training updates or stopping rules within each declared run. Target amplitudes
enter the scoring boundary.

`model.coordinate_features` defaults to the six-input `cmp_offset_azimuth`
representation. The optional `cmp_cartesian_half_offset` uses five inputs:
time, CMP X/Y, and source-minus-receiver half-offset X/Y. Both half-offset axes
share the scale derived from the maximum known offset. The existing
`cmp_cartesian_half_offset_radius` mode adds offset magnitude as a sixth input.
`model.input_features` must match the selected representation. Optional
`model.time_coordinate_scale` multiplies normalized time and defaults to 1.0.
Coordinate bounds use known crop geometry; changing these model coordinates
does not change amplitude normalization or the physical IDW distance metric.
Checkpoints retain the selected coordinate order, bounds and time scale.

The default SIREN amplitude mode remains `train_global_rms`, fitted on the
selected volume's observed samples. Explicit `training.amplitude_scaling:
per_trace_rms` uses each observed trace's RMS over the selected time samples.
This mode requires `prediction.scale_interpolation` with `neighbors`, `power`,
and four positive `distance_scales_m` values. Missing-trace physical RMS is a
convex inverse-distance-weighted average of nearby observed RMS values in
source X/Y and relative receiver X/Y coordinates divided by those distance
scales. Exact coordinate matches and equal distances use original array-row
order. Zero observed waveforms retain zero physical scale and zero normalized
targets; an entirely zero observed volume still fails the existing RMS guard.

Per-trace SIREN predictions are multiplied once by their measured or interpolated
scale before observed-data reinsertion and physical target scoring. Target RMS
and prediction self-normalization are not used. A per-trace fixed-step checkpoint
records the scale vector, original row IDs and interpolation contract; loading
exposes these fields for row-checked physical prediction. The scale vector is
also saved as `artifacts/trace_amplitude_scales.npy` in C-order spatial flattening.
The global RMS stored in normalization metadata is diagnostic in per-trace mode.
The default global-mode checkpoint payload and normalization behavior are unchanged.

The default `training.batch_mode` is `random_points`, where `training.batch_size`
counts sampled points per optimizer update. Opting into `random_complete_traces`
requires `training.traces_per_step`: a positive count selects distinct observed
traces within each update, while `null` uses every observed trace in fixed order.
Every selected trace contributes all selected time samples. In this mode,
`training.batch_size` limits points per forward/backward microbatch; losses are
weighted by each microbatch's fraction of the update's samples, including a
short final microbatch. Adam updates once after the complete trace batch.
`training.learning_rate_schedule` accepts `constant` or `cosine`; cosine requires
a positive `training.minimum_learning_rate` below the initial rate and spans
the declared `training.max_steps`. Each run saves its final fixed-step model.

`training.initial_time_weight_scale` defaults to 1.0. An explicit positive value
multiplies only column zero of the first SIREN layer's weight after the seeded
model is constructed. It consumes no random numbers and changes neither spatial
columns nor biases or subsequent layers. This is a training initialization;
checkpoint prediction restores the stored weights without applying it again.

Complete-trace training can additionally use `training.envelope_loss` with
exactly `weight`, `sigma_samples`, and `decay_steps`. Weight and sigma values
must be positive, finite numbers excluding booleans; the sigma sequence must
be nonempty, and decay steps must be an integer of at least two. Trace-length
and microbatch limits are checked before model initialization changes.
Each trace's local RMS envelope is
the square root of its Gaussian-smoothed squared waveform plus 1e-6. Gaussian
kernels sum to one, extend to `ceil(4 * sigma)` samples on each side, and use
reflection padding within that trace. The auxiliary loss is the mean envelope
MSE across the declared scales, added to the original waveform MSE with weight
`weight * max(0, 1 - (step - 1) / (decay_steps - 1))`. Decay requires at least two
steps and remains independent of a shortened preflight budget. Waveforms and
evaluation times are never shifted or resampled.

While the auxiliary weight is positive, each microbatch contains whole traces
within the existing point limit and contributes its fraction of the observed
batch to the gradient. The point limit must fit at least one trace, and each
kernel radius must be shorter than the trace. At zero weight, training uses
the existing point-MSE path. Opt-in histories separate waveform MSE, weighted
envelope MSE and the current weight from total training loss; an uncomputed raw
envelope error is not reported as zero. Loss components are means over each
reporting interval; the weight is the value at that interval's final step.
Without these options, existing random
sequences, updates and checkpoint payloads are preserved. SIREN architecture,
observed-only amplitude scales and the target-scoring boundary are unchanged.

[Study 031](../studies/study_031_c3_siren_10db/README.md) evaluates whether these
options can exceed 10 dB physical target SNR on the unchanged QC-derived case.
The common-method comparison excludes an externally estimated fixed temporal
shear supplied only to SIREN. Fitting a transform without target waveforms
prevents that source of leakage, but does not establish comparable preprocessing
across methods. The current SIREN condition uses Cartesian five-input coordinates,
omega 30, time factor 12, per-trace RMS with observed IDW scales, all 14,729 observed
traces per update, and 5,000 constant-rate updates with shear 0. It scores
-3.1486 dB. The shear-assisted 11.3422 dB result is diagnostic and does not meet
the common-condition goal; that goal remains unmet. The prior
10.6805 dB result cited in its [decision record](../studies/study_031_c3_siren_10db/decisions.md)
used oracle unit-RMS scoring and 80% observed traces, whereas this case has
approximately 20% observed traces and physical-amplitude scoring. Those results
are not directly comparable. Recorded validation scores may guide subsequent
declared variants; the test partition remains unused.

The primary metric is target-only `physical_amplitude_global_snr_db`: float64
reference and error energies are summed in physical units before taking dB.
Dense and query paths share the existing physical-amplitude metric contract.
Context-free queries remain evaluation targets. Zero reference and perfect
reconstruction retain their existing status/null-SNR strict-JSON representation.

To verify a frozen suite independently:

```bash
python -m seis_interp.cli data verify-c3-benchmark \
  --input data/processed/c3_na/study_027_c3_na_benchmark \
  --json
```

Paths resolve relative to the manifest directory, including relative references
to the interim dataset. Verification recalculates each distinct file's hash,
checks the full configured case list, rechecks geometry and crop signal QC,
compares physical trace-to-cell mappings semantically, and validates all
case/volume/mask bindings. Matching Parquet bytes and matching table semantics
are separate checks. The manifest is written only after validation succeeds.
Source configuration snapshots carry their original paths and exact hashes;
generation records capture the commit and dirty-worktree state at execution.
Later document edits do not rewrite the generation record or its snapshots.
The already frozen suite retains its original configuration snapshots; removing
the duplicate recipe declaration from the current study does not rewrite it.

C3 SIRENの5入力Cartesian条件では、`model.relative_receiver_y_time_shear_s_per_m`
（既定0）で `tau = time_s + coefficient * relative_receiver_y_m` を指定できる。
観測学習と全位置の予測に同じ可逆変換を用い、波形の再サンプリング・時刻範囲・RMS尺度は
変えない。非zero係数はcheckpointの座標metadataへ保存され、復元時に同じ変換を再構成する。
既定0では従来の座標演算・乱数・checkpoint payloadを維持する。
この固定shearは補助変換の効果を調べる診断用とし、SIRENだけへ与えた結果は共通条件の
手法比較および10 dB達成判定に採用しない。時間・空間の関係は主比較ではSIRENの
学習対象とする。他手法の前処理へ固定shearを導入することは、現在の比較契約に含めない。
