# study_025_c3_na_ccnet5d

## Status

`draft`

## Research question and information use

How well does a supervised, pretrained CCNet5D reconstruct missing traces on the same C3
validation benchmark used by [POCS](../study_022_c3_na_pocs/README.md),
[DRR](../study_023_c3_na_drr/README.md), and
[per-volume SIREN](../study_024_c3_na_siren_volume/README.md)?

This method uses additional complete labels from the train partition. Its
`training_regime: supervised_train_partition` must accompany comparisons: the benchmark case,
volume, and target metric are shared, but the available training information is not identical.
Training creates a reusable checkpoint; benchmark inference freezes that checkpoint and performs
no retraining or field-data self-supervised fine-tuning.

## Data and isolated regions

`inputs.yaml` reuses the same four SEG-Y checksums, all-FFID interim data, prepared split,
validation mask, case, and formal/smoke volume artifacts as Studies 022–024. It does not create
another mask or case. The current C3 dataset has 625 time samples and 50 source lines; the
paper specification describes 51 sail lines. We use the repository's existing geometry and do
not invent the additional line. Array axes are `(time, source_line, shot_in_line,
relative_receiver_x, relative_receiver_y)`.

Canonical duplicate physical cells retain the lowest `array_row`. Prepared source-line ranges
are train `[0,25)`, validation `[25,35)`, and test `[35,50)`. Complete labels are read only from
the two train-partition regions below; their full executable definitions are in the training
configs. Ranges are global, zero-based, half-open.

| Region | Time | Source line | Shot in line | Receiver x | Receiver y |
|---|---|---|---|---|---|
| Formal fit | `[0,384)` | `[0,16)` | `[27,59)` | `[0,8)` | `[18,34)` |
| Formal internal selection | `[0,384)` | `[0,16)` | `[59,75)` | `[0,8)` | `[18,34)` |
| Calibration fit | `[128,192)` | `[0,16)` | `[27,59)` | `[0,8)` | `[18,34)` |
| Calibration internal selection | `[128,192)` | `[0,16)` | `[59,75)` | `[0,8)` | `[18,34)` |
| Smoke fit | `[128,144)` | `[0,2)` | `[27,35)` | `[0,2)` | `[18,22)` |
| Smoke internal selection | `[128,144)` | `[0,2)` | `[35,43)` | `[0,2)` | `[18,22)` |
| Formal benchmark | `[0,384)` | `[25,35)` | `[27,59)` | `[0,8)` | `[18,50)` |
| Smoke benchmark | `[64,128)` | `[25,29)` | `[27,35)` | `[0,8)` | `[18,34)` |

Fit and internal selection are spatially disjoint, not merely separate time windows. Benchmark
validation/test amplitudes are excluded from labels, fit RMS, and internal checkpoint selection.
The source checks dense geometry and train membership at runtime. Formal/calibration teachers
use the first 16 train source lines and a dense 16-receiver-y subset; the original patch shape
is unchanged. The CPU smoke uses a verified nonzero teacher time interval. A hole or
unavailable range must fail, not be silently filled or replaced. Hashing whole source files is
provenance verification, not model access to nontrain amplitudes.

The benchmark mask is seed-42 `random_trace` with requested missing fraction 0.8, bound to case
`c3_na_validation_random_trace_80_seed42`. The formal and smoke volumes have shapes
`(384,10,32,8,32)` and `(64,4,8,8,16)`, respectively. Requested and realized missing fractions
are recorded separately. `project.random_seed` identifies prepared/benchmark data;
`patches.random_seed` fixes descriptors and masks; `training.random_seed` fixes model
initialization and epoch order. All start at 42 but have distinct roles.

## Paper correspondence and implementation choices

The primary reference is [Fang et al. (2023), CCNet-5D](https://doi.org/10.1190/GEO2022-0420.1).
The supplied method specification cites Figure 4, Figure 5, equations (12)–(15), and Appendix B.
The PDF has not been directly inspected for this implementation; the table follows that supplied
specification.
It distinguishes documented structure from choices not fixed by the available paper description:

| Topic | Paper description in the supplied specification | This baseline |
|---|---|---|
| Cross-convolution | Conv3D over the first three axes, then Conv2D over the remaining two | Same cascade with explicit axis permutations; intermediate channels reduced by Conv2D |
| Network | Four modules: `(Cin,R,Cout)` = `(1,64,64)`, `(64,64,64)` twice, `(64,64,1)` | Same full-width candidate; kernels `5×5×5` and `5×5`, bias in both convolutions |
| Activation | ReLU after both convolutions | ReLU everywhere except final Conv2D; optional final ReLU audit variant |
| Supervision | Complete patch squared error, Adam, batch 1, 20 epochs | Complete-patch mean squared error; no observed reinsertion in loss |
| Patches and learning rate | 10,000/1,000 patches of `16×16×16×8×16`, 80% random trace removal; `1e-4` then `1e-5` after epoch 15 | Same starting budget, with fixed descriptors/masks and explicit disjoint crop ranges |
| Model selection | Every 2,000 batches, maximize mean patch S/N | Every 2,000 updates and final update, maximize missing-sample global S/N over internal-selection patch instances |
| Padding, RMS, initialization, tiling | Not fully specified | Same zero padding, fit-label global RMS, PyTorch default Conv initialization, halo/core inference |

The main method is `ccnet5d`, variant `supervised_train_partition_linear_output`. Removing the
last ReLU is an explicit signed-amplitude modification, not a complete reproduction claim.
`output_activation: relu` selects `supervised_train_partition_paper_relu_output` for an audit.
Inverse RMS scaling alone cannot recover negative amplitudes from a final ReLU.

The full-width network has 1,853,249 parameters. Each axis' receptive field is 17 and halo
radius is 8, derived from four stride-one kernel-5 modules. There are no mask channels, skips,
normalization layers, pooling, or extra output heads. See [decisions.md](decisions.md) for the
reasoning behind the output, padding/RMS, selection, and halo choices.

## Training, inference, and evaluation

`config_train.yaml` is a formal starting candidate, not authorization to run 20 epochs.
Fit RMS uses all complete fit-region samples with float64 energy accumulation. Both training
and internal selection divide by this fixed scalar. Mean removal, shift, clipping, and
benchmark-target-fitted normalization are absent. Descriptors and whole-trace artificial masks
remain fixed across epochs; only their visitation order is shuffled. `best.pt` retains the earliest smallest
internal relative squared error; `final.pt` retains the last update. There is no early stopping.
Before model initialization or output creation, all fixed selection descriptors must have
positive finite aggregate reference energy on their artificial missing samples. Individual zero
patches are retained; there is no mask redraw, initial-model evaluation, or step-0 selection.

`config_train_smoke.yaml` retains four modules, kernel 5, and linear output but uses H/R=4,
patch `(8,2,4,2,4)`, four fit descriptors, two selection descriptors, and one CPU epoch.
Its score is not evidence of full-width model performance. `config_train_calibration.yaml`
retains 64/64 channels and the original patch, restricts teacher time to `[128,192)`, and performs
one GPU update plus one selection pass. It is resource calibration, not a performance run.

Inference uses only the observed volume and checkpoint RMS. Nonoverlapping output cores acquire
an 8-sample halo clipped to the real volume; each Conv handles its own outside padding. Only core
outputs are written, without Hann blending or empty-observation skipping. Predictions return to
physical float32 amplitudes, pre-reinsertion observed errors are recorded, then observed traces
are reinserted exactly. The checkpoint's source hashes must match the same dataset/prepared
partition, and its teacher regions must be spatially separate from the benchmark.

Benchmark evaluation first reads target truth after prediction. The primary score is physical
global S/N on unique `evaluation_target` samples, with RMSE, relative L2, zero-fill, and observed
error diagnostics. It is distinct from the paper's mean patch S/N and from this model's internal
selection patch-instance score. Published 14.87 dB or 8.65 dB scores are not acceptance targets;
neither a minimum smoke S/N nor cross-device bitwise identity is required.

## Reproduction

Use a clean checkout and make its `src` the imported package. Reuse valid existing shared inputs;
if absent, follow Study 022 preparation rather than silently downloading or regenerating data.
From the repository root:

```bash
TRAIN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_train_smoke"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$PWD/src" \
python -m seis_interp.cli train ccnet5d \
  --config studies/study_025_c3_na_ccnet5d/config_train_smoke.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --output "runs/study_025_c3_na_ccnet5d/$TRAIN_ID" --device cpu --json

PRED_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_infer_smoke"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$PWD/src" \
python -m seis_interp.cli interpolate ccnet5d \
  --config studies/study_025_c3_na_ccnet5d/config_smoke.yaml \
  --checkpoint "runs/study_025_c3_na_ccnet5d/$TRAIN_ID/artifacts/best.pt" \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --mask data/processed/c3_na/c3_source_line_blocks_seed42/masks/validation-random-trace-80-seed42 \
  --case data/processed/c3_na/c3_source_line_blocks_seed42/cases/c3_na_validation_random_trace_80_seed42 \
  --volume data/processed/c3_na/c3_source_line_blocks_seed42/volumes/c3_na_validation_smoke_t64_128_sl25_29_sh27_35_rx0_8_ry18_34 \
  --output "runs/study_025_c3_na_ccnet5d/$PRED_ID" --device cpu --json
```

For the one-update full-width calibration, use a separate immutable run:

```bash
CALIBRATION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_train_calibration"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$PWD/src" \
python -m seis_interp.cli train ccnet5d \
  --config studies/study_025_c3_na_ccnet5d/config_train_calibration.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/c3_source_line_blocks_seed42 \
  --output "runs/study_025_c3_na_ccnet5d/$CALIBRATION_ID" --device cuda:0 --json
```

Do not automatically launch formal training or formal inference. The initial inference core shape
requires memory calibration before formal use. Do not silently reduce full-width calibration
channels or patch shape after an out-of-memory failure.

## Outputs and acceptance

Training writes resolved config, input lock, metrics, run metadata, and
`artifacts/{patch_plan.json,best.pt,final.pt}`. Frozen inference writes the four records and
`artifacts/prediction.npy`. Checkpoints preserve constructor, CPU weights, fit RMS, teacher scope,
input hashes, plan hash, seeds, training-start Git commit/worktree state, role, epoch, step, and
selection metric. Inference records bind
the checkpoint SHA-256 and training provenance in addition to the shared benchmark lock.

Report pretraining/selection cost separately from frozen inference cost. Runtime records include
actual device, float32 parameter/input/target dtype, GPU identity and precision flags, stage
timings, maximum allocated/reserved memory, whole-process RSS, and Git worktree state. A single
full-width activation is 128 MiB; this is not a prediction of total training peak memory.

Acceptance covers independent cross-convolution values and gradients, signed output, supervised
region isolation, fixed plans/masks, RMS boundaries, LR/cadence, checkpoint reload, halo/full
forward equivalence, target-truth independence, strict records, and unchanged existing baselines.
Formal promotion requires both the inference run's `git_worktree_dirty` and its checkpoint's
`training_provenance.training_run.git_worktree_dirty` to be `false`. Dirty-checkpoint development
inference is allowed with a recorded warning; a clean inference checkout does not make its
training history clean. Generated data, runs, checkpoints, and the paper PDF are not committed.

## Current result

The current teacher crops pass real-C3 source verification and fixed-selection target preflight.
Formal and calibration fit/selection each contain 65,536/32,768 spatial traces with no shared
rows; all selected values are finite. Formal fit/selection RMS are 7.8755/8.2312, calibration
17.2473/16.9012, and CPU smoke 0.0658/0.0277 (rounded). The formal 1,000-descriptor selection
plan has positive finite aggregate reference energy. These are input-feasibility checks, not
formal training or model-performance results. See [decisions.md](decisions.md) for the reasons
behind the ranges.

### Clean CPU training and frozen benchmark smoke

Both runs completed successfully from clean commit
`d7836692712fd6bf13b0d1d0178f15c77ac1c301`, importing its isolated checkout with eight
OMP/MKL/PyTorch threads. Training and inference Git states are independently recorded; the
checkpoint also preserves the clean training state. Both use variant
`supervised_train_partition_linear_output` and regime `supervised_train_partition`.

Run IDs and local generated paths:

- Training: `20260907T140759Z_d783669_train_smoke` —
  `runs/study_025_c3_na_ccnet5d/20260907T140759Z_d783669_train_smoke/`
- Frozen inference: `20260907T140839Z_d783669_infer_smoke` —
  `runs/study_025_c3_na_ccnet5d/20260907T140839Z_d783669_infer_smoke/`

The four-module H/R=4 model has 7,829 parameters. Four fit descriptors and two fixed selection
descriptors produced four optimizer updates, selection at steps 2 and 4, and best/final
checkpoints at step 4. Both checkpoints reload on CPU with finite weights. Fit RMS is 0.0658;
best internal selection S/N is -0.4997 dB. The source was independently reverified as train-only
with disjoint fit/selection rows.

Frozen inference uses the unchanged shared seed-42 validation case and smoke volume. Its input
lock, excluding the added checkpoint binding, matches the POCS and SIREN smoke inputs exactly.
Prediction shape is `(64,4,8,8,16)`, float32, entirely finite. All 821 observed traces are
reinserted exactly, all 3,275 target traces are covered, and eight halo-8 tiles cover every
sample once. A fresh checkpoint reload and identical-core CPU prediction reproduced all
262,144 saved samples byte-for-byte in the same environment.

| Benchmark diagnostic | Result |
|---|---:|
| Target-only physical global S/N | approximately 0 dB (rounded) |
| Target RMSE / relative L2 | 4.0073 / 1.0000 |
| Zero-fill S/N / RMSE | 0.0000 dB / 4.0074 |
| Observed model RMSE / maximum absolute error before reinsertion | 4.1126 / 75.0110 |
| Observed maximum absolute error after reinsertion | 0 |
| Uncovered traces / samples | 0 / 0 |
| Training plus selection / frozen prediction time | 0.8119 s / 0.1193 s |
| Training / inference whole-process peak RSS | 3,513,176 / 589,504 KiB |

Full-precision metrics and timings remain in the immutable run records. Source loading and
verification are separate from the stage times above. This tiny four-update smoke establishes
the execution contract, not full-width performance or a meaningful zero-fill improvement.

An additional identical CPU training run is retained at
`runs/study_025_c3_na_ccnet5d/20260907T085141Z_d783669_train_smoke/`. Its directory timestamp
was mistyped; the authoritative `run.json` start is `2026-09-07T14:07:09Z`. It was not renamed
or edited. Its patch-plan bytes, training/selection histories, and best/final weights exactly
match the correctly time-named training run above.

### Clean full-width GPU calibration

Run ID: `20260907T141201Z_df66605_gpu_calibration`

Local generated path:
`runs/study_025_c3_na_ccnet5d/20260907T141201Z_df66605_gpu_calibration/`

This run succeeded from clean commit `df666052bf76fa8324c175762a8c44815973bab3` on
`cuda:0`, NVIDIA H100 NVL (compute capability 9.0), with eight CPU threads. The four-module,
64/64-channel, kernel-5 linear-output model retains all 1,853,249 parameters and the original
`(16,16,16,8,16)` patch, batch 1, float32 parameters/inputs/targets, and seed 42. One optimizer
update and one selection pass completed; both best/final checkpoints reload on CPU with
finite weights and clean training provenance. Fit RMS is 17.2473, training loss 1.2045, and
internal selection S/N -0.0002 dB. This checkpoint was not benchmarked.

Training plus selection took 15.0282 s. CUDA peak allocated/reserved memory was
3,597.6099/3,878.0000 MiB; whole-process peak RSS was 3,640,456 KiB. Peaks include forward,
backward, permutation copies, convolution workspaces, and selection. Timing is synchronized
but includes first-update initialization/algorithm warmup, so multiplying it by 200,000 is not
a reliable formal-training time estimate.

The recorded runtime is PyTorch `2.5.0a0+b465a5843b.nv24.09`, CUDA build `12.6`, cuDNN `90400`,
float32 matmul precision `high`, CUDA matmul/cuDNN TF32 enabled, cuDNN benchmark enabled,
and cuDNN deterministic disabled. These are measured execution conditions, not a cross-device
bitwise reproducibility guarantee. Formal inference core memory still requires its own check.

No formal training or inference has been run. No holes were filled, zero-label patches filtered,
seeds redrawn, or prescribed model/patch dimensions reduced. Full-width performance validation
remains a separate, explicitly authorized experiment.
