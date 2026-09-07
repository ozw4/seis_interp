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
| Formal fit | `[0,384)` | `[0,25)` | `[27,59)` | `[0,8)` | `[18,50)` |
| Formal internal selection | `[0,384)` | `[0,25)` | `[59,75)` | `[0,8)` | `[18,50)` |
| Smoke fit | `[128,144)` | `[0,2)` | `[27,35)` | `[0,2)` | `[18,22)` |
| Smoke internal selection | `[128,144)` | `[0,2)` | `[35,43)` | `[0,2)` | `[18,22)` |
| Formal benchmark | `[0,384)` | `[25,35)` | `[27,59)` | `[0,8)` | `[18,50)` |
| Smoke benchmark | `[64,128)` | `[25,29)` | `[27,35)` | `[0,8)` | `[18,34)` |

Fit and internal selection are spatially disjoint, not merely separate time windows. Benchmark
validation/test amplitudes are excluded from labels, fit RMS, and internal checkpoint selection.
The source checks dense geometry and train membership at runtime. The CPU smoke uses a verified
nonzero teacher time interval; formal/calibration spatial geometry still needs revision before
training. See Current result below. A hole or
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
retains 64/64 channels and the original patch, restricts teacher time to `[64,128)`, and performs
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

Both requested real-data preflights were attempted from clean commit
`f654f9f97a8a720229ac6d81b7805b5ad21cf615` with eight OMP/MKL/PyTorch CPU threads.
The isolated checkout imported that commit's `src`; the supplied ZIP files and existing runs
were preserved. Neither attempt reached model initialization, an optimizer update, or output
directory creation. The CPU teacher time is now `[128,144)` in both regions; a new clean run
is pending. Formal/calibration geometry remains unresolved.

| Attempted run ID | Requested condition | Preflight result |
|---|---|---|
| `20260907T073648Z_f654f9f_train_smoke` | CPU, H/R=4, four modules, kernel 5, four fit/two selection patches, four updates | Rejected: fit amplitude RMS is zero |
| `20260907T073756Z_f654f9f_gpu_calibration` | `cuda:0`, H/R=64, original patch `(16,16,16,8,16)`, one update plus selection | Rejected: selected shots in source line 16 contain a 160 m gap instead of the required contiguous 80 m grid |

These are attempted IDs, not generated run directories: no `run.json`, checkpoint, prediction,
loss, benchmark S/N, or CUDA peak-memory measurement was produced. CUDA availability was checked
on an NVIDIA H100 NVL, but no CCNet GPU forward/backward occurred. CPU checkpoint-to-benchmark
smoke and full-width resource calibration remain incomplete; no formal training or inference
has been run.

A bounded read-only amplitude diagnostic on the same smoke spatial rows confirmed that both
fit and internal selection contain 128 traces and 2,048 finite, exactly zero samples in
`time=[64,80)`. A separate single geometry candidate narrowed calibration source lines to
`[0,16)` while preserving other ranges and the original patch size; it also failed because the
fit crop lacks local spatial cell `(3,26,0,31)` (global source line 3, shot-in-line index 53,
receiver indices 0 and 49). This candidate was not adopted or trained.

For the CPU smoke only, a read-only check within the already declared formal teacher time
extent established `time=[128,144)` with the same spatial rows. All values are
finite, all 128 traces in each region contain a nonzero value, and fit/selection RMS are
0.0658/0.0277 (rounded). This interval is adopted in both smoke teacher regions, not yet a
trained or scored result; patch seeds, counts, shapes, and spatial ranges are unchanged.

The next operational steps are a new immutable CPU smoke and geometry-based selection of a
dense formal/calibration teacher pair. No holes were filled, no zero-label
patches were filtered, and no width, precision, or patch-size fallback was applied.
