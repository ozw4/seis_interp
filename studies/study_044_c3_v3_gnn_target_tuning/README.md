# v3 Proposed GNN target tuning

Status: `adopted_18db_achieved`, exploring toward 20 dB. Goal: physical-amplitude mean trace SNR at least 20 dB
over all 104750 target traces of the fixed v3 input lock. O-only Global RMS,
MSE and the relational GNN are retained. No time shear or edge time shift is
allowed. The v3 window, mask and evaluation population do not change.
By explicit user instruction, fair-comparison runs must retain the common
constant learning rate 0.001. Any changed learning rate or schedule is a
reference-only experiment and cannot establish achievement of the 20 dB goal.

The current GNN canonical result is **18.2795 dB**, adopted by explicit user
instruction. [Adopted result lock](gnn_v3_result.lock.json) fixes the run,
configuration, input lock, checkpoint, prediction, metrics and source-archive hashes.
The evaluated weights are `artifacts/ema.pt`; `final.pt` contains the raw final
weights and is not the source of the adopted score.

This is target-informed exploration, not independent test evidence. Existing
canonical runs are preserved. All completed and failed candidates are retained.

The Study 042 reference has mean trace SNR 11.2815 dB (global SNR 10.3771 dB).
`mask10_5k.yaml` reduces inner masking to 10% to narrow the training/inference
context-density gap. Prediction batches of 16 limit GPU memory without changing
the graph neighborhood contract. Architecture, learning rate and 5000 updates
remain those of the reference.

## Candidates toward 20 dB

The user authorized a 50000-update experiment using the highest measured
configuration, attention RMS with mean relation-gate pooling. Its
[`50k configuration`](mask10_fourier16_width128_ema999_neighbors6_geometry500_attention_rms_50k.yaml)
changes the update budget and execution device, starts from initialization and evaluates final
EMA. This explicitly authorized budget exception is separate from the original
20000-update comparison; it does not retroactively satisfy that budget's goal.
Launch on physical GPU 0 (the less occupied device):

```bash
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_attention_rms_50k
```

The completed experiments independently replace temporal mean pooling with
RMS pooling in (1) neighbor attention or (2) the learned relation gate. Both
inherit geometry500, retaining 363269 parameters, constant LR 0.001, 20000
updates and final EMA evaluation. The temporal kernels remain 7 / 5.
Attention RMS is the highest measured fair-comparison candidate at **19.2251 dB**;
20 dB remains unmet by 0.7749 dB. Canonical adoption remains unchanged.

| Candidate | Mean trace SNR [dB] | Difference from geometry500 [dB] | Training [min] | Peak CUDA allocated [GiB] |
|---|---:|---:|---:|---:|
| geometry500 | 19.0449 | 0.0000 | 144.5506 | 41.6814 |
| Attention RMS only | 19.2251 | +0.1802 | 135.4422 | 41.6913 |
| Relation gate RMS only | 18.7845 | -0.2604 | 171.2937 | 43.9766 |

Completed runs under `runs/study_044_c3_v3_gnn_target_tuning/`:

- Attention: `20260918T082525Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_attention_rms_20k`
- Gate: `20260918T105107Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_gate_rms_20k`

Both retain 1253344 supervised presentations and final EMA 0.999, covering all
104750 targets with exactly zero observed reinsertion error. Input locks,
artifact hashes, strict EMA loading (including pooling modes), query/coverage
arrays and per-step query/graph counts pass verification. Rescoring saved
predictions exactly reproduces all metrics. Peak CUDA reserved memory is
91.4727 GiB for attention RMS and 82.3594 GiB for gate RMS.
Reproduction commands, one at a time on physical GPU 1:

```bash
cd /workspace
# 1: neighbor attention only
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_attention_rms_20k
# 2: relation gate only, after the preceding run has exited
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_gate_rms_20k
```

The completed temporal-kernel candidates are
A: [`geometry500_stem3_temporal7`](mask10_fourier16_width128_ema999_neighbors6_geometry500_stem3_temporal7_20k.yaml)
and B: [`geometry500_stem11_temporal3`](mask10_fourier16_width128_ema999_neighbors6_geometry500_stem11_temporal3_20k.yaml),
based on geometry500 (19.0449 dB).
Each resolved configuration changes only the two temporal kernel widths and
retains 363269 parameters. Both evaluated the final EMA state at 20000 updates.
Neither temporal-kernel candidate improves geometry500.

| Candidate (stem / temporal kernel) | Mean trace SNR [dB] | Difference from geometry500 [dB] | Training [min] | Peak CUDA allocated [GiB] |
|---|---:|---:|---:|---:|
| geometry500 (7 / 5) | 19.0449 | 0.0000 | 144.5506 | 41.6814 |
| A (3 / 7) | 18.6468 | -0.3981 | 197.4045 | 41.6845 |
| B (11 / 3) | 18.8211 | -0.2238 | 134.2957 | 41.6833 |

Completed runs under `runs/study_044_c3_v3_gnn_target_tuning/`:

- A: `20260917T071318Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_stem3_temporal7_20k`
- B: `20260918T053928Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_stem11_temporal3_20k`

Both retain 1253344 supervised trace presentations and cover all 104750 targets,
with exactly zero observed reinsertion error. Input locks, artifact hashes,
strict EMA loading, query/coverage arrays and per-update query/graph counts were
verified against geometry500. Saved-prediction rescoring reproduces all metrics
exactly. Peak CUDA reserved memory was 91.4277 GiB (A) and 91.4707 GiB (B).
The adopted canonical result is unchanged. Reproduction commands (one at a time
on physical GPU 1):

```bash
cd /workspace
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_stem3_temporal7_20k
# B, after A has exited:
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_stem11_temporal3_20k
```

Both separate feature-scale comparisons completed successfully. Both inherit
neighbors6 with learned relation gates, constant LR 0.001, 20000 updates, 363269
parameters and final EMA 0.999. Run one at a time on physical GPU 1:

```bash
# Position 1000 m, offset 500 m:
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_offset500_20k
# Position 500 m, offset 1000 m:
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_position500_20k
```

| Position scale [m] | Offset scale [m] | Mean trace SNR [dB] | Difference from neighbors6 [dB] |
|---:|---:|---:|---:|
| 1000 | 1000 | 19.0190 | 0.0000 |
| 500 | 500 | 19.0449 | +0.0259 |
| 1000 | 500 | 19.0023 | -0.0167 |
| 500 | 1000 | 19.0417 | +0.0226 |

Offset-only run:
`20260917T010129Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_offset500_20k`.
Global SNR is 18.1068 dB and RMSE 1.1524; training took 10037.2620 s,
prediction 166.0723 s. Position-only run:
`20260917T035328Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_position500_20k`.
Global SNR is 18.1408 dB and RMSE 1.1479; training took 10108.5015 s,
prediction 165.0782 s. Each run peaked at 44757148672 allocated GPU bytes
(41.6833 GiB), with 20000 updates, 363269 parameters, 1253344 supervised
trace presentations and final EMA 0.999 updated 20000 times.
Both input locks, artifact hashes, strict EMA loads, query IDs, coverage and
per-update query/graph counts were verified against neighbors6. Each saved
prediction reproduces every saved metric exactly over all 104750 targets,
with zero observed reinsertion error. Verification reports are the run-named
JSON and `_audit.json` files under
`runs/gnn_training_speed_scratch/20260915_goal_verification/`.
Position-only is 0.0032 dB below geometry500. These small single-run differences
do not establish reproducible superiority. Geometry500 remains the best measured
fair-comparison candidate; the 20 dB goal is unmet and the adopted lock is unchanged.

The next exploratory goal is full-target physical-amplitude mean trace SNR
at least 20 dB. The adopted result remains the canonical reference.
B completed at 18.5387 dB; C completed at 17.9784 dB; neighbors6 completed at
19.0190 dB; neighbors8 completed at 18.8649 dB; neighbors6 dilation11 completed
at 18.7930 dB. The reference-only neighbors6 late-cosine run scored 18.9676 dB.
Neighbors6 geometry500 completed at 19.0449 dB, improving neighbors6 by 0.0259 dB.
Attention RMS improves geometry500 by 0.1802 dB to 19.2251 dB and is the
highest-scoring completed fair-comparison candidate. It has not replaced the
adopted result lock.

- B: [neighbors4](mask10_fourier16_width128_ema999_neighbors4_20k.yaml)
  changes only the number of neighbors per relation.
- C: [dilation14](mask10_fourier16_width128_ema999_dilation14_20k.yaml)
  changes only the temporal dilation sequence.
- [neighbors6](mask10_fourier16_width128_ema999_neighbors6_20k.yaml) completed.
  It inherits B and changes only neighbors per relation from 4 to 6.
  It starts from initialization and retains B's complete training and evaluation
  contract, including the existing 20000-update budget and query sampling.
- [neighbors8](mask10_fourier16_width128_ema999_neighbors8_20k.yaml) changes only
  neighbors per relation from 6 to 8. Its observed-only memory preflight is
  separate from the full run; no preflight weights are used in the full run.
- [neighbors6 dilation11](mask10_fourier16_width128_ema999_neighbors6_dilation11_20k.yaml)
  changes only the temporal dilation sequence from neighbors6. It starts from
  initialization with the same graph and training/evaluation contract.
- [neighbors6 geometry500](mask10_fourier16_width128_ema999_neighbors6_geometry500_20k.yaml)
  completed with position and offset feature scales reduced from 1000 to 500 m.
  It retains constant LR 0.001, 20000 updates and 363269 parameters. Graph search
  scales, amplitude normalization and teacher sampling remain unchanged. Node
  and edge feature values intentionally change; graph topology does not.
  An observed-only CPU check on the first three batches of the first training
  episode (192 queries) confirmed exact topology-array and waveform equality,
  doubled geometric length features, and unchanged other features. This is a
  sampled batch check, not an all-episode comparison. It performed no training
  or target evaluation. Evidence:
  `runs/gnn_training_speed_scratch/20260915_neighbors6_geometry500_preflight/geometry_result.json`.
- [neighbors6 late cosine](mask10_fourier16_width128_ema999_neighbors6_cosine15k_20k.yaml)
  is reference-only: it changes the learning-rate schedule from neighbors6, holding through
  update 15000 and decaying to 0.0001 on update 20000. It starts from initialization.
- [neighbors6 LR 0.002](mask10_fourier16_width128_ema999_neighbors6_lr002_20k.yaml)
  is reference-only: it changes the constant learning rate from 0.001 to 0.002. It starts from
  initialization and retains the full 20000-update and final EMA contract.

Each model retains 363269 parameters. None of these candidates implements exact-64
supervision or extends training to match the CCNet/NeRSI update budget.
Evaluation uses the final EMA state, without best-checkpoint selection.
After completion, verify the input lock, artifact hashes, complete target
coverage and exact observed reinsertion; compare SNR, actual supervised trace
presentations, peak GPU memory and training/prediction time with the adopted run.
Graph equality with the reference is expected only for C; the neighbor-count
candidates intentionally change graph neighborhoods. Neighbors6 training peak
allocation was 41.6833 GiB; B's was 23.6409 GiB.

Reproduction command for the completed geometry500 experiment. No subsequent
experiment is launched as part of the result review:

```bash
cd /workspace
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors6_geometry500_20k
```

The launcher prints each new run directory. Progress is in its `stderr.log`;
metrics are in `relational_trace_graph/metrics.json`.

B run: `20260915T042114Z_684c4e4c659a_mask10_fourier16_width128_ema999_neighbors4_20k`.
Completed C run: `20260915T073428Z_57b0fe24de30_mask10_fourier16_width128_ema999_dilation14_20k`.
The earlier C attempt at `20260915T065417Z` stopped after the last logged update
4200 without an evaluation result. All runs remain retained under this study.
Neighbors6 run: `20260915T083437Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_20k`.
Neighbors8 completed in
`20260915T112434Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors8_20k`.
Its separate 84-update observed-only preflight completed with a peak CUDA
allocation of 63807712256 bytes. No checkpoint or target evaluation was made
in that preflight; its records are in
`runs/gnn_training_speed_scratch/20260915_neighbors8_preflight/`.
The full neighbors8 run used 61.0413 GiB peak allocated GPU memory, trained
for 13098.4108 s and predicted in 267.1163 s. Its input lock, artifact hashes,
strict EMA loading, per-update query counts and full target coverage were verified.
Rescoring its saved prediction reproduced every saved metric exactly. Observed
reinsertion error is zero. Verification records are under
`runs/gnn_training_speed_scratch/20260915_goal_verification/`.
The neighbors6 dilation11 observed-only 84-update preflight passed, with peak
CUDA allocation 43942823424 bytes and 363269 parameters. No target evaluation
or checkpoint was produced. Its records are in
`runs/gnn_training_speed_scratch/20260915_neighbors6_dilation11_preflight/`.
The full neighbors6 dilation11 run completed in
`20260915T151123Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_dilation11_20k`.
Training took 10058.4811 s and prediction 258.5810 s; peak GPU allocation was
44755051520 bytes. Input lock, artifact hashes, strict EMA loading, query IDs,
coverage and per-update query/graph counts were verified against neighbors6.
Rescoring the saved prediction reproduced all metrics exactly, with zero observed
reinsertion error. This candidate does not replace neighbors6 or the adopted lock.
The neighbors6 late-cosine run completed in
`20260915T180709Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_cosine15k_20k`.
Mean trace SNR is 18.9676 dB, 0.0515 dB below constant-rate neighbors6;
global SNR is 18.0569 dB and RMSE is 1.1590. Training took 9990.4497 s,
prediction 154.2009 s and peak allocated GPU memory was 44755051520 bytes.
All 20000 optimizer updates and learning rates were verified, with 363269
parameters, 1253344 supervised trace presentations and 20000 EMA updates.
Input lock, query IDs, coverage and per-update query/graph counts match neighbors6.
Artifact hashes and strict EMA loading pass. Rescoring the saved prediction
reproduces all saved metrics exactly across all 104750 targets; observed
reinsertion error is zero. Verification records are in the directory above.
The 20 dB goal remains unmet; the adopted result lock is unchanged.
The reference-only neighbors6 LR 0.002 run completed successfully in
`20260915T210440Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_lr002_20k`.
Mean trace SNR is 19.3961 dB, global SNR 18.5140 dB and RMSE 1.0996.
It completed 20000 updates with 363269 parameters and 1253344 supervised trace
presentations. The final EMA decay is 0.999 with 20000 EMA updates. Training
took 8103.9273 s, prediction 135.3938 s and peak GPU allocation 44755051520 bytes.
Input lock, query IDs, coverage and per-update query/graph counts match neighbors6.
Checkpoint and prediction hashes and strict EMA loading pass. Rescoring its
saved prediction reproduces all metrics exactly across all 104750 targets;
observed reinsertion error is zero. Evidence is under the verification directory
above, including `neighbors6_lr002_audit.json`. This is reference evidence only.
The late-cosine run and the earlier width64 cosine run have the same reference-only
classification. They are excluded from fair-comparison rankings and the 20 dB
achievement decision. Constant-rate neighbors6 remains the fair-comparison
best before geometry500 was 19.0190 dB; the adopted canonical lock remains unchanged.

The geometry500 run completed successfully in
`20260915T232958Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_20k`.
Mean trace SNR is 19.0449 dB, global SNR 18.1412 dB and RMSE 1.1478.
Constant LR 0.001, 20000 updates, 363269 parameters and 1253344 supervised
trace presentations are retained. The final EMA has decay 0.999 and 20000 updates.
Input lock, query IDs, target coverage and per-update query/graph counts match
neighbors6. Checkpoint/prediction hashes and strict EMA loading pass. Rescoring
the saved prediction reproduces all metrics exactly over all 104750 targets,
with zero observed reinsertion error. Full graph arrays are not saved per update;
the history comparison checks counts, not whole-array graph equality.
Training took 8673.0332 s, prediction 126.4038 s and peak allocated GPU memory
was 44755051520 bytes (41.6814 GiB). Verification is recorded in
`runs/gnn_training_speed_scratch/20260915_goal_verification/neighbors6_geometry500_audit.json`
and the corresponding run-named rescore report. The 0.0259 dB gain is a single-run
result; repeatability has not been established. The 20 dB goal remains unmet.

## Completed candidates

| Candidate | Mean trace SNR [dB] | Global SNR [dB] | RMSE |
|---|---:|---:|---:|
| Study 042 reference | 11.2815 | 10.3771 | 2.8059 |
| mask10_5k | 13.4188 | 12.4653 | 2.2063 |
| mask10_spectral_q64_5k | 12.8047 | 11.8083 | 2.3797 |
| mask10_fourier16_fast_5k | 13.8887 | 12.7654 | 2.1314 |
| mask10_fourier16_direct8_5k | 12.7670 | 11.1316 | 2.5725 |
| mask10_fourier16_fast_20k | 16.7631 | 15.7665 | 1.5087 |
| mask10_fourier16_cosine_20k | 16.6233 | 15.5591 | 1.5452 |
| mask10_fourier16_width128_20k | 17.8599 | 17.0436 | 1.3024 |
| mask10_fourier16_width192_20k | 17.8046 | 17.0065 | 1.3080 |
| mask05_fourier16_width128_20k | 17.6942 | 16.8436 | 1.3327 |
| mask10_fourier16_width128_ema999_20k (adopted) | 18.2795 | 17.3982 | 1.2503 |
| mask10_fourier16_width128_ema999_neighbors4_20k | 18.5387 | 17.6256 | 1.2180 |
| mask10_fourier16_width128_ema999_dilation14_20k | 17.9784 | 17.0628 | 1.2995 |
| mask10_fourier16_width128_ema999_neighbors6_20k | 19.0190 | 18.1061 | 1.1525 |
| mask10_fourier16_width128_ema999_neighbors8_20k | 18.8649 | 17.9543 | 1.1728 |
| mask10_fourier16_width128_ema999_neighbors6_dilation11_20k | 18.7930 | 17.8628 | 1.1852 |

`mask10_5k` run: `20260914T022037Z_684c4e4c659a_mask10_5k`, under
`runs/study_044_c3_v3_gnn_target_tuning/`. Full v3 input lock, prediction/checkpoint
hashes and all 104750 target traces' coverage were verified. The O-only Global RMS
scale is 9.27911442626805. This candidate is below 18 dB. Training took 2933.96 s;
prediction took 1501.03 s (graph 174.00 s, input assembly 648.51 s, forward 674.80 s).
Peak allocated GPU memory was 11267991552 bytes. This motivates prediction batches
of 64 for the next spectral candidate, while retaining the full-target population.

The spectral candidate run is `20260914T033543Z_684c4e4c659a_mask10_spectral_q64_5k`.
It has the same verified v3 lock and complete target coverage, with exact observed
reinsertion. The strict checkpoint loader restores the spectral block and Global RMS.
At 5000 updates this candidate degrades mean trace SNR by 0.6140 dB. Training took
2612.21 s and prediction 469.78 s; peak allocated GPU memory was 64495426048 bytes.
The faster prediction also uses a larger batch, so this is not an isolated spectral
block speed comparison. Both completed candidates and their artifacts are retained.

The Fourier candidate run is `20260914T042804Z_684c4e4c659a_mask10_fourier16_fast_5k`.
Its v3 input lock, artifact hashes and full target coverage pass verification.
Training took 2103.2088 s and prediction 324.7522 s, with peak allocated GPU memory
4577711616 bytes. This candidate is below the highest verified mean trace SNR;
it is below 18 dB. Fourier features, CPU-side input validation,
cuDNN settings and prediction batch size differ from the mask10 reference,
so accuracy and speed changes are not isolated single-factor effects.

## Additional candidates

`mask10_fourier16_5k.yaml` adds NeRSI's Fourier mapping (16 components,
frequency base 1.25) to the four scaled Cartesian midpoint/offset-vector node
coordinates. The original nine node features are retained; neither visibility
nor amplitudes are Fourier encoded. No time coordinate or shear is introduced.
`mask10_bf16_5k.yaml` changes only training precision to BF16. These are separate
candidates so geometry encoding and precision effects can be measured separately.

`mask10_spectral_5k.yaml` adds a first rFFT → complex-spectrum graph aggregation
→ irFFT block. A rank-8 frequency basis and geometry/relation-conditioned complex
edge responses generate query spectra from O-only waveforms. The original temporal
GNN encodes these query waveforms and predicts their residual correction. Observed
rows stay unchanged, and the final pipeline still reinserts O exactly. Initialization
is a distance-weighted observed-spectrum average. This is a frequency-domain model
variant, not a time-shear preprocessing step; no explicit time-shift parameter exists.

`mask10_spectral_q64_5k.yaml` uses prediction batches of 64 instead of 16, with
identical learning and graph settings. Query-batch invariance is covered by
synthetic tests; prediction precision and full-target coverage remain unchanged.

`mask10_fourier16_fast_5k.yaml` uses the Fourier-coordinate candidate with prediction
batches of 64 and `training.cudnn_benchmark: false` to avoid per-shape convolution
algorithm searches on variable-size graphs. This execution setting is recorded in
resolved config, checkpoint metadata and runtime metadata; default runs retain their
existing setting. GPU algorithm changes need not produce bit-identical training.

`mask10_fourier16_direct8_5k.yaml` uses one message-passing round with eight
neighbors per relation. It retains the Fourier candidate's input, normalization,
optimizer and training budget.
Its completed run is `20260914T050907Z_684c4e4c659a_mask10_fourier16_direct8_5k`.
The full v3 lock, hashes and target coverage pass verification; observed reinsertion
is exact. Training took 879.2121 s and prediction 274.1699 s. Peak allocated GPU
memory was 2442909696 bytes. It is faster but less accurate than the two-round
Fourier candidate and does not replace it.

`mask10_fourier16_fast_20k.yaml` changes only the Fourier candidate's training
budget to 20000 updates. It starts from the same initialization seed, not a
selected intermediate checkpoint. Its completed run is
`20260914T052901Z_684c4e4c659a_mask10_fourier16_fast_20k`.

The 20000-update run completed successfully. Its run directory is
`runs/study_044_c3_v3_gnn_target_tuning/20260914T052901Z_684c4e4c659a_mask10_fourier16_fast_20k/`.
The verified mean trace SNR is 16.7631 dB and global SNR is 15.7665 dB, with
complete coverage of all 104750 targets. Training took 8363.9541 s and prediction
took 186.5183 s. Observed reinsertion error is exactly 0.0. This candidate is below the adopted EMA result and the 18 dB target.

## Cosine candidate

`mask10_fourier16_cosine_20k.yaml` changes only the learning-rate schedule from
the 20000-update Fourier candidate. Updates 1–10000 use 0.001; the remaining
updates decay by cosine to 0.00003 on update 20000. Training starts from the
fixed initialization, not the completed 20000-update checkpoint. Each candidate
has at most 20000 total optimizer updates, including any continuation.
The schedule and actual per-update learning rates are retained with the run.
Completed run: `20260914T103539Z_684c4e4c659a_mask10_fourier16_cosine_20k`,
under `runs/study_044_c3_v3_gnn_target_tuning/`.
Its v3 input lock, artifact hashes and complete 104750-target coverage pass
verification. The strict checkpoint loader restores its model and schedule.
Observed reinsertion error is exactly 0.0. Training took 3785.5042 s and
prediction took 212.0644 s. It does not replace the constant-rate best.

## Width128 result

`mask10_fourier16_width128_20k.yaml` changes only model width from 64 to 128
relative to `mask10_fourier16_fast_20k.yaml`. The constant learning rate, fixed
initialization seed, episodes, graph and 20000-update budget are retained.
Completed run: `20260914T150810Z_684c4e4c659a_mask10_fourier16_width128_20k`,
under `runs/study_044_c3_v3_gnn_target_tuning/`. Its mean trace SNR is 17.8599 dB.

## Width192 result

Completed run: `20260914T231049Z_684c4e4c659a_mask10_fourier16_width192_20k`,
under `runs/study_044_c3_v3_gnn_target_tuning/`. The final checkpoint scores
17.8046 dB mean trace SNR, below the width128 best by 0.0554 dB.
The input lock matches the width128 run; checkpoint and prediction hashes pass
verification. All 104750 targets are covered and observed reinsertion error is
exactly 0.0. Training took 3383.0183 s and prediction took 105.2704 s.
This candidate does not replace the raw width128 reference.

## Mask05 result

Completed run: `20260915T001620Z_684c4e4c659a_mask05_fourier16_width128_20k`,
under `runs/study_044_c3_v3_gnn_target_tuning/`. Its final checkpoint scores
17.6942 dB mean trace SNR, below the mask10 width128 best by 0.1657 dB.
Input lock, target coverage and query IDs match the reference. Checkpoint and
prediction hashes pass verification; all 104750 targets are covered and observed
reinsertion error is exactly 0.0. Training took 3639.1648 s and prediction took
99.3974 s. This candidate does not replace the raw mask10 width128 reference.

## Adopted EMA result

[`mask10_fourier16_width128_ema999_20k.yaml`](mask10_fourier16_width128_ema999_20k.yaml)
adds parameter averaging to the mask10 width128 reference. Training ends at the
fixed update budget without validation selection or early stopping. The final
EMA state in `artifacts/ema.pt` supplies the full-target prediction; `final.pt`
retains the raw final weights. Completed run:
`20260915T014733Z_684c4e4c659a_mask10_fourier16_width128_ema999_20k`, under
`runs/study_044_c3_v3_gnn_target_tuning/`.
Mean trace SNR is 18.2795 dB, improving the prior raw width128 result by 0.4195 dB.
All 104750 targets are covered, observed reinsertion error is exactly 0.0,
and the input lock and query population match the reference. Artifact hashes
and strict loading of both checkpoints were verified. Training took 3122.1342 s
and prediction took 121.5645 s. This is the current canonical GNN result.
The rationale and comparison limits are recorded in [decisions.md](decisions.md).

The launcher sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, clears
`CUDA_VISIBLE_DEVICES`, and uses the configured physical GPU 1. It creates a
unique run directory and prints its path; progress is written to `stderr.log`.

```bash
cd /workspace
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_20k
```

Run one candidate at a time on GPU 1, from the repository root:

```bash
bash scripts/run_c3_v3_gnn_tuning.sh mask10_5k
# Subsequent candidates, after the preceding process has exited:
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_5k
bash scripts/run_c3_v3_gnn_tuning.sh mask10_bf16_5k
bash scripts/run_c3_v3_gnn_tuning.sh mask10_spectral_5k
```

Each invocation uses a new output directory, retains stdout/stderr and snapshots
source and input conditions. It neither retries nor selects settings automatically.
