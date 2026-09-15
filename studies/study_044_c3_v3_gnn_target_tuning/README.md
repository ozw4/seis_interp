# v3 Proposed GNN target tuning

Status: `adopted_18db_achieved`. Goal: physical-amplitude mean trace SNR strictly above 18 dB
over all 104750 target traces of the fixed v3 input lock. O-only Global RMS,
MSE and the relational GNN are retained. No time shear or edge time shift is
allowed. The v3 window, mask and evaluation population do not change.

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

## Prepared candidates toward 20 dB

The next exploratory goal is full-target physical-amplitude mean trace SNR
strictly above 20 dB. The adopted result remains the reference. Both candidates
are prepared, not yet run, and inherit its complete training and evaluation
contract, including the existing 20000-update budget and query sampling.

- B: [neighbors4](mask10_fourier16_width128_ema999_neighbors4_20k.yaml)
  changes only the number of neighbors per relation.
- C: [dilation14](mask10_fourier16_width128_ema999_dilation14_20k.yaml)
  changes only the temporal dilation sequence.

Each model retains 363269 parameters. Neither candidate implements exact-64
supervision or extends training to match the CCNet/NeRSI update budget.
Evaluation uses the final EMA state, without best-checkpoint selection.
After completion, verify the input lock, artifact hashes, complete target
coverage and exact observed reinsertion; compare SNR, actual supervised trace
presentations, peak GPU memory and training/prediction time with the adopted run.
Graph equality with the reference is expected only for C; B intentionally
changes graph neighborhoods.

Run sequentially on physical GPU 1; C starts only if B exits successfully:

```bash
cd /workspace
bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_neighbors4_20k && \
  bash scripts/run_c3_v3_gnn_tuning.sh mask10_fourier16_width128_ema999_dilation14_20k
```

The launcher prints each new run directory. Progress is in its `stderr.log`;
metrics are in `relational_trace_graph/metrics.json`.

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
