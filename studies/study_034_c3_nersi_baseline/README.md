# Study 034: C3 NeRSI baseline

Status: `validation_complete_candidate_c_selected`

## Purpose

- Compare four predeclared profile-wise NeRSI repository-reimplementation candidates on the fixed QC validation case.

## Conditions

- suite SHA-256: `f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`
- case: `c3_benchmark_validation_random_trace_80_seed142`; shape `[384,9,32,8,32]`
- evaluation: 58,999 targets and 22,655,616 samples; test unused
- method: 2,304 `(384,32)` time/receiver-y profiles; exponential Fourier mapping `K=40`; two-layer encoder; three convolution/PixelShuffle/activation upsampling blocks; no nuclear-norm term
- normalization: one global RMS from observed samples in the selected volume
- training: 5,000 Adam updates, batch 16 profiles, 80,000 profile presentations, seed 20260908; final checkpoint only
- implementation boundary: [`implementation_contract.md`](implementation_contract.md)
- configs: [`config_candidate_a.yaml`](config_candidate_a.yaml), [`config_candidate_b.yaml`](config_candidate_b.yaml), [`config_candidate_c.yaml`](config_candidate_c.yaml), [`config_candidate_d.yaml`](config_candidate_d.yaml)

## Results

| Candidate | Encoder / latent | Learning rate | Target S/N | RMSE | Fit + prediction |
|---|---:|---:|---:|---:|---:|
| A | 256 / 64 | `1e-3` | 15.6996 dB | 1.6306 | 160.4926 s |
| B | 256 / 64 | `3e-4` | 13.8712 dB | 2.0126 | 166.5564 s |
| **C** | **384 / 96** | **`1e-3`** | **15.9189 dB** | **1.5899** | **170.0764 s** |
| D | 384 / 96 | `3e-4` | 14.6172 dB | 1.8470 | 160.1146 s |

- all saved predictions independently re-scored, preserved observed samples exactly, and were bound to their final checkpoints
- Candidate C exceeded A by 0.2193 dB
- selection artifact: [summary JSON](../../results/study_034_c3_nersi_baseline/20260910T032100000000Z_d8568ae_candidate_selection/summary.json)

## Decision

- Select Candidate C as the fixed-final validation primary.
- The optional first-results collector is `partial_results` only because older-suite POCS/DRR runs are not substituted into the QC suite; the four-candidate matrix itself is complete.
- No test result or paper-reproduction claim is recorded.
