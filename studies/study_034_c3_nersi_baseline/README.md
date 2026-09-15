# Study 034: C3 NeRSI baseline

Status: `validation_complete_candidate_c_selected`

## Purpose

- Compare four predeclared profile-wise NeRSI repository-reimplementation candidates on the fixed QC validation case.

## Conditions

Executable condition: [`config_candidate_a.yaml`](config_candidate_a.yaml), [`config_candidate_b.yaml`](config_candidate_b.yaml), [`config_candidate_c.yaml`](config_candidate_c.yaml), [`config_candidate_d.yaml`](config_candidate_d.yaml); implementation boundary in [`implementation_contract.md`](implementation_contract.md).

- 58,999 targets and 22,655,616 samples
- 2,304 `(384,32)` time/receiver-y profiles; 80,000 profile presentations
- method: exponential Fourier mapping, two-layer encoder, three convolution/PixelShuffle/activation upsampling blocks; no nuclear-norm term

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
