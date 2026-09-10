# C3 NeRSI repository reimplementation — fixed validation result

**The four predeclared Study 034 candidates completed on the fixed QC validation case.
Candidate C was selected by the predeclared final target-S/N rule at 15.9189 dB and RMSE
1.5899.** This is a one-case, one-seed result for the repository reimplementation; it is not
an official NeRSI run, a paper-value reproduction, or a test result.

## Candidate result

| Candidate | Encoder / latent | LR | Parameters | Target S/N (dB) | RMSE | Observed model RMSE before reinsertion | Fit + prediction (s) | Peak GPU allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 256 / 64 | `1e-3` | 3,459,793 | 15.6996 | 1.6306 | 1.2170 | 160.4926 | 2.338 GB |
| B | 256 / 64 | `3e-4` | 3,459,793 | 13.8712 | 2.0126 | 1.5980 | 166.5564 | 2.338 GB |
| **C** | **384 / 96** | **`1e-3`** | **7,728,697** | **15.9189** | **1.5899** | **1.0407** | **170.0764** | **4.307 GB** |
| D | 384 / 96 | `3e-4` | 7,728,697 | 14.6172 | 1.8470 | 1.3131 | 160.1146 | 4.307 GB |

Every row uses all 58,999 evaluation-target traces and all 384 time samples (22,655,616
samples). The reference energy is 2,237,830,414.4523 for every independently re-scored
prediction. Candidate C's error energy is 57,271,077.1562 and its margin over Candidate A is
0.2193 dB. The [full-precision table](../results/study_034_c3_nersi_baseline/20260910T032100000000Z_d8568ae_candidate_selection/candidate_comparison.csv)
and [selection decision](../results/study_034_c3_nersi_baseline/20260910T032100000000Z_d8568ae_candidate_selection/adoption_decision.json)
are the machine-readable records.

## Integrity and reproducibility

Each candidate's `prediction.npy` was read independently from its immutable native run.
The audit rechecked the suite/case/volume binding, checkpoint and prediction SHA-256,
float32 full-volume shape, finiteness, all-target coverage, native metric agreement, and
exact hard observed-data reinsertion. A fresh checkpoint instance, loaded through CPU and
run with the recorded CUDA numerical settings, reproduced A, B, C, and a repeated D audit
bitwise.

The first D restoration audit selected a different valid cuDNN path and differed from the
saved prediction by RMSE 0.0000313 and maximum 0.0064764, failing the deliberately strict
`rtol=atol=1e-6` restoration check. An unchanged recorded-settings repeat was bitwise equal.
Both the [failed first audit](../results/study_034_c3_nersi_baseline/20260910T031300000000Z_d8568ae_candidate_d_audit/result.json)
and [successful repeat](../results/study_034_c3_nersi_baseline/20260910T031400000000Z_d8568ae_candidate_d_audit_repeat/result.json)
are retained. No candidate was retrained and the tolerance was not relaxed.

Candidate C's [independent audit](../results/study_034_c3_nersi_baseline/20260910T031200000000Z_d8568ae_candidate_c_audit/result.json)
passed every check, including bitwise checkpoint restoration. The large checkpoints and
dense predictions remain in `runs/`; compact audit and selection records are checked in.

## Interpretation boundary

NeRSI fits the observed samples of the target validation volume itself. Its reported
`170.0764 s` is `volume_fit_seconds + prediction_seconds`; `frozen_inference_seconds` is not
applicable. In contrast, the adopted proposed GNN uses train-partition pretraining followed
by frozen validation inference. Candidate C's S/N is 3.9835 dB above that GNN's 11.9354 dB
on this case, but the methods use different training information and objectives, so this is
not evidence of architecture superiority or equal-compute performance.

The fixed shear, per-trace RMS restoration, and envelope loss used in prior SIREN work are
absent. Nuclear-norm regularization, NMO, off-grid support, multiple seeds, and the test
partition are also absent. Paper-public S/N values are context only and are not placed in
the repository result column.

The optional comparison output is intentionally
[`partial_results`](../results/study_034_c3_nersi_baseline/20260910T032100000000Z_d8568ae_candidate_selection/first_results.json):
it validates Candidate C and supplies fixed sections, fixed target traces, and its learning
curve, while leaving POCS, DRR, SIREN, CCNet, and GNN absent. Their older or separate-study
runs are not silently mixed into this exact explicit-path summary.
