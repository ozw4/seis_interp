# study_067 registered S-a / S-b sensitivity

Output: `results/study_067_forge_c3_common_grid/20261103T233134Z_ba805bd26f6a_sensitivity`. Human-readable numbers are rounded to four decimals; CSVs retain precision.

C3 uses each seed's own test set (7321, 7322, 7323), then equal seed weights. Source-cell percentile bootstrap: RNG 670703, B=10,000; weights are loaded from the primary analysis and exactly equal regenerated weights. p is null-centred, two-sided, with the plus-one convention. No Holm correction: these sensitivities do not form a registered multiplicity family. Validation-free excludes the primary union of all S1 validation identities, including pilots; sensitivity S1 partitions are identical to the corresponding production partition.

| C3 contrast | Population | Metric | Delta | Percentile 95% CI | p | Delta 7321 / 7322 / 7323 | Paired t 95% CI (df=2) | Trace counts |
|---|---|---|---:|---|---:|---|---|---|
| ReGSI minus CCNet@0.1 | all_eligible | snr_db | 0.9713 | [0.8864, 1.0584] | 9.9990001e-05 | 1.0397 / 0.9589 / 0.9153 | [0.8146, 1.1280] | 104858 / 104858 / 104858 |
| ReGSI minus CCNet@0.1 | all_eligible | relative_mse | -0.0035 | [-0.0042, -0.0028] | 9.9990001e-05 | -0.0037 / -0.0035 / -0.0032 | [-0.0041, -0.0028] | 104858 / 104858 / 104858 |
| ReGSI minus CCNet@0.1 | validation_free | snr_db | 0.9707 | [0.8860, 1.0578] | 9.9990001e-05 | 1.0391 / 0.9584 / 0.9147 | [0.8140, 1.1275] | 102763 / 102755 / 102762 |
| ReGSI minus CCNet@0.1 | validation_free | relative_mse | -0.0035 | [-0.0042, -0.0028] | 9.9990001e-05 | -0.0037 / -0.0035 / -0.0032 | [-0.0041, -0.0028] | 102763 / 102755 / 102762 |
| CCNet@0.1 minus CCNet@0.25 (descriptive) | all_eligible | snr_db | 0.1591 | [0.1521, 0.1664] | 9.9990001e-05 | 0.1157 / 0.1835 / 0.1782 | [0.0654, 0.2529] | 104858 / 104858 / 104858 |
| CCNet@0.1 minus CCNet@0.25 (descriptive) | all_eligible | relative_mse | -0.0004 | [-0.0005, -0.0004] | 9.9990001e-05 | -0.0002 / -0.0006 / -0.0005 | [-0.0009, approximately 0] | 104858 / 104858 / 104858 |
| CCNet@0.1 minus CCNet@0.25 (descriptive) | validation_free | snr_db | 0.1591 | [0.1521, 0.1664] | 9.9990001e-05 | 0.1159 / 0.1831 / 0.1781 | [0.0661, 0.2521] | 102763 / 102755 / 102762 |
| CCNet@0.1 minus CCNet@0.25 (descriptive) | validation_free | relative_mse | -0.0004 | [-0.0005, -0.0004] | 9.9990001e-05 | -0.0002 / -0.0006 / -0.0005 | [-0.0009, approximately 0] | 102763 / 102755 / 102762 |

CCNet@0.1 minus CCNet@0.25 test contrasts are descriptive. Positive SNR and negative relative MSE favour the left method.

| Registered single-cell pair | Dataset | Left SNR | Right SNR | Left minus right (dB) | >=0.2 dB | Descriptive paired source-cell 95% CI |
|---|---|---:|---:|---:|---|---|
| S-a mask 0.1 minus 0.25 | c3 | 19.0733 | 18.8033 | 0.2700 | True | not requested |
| S-a mask 0.1 minus 0.25 | forge | 1.4781 | 1.5923 | -0.1143 | False | [-0.1427, -0.0852] |
| S-b FP32 minus bf16 | forge | 1.5868 | 1.5923 | -0.0055 | False | [-0.0310, 0.0203] |

S-a: >=0.2 dB on either dataset: True. The historical C3 comparison uses study_064 a05_b02 (19.073273231598527 dB); it is the registered historical pair, not an identical modern RNG implementation. The new C3 seeds change only the mask under the current production implementation.
S-b: FP32 minus bf16 >=0.2 dB: False. The registered FP32 three-seed remedy is not triggered.
S-b timing deviation (already recorded in decisions.md): FORGE CCNet seeds had already been trained and predicted at preparation on 2026-11-01, while test scoring was waiting for the all-method S1 barrier. The requirement to report before starting those seeds was not met.

Self-check: both recomputed primary C3 ReGSI minus CCNet@0.25 rows exactly equal contrasts.csv, including all raw statistics and the original primary Holm fields recalculated using the seven other registered contrasts. See primary_self_check.csv. All original analysis code SHA256 values match the primary manifest.
