# Study 035: CCNet full-train patch sampling

Status: `completed_degraded`

## Purpose

- Isolate sampling 512 fixed fit patches across the full QC training dataset instead of the limited Study 033 dense fit region.

## Conditions

- train pool: 1,146,366 QC-authorized canonical train traces, time `[0,384)`
- fixed normalization: Study 033 checkpoint RMS 8.351643078934; checkpoint SHA-256 `b8fb3a07e7a9d4a5b3c5540667abbfff8d87f4e24a6358f8cc9cd5e6cfa48697`
- patches: 512 unique `(64,8,16,8,16)` descriptors, seed 20260908, fixed 80% whole-trace masks
- model / optimizer / budget: Study 033 width-32, kernel-3 CCNet; Adam; complete-patch MSE; batch 2; eight epochs / 2,048 updates; learning rate 0.001 then 0.0001
- checkpoint rule: declared final checkpoint; selection metric is diagnostic when training and selection traces overlap
- evaluation: fixed validation case, all 58,999 targets x 384 samples
- config: [`config.yaml`](config.yaml)

## Results

- target S/N: **6.7996 dB**; RMSE: **4.5430**
- change from Study 033: **-4.8612 dB**
- saved-output rescoring: exact match
- patch coverage: 976,504 / 1,146,366 train traces (85.1826%); every train source line
- accepted descriptors: 512 of 941 candidates; 429 rejected for absent or unauthorized cells; no duplicate start
- fit/selection overlap: 31,944 unique traces; final selection S/N 4.6045 dB, diagnostic only
- patch-presentation RMS: 12.221864869477, or 1.4634 times the fixed scale
- epoch-eight mean loss: 0.9701; Study 033 control: 0.0671
- artifacts: [comparison CSV](../../results/study_035_c3_ccnet_full_train_sampling/comparison.csv), [summary JSON](../../results/study_035_c3_ccnet_full_train_sampling/summary.json)

## Decision

- Full-train fixed-patch sampling degraded validation performance under this one-seed condition.
- Do not replace the Study 033 model; no follow-on experiment was started.
