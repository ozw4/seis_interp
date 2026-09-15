# Study 035: CCNet full-train patch sampling

Status: `completed_degraded` — Study 033 model not replaced

## Purpose

- Isolate sampling 512 fixed fit patches across the full QC training dataset instead of the limited Study 033 dense fit region.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- train pool 1,146,366 QC-authorized canonical train traces
- fixed normalization reuses the Study 033 checkpoint RMS 8.351643078934
- 58,999 targets x 384 samples
- checkpoint rule: declared final checkpoint; the selection metric is diagnostic when training and selection traces overlap

## Results

- target S/N **6.7996 dB**; RMSE **4.5430**; change from Study 033 **-4.8612 dB**
- saved-output rescoring: exact match
- patch coverage 976,504 / 1,146,366 train traces (85.1826%); every train source line
- accepted descriptors 512 of 941 candidates; 429 rejected for absent or unauthorized cells; no duplicate start
- fit/selection overlap 31,944 unique traces; final selection S/N 4.6045 dB, diagnostic only
- patch-presentation RMS 12.221864869477, or 1.4634 times the fixed scale
- epoch-eight mean loss 0.9701; Study 033 control 0.0671
- artifacts: [comparison CSV](../../results/study_035_c3_ccnet_full_train_sampling/comparison.csv), [summary JSON](../../results/study_035_c3_ccnet_full_train_sampling/summary.json)
