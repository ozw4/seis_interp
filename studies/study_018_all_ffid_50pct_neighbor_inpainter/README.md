# Study 018: 50% within-FFID neighbor inpainter

Status: `completed`

## Purpose

- Exceed 20 dB oracle unit-RMS validation S/N with 50% of traces in every eligible FFID assigned to training.

## Conditions

- dataset: all four SEG C3 NA sources, 4,780 eligible FFIDs; FFID 1746 excluded
- split: seed 42; 50% train, 12.5% validation, 37.5% test within each FFID
- prepared counts: 1,151,740 / 287,935 / 863,805; effective canonical counts: 1,151,731 / 287,933 / 863,801
- model: K274 same-source-line aperture, four target coordinates, width 384, full-trace temporal receptive field, FiLM, target-coordinate neighbor gate, identity-initialized 31-tap depthwise alignment FIR
- training: 50,000 updates with the replacement sampler; only train amplitudes enter neighbor inputs
- metric: `oracle_per_trace_unit_rms_global_snr_db` on raw validation predictions
- config / variants: [`config.yaml`](config.yaml), [`variants/`](variants/)

## Results

- Stage 09 diagnostic: 18.2484 dB at 10,000 updates
- width 384 + neighbor gate + FIR at 2,500 updates: 16.8037 dB; +0.4371 dB over width 256
- formal run: 20.4604 dB at 50,000 updates; first crossed 20 dB at 30,000 updates
- final checkpoint was best; exact revalidation and all scope checks passed
- threshold margin: +0.4604 dB; `metric_success=true`, `scope_success=true`, `success=true`
- runs: `runs/study_018_all_ffid_50pct_neighbor_inpainter/`
- accepted run: `20260829T075432Z_ee3d9e5_formal_50000_steps`
- formal model: 9,210,121 parameters; commit `ee3d9e5d5fce73e3ce0450b3471fe3284af616a1`; NVIDIA H100 NVL
- formal validation signal / error energy: 179,958,125.0042889 / 1,618,586.4733; train audit 20.7557 dB

## Decision

- Accept the 50,000-update formal result and frozen architecture.
- Detailed stage evidence is retained in the variant configs and [investigation record](../../reports/all_ffid_50pct_20db_investigation.md).
