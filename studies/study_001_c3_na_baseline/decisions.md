# Decisions

This file records rationale that may matter when changing study conditions.
The current study contract is defined by `README.md`, `config.yaml`, and
`inputs.yaml`. Current code and tests define implementation behavior.

## 2026-08-24 — POC scope and formal input

**Status:** active

**Decision:**
Use SEG C3 Narrow-Azimuth FFID 2348 and prioritize proof of interpolation over exact paper reproduction.

**Reason:**
A complete mid-survey shot provides a controlled ground-truth POC input without sail-line-end incompleteness.

**Evidence:**
- `SEG_C3NA_ffid_1201-2400.sgy`, FFID 2348: 544 traces of 625 samples at 8 ms.

## 2026-08-24 — Trace-table evaluation contract

**Status:** active

**Decision:**
Use a trace-table representation, whole-trace splits, and normalization fitted from training traces only.

**Reason:**
This preserves irregular physical coordinates and prevents time samples from the same trace leaking across splits.

## 2026-08-24 — Periodic azimuth representation

**Status:** active

**Decision:**
Keep physical `azimuth_deg` in interim data and use sine/cosine features for model and spatial-baseline inputs.

**Reason:**
This preserves physical provenance while avoiding the numerical discontinuity at 0°/360°.

## 2026-08-25 — Clean-data optimization choice

**Status:** active

**Decision:**
Keep L2 support and use `config.yaml` as the executable source of the current loss and learning rate.

**Reason:**
The first full-domain L1 run converged near a zero predictor, while an L2 one-trace diagnostic showed that a clean waveform could be fitted. Later full-domain convergence failures are investigated in separate studies.

**Evidence:**
- `runs/study_001_c3_na_baseline/20260825T053620Z_09d01f2_baseline`
- `runs/study_001_c3_na_baseline/20260825T070226Z_73694b8_l2_baseline`
