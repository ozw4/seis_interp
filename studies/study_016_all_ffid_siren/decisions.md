# Decisions

## 2026-08-28 — Exclude amplitude-invalid traces before splitting and normalization

**Status:** active

**Decision:**
Exclude a complete trace when all raw samples are exactly zero or when any raw sample has an
absolute amplitude greater than `1.0e4`. Apply this filter before split assignment and before
fitting coordinate or amplitude normalization, independently of the later training-target
amplitude scaling.

**Reason:**
The locked input contains one isolated invalid block at FFID 1746. Its 544 traces consist of 107
all-zero traces and 437 non-zero traces whose peak absolute amplitudes start at `115210.0` and
extend nearly to the float32 limit. Outside FFID 1746 the observed maximum is `4298.765625`, so
`1.0e4` lies in a wide measured gap and excludes no other FFID. Retaining these rows makes
per-trace RMS undefined for zero traces and contaminates the training-global RMS with extreme
values. The excluded rows remain explicit in the processed split artifact for auditability.

## 2026-08-28 — Include the observed final FFID 4782

**Status:** active

**Decision:**
Treat FFID 4782 in `SEG_C3NA_ffid_3601-4781.sgy` as part of the SEG C3 NA survey.

**Reason:**
The source object name ends in 4781, but its verified trace headers contain the continuous
range 3601-4782 with no missing FFIDs. FFID 4782 has 544 traces with `trace_index_in_ffid`
values 0-543, geometry continuous with FFIDs 4779-4781, and finite amplitudes with non-zero
energy. It is therefore treated as a valid complete FFID rather than a corrupt trailing record.

## 2026-08-27 — Use the next available study identifier

**Status:** active

**Decision:**
Use `study_016_all_ffid_siren` rather than the originally proposed Study 015 name.

**Reason:**
Study 015 already contains a completed strong-fit budget-extension experiment and its recorded
result. Preserving that history takes precedence over the earlier proposed identifier.

## 2026-08-27 — Start with complete FFID batches and a global RMS target

**Status:** active

**Decision:**
Use one complete training FFID per optimizer update, point-wise L2, and one training-global RMS.
Do not use per-trace RMS balancing or a correlation term.

**Reason:**
Study 014 established that complete-trace sample density can escape the near-zero predictor with
global RMS alone. A per-trace scale is unavailable at a held-out trace coordinate, so it cannot be
used directly as the first interpolation target contract.

## 2026-08-27 — Treat ten epochs as a provisional comparable budget

**Status:** active

**Decision:**
Run at most ten epochs with patience of three epochs and select only by streaming global
validation S/N.

**Reason:**
With 4,780 amplitude-eligible FFIDs per epoch, ten epochs allow at most 47,800 optimizer updates,
which remains comparable to Study 014's 50,000 updates. This is a budget anchor, not an optimum or
convergence claim.
