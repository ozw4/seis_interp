# Decisions

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
Approximately 4,780 FFIDs per epoch makes ten epochs comparable to Study 014's 50,000 optimizer
updates. This is a budget anchor, not an optimum or convergence claim.
