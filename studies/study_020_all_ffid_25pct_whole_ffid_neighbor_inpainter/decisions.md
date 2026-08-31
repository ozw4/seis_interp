# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-31 — Interpret 25% FFID as a disjoint whole-FFID split

**Status:** active

**Decision:** Select exactly 25% of the 4,780 amplitude-eligible FFIDs for
training. Assign every eligible trace of one FFID to the same split. Allocate
25% of the held-out FFIDs to validation and the remainder to test.

**Reason:** The user explicitly corrected the experimental condition from a
25% within-FFID trace ratio to selecting 25% of FFIDs. A separate study keeps
the earlier Study 019 run facts immutable and prevents the two estimands from
being conflated.

## 2026-08-31 — Start with the accepted K274 architecture and probe source coverage

**Status:** active

**Decision:** Stage 01 changes only the split relative to the accepted Study 018
architecture and trains for 2,500 updates. Geometry-only diagnostics precede
aperture changes. Subsequent stages change one mechanism at a time and use
fresh initialization.

**Reason:** Whole-FFID withholding changes source coverage qualitatively. The
matched baseline measures that shift, while the geometry probe identifies
whether same-line or crossline aperture is the first bottleneck without using
validation amplitudes.

**Geometry evidence:** K274 has 132,336 zero-neighbor validation traces
(30.28%). Same-line K494 and K934 reduce this to 10.17% and 1.21% respectively.
Crossline K714 reduces it to 3.56%, while crossline K1374 reaches zero and has a
minimum of 9 available train traces. Stage 02 isolates crossline support at the
original source-y radius; Stage 03 then isolates complete source-y coverage.

## 2026-08-31 — Match training context to held-out-FFID evaluation

**Status:** active

**Decision:** For every training target, mask all neighbor offsets whose source
coordinates equal the target source coordinates. Apply the same deterministic
mask during training audit and validation.

**Reason:** A validation FFID contributes no neighbor amplitudes, whereas an
unmasked training target can use hundreds of traces from its own FFID. That
train/evaluation context shift would optimize receiver interpolation within a
known shot instead of the requested held-out-shot interpolation. The mask
creates a pseudo-held-out FFID using train amplitudes only.

## 2026-08-31 — Use evidence-based promotion gates

**Status:** active

**Decision:** A 2,500-step model change must improve the matched Stage 01 result
by at least 0.20 dB, have a finite reproducible checkpoint metric, and preserve
all scope/leakage checks to be promoted. A 10,000-step winner is extended only
if its measured late slope and a documented empirical tail leave a credible
path to the strict 25 dB target.

**Reason:** Full validation covers 437,087 traces and longer GPU runs are
expensive. Predeclared gates keep the staged search interpretable and prevent
budget-only escalation after a clearly flattened curve.

## 2026-08-31 — Isolate an exact-receiver shot-axis prediction reference

**Status:** active

**Decision:** After the finite-aperture coverage stages, compare Stage 01 with a
K274 model whose initial prediction is the distance-weighted interpolation of
the nearest strict lower/upper train shots on the same source-x line and at the
same relative receiver coordinates. Use the nearest side without extrapolation
when only one side exists. Append the synthesized reference after local-neighbor
dropout and zero-initialize the residual head.

**Reason:** Whole-FFID withholding removes an entire source point. A
geometry-only diagnostic found exact-receiver two-sided brackets for 397,535 of
437,087 validation traces and a one-sided train source for every remaining
trace. The deterministic reference alone measured 5.4785273632 dB, versus
4.3999236270 dB for nearest-shot copying. This is not a 25 dB candidate by
itself, but it supplies a source-axis-aligned waveform while leaving the K274
CNN to learn the residual. Formal audits require zero unresolved rows, zero
non-train sources, zero target-FFID sources, and zero same-source-y sources.
