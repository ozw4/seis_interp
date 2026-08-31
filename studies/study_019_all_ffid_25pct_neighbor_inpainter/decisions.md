# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-31 — Interpret 25% as a within-FFID trace ratio

**Status:** active

**Decision:** Assign 25% of amplitude-eligible traces independently inside every
FFID using the seed-42 whole-trace permutation. Keep validation at 25% of the
75% holdout and test at the remaining 75% of holdout.

**Reason:** This preserves the interpolation question and the predecessor
studies' split meaning while changing only observed-trace density. Holding out
whole FFIDs would instead test shot extrapolation.

## 2026-08-31 — Start from the accepted Study 018 architecture

**Status:** active

**Decision:** Stage 01 reuses the Study 018 formal architecture and optimization
condition for 2,500 updates, changing only the split and success threshold.

**Reason:** A direct density-controlled baseline is necessary before attributing
later gains to neighborhood, model, objective, or budget changes.

## 2026-08-31 — Test a train-neighbor reference before a new model family

**Status:** rejected

**Decision:** Stage 02 adds the availability-masked, coordinate-gated, aligned
neighbor mean directly to a zero-initialized CNN residual. All split, aperture,
capacity, loss, sampler, and budget values remain fixed from Stage 01.

**Reason:** At 25% density the K274 validation neighborhood retains a mean of
only about 55 train traces. Supplying their coherent component as an explicit
reference tests whether the existing CNN is spending its limited budget
reconstructing a baseline waveform from zero. The mode is optional so legacy
models and checkpoints retain their exact behavior.

**Result:** Stage 02 reached `14.22890961173312 dB`, only
`+0.006065316774276797 dB` above Stage 01 at the same 2,500-update budget. This
is not a material gain, so the condition is not promoted to longer training.

## 2026-08-31 — Isolate geometry-aware fusion from aperture density

**Status:** rejected

**Decision:** After the residual-reference control, Stage 03 keeps K274 and
replaces the offset-specific input stem with a shared temporal encoder,
zero-padded receiver-y coarse shifts, and offset/target/time/content-conditioned
masked attention. Stage 04 separately keeps the accepted temporal CNN and grows
only the same-source-line aperture from K274 to K734.

**Reason:** The K274 validation neighborhood falls from about 110 available
train traces at 50% density to about 55 at 25%. Shared offset attention tests
whether explicit geometry and time-dependent fusion improve use of those traces;
K734 independently raises mean availability to about 133. Keeping the two
changes separate identifies representation versus neighbor-density effects.
The Stage 03 distance-prior scale is 0.1: at scale 1.0 the sparse K274 mask
would leave an effective attention support of only about 2.5 neighbors and
nearly eliminate gradients to distant offsets before learning.

**Result:** Stage 03 reached `9.819645233036228 dB`, which is
`-4.403199061922615 dB` relative to Stage 01. Its training audit was
`9.822922017404714 dB`, so the loss is not explained by overfitting; compressing
all K274 inputs into one shared attended feature is the observed bottleneck.
Stage 04 raised mean validation availability from 54.788 to 132.690 but reached
only `14.089875885195529 dB`, or `-0.132968409763314 dB` relative to Stage 01.
Neither condition is promoted.

## 2026-08-31 — Isolate deterministic receiver-y moveout alignment

**Status:** rejected

**Decision:** Stage 05 returns to the Stage 01 K274 architecture and changes
only neighbor preprocessing. Offset `(drx, dsx, dsy, dry)` is shifted by
`3 * dry` samples with zero padding before the existing target-coordinate gate,
depthwise FIR, and temporal CNN. The source index convention is
`source_sample = output_sample - shift`; circular wrap is forbidden.

**Reason:** A train-only cross-correlation probe identifies three samples per
relative receiver-y index as the stable coarse moveout. In the Stage 01
checkpoint, all 274 learned FIR channels still have their largest coefficient
at the center tap, and only 8.76% of expected moveout taps coincide with that
maximum. Hard alignment therefore tests a mechanism that the accepted FIR did
not absorb. The legacy default remains zero so existing models and checkpoints
retain exact behavior.

**Result:** Stage 05 reached `14.204319211934315 dB`, or
`-0.018525083024528 dB` relative to Stage 01. The checkpoint metric reproduced
exactly and the training audit reached `14.278501922544663 dB`; the condition
therefore fails the `+0.10 dB` continuation gate and is not promoted.

## 2026-08-31 — Isolate model width after mechanism tests

**Status:** accepted

**Decision:** Stage 06 returns to the unshifted Stage 01 K274 condition and
changes only `hidden_width` from 384 to 512. Split, aperture, gate, FIR, loss,
sampler, and 2,500-update horizon remain fixed.

**Reason:** Residual reference, shared fusion, wider aperture, and deterministic
alignment did not improve the matched baseline. Width is the remaining
low-complexity capacity control supported by the existing pipeline. It must gain
at least `0.20 dB` at 2,500 updates to become the longer-budget candidate.

**Result:** Stage 06 reached `14.438497913078372 dB`, or
`+0.215653618119529 dB` relative to Stage 01, with the best checkpoint at the
final step. The gain narrowly passes the promotion gate, so width 512 is carried
forward without any rejected mechanism changes.

## 2026-08-31 — Measure the promoted candidate at 10,000 updates

**Status:** active

**Decision:** Stage 07 reruns the unshifted width-512 K274 condition from a fresh
initialization with `total_steps: 10000`, evaluation every 2,500 steps, and a
10,000-trace training audit. It does not continue the 2,500-step checkpoint,
whose cosine schedule has already reached its minimum learning rate.

**Reason:** Study 018 gained 1.7497346480585065 dB from 10,000 to 50,000 updates.
Under that observed tail, Stage 07 must reach at least `23.250265351941493 dB`
to leave a budget-only path to the strict 25 dB target. A lower result stops the
50,000-step formal run unless the measured tail is demonstrably much steeper.
