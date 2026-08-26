# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Probe shared 64-sample temporal patches

**Context:** Study 008 remained near zero with pure-MSE batches of eight complete traces, and
Study 009 remained near zero after adding the Study 005 trace-wise correlation loss to the same
batching structure. Both conditions exposed all 625 samples of each selected trace in every
update.

**Decision:** Run one seed-42 pure-MSE condition over the full 435-trace training pool. At each
update, uniformly select 78 distinct traces without replacement and one shared 64-sample patch
from the fixed starts `[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
480, 512, 544, 561]`. Use batch size 4,992, 50,000 updates, and the existing full-training
metrics and classification thresholds.

**Evidence:** Study 008 run `20260826T065417Z_fa548ba_tracebatch8_trace435` and Study 009 run
`20260826T071528Z_3013774_tracebatch8_corr0p1_trace435` both produced
`full_ffid_near_zero`. A 64-sample temporal patch with nominal 50% overlap is the next focused
batch-structure condition, and the terminal start 561 preserves coverage through sample 624.

**Impact:** This diagnostic tests whether the specified shared temporal-patch condition can fit
the full training split at a nearly matched point budget. It does not isolate the effects of
patch length, trace count per update, overlap, or the small point-budget difference, and it does
not justify broader causal claims or change production training.

## 2026-08-26 — Shared temporal patches remain near zero

**Context:** The single fixed condition completed all 50,000 updates and 249,600,000 point
evaluations. Each update used 78 distinct training traces and one shared 64-sample window from
the 19 predefined starts. All 100 full-training report points were finite.

**Decision:** Record `full_ffid_near_zero`. The best report occurred at step 10,000 with
-0.001837 dB median training-trace S/N, -0.000147 dB global S/N, 0.000669 median trace
correlation, and a 0.007035 prediction/target RMS ratio. Final median training-trace S/N was
-0.146711 dB.

**Evidence:** Run `20260826T074127Z_b7bfe87_patch64_trace78_trace435`; summary
`20260826T074127Z_b7bfe87_summary.json`.

**Impact:** This shared temporal-patch condition did not make the full training split fit under
the fixed seed and update budget. The result does not isolate patch length, trace count,
overlap, or point-budget effects and does not establish that other patch sampling strategies
will fail.
