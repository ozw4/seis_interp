# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Continue one fitted model through nested trace pools

**Context:** Study 006 strongly fit its seed-42 eight-trace subset with a 5,000-point
random-replacement batch, while Study 007 remained near zero when the identical batch mode and
50,000-update budget started fresh on all 435 training traces. Complete-trace batches in Study
008 and shared temporal patches in Study 010 also remained near zero on the full training pool.

**Decision:** Initialize one model and Adam optimizer once, then train for 50,000 updates at
each nested pool size `[8, 16, 32, 64, 128, 256, 435]`. Use pure L2, learning rate `1.0e-3`, and
5,000-point random-replacement batches throughout. Seed each stage sampler with the project
seed plus its zero-based index. Carry the final model and optimizer state forward without
resetting, rewinding, or checkpoint selection. Require the first-stage final median
training-trace S/N to reach 20 dB; otherwise stop before later stages and record
`stage8_anchor_failed`.

**Evidence:** Study 006 run `20260826T045538Z_55b4b9d_random_replacement_5000` reached 30.1410
dB best and 26.8248 dB final median training-trace S/N. Study 007 run
`20260826T061901Z_b0af699_random5000_trace435` reached only -0.01808 dB best median
training-trace S/N. Study 008 run `20260826T065417Z_fa548ba_tracebatch8_trace435` and Study 010
run `20260826T074127Z_b7bfe87_patch64_trace78_trace435` also produced
`full_ffid_near_zero`.

**Impact:** The run tests whether this fixed continuation path can reach a full-training fit
while preserving the Study 006/007 random-point batching contract. Its 1,750,000,000 point
evaluations are seven times the Study 007 budget, and no compute-matched fresh full-pool control
is included, so a positive result cannot by itself establish that continuation rather than
additional compute caused the difference. Production training remains unchanged.

## 2026-08-26 — Continuation loses the strong fit at 128 traces

**Context:** The fixed continuation run completed all seven stages, 350,000 updates, and
1,750,000,000 sampled point evaluations. The eight-trace final state reproduced the Study 006
anchor at 26.8248 dB median training-trace S/N. The 16-, 32-, and 64-trace stages remained
`strong_fit`, with final median S/N values of 30.0633, 21.1843, and 20.9149 dB.

**Decision:** Record `full_ffid_near_zero`. At the 128-trace stage, entry median S/N was 6.1350
dB, but the best post-update report was -0.02026 dB and the final state was -0.03638 dB. The
256- and 435-trace stages also remained `near_zero`. The final 435-trace stage's best report was
at cumulative step 307,000, with -0.02189 dB median trace S/N, -0.00235 dB global S/N,
-0.00234 median trace correlation, and a 0.02230 prediction/target RMS ratio; final median trace
S/N was -0.06302 dB.

**Evidence:** Run `20260826T232352Z_1aa9755_continuation8to435_random5000`; summary
`20260826T232352Z_1aa9755_summary.json`. All seven stages completed, all 700 report points were
finite, and the generated run contains only the four contracted files.

**Impact:** This fixed continuation path did not fit all 435 training traces. It localized the
observed loss of fit to the expansion from 64 to 128 traces under this schedule, seed, optimizer
state, and per-stage budget. It does not establish why the collapse occurred or separate
continuation effects from the seven-times-larger point budget relative to Study 007.
