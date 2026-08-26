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
