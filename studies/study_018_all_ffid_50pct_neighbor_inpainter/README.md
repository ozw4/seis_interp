# study_018_all_ffid_50pct_neighbor_inpainter

## Status

`in_progress`

## Research question

Can a leakage-safe geometry-conditioned trace inpainter exceed
`oracle_per_trace_unit_rms_global_snr_db > 20 dB` when exactly half of the
amplitude-eligible traces in every FFID are assigned to training?

## Split contract

"50% FFID" is operationalized as a deterministic whole-trace split inside every
eligible FFID. Seed 42 assigns 50% to train. The remaining 50% retains Study 017's
holdout allocation rule: 25% of holdout (12.5% overall) is validation and 75% of
holdout (37.5% overall) is test. Every canonical trace belongs to one split only;
no samples from a trace cross splits.

All four locked SEG C3 Narrow-Azimuth sources and all 4,780 amplitude-eligible
FFIDs remain in scope. FFID 1746 is wholly excluded by amplitude QC. The 15
repeated physical cells are canonicalized globally by retaining the lowest
`array_row`, without consulting split or amplitude values.

Prepared train/validation/test counts are 1,151,740 / 287,935 / 863,805.
Canonicalization removes 9 / 2 / 4 rows respectively, giving the formal effective
counts 1,151,731 / 287,933 / 863,801.

## Evaluation and staged method

Only train amplitudes may populate neighbor inputs. Validation amplitudes are used
only for checkpoint selection and metrics; test and excluded amplitudes are not
materialized. Raw model predictions are compared with oracle per-trace unit-RMS
validation targets. Prediction self-normalization is diagnostic only.

The investigation started from the accepted Study 017 temporal CNN and isolated
target coordinates, neighborhood extent, training budget, capacity, conditioning,
sampling, temporal alignment, and objective regularization. The frozen formal
candidate uses a K274 same-source-line aperture, four target coordinates, width
384, full-trace temporal receptive field, FiLM, a target-coordinate neighbor gate,
and an identity-initialized 31-tap depthwise alignment FIR. It trains for 50,000
updates with the established replacement sampler. Only its fresh full-scope run
may be accepted.

## Acceptance criteria

- All 4,780 eligible FFIDs contribute train, validation, and test traces.
- Effective train/validation/test counts equal 1,151,731 / 287,933 / 863,801.
- Duplicate physical cells are canonicalized before amplitude routing.
- The target offset is absent and every neighbor amplitude comes from train.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 20 dB.
- Run provenance is complete and repository quality gates pass.

## Current result

The formal condition is frozen and its fresh run is pending. The strongest
completed diagnostic is Stage 09 at 18.24842244059429 dB after 10,000 updates.
At 2,500 updates, combining width 384, coordinate-dependent neighbor gating, and
the lightweight alignment FIR reached 16.80368309012617 dB, 0.437127913217366 dB
above the width-256 comparison. Staged immutable runs are recorded under
`runs/study_018_all_ffid_50pct_neighbor_inpainter/`.

Historical rationale is recorded in [decisions.md](decisions.md).
