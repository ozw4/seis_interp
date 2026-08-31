# study_019_all_ffid_75pct_neighbor_inpainter

## Status

`in_progress`

## Research question

Can a leakage-safe geometry-conditioned trace inpainter exceed
`oracle_per_trace_unit_rms_global_snr_db > 25 dB` when 75% of the
amplitude-eligible traces in every FFID are assigned to training?

## Split contract

"75% FFID" means a deterministic whole-trace split inside every eligible FFID.
Seed 42 assigns 75% to train. The remaining 25% retains Study 018's holdout
allocation rule: 25% of holdout (6.25% overall) is validation and 75% of holdout
(18.75% overall) is test. No time samples from a trace cross splits.

All four locked SEG C3 Narrow-Azimuth sources and all 4,780 amplitude-eligible
FFIDs remain in scope. FFID 1746 is wholly excluded by amplitude QC. The 15
repeated physical cells are canonicalized globally by retaining the lowest
`array_row`, without consulting split or amplitude values.

Prepared train/validation/test counts are 1,727,610 / 143,980 / 431,890.
Canonicalization removes 12 / 2 / 1 rows respectively, giving formal effective
counts 1,727,598 / 143,978 / 431,889.

## Evaluation and staged method

Only train amplitudes may populate neighbor inputs. Validation amplitudes are used
only for checkpoint selection and metrics; test and excluded amplitudes are not
materialized. Raw model predictions are compared with oracle per-trace unit-RMS
validation targets. Prediction self-normalization is diagnostic only.

The first diagnostic transfers Study 018's accepted architecture without changing
its model, optimizer, or loss, so the effect of the denser split is measurable.
Later stages isolate one model, neighborhood, initialization, or optimization
change at a time. A final condition is frozen only after the staged evidence is
reviewed, and only a fresh full-scope run of that frozen condition may be accepted.

## Acceptance criteria

- All 4,780 eligible FFIDs contribute train, validation, and test traces.
- Effective train/validation/test counts equal 1,727,598 / 143,978 / 431,889.
- Duplicate physical cells are canonicalized before amplitude routing.
- The target offset is absent and every neighbor amplitude comes from train.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 25 dB.
- Run provenance is complete and repository quality gates pass.

## Current result

The split contract and first diagnostic candidate are defined. No 75% run has yet
been accepted. Immutable runs will be recorded under
`runs/study_019_all_ffid_75pct_neighbor_inpainter/`, and the completed staged
investigation will be summarized in
`reports/all_ffid_75pct_25db_investigation.md`.

Historical rationale is recorded in [decisions.md](decisions.md).
