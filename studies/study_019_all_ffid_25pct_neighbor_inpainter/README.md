# study_019_all_ffid_25pct_neighbor_inpainter

## Status

`in_progress`

## Research question

Can a leakage-safe geometry-conditioned trace inpainter exceed
`oracle_per_trace_unit_rms_global_snr_db > 25 dB` when exactly one quarter of
the amplitude-eligible traces in every FFID are assigned to training?

## Split contract

"25% FFID" means a deterministic whole-trace split inside every eligible FFID,
not a split of whole FFIDs. Seed 42 assigns 25% to train. The remaining 75%
keeps the established allocation: 25% of holdout (18.75% overall) is validation
and 75% of holdout (56.25% overall) is test. No time samples from a trace cross
split boundaries.

All four locked SEG C3 Narrow-Azimuth sources and all 4,780 amplitude-eligible
FFIDs remain in scope. FFID 1746 is wholly excluded by amplitude QC. Global
physical-coordinate canonicalization retains the lowest `array_row` without
consulting split or amplitude values.

Prepared train/validation/test counts are 575,870 / 431,890 / 1,295,720.
Canonicalization removes 6 / 3 / 6 rows, giving effective counts of 575,864 /
431,887 / 1,295,714. Every eligible FFID contributes all three splits.

## Evaluation and staged method

Only train amplitudes may populate neighbor inputs. Validation amplitudes are
used only for checkpoint selection and metrics; test and excluded amplitudes are
not materialized. Raw predictions are compared with oracle per-trace unit-RMS
validation targets.

Stage 01 changes only the split relative to the accepted Study 018 architecture
and evaluates it after 2,500 updates. Stage 02 keeps that condition fixed and
tests an explicit aligned-neighbor reference with a zero-initialized CNN
residual. Subsequent stages will isolate the most promising neighborhood,
alignment, capacity, objective, and training-budget changes. A final candidate
must be frozen before a fresh full-scope acceptance run.

## Acceptance criteria

- All 4,780 eligible FFIDs contribute train, validation, and test traces.
- Effective train/validation/test counts equal 575,864 / 431,887 / 1,295,714.
- Duplicate physical cells are canonicalized before amplitude routing.
- The target offset is absent and every neighbor amplitude comes from train.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 25 dB.
- Run provenance is complete and repository quality gates pass.

## Current result

No 25% training run has been accepted yet. Immutable staged runs will be stored
under `runs/study_019_all_ffid_25pct_neighbor_inpainter/` and summarized in
`reports/all_ffid_25pct_25db_investigation.md`.

Historical rationale is recorded in [decisions.md](decisions.md).
