# study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter

## Status

`completed` — the 25 dB threshold was not reached

## Research question

Can a leakage-safe geometry-conditioned trace inpainter exceed
`oracle_per_trace_unit_rms_global_snr_db > 25 dB` when exactly one quarter of
the amplitude-eligible FFIDs, rather than one quarter of traces within every
FFID, are assigned wholly to training?

## Split contract

Seed 42 permutes the 4,780 amplitude-eligible FFID identifiers. Exactly 1,195
FFIDs are assigned wholly to train. The 3,585 held-out FFIDs retain the
established allocation: 896 FFIDs to validation and 2,689 FFIDs to test. A
single eligible FFID cannot occur in more than one split. Amplitude-QC failures
remain `excluded`; FFID 1746 is wholly excluded before FFID selection.

Prepared train/validation/test trace counts are 578,688 / 437,088 / 1,287,704.
Global physical-coordinate canonicalization removes 3 / 1 / 11 rows, giving
effective counts of 578,685 / 437,087 / 1,287,693.

## Evaluation and staged method

Only train-FFID amplitudes may populate model inputs. Validation-FFID amplitudes
are used only for checkpoint selection and metrics. Test and excluded
amplitudes are not materialized by training runs. Raw predictions are compared
with oracle per-trace unit-RMS validation targets. During training, every
neighbor whose exact FFID ID matches the target FFID is masked. This
pseudo-held-out-FFID context matches validation, where the target FFID is absent
from the train pool.

Stage 01 moves the accepted Study 018 architecture to the corrected whole-FFID
split and matched target-FFID mask for 2,500 updates. Before changing the model,
deterministic geometry probes measure zero-neighbor coverage and the useful source
aperture. K274 leaves
132,336 validation traces (30.28%) without any train neighbor. Stage 02 changes
only source-line support to crossline K714, reducing this to 15,560 traces
(3.56%). Stage 03 changes only its source-y radius to crossline K1374, yielding
zero missing-neighbor validation traces. Stage 04 returns to K274 and changes
only the prediction reference: it linearly interpolates the nearest strict
lower/upper train shots at the exact same-line receiver geometry, then learns a
zero-initialized CNN residual. The reference is built only from train amplitudes,
is not dropped out, and excludes the target FFID and same source-y. Later stages
change one mechanism at a time. If both complete K1374 coverage and the
bracketing reference improve Stage 01 by at least 0.20 dB, Stage 05 combines
only those two promoted changes. A longer run is promoted only when its matched
diagnostic shows a material gain and the measured curve leaves an
evidence-backed path to 25 dB.

## Results

All five full-scope runs passed their FFID isolation, amplitude access,
collision, target-FFID masking, and checkpoint-revalidation checks. The best
result was Stage 03, which removed every zero-neighbor validation row with
crossline K1374:

| Stage | Isolated condition | Validation S/N |
|---:|---|---:|
| 01 | K274 matched baseline | 4.431249374754326 dB |
| 02 | K714 crossline support | 7.783543855019937 dB |
| 03 | K1374 complete validation coverage | **8.719953365995504 dB** |
| 04 | K274 plus shot-bracketing residual | 8.51333997509688 dB |
| 05 | K1374 plus shot-bracketing residual | 8.595997409114656 dB |

Stage 02 improved Stage 01 by 3.352294480265612 dB, and Stage 03 added
0.936409510975567 dB. The bracketing reference improved matched K274 by
4.082090600342554 dB, but combining it with K1374 was 0.123955956880849 dB
worse than Stage 03. The best result remained 16.280046634004496 dB below the
strict threshold.

The matching Study 018 architecture gained 3.656672439472694 dB between its
2,500- and 50,000-step results. A 2,500-step candidate therefore needed at
least 21.343327560527307 dB to leave an evidence-backed path to 25 dB. Stage 03
was 12.623374194531802 dB below that promotion gate, so no budget-only 10,000-
or 50,000-step extension was run.

## Acceptance criteria

- Train/validation/test contain exactly 1,195 / 896 / 2,689 disjoint FFIDs.
- Their union is all 4,780 amplitude-eligible FFIDs.
- Effective train/validation/test counts equal 578,685 / 437,087 / 1,287,693.
- Duplicate physical cells are canonicalized without consulting split or
  amplitude values.
- Neighbor amplitudes come only from train FFIDs and the target center is absent.
- Training masks every same-source/target-FFID neighbor to match held-out-FFID evaluation.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 25 dB.
- Run provenance is complete and repository quality gates pass.

## Reproduction

Prepare the corrected split:

```bash
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_whole_ffid_25pct_train_amplitude_qc \
  --config studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/config.yaml \
  --json
```

Reproduce the best Stage 03 condition:

```bash
python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/variants/stage03_crossline_k1374.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_whole_ffid_25pct_train_amplitude_qc \
  --output runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/<run-id> \
  --device cuda:1 \
  --json
```

Historical rationale is recorded in [decisions.md](decisions.md).
The complete Japanese work and experiment report is
[`reports/all_ffid_25pct_whole_ffid_25db_investigation.md`](../../reports/all_ffid_25pct_whole_ffid_25db_investigation.md).
