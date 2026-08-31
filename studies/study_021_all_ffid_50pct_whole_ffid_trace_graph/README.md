# study_021_all_ffid_50pct_whole_ffid_trace_graph

## Status

`running`

## Research question

Can a graph neural network that treats one trace as one node — with time handled
as an in-trace latent sequence, never as a graph coordinate — exceed
`oracle_per_trace_unit_rms_global_snr_db > 20 dB` when exactly half of the
amplitude-eligible FFIDs are assigned wholly to training and every validation
FFID must be reconstructed as a completely unobserved shot?

## Split contract

Seed 42 permutes the 4,780 amplitude-eligible FFID identifiers with the
established whole-FFID rule. Exactly 2,390 FFIDs are assigned wholly to train.
The 2,390 held-out FFIDs retain the established allocation:
`round(2390 * 0.25) = 598` FFIDs to validation and 1,792 FFIDs to test. A
single eligible FFID cannot occur in more than one split. Amplitude-QC failures
remain `excluded`; FFID 1746 is wholly excluded before FFID selection.

Prepared train/validation/test trace counts are 1,155,312 / 293,152 / 855,016.
Global physical-coordinate canonicalization removes 8 / 1 / 6 rows, giving
effective counts of 1,155,304 / 293,151 / 855,010.

## Model and objective

The candidate model is `trace_graph_interpolator`. Nodes are traces: the 544
receiver cells of the target FFID plus the 544 cells of each of the `K`
nearest train source gathers. Waveforms are encoded per node into a
time-downsampled latent sequence, so time is an intra-node feature axis.
Message passing alternates over two factorized edge sets: receiver-lattice
edges inside each shot and source-axis edges between shots at the same
relative-receiver cell, conditioned on inter-source geometry. A
`source_receiver_bipartite` graph mode replaces the trace-node message passing
with explicit source-node and receiver-node aggregation where observed traces
are edge features and the missing target shot is a missing-edge set.
Prediction starts from the deterministic inverse-source-distance reference and
adds a zero-initialized decoded residual.

The training objective is

`L = L_mask + lambda_spec * L_spectrum + lambda_slope * L_slope + lambda_amp * L_amplitude`

where `L_mask` is the masked reconstruction MSE of the artificially hidden
gather, `L_spectrum` matches log-magnitude spectra plus magnitude-weighted
phase, `L_slope` matches local plane-wave (slope) consistency along the
receiver axis, and `L_amplitude` matches windowed RMS envelopes. Loss weights
are executable conditions in `config.yaml`; stages isolate one term at a time.

## Evaluation and staged method

Only train-FFID amplitudes may populate model inputs. Validation-FFID
amplitudes are used only for checkpoint selection and metrics. Test and
excluded amplitudes are not materialized by training runs. Raw predictions are
compared with oracle per-trace unit-RMS validation targets. During training,
every neighbor whose exact FFID matches the target FFID is excluded, matching
validation where the target FFID is absent from the train pool.

Stages change one mechanism at a time from a frozen baseline. Control stages
first calibrate the split difficulty with the existing accepted architectures
(joint shot-gather CNN and K-aperture trace inpainter) so every GNN gain is
attributable. A stage is promoted when it improves its matched comparison by
at least 0.20 dB at 2,500 updates. Budget extensions are granted only when the
measured curve leaves an evidence-backed path to 20 dB.

## Acceptance criteria

- Train/validation/test contain exactly 2,390 / 598 / 1,792 disjoint FFIDs.
- Their union is all 4,780 amplitude-eligible FFIDs.
- Effective train/validation/test counts equal 1,155,304 / 293,151 / 855,010.
- Duplicate physical cells are canonicalized without consulting split or
  amplitude values.
- Neighbor amplitudes come only from train FFIDs and the target FFID is
  excluded from its own inputs.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 20 dB.
- Run provenance is complete and repository quality gates pass.

## Reproduction

Prepare the split:

```bash
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_whole_ffid_50pct_train_amplitude_qc \
  --config studies/study_021_all_ffid_50pct_whole_ffid_trace_graph/config.yaml \
  --json
```

Train a stage:

```bash
python -m seis_interp.cli train trace-graph \
  --config studies/study_021_all_ffid_50pct_whole_ffid_trace_graph/variants/<stage>.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_whole_ffid_50pct_train_amplitude_qc \
  --output runs/study_021_all_ffid_50pct_whole_ffid_trace_graph/<run-id> \
  --device cuda:0 \
  --json
```

## Current result

The study is running; staged results are recorded under
`runs/study_021_all_ffid_50pct_whole_ffid_trace_graph/` and summarized in the
study report when stages complete.

Historical rationale is recorded in [decisions.md](decisions.md).
