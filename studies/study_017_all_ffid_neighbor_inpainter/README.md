# Study 017: all-FFID neighbor inpainter

Status: `completed`

## Purpose

- Evaluate leakage-safe reconstruction from training-trace neighbors at a strict oracle unit-RMS validation S/N threshold above 15 dB.

## Conditions

- dataset: all four SEG C3 NA sources and 4,780 eligible FFIDs; FFID 1746 excluded by QC
- prepared split counts: 1,842,102 train / 114,492 validation / 346,886 test
- effective counts after lowest-`array_row` canonicalization of 15 repeated physical cells: 1,842,090 / 114,490 / 346,885
- inputs: 104 same-source-line lattice neighbors; only train amplitudes, each divided by its RMS; unavailable neighbors are zero with a false availability channel
- model: width-128 temporal CNN, kernel-15 stem, 11 gated dilated blocks, 983,041 parameters
- loss: trace MSE + 0.1 first-time-difference MSE
- training: seed 42, AdamW, 2,500 updates, batch 96, learning rate `5e-4` with cosine decay to `1.5e-5`, weight decay `1e-5`, 5% neighbor dropout
- metric: `oracle_per_trace_unit_rms_global_snr_db`; raw prediction selects the checkpoint
- config / inputs: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml)

## Results

- staged diagnostics before the formal model: coordinate-only SIREN 10.6805 dB on FFID 2348-2363; leave-one-trace-out temporal neighbor model 16.1824 dB on 16 FFIDs, 16.5134 dB on a frozen 69-FFID replication, and 18.0608 dB on all eligible FFIDs
- validation: 18.1119 dB at step 2,500 over 114,490 traces and 71,556,250 samples
- threshold margin: +3.1119 dB; `metric_success=true`, `scope_success=true`, `success=true`
- independent training audit: 18.1044 dB
- audit: zero duplicate cells, train-coordinate collisions, train-validation coordinate overlaps, and target-center neighbor offsets; checkpoint reload reproduced the accepted metric
- run: [`20260828T194620Z_edb2561_all_ffids`](../../runs/study_017_all_ffid_neighbor_inpainter/20260828T194620Z_edb2561_all_ffids/metrics.json)
- commit: `edb2561ffa731f02e2c87325ba340ebce9671104`; device: NVIDIA H100 NVL

## Decision

- Accept the formal run.
- The result is an oracle unit-RMS waveform metric; it does not provide physical held-out trace gains.
