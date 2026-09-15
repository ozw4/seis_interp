# Study 017: all-FFID neighbor inpainter

Status: `completed` — formal run accepted

## Purpose

- Evaluate leakage-safe reconstruction from training-trace neighbors at a strict oracle unit-RMS validation S/N threshold above 15 dB.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml).

- prepared split counts 1,842,102 train / 114,492 validation / 346,886 test; effective counts after lowest-`array_row` canonicalization of 15 repeated physical cells 1,842,090 / 114,490 / 346,885
- 104 same-source-line lattice neighbors per target
- 983,041 model parameters
- unavailable neighbors are zero with a false availability channel
- raw prediction selects the checkpoint

## Results

- staged diagnostics before the formal model: coordinate-only SIREN 10.6805 dB on FFID 2348-2363; leave-one-trace-out temporal neighbor model 16.1824 dB on 16 FFIDs, 16.5134 dB on a frozen 69-FFID replication, 18.0608 dB on all eligible FFIDs
- validation: 18.1119 dB at step 2,500 over 114,490 traces and 71,556,250 samples; margin +3.1119 dB; `metric_success=true`, `scope_success=true`, `success=true`
- independent training audit: 18.1044 dB; duplicate-cell, collision, overlap, neighbor-offset, and checkpoint-reload checks passed
- run: [`20260828T194620Z_edb2561_all_ffids`](../../runs/study_017_all_ffid_neighbor_inpainter/20260828T194620Z_edb2561_all_ffids/metrics.json)
- commit `edb2561ffa731f02e2c87325ba340ebce9671104`; NVIDIA H100 NVL
