# Study 021: 50% whole-FFID trace graph

Status: `running`; strict 20 dB threshold not reached

## Purpose

- Evaluate a trace-node GNN for reconstructing completely unobserved validation shots with 50% of eligible FFIDs assigned wholly to training.

## Conditions

- dataset: 4,780 eligible SEG C3 NA FFIDs; FFID 1746 excluded
- split: seed 42; 2,390 train / 598 validation / 1,792 test FFIDs, mutually disjoint
- prepared trace counts: 1,155,312 / 293,152 / 855,016; effective canonical counts: 1,155,304 / 293,151 / 855,010
- leakage contract: train-FFID amplitudes only; target FFID excluded from inputs; test and excluded amplitudes not materialized
- model family: trace-node latent temporal sequence with receiver-lattice and source-axis relations; inverse-distance reference plus decoded residual
- tested objective: masked MSE, with spectrum, slope, and amplitude terms isolated separately
- training baseline: AdamW, learning rate `5e-4` with cosine decay, weight decay `1e-5`, seed 42, K8 train shots, bfloat16, 2,500 updates per short stage
- metric: `oracle_per_trace_unit_rms_global_snr_db`; strict threshold above 20 dB
- config / variants: [`config.yaml`](config.yaml), [`variants/`](variants/)

## Results

| Stage | Condition | Validation S/N | Decision |
|---:|---|---:|---|
| — | inverse-distance K8 reference | 7.089 dB | reference |
| 01 | joint-shot CNN control | 7.768 dB | control |
| 02 | Study 018 K1374 control | 11.318 dB | strongest control |
| 03 | trace lattice, width 64, 4 rounds, mask loss | 8.734 dB | GNN baseline |
| 04 | + spectrum loss | 8.642 dB | reject |
| 05 | + slope loss | 8.712 dB | reject |
| 06 | + amplitude loss | 8.483 dB | reject |
| 07 | source-receiver bipartite graph | 8.181 dB | reject |
| 08 | width 128, 6 rounds | 9.003 dB | promote |
| 09 | K16 shots | 8.647 dB | reject |
| 10 | per-frame attention | 8.026 dB | reject |
| 11 | shifted per-frame attention | 7.752 dB | reject |
| 12 | full time resolution, batch 1 | 8.554 dB | reject |
| 13 | width 128, 6 rounds, 10,000 updates | 10.691 dB | budget diagnostic |
| 15 | width 64, 12 rounds | 9.044 dB | promote |
| 17 | width 128, 12 rounds | 9.280 dB | promote |
| 18 | width 128, 12 rounds, batch 4 | 10.322 dB | promote |
| 19 | Stage 18, 10,000 updates | 12.2393 dB | below threshold |
| 20 | two-pass refinement | 8.777 dB | reject |
| 21 | width 128, 6 rounds, batch 4, 25,000 updates | **12.4692 dB** | study best |

- Stage 21 training audit: 12.793 dB; runtime 19.9 hours; 1.43M parameters
- Stage 21 exceeds the strongest control by 1.15 dB and remains 7.53 dB below threshold
- every completed run passed FFID-isolation, amplitude-access, collision, and exact checkpoint-revalidation audits
- OOM conditions: width192/rounds6/batch2, width64/full-time/batch2, width64/batch8, width128/rounds12/batch8 on a 93 GB GPU

## Decision

- Retain mask-only trace-lattice message passing; reject the tested auxiliary losses, bipartite graph, K16, per-frame attention, full-time-resolution condition, and two-pass refinement at their recorded budgets.
- Promote width, depth, batch scaling, and activation checkpointing.
- Stage 21 is the current best; the 20 dB requirement remains unmet and the study remains open.
- Full supporting record: [investigation report](../../reports/all_ffid_50pct_whole_ffid_trace_graph_20db_investigation.md).
