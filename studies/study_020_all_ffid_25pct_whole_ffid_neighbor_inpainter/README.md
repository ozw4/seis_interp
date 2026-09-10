# Study 020: 25% whole-FFID neighbor inpainter

Status: `blocked`; strict 25 dB threshold not reached and no candidate met the recorded promotion rule

## Purpose

- Evaluate held-out-shot reconstruction when 25% of eligible FFIDs are assigned wholly to training.

## Conditions

- dataset: 4,780 eligible SEG C3 NA FFIDs; FFID 1746 excluded before FFID selection
- split: seed 42; 1,195 train / 896 validation / 2,689 test FFIDs, mutually disjoint
- prepared trace counts: 578,688 / 437,088 / 1,287,704; effective canonical counts: 578,685 / 437,087 / 1,287,693
- leakage contract: train-FFID amplitudes only; same-target-FFID neighbors masked during training; test and excluded amplitudes not materialized
- metric: `oracle_per_trace_unit_rms_global_snr_db` on raw validation predictions; strict threshold above 25 dB
- config / variants: [`config.yaml`](config.yaml), [`variants/`](variants/)

## Results

| Stage | Isolated condition | Validation S/N |
|---:|---|---:|
| 01 | K274 matched baseline | 4.4312 dB |
| 02 | K714 crossline support | 7.7835 dB |
| 03 | K1374 complete validation coverage | 8.7200 dB |
| 04 | K274 + shot-bracketing residual | 8.5133 dB |
| 05 | K1374 + shot-bracketing residual | 8.5960 dB |
| 06 | Stage 03 + epoch-without-replacement sampling | 8.7156 dB |
| 07 | Stage 06 + 6,030 updates | **9.0998 dB** |
| 08 | uncollapsed bracket channels | 8.4743 dB |
| 09 | K8 joint shot gather | 6.7827 dB |
| 10 | Stage 09 + receiver-y dilation | 6.7757 dB |
| 11 | Stage 09 + ordered raw source channels | 6.8000 dB |
| 12 | Stage 09 + width 128 | 7.0106 dB |
| 13 | Stage 09 + full temporal field | 6.8193 dB |
| 14 | Stage 12 + receiver-cell FiLM | 7.0281 dB |
| 16 | Stage 09 + inverse-distance power 2 | 6.9982 dB |
| 17 | Stage 09 + pure MSE | 6.7794 dB |
| 18 | Stage 09 without neighbor dropout | 6.7828 dB |
| 19 | Stage 12 + 6,000 updates | 7.4092 dB |
| 20 | Stage 12 + inverse-distance power 2 | 7.2845 dB |
| 21 | Stage 09 + dynamic source attention | 7.0227 dB |
| 22 | Stage 20 + 6,000 updates | 7.7007 dB |

- K274/K714/K1374 validation traces without a train neighbor: 132,336 (30.28%) / 15,560 (3.56%) / 0
- Stage 07 best remained 15.9002 dB below the strict threshold
- Stage 15 K16 was rejected before a formal run from fixed geometry diagnostics; its stage number remains reserved
- target-optimized 384-512-neighbor linear-span diagnostic: approximately 23.36 dB; diagnostic only because it uses target amplitudes
- every completed full-scope run passed FFID isolation, amplitude-access, collision, target-FFID masking, and checkpoint-revalidation checks
- Stage 07: 21,921,721 parameters; commit `7343bb0031a7713c55a19a691f47ef1d8b57e0ad`; train audit 11.9772 dB
- Stage 07 validation signal / error energy: 273,179,375.0032627 / 33,609,934.5625
- Stage 07 runtime: 1,652 s; CUDA peak allocated/reserved: 15,804,441,088 / 22,779,265,024 bytes

## Decision

- Keep Stage 07 as the study best.
- Reject the bracketing/K1374 combination, K16, and unchanged-scope budget extensions under the recorded gates.
- No Stage 23 or 50,000-update formal extension is promoted; the study remains blocked.
- Full supporting record: [investigation report](../../reports/all_ffid_25pct_whole_ffid_25db_investigation.md).
