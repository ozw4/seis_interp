# Study 027: fixed C3 NA benchmark

Status: `materialized_locked`

## Purpose

- Define the immutable data contract for POCS, DRR, SIREN-5D, CCNet-5D, and RelationalTraceGraphInterpolator comparisons.

## Conditions

- primary test volume shape: `[384,16,32,8,32]`
- ranges: time `[0,384)`; source lines `[25,41)`; shots `[28,60)`; receiver x `[0,8)`; receiver y `[18,50)`
- time: 0 to 3.064 s at 8 ms
- source-line partitions and canonical trace counts:

| Partition | Source lines | Traces |
|---|---|---:|
| train | `[0,25)` | 1,146,803 |
| test | `[25,41)` | 740,542 |
| validation | `[41,50)` | 416,664 |

- validation volume: `[384,9,32,8,32]`, source lines `[41,50)`, shots `[32,64)`
- canonicalization: remove 15 duplicate physical-coordinate rows by lowest `array_row`
- cases: test has random-trace 50/80/90% and whole-FFID 50/80%, each with seeds 42/43/44; validation uses the same five masks with seed 142
- model training seed: 20260908, distinct from mask seeds
- primary metric: `physical_amplitude_global_snr_db` on every `evaluation_target` sample
- information contract: POCS/DRR use crop observations; SIREN fits crop observations; CCNet/GNN may train on canonical train rows and time `[0,384)`; every method infers from the same crop observations
- config / cases: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml)

## Results

- benchmark suite: [`benchmark_suite.json`](../../data/processed/c3_na/study_027_c3_na_benchmark/benchmark_suite.json)
- suite SHA-256: `6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`
- 144 file hashes, all case bindings, physical mappings, mask seeds, and training ranges independently reverified
- 20 cases materialized; the later proposed contiguous-missing condition is not included

## Decision

- `inputs.yaml:cases` is the canonical mask/seed list.
- Target waveforms are evaluation-only and do not enter normalization, inputs, neighbor selection, training, or stopping.
- Keep the suite immutable; generated arrays, masks, and volumes remain outside Git.
