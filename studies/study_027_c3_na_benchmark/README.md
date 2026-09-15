# Study 027: fixed C3 NA benchmark

Status: `materialized_locked`

## Purpose

- Define the immutable data contract for POCS, DRR, SIREN-5D, CCNet-5D, and RelationalTraceGraphInterpolator comparisons.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml).

- time 0 to 3.064 s at 8 ms
- validation volume `[384,9,32,8,32]`, source lines `[41,50)`, shots `[32,64)` (the config defines the test volume only)
- canonical trace counts: train 1,146,803 / test 740,542 / validation 416,664
- canonicalization removes 15 duplicate physical-coordinate rows by lowest `array_row`
- 20 cases materialized; the later proposed contiguous-missing condition is not included

## Results

- benchmark suite: [`benchmark_suite.json`](../../data/processed/c3_na/study_027_c3_na_benchmark/benchmark_suite.json)
- suite SHA-256: `6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`
- 144 file hashes, all case bindings, physical mappings, mask seeds, and training ranges independently reverified
- 20 cases materialized; the later proposed contiguous-missing condition is not included

## Decision

- `inputs.yaml:cases` is the canonical mask/seed list; target waveforms are evaluation-only and do not enter normalization, inputs, neighbor selection, training, or stopping.
- The suite is immutable; generated arrays, masks, and volumes remain outside Git.
