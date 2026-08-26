# study_004_domain_scaling

## Status

`active`

## Research question

With all 625 time samples fixed, how does empirical training fit change as the number of
training traces increases?

## Fixed conditions

Experiment A uses the existing FFID 2348 train/validation/test split and its training-only
global-RMS normalization. Each condition trains a fresh 6-input SIREN with width 256, four sine
layers, `omega_0: 300.0`, Adam, L2 loss, learning rate `1.0e-3`, batch size 1024, and 50,000
updates on `cuda:0`. It reports full-subset training fit every 500 updates without early stopping
or checkpoints. Validation and test amplitudes are not used.

One seed-42 permutation of sorted training `array_row` values supplies the nested prefixes:

```text
1 ⊂ 8 ⊂ 32 ⊂ 128 ⊂ 435 traces
```

## Acceptance and interpretation

- `strong_fit`: best median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, best median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that report exceeds 0.1.
- `near_zero`: otherwise.

The result identifies the largest strong-fit subset, the largest subset that escaped the zero
predictor, and the first larger nested subset that returned to near-zero behavior. These are POC
diagnostic thresholds under one fixed setup; they do not attribute the boundary to parameter
count or select an `omega_0` optimum.

## Reproduction

```bash
python scripts/run_study_004_experiment_a.py \
  --config studies/study_004_domain_scaling/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_004_domain_scaling \
  --device cuda:0
```

## Decision log

- 2026-08-26: Keep the complete 625-sample time domain and vary only deterministic nested
  training-trace counts. Temporal patching and interpolation evaluation remain outside
  Experiment A because this probe measures training fit only.
