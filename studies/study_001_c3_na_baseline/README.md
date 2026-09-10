# Study 001: C3 NA baseline

Status: `draft`

## Purpose

- Compare a coordinate SIREN with nearest-neighbor and inverse-distance trace interpolation on held-out traces.

## Conditions

- dataset: SEG C3 Narrow-Azimuth, `SEG_C3NA_ffid_1201-2400.sgy`, FFID 2348
- data: 544 traces, 625 samples per trace, 8 ms sampling, no time window
- split: complete-trace train/validation/test; random and structured masks are separate conditions
- model: six-feature SIREN input derived from the stored five-dimensional physical mapping
- normalization: coordinate bounds and amplitude RMS fitted from training traces only
- inputs: [`inputs.yaml`](inputs.yaml)
- config: [`config.yaml`](config.yaml)

## Results

- No accepted comparison result or metric is recorded.
- preliminary runs: `20260825T053620Z_09d01f2_baseline` (L1) and `20260825T070226Z_73694b8_l2_baseline` (L2)

## Decision

- Keep complete-trace splits, train-only normalization, physical `azimuth_deg` in interim data, and sine/cosine azimuth model features.
- Use `config.yaml` as the source of the current L2 and learning-rate settings.
