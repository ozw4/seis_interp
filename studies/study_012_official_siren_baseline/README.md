# study_012_official_siren_baseline

## Status

`active`

## Research question

Can the official SIREN frequency and initialization parameterization escape the near-zero
predictor and fit all 435 FFID 2348 training traces under the same 5,000-point
random-replacement and 50,000-update budget as Study 007?

## Motivation

Study 007 remained near zero when training this full 435-trace pool from a fresh legacy model.
Study 011 reproduced the eight-trace strong-fit anchor but lost the fit while expanding the
pool to 128 traces. This direct paired comparison tests whether the shared SIREN
parameterization, rather than batching or continuation, is the limiting condition here.

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split and all 625 time
samples. Coordinates use training-min/max linear scaling plus azimuth sine/cosine features, and
amplitudes use the training-only global RMS. Validation and test amplitudes are used only to
validate the existing split contract; they do not enter training, model selection, or metrics.
The split contract fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to
validation.

Both conditions use a 6-input SIREN with width 256 and four total sine-activated layers. The
first layer computes `sin(omega_0 * linear)`, and every later sine layer computes
`sin(hidden_omega * linear)`. First-layer weights are initialized uniformly in
`[-1 / fan_in, 1 / fan_in]`. Every later sine-layer linear and the final linear are initialized
uniformly in
`[-sqrt(6 / fan_in) / hidden_omega, sqrt(6 / fan_in) / hidden_omega]`. Biases retain the
PyTorch defaults.

Each condition starts with a fresh model, Adam optimizer, and `RandomPointSampler`. NumPy,
PyTorch, and CUDA are reseeded to 42 immediately before model construction, and each sampler
uses seed 42. Training uses pure point-wise L2 loss, learning rate `1.0e-3`, uniform sampling
with replacement, batch size 5,000, and 50,000 updates on `cuda:0`. Thus each condition uses
250,000,000 sampled point evaluations. Every 500 updates, chunked prediction in batches of
65,536 evaluates all 435 by 625 training points. Each report records median training-trace
S/N, global training S/N, median training-trace correlation, prediction/target RMS ratio, and
mean training loss since the previous report. No correlation loss, temporal patching,
trace-pool continuation, checkpointing, or validation/test interpolation evaluation is used.

## Conditions

Exactly two conditions are run in this order:

| Condition | `omega_0` | `hidden_omega` |
|---|---:|---:|
| `legacy_control` | 300.0 | 1.0 |
| `official_siren_30` | 30.0 | 30.0 |

The legacy condition directly checks that the Study 007 near-zero control is reproduced. The
official condition changes the frequency and initialization parameterization as one package;
this study does not decompose their individual effects.

## Acceptance and interpretation

For each condition, the report with maximum median training-trace S/N is classified as follows:

- `strong_fit`: median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

If `legacy_control` is not `near_zero`, the summary decision is
`legacy_control_not_reproduced`, regardless of the official condition. Otherwise,
`official_siren_30` maps to `official_siren_strong_fit`,
`official_siren_escaped_zero_predictor`, or `official_siren_near_zero` according to its
classification. Best reports are used only for diagnostic classification; no report selects a
checkpoint or changes training.

## Reproduction

```bash
python scripts/run_study_012_official_siren_baseline.py \
  --config studies/study_012_official_siren_baseline/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_012_official_siren_baseline \
  --device cuda:0
```

## Expected outputs

Each of the two conditions produces one immutable run directory containing only
`config.resolved.yaml`, `inputs.lock.json`, `metrics.json`, and `run.json`. One sibling summary
JSON records both condition outcomes and the summary decision. No checkpoint, plot, table, or
notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only and does not evaluate validation/test
interpolation. The official frequency and initialization parameterization is tested as one
package, so the result cannot identify an individual causal contribution. Study 007's
full-training random-replacement run remained near zero, while Study 011's continuation run
collapsed at 128 traces; this paired experiment tests a new parameterization under the Study
007 training budget but does not establish generality beyond that setting.

Historical rationale belongs in [`decisions.md`](decisions.md).
