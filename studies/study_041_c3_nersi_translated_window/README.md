# NeRSI translated-window experiment

Status: target_reached. Primary target: full-T mean trace SNR >= 15 dB in physical amplitude.
This is target-informed window selection, not independent generalization evidence.
It does not establish success on the original fixed PoC window.

## Result

`shot18_ry18` reached **15.6282 dB mean trace SNR**, with global SNR 14.4326 dB
and physical RMSE 1.7591. All 104,750 target traces were evaluated; no trace SNR
was nonfinite. O=26,322, total traces=131,072, realized missing fraction=79.9179%.
Source-line [25,41), shot [18,50), receiver-x [0,8), receiver-y [18,50).
The original window's measured mean trace SNR was 13.7364 dB; this is a different
evaluation window, not a 1.8918 dB improvement on the original fixed target.

The 50,000-update run has full coverage and exact observed reinsertion.
Strict checkpoint restoration matched prediction exactly (maximum absolute error 0),
and restored evaluation matched metrics.json. Frozen Study 036/037 hashes remain unchanged.
The other two candidates were not trained because the first candidate reached the target.

- [Run](../../runs/study_041_c3_nersi_translated_window/20260913T232356Z_03c2fc83df18_shot18_ry18/metadata.json)
- [Verified summary](../../runs/study_041_c3_nersi_translated_window/20260913T232356Z_03c2fc83df18_shot18_ry18_summary.json)
- [Restoration audit](../../runs/study_041_c3_nersi_translated_window/20260913T232356Z_03c2fc83df18_shot18_ry18.audit.json)

## Conditions

The input layout, verified artifacts, outer mask and physical mean trace SNR metric
are fixed as **c3_random80_v3** in [v3_conditions.lock.json](v3_conditions.lock.json).
This is a condition label; existing run benchmark IDs and metadata are not renamed.
Model and normalization choices are not fixed by this input-condition lock.

The recorded 15.6282 dB reference uses O-only **Global RMS = 9.27911442626805**,
not independent trace RMS normalization or RMS-IDW restoration. Prediction is restored
to physical amplitude before averaging per-trace SNR in dB. The mean-trace evaluation
metric does not imply trace-normalized training inputs.

Time [0,384), source-line [25,41), and shape [384,16,32,8,32] remain fixed.
Receiver-x has only eight indices and stays [0,8).
Standalone method configurations retain aligned_ema0999 architecture, MSE, seeds,
LR 0.001, EMA 0.999, 50,000 optimizer updates, batch sizes and time alignment.
Only input identity/selection/protocol and primary evaluation metric differ.

Candidate order:

| Candidate | Shot range | Receiver-y range |
|---|---|---|
| shot18_ry18 | [18,50) | [18,50) |
| shot10_ry34 | [10,42) | [34,66) |
| shot27_ry00 | [27,59) | [0,32) |

Shot [0,32) with receiver-y [18,50) is not dense and is not a training candidate.
Every candidate must pass existing canonical geometry/QC and input-hash checks.
Study 029 partition and seed42 nominal 80% random mask are reused unchanged;
overlapping cells retain their O/T roles. Realized missing fractions are recorded per window.
Each window has a separate case, volume, benchmark ID and full input lock.
No T is excluded after prediction. No target amplitudes fit preprocessing or training.

The runner preserves the source archive, log, config, checkpoint and prediction,
then strict-loads the checkpoint, verifies the input binding and re-predicts all T.
Audit results are written separately, without modifying the completed run.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
unset CUDA_VISIBLE_DEVICES
study=studies/study_041_c3_nersi_translated_window
candidate=shot18_ry18
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_$candidate"
.venv/bin/python "$study/run.py" \
  --config "$study/$candidate.yaml" \
  --data-output "data/processed/c3_na/study_041_c3_nersi_translated_window/$candidate" \
  --output "runs/study_041_c3_nersi_translated_window/$run_id"
```

Use a new output for each run; existing data/run outputs are rejected.
Run candidates sequentially on cuda:1, without automatic retries or changes to training settings.
