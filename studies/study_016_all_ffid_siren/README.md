# Study 016: all-FFID SIREN

Status: `draft`

## Purpose

- Train one SIREN across all amplitude-eligible SEG C3 NA FFIDs and evaluate traces held out within each FFID.

## Conditions

- dataset: four manifest sources, FFIDs 2-4782; 4,780 eligible FFIDs
- QC: exclude a trace if all 625 samples are zero or any absolute amplitude exceeds `1e4`; 544 traces from FFID 1746 are excluded (107 zero, 437 excessive), leaving FFID 1746 in no split
- split: seed-42 split independently within each FFID; 80% train, 5% validation, 15% test
- normalization: coordinate bounds and one global amplitude RMS fitted on eligible training traces only
- model: six-input SIREN, width 256, four sine layers, `omega_0=30`, `hidden_omega=30`
- training: Adam, L2, learning rate `1e-4`, one complete eligible FFID per update, 4,780 updates per epoch, at most 10 epochs, early-stopping patience 3
- selection: streaming global validation S/N; test and excluded traces unused
- config / inputs: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml)

## Results

- Not run; no run, checkpoint, or result is recorded.

## Decision

- This is within-FFID trace interpolation, not whole-shot interpolation.
- Per-trace RMS targets, correlation loss, and Huber loss are excluded from this condition.
