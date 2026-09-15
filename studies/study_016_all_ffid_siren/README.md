# Study 016: all-FFID SIREN

Status: `draft`

## Purpose

- Train one SIREN across all amplitude-eligible SEG C3 NA FFIDs and evaluate traces held out within each FFID.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`inputs.yaml`](inputs.yaml).

- FFIDs 2-4782; 4,780 eligible after QC; 625 samples per trace
- split within each FFID: 80% train, 5% validation, 15% test
- QC excluded 544 traces of FFID 1746 (107 all-zero, 437 above the amplitude limit), leaving FFID 1746 in no split
- 4,780 updates per epoch (one complete eligible FFID per update)
- selection: streaming global validation S/N; test and excluded traces unused

## Results

- Not run; no run, checkpoint, or result is recorded.
