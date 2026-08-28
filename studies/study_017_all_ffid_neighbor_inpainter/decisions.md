# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`, code, and tests.
This file records only rationale that may matter when changing that contract.

## 2026-08-28 — Pivot from an unconditional coordinate field to train-neighbor conditioning

**Status:** active

**Decision:**
Use a temporal CNN conditioned on fixed physical neighbors from the train split. Preserve Study
016's survey and split, but do not describe this condition as a SIREN result.

**Reason:**
The best fixed coordinate-only SIREN diagnostic reached 10.6805 dB on FFID 2348-2363. Cartesian
coordinates, width, time scaling, layer-frequency schedules, dense/deep connections, Fourier-ReLU,
trace/profile outputs, low-rank completion, moveout, and non-local retrieval all remained below
the 15 dB target. A leave-one-trace-out temporal neighbor model reached 16.1824 dB on 16 FFIDs,
16.5134 dB on a frozen 69-FFID replication, and 18.0608 dB on all eligible FFIDs in staged
proxies. The scale progression justified formalizing the changed model family.

## 2026-08-28 — Keep the target trace out of every neighbor lookup

**Status:** active

**Decision:**
Exclude offset `(0, 0, 0)`, restrict lookup amplitudes to train rows, never cross source-x lines,
and encode unavailable positions explicitly.

**Reason:**
The model must reproduce a completely held-out trace rather than copy it. This rule also prevents
the train twin at either of the two duplicate train-validation physical coordinates from entering
the matching validation prediction, while a separately reported clean metric quantifies any
remaining concern.

## 2026-08-28 — Accept oracle per-trace unit-RMS only as the requested waveform criterion

**Status:** active

**Decision:**
Select the checkpoint by raw predictions against per-trace unit-RMS validation targets and require
a point-weighted global S/N strictly greater than 15 dB. Do not normalize predictions for the
primary metric.

**Reason:**
This is the success rule requested for the staged investigation. It isolates waveform recovery
from trace gain, but cannot reconstruct physical amplitudes because target RMS is unavailable at
an unseen coordinate. Keeping raw predictions avoids a second prediction-dependent postprocess.
