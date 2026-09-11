#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study=studies/study_039_c3_drr_target_tuning
qc=data/processed/c3_na/study_029_c3_amplitude_qc
case_id=c3_benchmark_test_random_trace_80_seed42
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_rank12_iterations60"
run_root="runs/study_039_c3_drr_target_tuning/$run_id"
mkdir -p "$run_root"
python -m seis_interp.cli interpolate drr \
  --config "$study/rank12_iterations60.yaml" \
  --interim data/interim/c3_na/all_ffids \
  --processed "$qc/partition" --mask "$qc/masks/$case_id" \
  --case "$qc/cases/$case_id" --volume "$qc/volumes/$case_id" \
  --output "$run_root/drr" --json 2>&1 | tee "$run_root/execution.log"
