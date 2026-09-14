#!/usr/bin/env bash
set -u
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study=studies/study_040_c3_nersi_target_tuning
qc=data/processed/c3_na/study_029_c3_amplitude_qc
case_id=c3_benchmark_test_random_trace_80_seed42
poc_inputs=(
  --interim data/interim/c3_na/all_ffids
  --processed "$qc/partition"
  --mask "$qc/masks/$case_id"
  --case "$qc/cases/$case_id"
  --volume "$qc/volumes/$case_id"
)
.venv/bin/python -m seis_interp.cli poc check "${poc_inputs[@]}" --json || exit 1
run_root="runs/study_040_c3_nersi_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_exploration"
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root" || exit 1
tar -cf "$run_root/nersi_source.tar" \
  src/seis_interp/models/nersi.py \
  src/seis_interp/nersi_config.py \
  src/seis_interp/pipelines/interpolate_nersi.py \
  src/seis_interp/training/fixed_step_nersi.py \
  src/seis_interp/training/trace_relative_loss.py \
  src/seis_interp/processing/trace_rms_idw.py \
  src/seis_interp/processing/trace_time_alignment.py \
  src/seis_interp/processing/nersi_coordinate_mapping.py \
  src/seis_interp/training/nersi_optimization.py \
  src/seis_interp/training/c3_volume_nersi_data.py \
  src/seis_interp/training/c3_volume_nersi_prediction.py \
  src/seis_interp/training/nersi_checkpoints.py \
  src/seis_interp/training/nersi_profile_mixup.py \
  src/seis_interp/training/nersi_coordinate_jitter.py || exit 1
echo "$run_root"
status=0
if [ "$#" -eq 0 ]; then
  set -- frequency_105 frequency_110 frequency_115 learning_rate_003
fi
for candidate in "$@"; do
  .venv/bin/python -m seis_interp.cli interpolate nersi \
    "${poc_inputs[@]}" --config "$study/$candidate.yaml" \
    --output "$run_root/$candidate" --json > "$run_root/$candidate.log" 2>&1 || status=1
  echo "$candidate finished; log: $run_root/$candidate.log"
done
exit "$status"
