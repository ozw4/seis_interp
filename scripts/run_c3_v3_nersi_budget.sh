#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
study=studies/study_046_c3_v3_nersi_gnn_parameter_budget
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
unset CUDA_VISIBLE_DEVICES
test "$#" -gt 0
status=0
for candidate in "$@"; do
  [[ "$candidate" =~ ^(fourier(160|40|16)|fourier48_profiles10_adamw_axis_nyquist|encoder_heavy|decoder_heavy|kernel3|base(105|110|115)|base115_profiles10(_adamw(_(shot_profile|wd0001|wd001|depthwise|nyquist_bandlimit|axis_rx105|axis_nyquist(_cartesian|_seed10[23]|_profile_embedding|_rank2_latent|_temporal_basis128)?|axis_source107(_shot109)?|axis_shot109))?)?)$ ]] || exit 2
  .venv/bin/python - "$study/$candidate.yaml" <<'PY'
import sys
from seis_interp.configuration import load_resolved_config
from seis_interp.models.nersi import Nersi
from seis_interp.nersi_config import validate_nersi_poc_config
c = load_resolved_config(sys.argv[1])
s = validate_nersi_poc_config(c)
m = Nersi(**s.model_constructor_config((384, 32)))
assert sum(p.numel() for p in m.parameters()) == 363269
assert s.training.max_steps == 50000
assert s.training.learning_rate == 0.001
assert c['training']['amplitude_scaling'] == 'observed_volume_global_rms'
assert s.time_alignment is None
assert s.optimization['learning_rate_schedule'] == 'constant'
PY
  run_root="runs/study_046_c3_v3_nersi_gnn_parameter_budget/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_${candidate}_50k"
  mkdir -p runs/study_046_c3_v3_nersi_gnn_parameter_budget
  mkdir "$run_root"
  printf '%s\n' "$run_root"
  tar -cf "$run_root/source.tar" src "$study" scripts/run_c3_v3_nersi_budget.sh \
    studies/study_042_c3_v3_five_method_comparison/nersi_no_time_shear.yaml \
    studies/study_041_c3_nersi_translated_window/v3_conditions.lock.json
  if ! .venv/bin/python -u -m seis_interp.cli interpolate nersi \
    --interim data/interim/c3_na/all_ffids \
    --processed data/processed/c3_na/study_029_c3_amplitude_qc/partition \
    --mask data/processed/c3_na/study_029_c3_amplitude_qc/masks/c3_benchmark_test_random_trace_80_seed42 \
    --case data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18/case \
    --volume data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18/volume \
    --config "$study/$candidate.yaml" --output "$run_root/nersi" --json \
    > "$run_root/stdout.log" 2> "$run_root/stderr.log"; then
    status=1
  fi
done
exit "$status"
