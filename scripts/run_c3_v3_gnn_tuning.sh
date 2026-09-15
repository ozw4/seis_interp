#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
candidate=${1:?Usage: bash scripts/run_c3_v3_gnn_tuning.sh candidate_stem}
[[ "$candidate" =~ ^[a-z0-9_]+$ ]] || exit 2
study=studies/study_044_c3_v3_gnn_target_tuning
test -f "$study/$candidate.yaml"
.venv/bin/python - "$study/$candidate.yaml" <<'PY'
import sys

from seis_interp.configuration import load_resolved_config
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config

settings = validate_relational_trace_graph_poc_config(load_resolved_config(sys.argv[1]))
if settings.training["max_steps"] > 20000:
    raise SystemExit("Study 044 permits at most 20000 optimizer updates per run")
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
unset CUDA_VISIBLE_DEVICES

run_root="runs/study_044_c3_v3_gnn_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_$candidate"
mkdir -p runs/study_044_c3_v3_gnn_target_tuning
mkdir "$run_root"
printf '%s\n' "$run_root"
tar -cf "$run_root/source.tar" src "$study" \
  studies/study_041_c3_nersi_translated_window/v3_conditions.lock.json \
  studies/study_042_c3_v3_five_method_comparison/gnn.yaml \
  scripts/run_c3_v3_gnn_tuning.sh

.venv/bin/python -m seis_interp.cli interpolate relational-trace-graph \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/study_029_c3_amplitude_qc/partition \
  --mask data/processed/c3_na/study_029_c3_amplitude_qc/masks/c3_benchmark_test_random_trace_80_seed42 \
  --case data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18/case \
  --volume data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18/volume \
  --config "$study/$candidate.yaml" \
  --output "$run_root/relational_trace_graph" --json \
  > "$run_root/stdout.log" 2> "$run_root/stderr.log"
