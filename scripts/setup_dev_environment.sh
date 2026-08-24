#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${SEIS_INTERP_VENV:-${repo_root}/.venv}"
bootstrap_python="${PYTHON_BOOTSTRAP_BIN:-python}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to set up the development environment." >&2
  echo "Run this script inside the repository Dev Container." >&2
  exit 1
fi

if [[ -f "${venv_dir}/pyvenv.cfg" ]] \
  && grep -q '^include-system-site-packages = false$' "${venv_dir}/pyvenv.cfg"; then
  echo "Recreating outdated virtual environment: ${venv_dir}"
  echo "It cannot see the PyTorch bundled with the NGC image."
  rm -rf "${venv_dir}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "Creating virtual environment: ${venv_dir}"
  uv venv --python "${bootstrap_python}" --system-site-packages "${venv_dir}"
fi

cd "${repo_root}"
uv pip install \
  --python "${venv_dir}/bin/python" \
  --editable ".[dev,segy,data,visualization]"

"${venv_dir}/bin/python" - <<'PY'
from importlib.metadata import version

import segyio  # noqa: F401
import torch

print(f"Environment ready: Python with segyio {version('segyio')}")
print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
