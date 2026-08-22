#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="python"
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${python_bin}" -c "import segyio" >/dev/null 2>&1; then
  cat >&2 <<EOF
segyio is not available in: ${python_bin}
Set up the repository environment, then rerun this script:

  bash ${repo_root}/scripts/setup_dev_environment.sh
EOF
  exit 1
fi

exec "${python_bin}" -m seis_interp.cli data inspect seg_c3_na "$@"
