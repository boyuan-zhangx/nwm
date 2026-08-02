#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
  echo "Usage: bash scripts/train.sh EXPERIMENT_CONFIG PATHS_CONFIG [train.py args...]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${NAVWARE_PYTHON:-}" ]]; then
  PYTHON_BIN="${NAVWARE_PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv-wsl/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv-wsl/bin/python"
else
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
EXPERIMENT_CONFIG="$1"
PATHS_CONFIG="$2"
shift 2

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/navware.py doctor \
  --profile nwm \
  --config "${EXPERIMENT_CONFIG}" \
  --paths-config "${PATHS_CONFIG}"
exec "${PYTHON_BIN}" train.py \
  --config "${EXPERIMENT_CONFIG}" \
  --paths-config "${PATHS_CONFIG}" \
  "$@"
