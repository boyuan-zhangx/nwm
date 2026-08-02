#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh EXPERIMENT_CONFIG PATHS_CONFIG [train.py arguments...]

Paths may be absolute, relative to the caller's directory, or relative to the
repository root. Activate a Python 3.10 environment first, or set
NAVWARE_PYTHON to the interpreter executable.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

if (($# < 2)); then
  usage >&2
  exit 2
fi

CALLER_DIR="${PWD}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

select_python() {
  local candidate=""

  if [[ -n "${NAVWARE_PYTHON:-}" ]]; then
    candidate="${NAVWARE_PYTHON}"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    candidate="${VIRTUAL_ENV}/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    candidate="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv-wsl/bin/python" ]]; then
    candidate="${REPO_ROOT}/.venv-wsl/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    candidate="${REPO_ROOT}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    candidate="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    candidate="$(command -v python)"
  else
    echo "No Python interpreter found." >&2
    echo "Activate the project environment or set NAVWARE_PYTHON=/path/to/python." >&2
    return 1
  fi

  if [[ "${candidate}" == */* ]]; then
    if [[ ! -x "${candidate}" ]]; then
      echo "Python interpreter is not executable: ${candidate}" >&2
      return 1
    fi
  else
    candidate="$(command -v "${candidate}")" || {
      echo "Python command not found: ${candidate}" >&2
      return 1
    }
  fi

  printf '%s\n' "${candidate}"
}

resolve_config() {
  local label="$1"
  local supplied="$2"
  local candidate=""

  if [[ "${supplied}" == /* ]]; then
    candidate="${supplied}"
  elif [[ -f "${CALLER_DIR}/${supplied}" ]]; then
    candidate="${CALLER_DIR}/${supplied}"
  elif [[ -f "${REPO_ROOT}/${supplied}" ]]; then
    candidate="${REPO_ROOT}/${supplied}"
  fi

  if [[ -z "${candidate}" || ! -r "${candidate}" ]]; then
    echo "${label} not found or not readable: ${supplied}" >&2
    echo "Checked relative to: ${CALLER_DIR} and ${REPO_ROOT}" >&2
    return 1
  fi

  printf '%s\n' "${candidate}"
}

PYTHON_BIN="$(select_python)"
EXPERIMENT_CONFIG="$(resolve_config "Experiment config" "$1")"
PATHS_CONFIG="$(resolve_config "Paths config" "$2")"
shift 2

printf '[navware] Python: %s\n' "${PYTHON_BIN}"
printf '[navware] Experiment config: %s\n' "${EXPERIMENT_CONFIG}"
printf '[navware] Paths config: %s\n' "${PATHS_CONFIG}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/navware.py doctor \
  --profile nwm \
  --config "${EXPERIMENT_CONFIG}" \
  --paths-config "${PATHS_CONFIG}"

exec "${PYTHON_BIN}" train.py \
  --config "${EXPERIMENT_CONFIG}" \
  --paths-config "${PATHS_CONFIG}" \
  "$@"
