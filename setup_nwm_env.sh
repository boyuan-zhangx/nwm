#!/usr/bin/env bash
# Reproducible Python 3.10 environment for local Linux/WSL and GPU clusters.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${REPO_ROOT}/.venv-wsl"
PYTHON_BIN="python3.10"
PROFILE="nwm"
TORCH_BACKEND="cpu"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./setup_nwm_env.sh [options]

Options:
  --profile core|nwm|worldmem|all  Dependency group (default: nwm)
  --backend cpu|cu121|cu124        PyTorch wheel backend (default: cpu)
  --python COMMAND                 Python 3.10 executable (default: python3.10)
  --venv PATH                     Virtual environment path (default: .venv-wsl)
  --dry-run                       Print the resolved setup without installing
  -h, --help                      Show this help

Examples:
  ./setup_nwm_env.sh --profile nwm --backend cpu
  ./setup_nwm_env.sh --profile all --backend cu124
EOF
}

while (($#)); do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --backend)
      TORCH_BACKEND="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      ENV_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PROFILE}" in
  core|nwm|worldmem|all) ;;
  *)
    echo "Unsupported profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

case "${TORCH_BACKEND}" in
  cpu|cu121|cu124) ;;
  *)
    echo "Unsupported PyTorch backend: ${TORCH_BACKEND}" >&2
    exit 2
    ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python command not found: ${PYTHON_BIN}" >&2
  echo "Install Python 3.10 in the WSL distribution/cluster, then rerun." >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
  echo "Expected Python 3.10, found ${PYTHON_VERSION} via ${PYTHON_BIN}." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import ensurepip' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Python 3.10 is installed, but ensurepip/venv support is missing.
On Ubuntu 22.04 run:

  sudo apt update
  sudo apt install -y python3.10-venv

Then rerun this setup script. Windows and WSL environments are intentionally
separate: WSL uses .venv-wsl; the Windows CPU test environment may use .venv.
EOF
  exit 1
fi

if [[ -e "${ENV_DIR}" && ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Incomplete or non-Linux environment found at: ${ENV_DIR}" >&2
  echo "Choose a new --venv path or remove that environment directory, then rerun." >&2
  exit 1
fi

echo "Repository: ${REPO_ROOT}"
echo "Environment: ${ENV_DIR}"
echo "Profile: ${PROFILE}"
echo "PyTorch backend: ${TORCH_BACKEND}"

if ((DRY_RUN)); then
  exit 0
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi
VENV_PYTHON="${ENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade "pip>=24,<26" wheel setuptools

TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_BACKEND}"
"${VENV_PYTHON}" -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url "${TORCH_INDEX_URL}"

"${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements/dev.txt"

if [[ "${PROFILE}" == "nwm" || "${PROFILE}" == "all" ]]; then
  "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements/nwm.txt"
fi

if [[ "${PROFILE}" == "worldmem" || "${PROFILE}" == "all" ]]; then
  "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/WorldMem/requirements.txt"
fi

"${VENV_PYTHON}" -m pip check

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg is missing. Install it with the OS/cluster package manager." >&2
fi

DOCTOR_PROFILE="core"
if [[ "${PROFILE}" == "nwm" || "${PROFILE}" == "all" ]]; then
  DOCTOR_PROFILE="nwm"
elif [[ "${PROFILE}" == "worldmem" ]]; then
  DOCTOR_PROFILE="worldmem"
fi

"${VENV_PYTHON}" "${REPO_ROOT}/scripts/navware.py" doctor --profile "${DOCTOR_PROFILE}"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/navware.py" smoke

echo "Environment ready. Activate with: source '${ENV_DIR}/bin/activate'"
