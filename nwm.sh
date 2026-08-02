#!/usr/bin/env bash
# Generic Slurm adapter. Supply site-specific partition, account, and node
# constraints as `sbatch` command-line options rather than editing this file.
#SBATCH --job-name=navware-train
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch [site options] nwm.sh EXPERIMENT_CONFIG PATHS_CONFIG [train.py arguments...]

Optional environment variables:
  NAVWARE_VENV    Virtual environment directory to activate inside the job.
  NAVWARE_PYTHON  Python executable consumed by scripts/train.sh.

Example:
  export NAVWARE_VENV="$HOME/.venvs/navware-nwm"
  sbatch --partition=GPU nwm.sh \
    config/nwm_cdit_xl.yaml \
    config/paths.local.yaml \
    --epochs 1
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

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${NAVWARE_VENV:-}" ]]; then
  if [[ ! -r "${NAVWARE_VENV}/bin/activate" ]]; then
    echo "NAVWARE_VENV is not a readable virtual environment: ${NAVWARE_VENV}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${NAVWARE_VENV}/bin/activate"
fi

printf '[navware] Job ID: %s\n' "${SLURM_JOB_ID:-local-shell}"
printf '[navware] Host: %s\n' "$(hostname)"
printf '[navware] Started: %s\n' "$(date --iso-8601=seconds)"

bash scripts/train.sh "$@"

printf '[navware] Finished: %s\n' "$(date --iso-8601=seconds)"
