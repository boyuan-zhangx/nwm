# Cluster Setup and Job Contract

## One-time installation

```bash
git clone https://github.com/boyuan-zhangx/nwm.git
cd nwm

export NAVWARE_VENV="${NAVWARE_VENV:-$HOME/.venvs/navware-nwm}"
bash setup_nwm_env.sh \
  --profile nwm \
  --backend cu124 \
  --venv "$NAVWARE_VENV"

source "$NAVWARE_VENV/bin/activate"
```

Run `nvidia-smi` before selecting a backend. Use `cu121` only when the cluster
driver cannot support the repository's `cu124` wheel. Do not use an unpinned
nightly PyTorch wheel from an old README.

The LT-NWM path does not import the vendored `WorldMem/` package. Install that
package only when reproducing the upstream WorldMem Minecraft baseline:

```bash
python worldmem_setup_and_test.py install --dry-run
python worldmem_setup_and_test.py install
```

## Machine-local paths

Create one ignored overlay per machine or cluster account:

```bash
cp config/paths.example.yaml config/paths.local.yaml
```

Edit only `config/paths.local.yaml` to point at datasets, checkpoints, and
result storage. Shared experiment YAML files must not contain account names,
home directories, scratch roots, tokens, or host-specific mount points.

## Required metadata for every job

Run these commands inside the job output directory:

```bash
git -C /path/to/nwm rev-parse HEAD > run_commit.txt
python -m pip freeze > environment.txt
nvidia-smi > nvidia_smi.txt
cp "$EXPERIMENT_CONFIG" resolved_experiment.yaml
cp "$PATHS_CONFIG" resolved_paths.yaml
```

Never commit tokens, W&B keys, or personal directories. Every run directory
must contain at least:

```text
checkpoints/
logs/
metrics/
visualizations/
metadata/
```

## Gate before a long job

Set the paths once, then run all checks:

```bash
export EXPERIMENT_CONFIG=config/nwm_cdit_xl.yaml
export PATHS_CONFIG=config/paths.local.yaml
export DATASET_ROOT=/path/to/dataset
export SPLIT_FILE=data_splits/recon/train

python scripts/navware.py doctor \
  --profile nwm \
  --config "$EXPERIMENT_CONFIG" \
  --paths-config "$PATHS_CONFIG"
python scripts/navware.py smoke
python scripts/validate_dataset.py \
  --data-root "$DATASET_ROOT" \
  --split "$SPLIT_FILE" \
  --max-trajectories 20
```

A hybrid job additionally requires a tiny-overfit loss curve, fixed samples,
finite non-zero memory gradients, and an initial correct-vs-random comparison.

## Submit through Slurm

`nwm.sh` is a site-neutral Slurm adapter. Do not edit it to insert a personal
directory, partition, account, node name, or Conda installation. Pass
site-specific scheduler values to `sbatch`:

```bash
export NAVWARE_VENV="$HOME/.venvs/navware-nwm"
export EXPERIMENT_CONFIG=config/nwm_cdit_xl.yaml
export PATHS_CONFIG=config/paths.local.yaml

sbatch \
  --partition=YOUR_GPU_PARTITION \
  --account=YOUR_ACCOUNT \
  nwm.sh \
  "$EXPERIMENT_CONFIG" \
  "$PATHS_CONFIG" \
  --epochs 1 \
  --bfloat16 1
```

Add `--constraint`, `--nodelist`, or a different time limit on the `sbatch`
command only when required by the site. The job adapter delegates preflight and
training to `scripts/train.sh`, so local and cluster runs use the same contract.
