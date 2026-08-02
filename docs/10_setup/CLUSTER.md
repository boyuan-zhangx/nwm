# Cluster Setup and Job Contract

The cluster is a scaling tool for a Phase A experiment that already passed on
CDiT/S. Do not allocate a long training job for context replacement: the
current method freezes every NWM checkpoint and performs inference only.

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

The Phase A path does not import the vendored `WorldMem/` package. Install that
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

Every run directory must preserve:

```text
predictions/
metrics/
visualizations/
metadata/
```

Metadata must include the commit, resolved experiment and path configurations,
environment snapshot, GPU information, checkpoint, query manifest, policies,
and seeds. Never commit tokens, W&B keys, or personal directories.

## Gate before a scaled inference job

Set the paths once, then run the general checks:

```bash
export EXPERIMENT_CONFIG=config/nwm_cdit_xl.yaml
export PATHS_CONFIG=config/paths.local.yaml
export DATASET_ROOT=/path/to/dataset
export SPLIT_FILE=data_splits/recon/test

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

Before requesting several GPUs or a long wall time, preserve local outputs that
show all of the following:

1. the frozen baseline runs on a tiny revisit manifest;
2. `oracle_manifest` beats `recent` and `random_history` on the primary metric;
3. every policy emits exactly four conditioning frames;
4. source frame indices and matched seeds are saved;
5. no history crosses a trajectory boundary.

If the context-policy runner is not yet present, only baseline inference is
executable. Do not substitute the hybrid configuration as a workaround.

## Scale-up order

1. Complete all debugging and ablations on official CDiT/S.
2. Run the core four policies on CDiT/B with matched queries and seeds.
3. Run the same core comparison on CDiT/XL only after S and B are interpretable.
4. Do not spend cluster time on CDiT/L unless a specific result requires it.

The core policies are `recent`, `random_history`, `oracle_manifest`, and
`pose_aligned`. A larger checkpoint does not justify changing thresholds,
queries, context construction, sampler steps, or metrics.

## Upstream baseline training only

`nwm.sh` is the site-neutral upstream training adapter. It is retained for
baseline reproduction but is not the Phase A paper entry point. Do not edit it
to insert a personal directory, partition, account, node name, or Conda
installation. If baseline training is explicitly required, pass site-specific
scheduler values to `sbatch`:

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

Add `--constraint`, `--nodelist`, or a different time limit only when required
by the site. Do not use this training example for context replacement.
