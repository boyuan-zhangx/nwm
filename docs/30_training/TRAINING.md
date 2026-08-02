# Training

## Current paper decision: no training in Phase A

Retrieval-based context replacement uses the pretrained NWM checkpoint without
updating any parameter. Do not launch baseline retraining, hybrid training, or
a learned retrieval scorer for the initial ICRA experiment.

This rule applies at every scale. CDiT/S, CDiT/B, and CDiT/XL use their official
pretrained weights as frozen backbones; changing scale is not authorization to
fine-tune or retrain them.

The first decision gate is inference-only:

```text
recent vs. random_history vs. oracle_manifest
```

If oracle replacement cannot help the frozen model, training a larger memory
system is not justified by the current paper question.

## Upstream baseline training wrapper

The retained training wrapper accepts two required config files followed by
unchanged `train.py` arguments:

```text
bash scripts/train.sh EXPERIMENT_CONFIG PATHS_CONFIG [train.py arguments...]
```

Both config paths may be absolute, relative to the caller's directory, or
relative to the repository root. The wrapper validates both files, runs doctor,
and starts training only if the preflight succeeds.

Interpreter selection order:

1. `NAVWARE_PYTHON`;
2. the active `VIRTUAL_ENV`;
3. the active `CONDA_PREFIX`;
4. a repository-local `.venv-wsl` or `.venv`;
5. `python3` or `python` from `PATH`.

Interns should activate the project environment or set `NAVWARE_PYTHON`; they
must not edit the wrapper to insert a personal interpreter path.

## Baseline reproduction command

```bash
cp config/paths.example.yaml config/paths.local.yaml
export EXPERIMENT_CONFIG=config/nwm_cdit_xl.yaml
export PATHS_CONFIG=config/paths.local.yaml

bash scripts/train.sh \
  "$EXPERIMENT_CONFIG" \
  "$PATHS_CONFIG" \
  --epochs 1 \
  --ckpt-every 2000 \
  --eval-every 10000 \
  --bfloat16 1 \
  --torch-compile 0
```

This command reproduces or debugs upstream training; it is not required for the
Phase A method. Do not use the XL model to debug a new data pipeline.

## Deferred HybridCDiT boundary

`HybridCDiT` currently provides:

- explicit `memory_latents` with shape `[B, M, C, H, W]`;
- a padding mask for memory attention;
- baseline checkpoint loading with `strict=False`, where missing keys are
  limited to the memory branch;
- a zero-initialized memory gate, making the initial output match the baseline.

The missing path is critical: `TrainingDataset` does not yet produce historical
memory candidates, poses, or masks, and `train.py` does not yet pass those
tensors through diffusion `model_kwargs`. A hybrid YAML alone therefore cannot
train the memory branch. In addition, the zero-initialized gate intentionally
makes an untrained branch reproduce the baseline, so it is not a useful
inference-only shortcut.

## Phase B reactivation gate

Do not connect hybrid training merely because compute is available. Reconsider
it only after oracle and pose-aligned context replacement are positive and the
remaining failure specifically requires learned fusion.

If Phase B is approved, first use one scenario and 8-32 trajectories that
contain revisits. Freeze the VAE and most of CDiT, train only memory attention
and the gate, fix every seed, and keep a fixed visualization set.

The tiny-overfit gate passes only if all conditions hold:

1. training loss decreases substantially;
2. memory gate and attention parameters receive finite non-zero gradients;
3. predictions improve on fixed training revisit samples;
4. correct history outperforms random history;
5. no-memory output matches the baseline at initialization.

Do not scale Phase B before this gate passes.
