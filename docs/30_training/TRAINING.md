# Training

## Wrapper contract

The training wrapper accepts two required config files followed by unchanged
`train.py` arguments:

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

## Baseline command

Create and edit the ignored path overlay first:

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

Do not use the XL model to debug the data pipeline. A local smoke run uses an
S/B model, reduced image size, `num_workers: 0`, a tiny split, and a dedicated
experiment YAML.

## Current LT-NWM boundary

`HybridCDiT` currently provides:

- a memory buffer lifecycle separate from the diffusion denoiser;
- explicit `memory_latents` with shape `[B, M, C, H, W]`;
- a padding mask for memory attention;
- baseline checkpoint loading with `strict=False`, where missing keys are
  limited to the memory branch;
- a zero-initialized memory gate, making the initial output match the baseline.

The missing path is critical: `TrainingDataset` does not yet produce historical
memory candidates, poses, or masks, and `train.py` does not yet pass those
tensors through diffusion `model_kwargs`. A hybrid YAML alone therefore cannot
train the memory branch. Do not submit a long hybrid job yet.

## Tiny-subset overfit gate

Use one scenario and 8-32 trajectories that contain revisits. Freeze the VAE
and most of CDiT, train only the memory attention and gate, fix every seed, and
keep a fixed visualization set.

The gate passes only if all conditions hold:

1. training loss decreases substantially;
2. memory gate and attention parameters receive finite non-zero gradients;
3. predictions improve on fixed training revisit samples;
4. correct memory outperforms random memory;
5. no-memory output matches the baseline at initialization.

Do not increase model size, batch size, or node count before this gate passes.
