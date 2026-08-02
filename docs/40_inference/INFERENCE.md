# Inference

## Wrapper contract

The inference wrapper accepts two required config files followed by unchanged
`isolated_nwm_infer.py` arguments:

```text
bash scripts/infer.sh EXPERIMENT_CONFIG PATHS_CONFIG [inference arguments...]
```

It uses the same interpreter and config resolution rules as `scripts/train.sh`,
validates the environment with doctor, and refuses to launch when a config file
is missing. Do not insert a personal path into the script.

## Baseline command

```bash
export EXPERIMENT_CONFIG=config/nwm_cdit_xl.yaml
export PATHS_CONFIG=config/paths.local.yaml
export RUN_OUTPUT=/path/to/results/nwm-baseline

bash scripts/infer.sh \
  "$EXPERIMENT_CONFIG" \
  "$PATHS_CONFIG" \
  --output_dir "$RUN_OUTPUT" \
  --ckp 0100000 \
  --datasets recon \
  --eval_type rollout \
  --rollout_fps_values 1,4 \
  --batch_size 1 \
  --num_workers 0
```

Use a new output directory for every commit/config/seed combination. Never
overwrite the source checkpoint or a previous run's raw outputs.

## Correct LT-NWM memory lifecycle

The memory buffer must not update inside `HybridCDiT.forward`. The denoiser is
called hundreds of times for one generated frame; updating there would treat
diffusion steps as historical video frames.

Each rollout step must follow this order:

1. Query the buffer once using the current global pose and target action.
2. Produce top-k VAE latents, source frame indices, retrieval scores, and mask.
3. Reuse that exact memory set for every diffusion step of the generated frame.
4. After decoding, write exactly one real observation or final predicted latent
   to the buffer.
5. Clear the buffer at the trajectory boundary. Never leak memory across batch
   items or trajectories.

## Required causal groups

- `no_memory`
- `correct_memory`
- `random_memory`
- `temporally_wrong_memory`
- `heading_wrong_memory`

Every run must save the query, top-k source frames, source indices, each score
component, memory mask, gate statistics, prediction, and seed. A qualitative
video without those diagnostics cannot support a mechanism claim.

The frozen group specification is
`experiments/e04_memory_causal_ablation.yaml`. Its current status is
`specification`. Change that status only after a runner consumes and validates
every field.
