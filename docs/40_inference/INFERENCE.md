# Inference

## Current Phase A method

The pretrained NWM accepts four conditioning frames. Context replacement must
preserve that interface and all checkpoint weights:

```text
recent:              [t-3, t-2, t-1, t]
random_history:      [random_old, t-2, t-1, t]
pose_aligned:        [pose_match, t-2, t-1, t]
oracle_manifest:     [oracle_match, t-2, t-1, t]
```

Exactly one real historical observation replaces the oldest native context
frame. Do not append tokens, change positional embeddings, instantiate
`HybridCDiT`, or retrieve top-k frames in Phase A.

Use CDiT/S for implementation and the full policy matrix. Once the same runner
is frozen, change only the model config and matching official checkpoint for
CDiT/B and CDiT/XL confirmation. A scale change must not alter query selection,
context indices, diffusion seeds, sampler steps, or metrics.

## Wrapper contract

The baseline inference wrapper accepts two required config files followed by
unchanged `isolated_nwm_infer.py` arguments:

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

## Implementation status

The baseline command above is executable. The four-policy context selector is
the next implementation task and must not be documented as executed until its
runner and tests land.

Each query must follow this order:

1. Read the query and its past real observations from one trajectory.
2. Select one source index according to the requested policy.
3. Construct exactly four conditioning frames or latents.
4. Reuse that exact context for every diffusion step of the prediction.
5. Save the selected source index and retrieval diagnostics.
6. Clear history at the trajectory boundary.

Do not update history inside a diffusion-model `forward` call. Do not add the
decoded prediction to Phase A history; self-generated memory introduces drift
and changes the research question.

## Required policy groups

- `recent`
- `random_history`
- `oracle_manifest`
- `pose_aligned`

Run `recent`, `random_history`, and `oracle_manifest` first. Implement and run
`pose_aligned` only if oracle passes the gate.

Every run must save the query, selected source frame and index, pose distance,
yaw difference, temporal gap, policy, prediction, ground-truth future, and
seed. A qualitative video without those diagnostics cannot support a causal
claim.

The existing `experiments/e04_memory_causal_ablation.yaml` describes the older
hybrid-memory study and is not the Phase A run configuration.
