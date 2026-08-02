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

`random_history` deterministically samples an eligible frame that is not a
manifest positive. `oracle_manifest` uses the first eligible positive.
`pose_aligned` minimizes normalized position distance plus wrapped yaw
difference over history outside the frozen temporal gap; its default scales are
0.75 meters and 20 degrees. Each policy fails on an invalid candidate set
instead of silently falling back to another policy.

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

## CDiT/S baseline command

```bash
export EXPERIMENT_CONFIG=config/nwm_cdit_s.yaml
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

## Copy-paste oracle gate

The first gate predicts one second at 4 Hz for at most 50 frozen manifest
queries. Generate the matching ground truth once:

```bash
export EXPERIMENT_CONFIG=config/nwm_cdit_s.yaml
export PATHS_CONFIG=config/paths.local.yaml
export REVISIT_MANIFEST=artifacts/manifests/recon_revisit_geometry_v1.jsonl
export RUN_OUTPUT=/path/to/results/cdit_s_gate_001

bash scripts/infer.sh \
  "$EXPERIMENT_CONFIG" \
  "$PATHS_CONFIG" \
  --output_dir "$RUN_OUTPUT" \
  --datasets recon \
  --eval_type time \
  --gt 1 \
  --revisit-manifest "$REVISIT_MANIFEST" \
  --num_sec_eval 1 \
  --input_fps 4 \
  --max-revisit-queries 50 \
  --batch_size 1 \
  --num_workers 0
```

Then run the three decision policies with identical seeds:

```bash
for POLICY in recent random_history oracle_manifest; do
  bash scripts/infer.sh \
    "$EXPERIMENT_CONFIG" \
    "$PATHS_CONFIG" \
    --output_dir "$RUN_OUTPUT" \
    --ckp 0100000 \
    --datasets recon \
    --eval_type time \
    --revisit-manifest "$REVISIT_MANIFEST" \
    --context-policy "$POLICY" \
    --context-seed 0 \
    --diffusion-seed 0 \
    --num_sec_eval 1 \
    --input_fps 4 \
    --max-revisit-queries 50 \
    --batch_size 1 \
    --num_workers 0 \
    --torch-compile 0
done
```

Run `pose_aligned` with the same command only after oracle passes. Change
`--diffusion-seed` to 1 and 2 for the paper repeats; never change only one
policy's seed.

Predictions are isolated under:

```text
RUN_OUTPUT/nwm_cdit_s/phase_a/POLICY/context_seed_0/diffusion_seed_0/
```

Ground truth is stored under `RUN_OUTPUT/gt/phase_a/`. Each run writes one
rank-local retrieval JSONL file under its `metadata/` directory.

## Runtime contract

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

Run `recent`, `random_history`, and `oracle_manifest` first. Run `pose_aligned`
only if oracle passes the gate.

Every run must save the query, selected source frame and index, pose distance,
yaw difference, temporal gap, policy, prediction, ground-truth future, and
seed. A qualitative video without those diagnostics cannot support a causal
claim.

The existing `experiments/e04_memory_causal_ablation.yaml` describes the older
hybrid-memory study and is not the Phase A run configuration.
