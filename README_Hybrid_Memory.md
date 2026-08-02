# Hybrid memory status: deferred Phase B

`HybridCDiT` is not the current ICRA 2027 implementation path. The Phase A
question is cheaper and more fundamental: can a frozen pretrained NWM benefit
when one correctly retrieved real observation replaces the oldest item in its
existing four-frame context?

## Why it is deferred

The hybrid branch adds cross-attention parameters and a zero-initialized gate.
At initialization the gate deliberately preserves baseline behavior, so the
new branch cannot provide a useful inference-only intervention without
training. Training it would also require memory-aware batches, masks, paired
revisit data, loss validation, checkpoint handling, and additional ablations.
Those are avoidable costs until Phase A establishes that historical context is
useful to the frozen model.

## What remains available

- pose/action-aware `MemoryBuffer` experiments;
- explicit `memory_latents` accepted by `HybridCDiT`;
- checkpoint-compatible memory blocks and padding masks;
- CPU tests for shape, masking, and baseline compatibility.

The training dataset and rollout runner do not yet connect these pieces. Do not
run `config/hybrid_l40s_inference_memory.yaml` as a paper experiment.

## Reactivation gate

Consider Phase B only when all of the following are true:

1. oracle context replacement beats `recent` and `random_history` on CDiT/S;
2. `pose_aligned` replacement shows a stable gain on held-out revisits;
3. the core gain transfers to at least one larger official NWM checkpoint;
4. the gain survives three diffusion seeds and non-revisit evaluation;
5. the remaining failure can plausibly be solved by learned fusion rather than
   better retrieval or labels;
6. the paper schedule still allows a tiny-overfit training gate.

Until then, start from [the current handbook](docs/README.md).
