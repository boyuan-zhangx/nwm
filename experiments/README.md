# Experiment manifests

These YAML files freeze scientific comparisons independently of Python code.
They are specifications until a runner explicitly validates and consumes every
field. A manifest must never be presented as an executed experiment by itself.

## Current Phase A contract

The active study uses a frozen NWM checkpoint and exactly four context slots.
Its policies are `recent`, `random_history`, `oracle_manifest`, and
`pose_aligned`. Each replacement policy selects one real historical frame and
keeps the three most recent frames. No Phase A manifest may enable
`HybridCDiT`, top-k memory attention, or training.

Use separate manifests or resolved run configurations for CDiT/S, CDiT/B, and
CDiT/XL. CDiT/S carries the complete ablation matrix; B and XL carry only the
core four policies. Model scale must never be encoded by a Python source edit.

The existing `e04_memory_causal_ablation.yaml` belongs to the older hybrid
study. It remains a specification for possible Phase B work and must not be run
or cited as the current experiment.

Rules:

- the same seeds, frozen checkpoint, sampler steps, and query manifest apply to
  all groups;
- group selection is a config value, never a source-code edit;
- every run saves the selected source index, pose/yaw diagnostics, prediction,
  ground-truth future, and resolved configuration;
- missing required output invalidates the run.
