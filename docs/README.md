# NavWare Context-Replacement Experiment Handbook

These documents are the operational source of truth for interns and cluster
users. The current ICRA 2027 route uses a frozen pretrained NWM and replaces
one item in its existing four-frame context with one real historical
observation. It does not train a new backbone or a hybrid memory branch.

The four frozen policies are `recent`, `random_history`, `pose_aligned`, and
`oracle_manifest`. Policy selection must eventually be a command-line or YAML
value, never a Python source edit.

Use official CDiT/S (50M parameters; repository key `CDiT-S/2`) for local
development, the oracle gate, and the complete ablation matrix. After the
effect is stable, repeat only the core four-policy comparison on CDiT/B
(`CDiT-B/2`) and CDiT/XL (`CDiT-XL/2`). Do not replace NWM with an unrelated
world-model family during Phase A.

## Phase A invariant

Every method must provide exactly four conditioning frames to the unchanged
NWM checkpoint:

```text
recent:              [t-3, t-2, t-1, t]
replacement policy:  [history, t-2, t-1, t]
```

Use one historical frame, not top-k memory tokens. Use only real observed
history in the first benchmark. Do not use generated frames as memory.

## Current implementation status

| Component | Status | Supported entry point |
| --- | --- | --- |
| Python 3.10 setup and dependency checks | Ready | `python scripts/navware.py doctor` |
| Existing model compatibility smoke | Ready | `python scripts/navware.py smoke` |
| Dataset contract validation | Ready | `python scripts/validate_dataset.py` |
| Geometric revisit manifest builder | Ready | `python scripts/build_revisit_manifest.py` |
| Baseline NWM training and inference wrappers | Ready when data and a checkpoint are available | `scripts/train.sh`, `scripts/infer.sh` |
| CDiT/S and CDiT/B experiment YAMLs | Not added | Use `CDiT-S/2` or `CDiT-B/2` and validate against the matching official checkpoint |
| Four-slot context-policy selector | Not connected | Next implementation task; do not claim results yet |
| Oracle/random/pose-aligned evaluation runner | Not connected | Implement after freezing a tiny revisit manifest |
| `HybridCDiT` training and memory attention | Deferred Phase B | Do not submit a hybrid training job |

If `doctor` reports `config:hybrid-training-data`, the selected configuration
belongs to the deferred branch. Switch back to the baseline NWM config; do not
hide the warning or implement hybrid training during Phase A.

## Reading order

1. [ICRA research plan](00_research/ICRA_PLAN.md)
2. [Local WSL setup](10_setup/LOCAL_WSL.md)
3. [Cluster setup](10_setup/CLUSTER.md)
4. [Dataset contract](20_data/DATA_CONTRACT.md)
5. [Training policy](30_training/TRAINING.md)
6. [Inference and context policies](40_inference/INFERENCE.md)
7. [Evaluation](50_evaluation/EVALUATION.md)
8. [Intern handoff checklist](60_handoff/INTERN_CHECKLIST.md)

## Reproducibility rule

Every paper number must be traceable to all of the following:

- a Git commit;
- the complete experiment YAML;
- the machine-local path overlay used for the run;
- the input checkpoint;
- every random seed;
- an environment snapshot;
- the untouched raw output directory.

Results produced after an uncommitted Python edit do not enter a paper table.
