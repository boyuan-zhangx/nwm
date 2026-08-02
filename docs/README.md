# NavWare / LT-NWM Experiment Handbook

These documents are the operational source of truth for interns and cluster
users. A ready workflow should require changes only to a machine-local path
overlay and an experiment YAML. Do not edit Python source to select a dataset,
checkpoint, output directory, or ablation group.

## Current implementation status

| Component | Status | Supported entry point |
| --- | --- | --- |
| Python 3.10 setup and dependency checks | Ready | `python scripts/navware.py doctor` |
| Memory and checkpoint compatibility tests | Ready | `python scripts/navware.py smoke` |
| Dataset contract validation | Ready | `python scripts/validate_dataset.py` |
| Baseline NWM training and inference wrappers | Ready when data and a checkpoint are available | `scripts/train.sh`, `scripts/infer.sh` |
| LT-NWM tensor flow and baseline checkpoint compatibility | Covered by CPU tests | `tests/test_hybrid_models.py` |
| LT-NWM memory-aware training batches | Not connected | Do not submit a long hybrid training job |
| LT-NWM rollout memory lifecycle | Not connected | Do not claim memory-aware inference results |

If `doctor` reports `config:hybrid-training-data`, the hybrid configuration is
not receiving a real memory batch. Do not hide that warning by changing YAML.
Implement and test the missing data path first.

## Reading order

1. [ICRA research plan](00_research/ICRA_PLAN.md)
2. [Local WSL setup](10_setup/LOCAL_WSL.md)
3. [Cluster setup](10_setup/CLUSTER.md)
4. [Dataset contract](20_data/DATA_CONTRACT.md)
5. [Training](30_training/TRAINING.md)
6. [Inference](40_inference/INFERENCE.md)
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
