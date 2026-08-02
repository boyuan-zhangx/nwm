# Intern Handoff Checklist

## Day-one checklist

1. Read `docs/README.md` and the chapter for the assigned task.
2. Use Python 3.10 and activate the project Linux/WSL/cluster environment.
3. Run `python scripts/navware.py doctor --profile nwm`.
4. Run `python scripts/navware.py smoke`; every test must pass.
5. Copy `config/paths.example.yaml` to `config/paths.local.yaml` and edit only
   machine-specific paths.
6. Run the dataset validator on the assigned split.
7. Build or validate the assigned revisit manifest.
8. Run a tiny frozen CDiT/S baseline before requesting a scaled cluster job.

Stop and report the exact command and complete error when any gate fails. Do not
work around a failed gate by editing source, removing a check, or changing data.

## Independent work packages

### A. Data and benchmark

Deliver a versioned revisit-event manifest, oracle/random historical indices,
ground-truth future indices, a non-revisit query list, and a dataset validation
report. Freeze thresholds before inspecting model outputs.

### B. Context-policy verification

The selector is implemented in `retrieval_context.py`. Run its tests and verify
that every assigned dataset preserves:

- `recent`, `random_history`, `oracle_manifest`, and `pose_aligned`;
- deterministic random selection under a seed;
- no current/future-frame leakage;
- no cross-trajectory history;
- identical shapes for all policies.

Do not edit the selector, model architecture, or training code merely to make a
dataset pass. Report an invalid manifest with the failing trajectory and query.

### C. Oracle gate

Run `recent`, `random_history`, and `oracle_manifest` on 20-50 frozen revisit
queries with official CDiT/S. Use matched diffusion seeds. Deliver raw
predictions, selected indices, resolved config, metrics, and success and failure
panels.

Stop and report if oracle does not beat both controls. Do not proceed to learned
retrieval or hybrid training.

### D. Scale confirmation

After the CDiT/S gate passes, repeat only the four frozen policies on CDiT/B and
CDiT/XL. Change only the model configuration and matching checkpoint. Deliver a
scale-by-policy table and report an S-only gain explicitly.

### E. Evaluation

Compare predictions with ground-truth future frames using
`scripts/evaluate_phase_a.py`. Add non-revisit query manifests when assigned.
Every policy must use the same directory schema, and aggregation must read raw
files rather than contain hand-entered numbers.

### Deferred work: HybridCDiT

Do not connect or train `HybridCDiT` unless the maintainer explicitly reopens
Phase B after the oracle and pose-aligned gates pass.

## Required content for every pull request

- one sentence stating the scientific question;
- every modified config;
- an exact copy-paste command;
- automated tests;
- output location and commit hash;
- representative success and failure cases;
- an explicit list of questions that remain unanswered.

## Prohibited practices

- hard-coding a personal path or interpreter;
- selecting experiment groups inside Python source;
- changing the model architecture or checkpoint for Phase A;
- retrieving more than one historical frame in Phase A;
- using generated predictions as Phase A history;
- overwriting a source checkpoint or prior raw output;
- reusing history across trajectories;
- reporting only the best seed;
- tuning ground-truth thresholds after viewing model results;
- describing an unconnected or untested component as complete.

## Copy-paste preflight

```bash
source /path/to/navware-venv/bin/activate
cd /path/to/nwm

cp config/paths.example.yaml config/paths.local.yaml
# Edit config/paths.local.yaml before continuing.

python scripts/navware.py doctor \
  --profile nwm \
  --config config/nwm_cdit_s.yaml \
  --paths-config config/paths.local.yaml
python scripts/navware.py smoke
```

The preflight is complete only when doctor and smoke both succeed.
