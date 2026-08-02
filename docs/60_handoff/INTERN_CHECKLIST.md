# Intern Handoff Checklist

## Day-one checklist

1. Read `docs/README.md` and the chapter for the assigned task.
2. Use Python 3.10 and activate the project Linux/WSL/cluster environment.
3. Run `python scripts/navware.py doctor --profile nwm`.
4. Run `python scripts/navware.py smoke`; every test must pass.
5. Copy `config/paths.example.yaml` to `config/paths.local.yaml` and edit only
   machine-specific paths.
6. Run the dataset validator on the assigned split.
7. Run a tiny baseline smoke before touching hybrid code or requesting a long
   cluster job.

Stop and report the exact command and complete error when any gate fails. Do not
work around a failed gate by editing source, removing a check, or changing data.

## Independent work packages

### A. Data and benchmark

Deliver a versioned revisit-event manifest, correct/wrong memory labels, and a
dataset validation report. Freeze thresholds before inspecting model outputs.

### B. Retrieval

Deliver Recall@K/mAP, top-k visualizations, and pose-only/action-only/combined/
random comparisons. Modify only the retrieval module and config, and add tests.

### C. Training integration

Connect memory tensors, masks, and the freeze policy. Pass the tiny-overfit gate
before any scale-up. Deliver loss curves, gradient statistics, fixed samples,
and the complete YAML.

### D. Evaluation

Implement non-regression and revisit-only metrics. Every causal group must use
the same directory schema, and aggregation code must read raw files rather than
contain hand-entered numbers.

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
- overwriting a source checkpoint or prior raw output;
- reusing memory across trajectories;
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
  --config config/nwm_cdit_xl.yaml \
  --paths-config config/paths.local.yaml
python scripts/navware.py smoke
```

The preflight is complete only when doctor and smoke both succeed.
