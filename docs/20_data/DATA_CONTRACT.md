# Dataset Contract

## Minimum NWM trajectory layout

```text
DATA_ROOT/
└── trajectory_name/
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
    └── traj_data.pkl
```

Each `traj_data.pkl` must contain at least:

- `position`: shape `[T, >=2]`, expressed in a consistent global frame;
- `yaw`: shape `[T]` or `[T, 1]`, in radians;
- one numbered image per pose entry.

The split path must contain `traj_names.txt` with one trajectory name per line.

## Validate before any experiment

```bash
export DATASET_ROOT=/path/to/dataset
export SPLIT_PATH=data_splits/recon/test

python scripts/validate_dataset.py \
  --data-root "$DATASET_ROOT" \
  --split "$SPLIT_PATH" \
  --max-trajectories 20
```

Inspect 20 trajectories first. Remove `--max-trajectories` only after the sample
passes. Load pickle files only from trusted project datasets.

## Revisit benchmark labels

Every benchmark record must preserve:

- trajectory and query frame;
- candidate historical frames;
- pose distance and wrapped heading difference;
- temporal gap;
- scenario or landmark identity when annotated;
- the labeling rule and its version.

Define a correct historical frame before inspecting model outputs. A geometric
candidate normally satisfies all of the following:

- its temporal gap exceeds the native four-frame context;
- its position distance is below a frozen threshold;
- its wrapped yaw difference is below a frozen threshold;
- it belongs to the same pre-annotated revisit event or landmark when semantic
  annotations are available.

These independent labels support retrieval diagnostics. Phase A uses the first
valid positive as a top-1 oracle candidate; a model's own retrieval score must
never become its ground truth.

Each query must also identify the ground-truth future frame used by the
baseline evaluation path. The primary generation metrics compare the model
prediction with that future frame, not with the historical frame selected as
context. Historical-frame similarity is only an auxiliary, pose-matched
consistency measure.

## Build the first geometric manifest

```bash
export DATASET_ROOT=/path/to/dataset
export SPLIT_PATH=data_splits/recon/test
export MANIFEST_OUTPUT=artifacts/manifests/recon_revisit_geometry_v1.jsonl

python scripts/build_revisit_manifest.py \
  --data-root "$DATASET_ROOT" \
  --split "$SPLIT_PATH" \
  --output "$MANIFEST_OUTPUT" \
  --min-temporal-gap 8 \
  --position-threshold 0.75 \
  --heading-threshold-deg 20 \
  --heading-wrong-threshold-deg 90 \
  --required-future-steps 4 \
  --seed 0
```

Four future steps match the first one-second gate at the default 4 Hz input
rate. Use `--required-future-steps 64` before a full 16-second or rollout
evaluation. The inference dataset rejects queries that do not contain the
requested future horizon rather than silently shortening them.

`geometry_v1` provides pose/heading positives and geometric negatives only. For
paper examples, freeze landmark and scenario annotations before viewing model
results. Geometric proximity is not automatically task relevance.

## Phase A sampling rules

- Use only historical real observations from the same trajectory.
- Require the historical index to lie outside the native four-frame context.
- Select exactly one historical frame for replacement policies.
- Clear history between trajectories and batch items.
- Freeze revisit and non-revisit query lists before comparing model outputs.
- Record the query index, selected source index, pose distance, wrapped yaw
  difference, temporal gap, policy name, and seed for every prediction.
