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

## Validate before training

```bash
export DATASET_ROOT=/path/to/dataset
export SPLIT_PATH=data_splits/recon/train

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
- candidate memory frames;
- pose distance and wrapped heading difference;
- temporal gap;
- scenario or landmark identity;
- the labeling rule and its version.

Define a correct memory before inspecting model outputs. A geometric candidate
normally satisfies all of the following:

- its temporal gap exceeds the native context;
- its position distance is below a frozen threshold;
- its wrapped yaw difference is below a frozen threshold;
- it belongs to the same pre-annotated revisit event or landmark.

These independent labels support Recall@K and mAP. A model's own retrieval score
must never become its ground truth.

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
  --seed 0
```

`geometry_v1` provides pose/heading positives and geometric negatives only. For
paper examples, freeze landmark and scenario annotations before viewing model
results. Geometric proximity is not automatically task relevance.
