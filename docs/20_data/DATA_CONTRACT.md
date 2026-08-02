# 数据契约

## NWM trajectory 最小结构

```text
DATA_ROOT/
└── trajectory_name/
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
    └── traj_data.pkl
```

`traj_data.pkl` 至少包含：

- `position`: `[T, >=2]`，全局位置；
- `yaw`: `[T]` 或 `[T,1]`，弧度；
- 图像编号与 pose 长度一致。

split 目录包含 `traj_names.txt`，每行一个 trajectory name。

## 验证命令

```bash
python scripts/validate_dataset.py \
  --data-root /data/recon \
  --split data_splits/recon/train \
  --max-trajectories 20
```

先抽查 20 条，再去掉 `--max-trajectories` 做全量验证。Pickle 只能来自可信项目数据。

## Revisit benchmark 还需要的标注

每个样本必须保存：trajectory、query frame、候选 memory frame、pose distance、heading difference、temporal gap 和 scenario type。正确 memory 的定义要在看模型结果前固定，例如：

- temporal gap 大于 native context；
- position distance 小于阈值；
- wrapped yaw difference 小于阈值；
- 属于同一 landmark/revisit event。

这些标签用于 Recall@K/mAP，不能使用模型自己的 retrieval score 作为 ground truth。

## 生成第一版几何 revisit manifest

```bash
python scripts/build_revisit_manifest.py \
  --data-root /data/recon \
  --split data_splits/recon/test \
  --output artifacts/manifests/recon_revisit_geometry_v1.jsonl \
  --min-temporal-gap 8 \
  --position-threshold 0.75 \
  --heading-threshold-deg 20 \
  --heading-wrong-threshold-deg 90 \
  --seed 0
```

`geometry_v1` 只给出 pose/heading positives 与几何 negatives。主论文案例还需要在看模型结果前冻结 landmark/scenario 标注；不要把几何 proximity 自动等同于 task relevance。
