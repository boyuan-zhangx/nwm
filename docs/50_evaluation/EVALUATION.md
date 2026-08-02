# 评估

## 两类指标

Non-regression：LPIPS、DreamSim、FID/FVD 等原始生成质量指标。目标是不明显破坏旧能力，不要求所有指标都刷新 SOTA。

Memory-specific：

- Retrieval Recall@K / mAP：相对独立的 pose-heading-revisit 标签；
- Revisit prediction error：只在预先定义的 revisit frames 上计算 LPIPS/DreamSim；
- Correct-memory causal gain：`metric(wrong/random) - metric(correct)`；
- Failure rate：teleportation、landmark identity break、mode collapse，由盲评规则或独立 estimator 判定。

不要用内部 retrieval score 评价 retrieval score，这会形成循环论证。

## 公平比较

Baseline 与 LT-NWM 必须使用相同 dataset split、checkpoint initialization、diffusion steps、seed、输入 context、image size 和算力预算。至少三个 seed；先报告每个 seed，再报告 mean ± std。

## 最小主表

| Method | LPIPS ↓ | DreamSim ↓ | Recall@1 ↑ | Recall@5 ↑ | Revisit error ↓ | Failure rate ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NWM | | | N/A | N/A | | |
| LT-NWM no memory | | | | | | |
| LT-NWM random memory | | | | | | |
| LT-NWM correct memory | | | | | | |

若 correct 与 wrong/random 没有稳定差异，就不能把 improvement 归因于 retrieval 或 memory。
