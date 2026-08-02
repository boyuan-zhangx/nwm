# ICRA 2027 研究计划

## 判断

计划是有条件可行，不是稳妥可行。官方投稿截止时间是 2026-09-15 23:59 PST，当前从 2026-08-02 起约有 44 天。这个时间只够完成一个窄主张：

> NWM 在 revisit-like navigation 中因短上下文产生长期空间不一致；外部 memory、正确 retrieval 与可学习的 selective fusion 能缓解该 failure mode，同时不显著破坏短期生成质量。

官方日期：https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/

不进入主线：换 backbone、完整 MPC、multi-sensor、online learning、object-centric learning、全面 SOTA。

## 必须闭合的四段证据

1. Failure existence：固定数据、checkpoint、seed，证明 revisit/turn-back/loop 中 baseline failure 可复现。
2. Mechanism validity：retrieval Recall@K 与可视化证明取到了正确历史，而不只是外观相似帧。
3. Causal evidence：correct memory 优于 no/random/temporally-wrong/visually-similar-wrong memory。
4. Non-regression：LPIPS、DreamSim 等旧指标没有明显恶化。

## 硬性 go/no-go 日期

| 日期 | 必须交付 | 失败后的动作 |
| --- | --- | --- |
| 08-06 | setup、doctor、dataset validator、baseline smoke 全通过 | 暂停模型开发，先修基础设施 |
| 08-12 | tiny subset 能 overfit；memory gate/attention 有非零梯度 | 放弃 full training，转 retrieval/benchmark 备线 |
| 08-20 | correct memory 明显优于 random/wrong memory | 不得宣称 memory mechanism 有效 |
| 08-30 | 主实验和三随机种子完成，原始输出冻结 | 只补关键缺口，不增加模块 |
| 09-05 | 全部表格、主图、定性视频定稿 | 冻结实验代码 |
| 09-09 | 第一版完整论文和 accompanying video | 只做复核与文字修改 |
| 09-15 | 投稿 | 不在最后 48 小时启动新训练 |

## 最小实验矩阵

| ID | 目的 | 组别 |
| --- | --- | --- |
| E00 | 工程闭环 | doctor, tests, one-batch baseline |
| E01 | failure census | NWM on turn-return, full rotation, revisit, loop |
| E02 | retrieval validity | pose-only, action-only, pose+action, random |
| E03 | learnability | tiny subset, frozen backbone, memory branch only |
| E04 | causal ablation | no/correct/random/temporal-wrong/visual-wrong memory |
| E05 | main result | baseline vs LT-NWM, same data/compute/seeds |
| E06 | minimal ablation | top-k and memory layer location；最多两项 |

先回答机制问题，再扩大训练。Cluster 只放大已经在 tiny experiment 中通过的实验。
