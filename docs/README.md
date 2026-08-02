# NavWare / LT-NWM 实验手册

这组文档的目标是让实习生只修改路径配置和实验 YAML，不修改 Python 源码即可执行已经就绪的流程。

## 当前状态

| 模块 | 状态 | 可执行入口 |
| --- | --- | --- |
| Python 3.10 环境与依赖检查 | 可用 | `python scripts/navware.py doctor` |
| Memory retrieval 单测 | 可用 | `python scripts/navware.py smoke` |
| Dataset contract 检查 | 可用 | `python scripts/validate_dataset.py` |
| Baseline NWM 训练/推理 | 原代码入口可用，仍需要数据与 checkpoint | `scripts/train.sh`, `scripts/infer.sh` |
| LT-NWM tensor flow 与 baseline checkpoint compatibility | CPU 单测可用 | `tests/test_hybrid_models.py` |
| LT-NWM memory-aware training batches | 尚未接入 | 不得提交大训练 |
| LT-NWM rollout memory lifecycle | 尚未接入 | 不得宣称 inference-only 结果 |

`doctor` 出现 `config:hybrid-training-data` 警告时，表示 hybrid 配置还没有拿到真实 memory batch。不要通过改 YAML 绕过这个警告。

## 阅读顺序

1. [ICRA 研究计划](00_research/ICRA_PLAN.md)
2. [本地 WSL 环境](10_setup/LOCAL_WSL.md)
3. [Cluster 环境](10_setup/CLUSTER.md)
4. [数据契约](20_data/DATA_CONTRACT.md)
5. [训练](30_training/TRAINING.md)
6. [推理](40_inference/INFERENCE.md)
7. [评估](50_evaluation/EVALUATION.md)
8. [实习生交接清单](60_handoff/INTERN_CHECKLIST.md)

## 一条原则

任何论文数字必须能追溯到：Git commit、完整 YAML、path overlay、checkpoint、随机种子、环境快照和原始输出目录。手工改 Python 后得到的结果不进入论文表格。
