# 实习生交接清单

## 第一天

1. 阅读 `docs/README.md` 和本人任务对应章节。
2. 激活 Python 3.10 环境：Linux/WSL 使用 `.venv-wsl`，Windows CPU 检查使用 `.venv`。
3. 运行 `python scripts/navware.py doctor --profile nwm`。
4. 运行 `python scripts/navware.py smoke`，必须全部通过。
5. 创建自己的 `config/paths.local.yaml`，只改路径，不改 Python。
6. 对分配的数据运行 dataset validator。
7. 先跑 baseline tiny smoke，再接触 hybrid 或 cluster。

## 可并行任务包

### A. Data / benchmark

输出 revisit event manifest、正确/错误 memory labels、dataset validation report。不得根据模型输出反向挑选阈值。

### B. Retrieval

输出 Recall@K/mAP、top-k 可视化、pose-only/action-only/combined/random 对照。只修改独立 retrieval 模块和 config，并补单测。

### C. Training

接入 memory batch、mask 和冻结策略；先 tiny overfit。输出 loss、gradient statistics、fixed samples、完整 YAML。

### D. Evaluation

实现 non-regression 与 revisit-only 统计；对照组目录结构一致，汇总脚本不能手工填数字。

## 每次提交必须包含

- 一句话 scientific question；
- 修改过的 config；
- 可复制命令；
- 自动测试；
- 输出路径和 commit hash；
- 成功和失败案例；
- 明确写出仍未回答的问题。

禁止：硬编码个人路径、在 Python 中切换实验组、覆盖原 checkpoint、跨 trajectory 复用 memory、只汇报最好 seed、把尚未接通的功能写成已验证结果。
