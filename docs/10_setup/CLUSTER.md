# Cluster 环境与提交规范

## 首次安装

```bash
git clone https://github.com/boyuan-zhangx/nwm.git
cd nwm
bash setup_nwm_env.sh --profile nwm --backend cu124
source .venv-wsl/bin/activate
```

如果 cluster module 提供的 driver 不支持 cu124，先记录 `nvidia-smi`，再选择 `cu121`；不要使用旧 README 中未固定日期的 nightly wheel。

LT-NWM 不直接 import vendored `WorldMem/`。只有复现 upstream WorldMem Minecraft baseline 时才运行：

```bash
python worldmem_setup_and_test.py install --dry-run
python worldmem_setup_and_test.py install
```

## 每个 job 必须记录

```bash
git rev-parse HEAD > run_commit.txt
python -m pip freeze > environment.txt
nvidia-smi > nvidia_smi.txt
cp EXPERIMENT.yaml resolved_experiment.yaml
cp config/paths.local.yaml resolved_paths.yaml
```

不要把 token、W&B key 或个人目录提交进 Git。结果目录至少包含：`checkpoints/`、`logs/`、`metrics/`、`visualizations/`、`metadata/`。

## 提交前 gate

只有以下全部通过才允许申请长任务：

```bash
python scripts/navware.py doctor --profile nwm --config EXPERIMENT.yaml --paths-config config/paths.local.yaml
python scripts/navware.py smoke
python scripts/validate_dataset.py --data-root DATASET --split SPLIT --max-trajectories 20
```

Hybrid job 还必须先提交 tiny-overfit 的 loss curve、fixed sample 和 correct-vs-random 初步结果。
