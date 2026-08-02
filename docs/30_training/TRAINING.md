# 训练

## Baseline

配置好 `config/paths.local.yaml` 后：

```bash
bash scripts/train.sh \
  config/nwm_cdit_xl.yaml \
  config/paths.local.yaml \
  --epochs 1 \
  --ckpt-every 2000 \
  --eval-every 10000 \
  --bfloat16 1 \
  --torch-compile 0
```

本地 smoke 使用 S/B model、较小 image size、`num_workers: 0` 和 tiny split。XL 不用于验证数据管线。

## LT-NWM 当前边界

`HybridCDiT` 已做到：

- memory buffer 与 diffusion denoiser 生命周期分离；
- memory 以显式 `memory_latents [B,M,C,H,W]` 输入；
- memory attention 接受 padding mask；
- baseline checkpoint 可 `strict=False` 加载，missing keys 仅为 memory branch；
- memory gate 零初始化，初始输出与 baseline 一致。

尚未做到：`TrainingDataset` 产生历史 memory candidates、pose 和 mask，`train.py` 把它们送入 diffusion `model_kwargs`。因此现在运行 hybrid YAML 不能训练 memory branch。

## Tiny-subset overfit 验收

数据：单场景 8-32 条包含 revisit 的轨迹。训练：冻结 VAE 和大部分 CDiT，只打开 memory attention/gate，固定 seed 与 fixed visualization samples。

通过条件：

1. train loss 明显下降；
2. memory gate 与 attention 参数有非零、有限梯度；
3. fixed train sample 的 revisit 预测改善；
4. correct memory 优于 random memory；
5. no-memory 输出在初始化时与 baseline 数值一致。

未通过不得扩大 batch、模型或节点数。
