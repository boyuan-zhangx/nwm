# 推理

## Baseline 入口

```bash
bash scripts/infer.sh \
  config/nwm_cdit_xl.yaml \
  config/paths.local.yaml \
  --output_dir /results/nwm \
  --ckp 0100000 \
  --datasets recon \
  --eval_type rollout \
  --rollout_fps_values 1,4 \
  --batch_size 1 \
  --num_workers 0
```

## LT-NWM 正确 lifecycle

Memory buffer 不能在 `HybridCDiT.forward` 内更新。一次视频 frame 会调用 denoiser 数百次，在 forward 内更新会把 diffusion steps 当成历史视频帧。

每个 rollout step 必须按以下顺序执行：

1. 根据 current global pose、target action 查询一次 buffer；
2. 得到 top-k VAE latents、frame indices、retrieval scores；
3. 同一组 memory latents 供该 frame 的全部 diffusion steps 使用；
4. 完成 decode 后，只把一个真实 observation 或最终预测 latent 写入 buffer；
5. trajectory 结束立即 clear，禁止 batch/trajectory 间泄漏。

## 必须支持的推理组

- `no_memory`
- `correct_memory`
- `random_memory`
- `temporal_wrong_memory`
- `visual_similar_wrong_memory`

每次推理保存 query、top-k frames、indices、各 score component、memory mask、gate statistics、prediction 和 seed。没有这些 diagnostics 的视频不能用于机制结论。

五组的冻结规范位于 `experiments/e04_memory_causal_ablation.yaml`。当前状态是 `specification`；只有 runner 真正消费并验证全部字段后才能改成 runnable/executed。
