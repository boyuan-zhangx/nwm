# LT-NWM hybrid memory implementation status

The current implementation is an experimental research scaffold, not a completed training/inference pipeline.

Implemented and tested:

- pose/action-aware external `MemoryBuffer` with retrieval diagnostics;
- explicit memory latents passed into `HybridCDiT`;
- checkpoint-compatible memory blocks with a zero-initialized gate;
- padding masks and CPU unit tests.

Not yet connected:

- memory-aware dataset batches in `train.py`;
- trajectory-local buffer orchestration in rollout inference;
- correct/random/wrong-memory evaluation runners.

Start at [docs/README.md](docs/README.md). Do not use `config/hybrid_l40s_inference_memory.yaml` for a long cluster job until the doctor warning about hybrid training data is cleared.
