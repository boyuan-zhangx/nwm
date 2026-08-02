# Evaluation

## Metric families

**Non-regression metrics** measure the original generation capability: LPIPS,
DreamSim, FID/FVD, and any metric already used by the baseline. The goal is to
avoid a material regression, not to claim SOTA on every generic metric.

**Memory-specific metrics** test the proposed mechanism:

- Retrieval Recall@K and mAP against independent pose-heading-revisit labels.
- Revisit prediction error, measured only on pre-declared revisit frames.
- Correct-memory causal gain, for example
  `metric(wrong_or_random) - metric(correct)` when lower is better.
- Failure rate for teleportation, landmark identity breaks, and mode collapse,
  measured by a frozen blind-review rule or independent estimator.

Never evaluate a retrieval score against labels derived from the same retrieval
score. That would be circular evidence.

## Fair comparison contract

Baseline and LT-NWM must use identical:

- dataset split;
- checkpoint initialization;
- diffusion step count;
- random seeds;
- input context;
- image size;
- compute budget.

Use at least three seeds. Preserve and report every seed before reporting mean
and standard deviation. Do not discard a run because its result is unfavorable.

## Minimum main table

| Method | LPIPS lower | DreamSim lower | Recall@1 higher | Recall@5 higher | Revisit error lower | Failure rate lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NWM | | | N/A | N/A | | |
| LT-NWM, no memory | | | | | | |
| LT-NWM, random memory | | | | | | |
| LT-NWM, correct memory | | | | | | |

If correct memory does not show a stable advantage over wrong and random memory,
the improvement cannot be attributed to retrieval or external memory.
