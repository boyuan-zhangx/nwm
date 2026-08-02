# Evaluation

## Primary endpoint

Compare every generated revisit prediction with the ground-truth future frame
at that query. Do not use the retrieved historical frame as the primary target;
that can reward copying even when viewpoint, action, or appearance changed.

Use matched seeds and report LPIPS, DreamSim, and SSIM on the same frozen query
set. Historical-frame similarity may be reported only as a secondary measure
for pose/yaw-matched revisits.

## Metric families

**Non-regression metrics** measure the original generation capability: LPIPS,
DreamSim, SSIM, FID/FVD where appropriate, and any metric already used by the
baseline. The goal is to avoid a material regression, not to claim SOTA on
every generic metric.

**Revisit-specific metrics** test the proposed mechanism:

- top-1 retrieval accuracy against independent pose-heading-revisit labels;
- revisit prediction error on pre-declared revisit frames;
- oracle and pose-aligned causal gain over random history;
- failure rate for teleportation, landmark identity breaks, and mode collapse,
  measured by a frozen blind-review rule or independent estimator.

Never evaluate a retrieval score against labels derived from the same retrieval
score. That would be circular evidence.

## Fair comparison contract

All context policies must use identical:

- dataset and frozen query split;
- checkpoint;
- diffusion step count;
- random seeds;
- three most recent frames, differing only in the oldest slot;
- image size;
- compute budget.

Use at least three seeds for paper results. Preserve and report every seed
before reporting mean and standard deviation. Do not discard a run because its
result is unfavorable.

## Minimum main table

| Policy | LPIPS lower | DreamSim lower | SSIM higher | Top-1 valid higher | Failure rate lower |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recent` | | | | N/A | |
| `random_history` | | | | | |
| `oracle_manifest` | | | | 1.0 | |
| `pose_aligned` | | | | | |

Report the same generation metrics on a frozen non-revisit set. If oracle does
not show a stable advantage over recent and random history, stop: the frozen
model has not demonstrated that it can exploit context replacement. If oracle
helps but pose-aligned retrieval does not, diagnose retrieval rather than add a
new model.

## Scale reporting

Run the complete table and detailed failure bins on CDiT/S. For CDiT/B and
CDiT/XL, report the same core four rows using the same frozen queries and seeds.
Include a scale-by-policy summary so an S-only gain cannot be presented as a
general NWM result. Do not pool metrics across scales.
