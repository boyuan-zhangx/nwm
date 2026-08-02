# ICRA 2027 Research Plan

## One-sentence paper

> Test whether replacing one stale item in a frozen Navigation World Model's
> four-frame context with one pose-aligned real historical observation improves
> long-horizon revisit prediction without harming ordinary prediction, and
> determine whether the effect transfers across NWM model scales.

This is a diagnostic benchmark plus a zero-training intervention. It is not a
new world-model architecture.

At the planning date of 2026-08-02, approximately 44 days remained before the
official submission deadline of 2026-09-15 at 23:59 PST.

Official deadline:
<https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/>

## Claim discipline

Before the first gate, phrase the hypothesis as a question:

> Can retrieval-based context replacement improve revisit consistency in a
> frozen NWM?

Only after positive held-out results may the paper claim:

> Pose-aligned context replacement improves revisit prediction in a frozen NWM
> with no additional training.

Do not claim restored spatial consistency before the experiment supports it.

## Frozen Phase A design

- Backbone and checkpoint: unchanged pretrained NWM.
- Native context size: four frames.
- Intervention: replace the oldest context frame with one historical frame.
- History: real observations only; clear it at every trajectory boundary.
- Retrieval size: top-1 only.
- Policies: `recent`, `random_history`, `pose_aligned`, `oracle_manifest`.
- Seeds: at least three matched diffusion seeds for paper results.
- Primary target: the ground-truth future at the revisit query.

## Model-scale strategy

Use only official pretrained checkpoints from the NWM family so the dataset,
action representation, VAE, output space, and evaluation contract remain
matched.

| Official scale | Repository model key | Parameters | Role in this paper |
| --- | --- | ---: | --- |
| CDiT/S | `CDiT-S/2` | 50M | Local development, oracle gate, complete ablation matrix, and failure analysis |
| CDiT/B | `CDiT-B/2` | 200M | Medium-scale confirmation of the core four policies |
| CDiT/XL | `CDiT-XL/2` | 1B | Final cluster confirmation of the core four policies only |

All checkpoints remain frozen. CDiT/L is optional and adds no required paper
claim. The official weight access conditions must be accepted before assigning
the download to an intern.

Do not introduce another world-model family merely because it is described as
small or state of the art. A different action space, latent target, training
dataset, or decoder would turn the project into a backbone-comparison paper and
invalidate the rapid shared evaluation contract.

CDiT/S is the development backbone, not the only scientific evidence. If the
gain appears only on CDiT/S and disappears on B/XL, report it as a capacity
effect rather than a scale-independent memory result.

Out of scope: `HybridCDiT` training, new cross-attention, learned retrievers,
top-k fusion, a new backbone, RL, MPC, online learning, self-generated memory,
object-centric memory, multi-sensor fusion, and a broad SOTA sweep.

## Required evidence chain

1. **Failure existence:** reproduce baseline failures on fixed revisit,
   turn-back, and loop queries using fixed data, checkpoint, and seeds.
2. **Capability upper bound:** oracle replacement must beat `recent` and
   `random_history`. If not, the frozen NWM cannot exploit this intervention.
3. **Retrieval validity:** `pose_aligned` must retrieve the intended historical
   observation and approach the oracle gain.
4. **Causal evidence:** pose-aligned history must beat random history under
   identical sampling seeds.
5. **Non-regression:** standard future-frame quality must not materially regress
   on pre-declared non-revisit queries.

## Hard go/no-go gates

| Date | Required deliverable | Action if the gate fails |
| --- | --- | --- |
| 2026-08-04 | Freeze 20-50 revisit queries; run CDiT/S with `recent`, `random_history`, and `oracle_manifest` | If oracle has no useful gain, stop the memory direction |
| 2026-08-07 | CDiT/S `pose_aligned` runs end to end and saves source indices and pose/yaw diagnostics | Repair retrieval only; do not add a learned module |
| 2026-08-12 | CDiT/S pose-aligned history beats random history on held-out revisits | Reframe as a negative diagnostic study or stop |
| 2026-08-16 | Core four-policy result transfers to CDiT/B | Treat an S-only result as a capacity-specific finding |
| 2026-08-20 | Three-seed revisit and non-revisit evaluation is complete; XL core run is scheduled | Add no architecture; fill only measurement gaps |
| 2026-08-30 | Main runs complete and raw outputs frozen | Add no new module |
| 2026-09-05 | Tables, main figures, and qualitative video frozen | Freeze experiment code |
| 2026-09-09 | Complete paper draft and accompanying video | Perform verification and writing edits only |
| 2026-09-15 | Submission | Start no new training in the final 48 hours |

## Minimum experiment matrix

| ID | Question | Required comparison |
| --- | --- | --- |
| E00 | Does the engineering loop work? | doctor, tests, manifest builder, baseline inference |
| E01 | Does history have an exploitable upper bound? | recent vs. random vs. oracle replacement |
| E02 | Does the simple retriever work? | pose-aligned vs. random vs. oracle |
| E03 | Is the effect stable? | three matched seeds on held-out revisit queries |
| E04 | Is the effect scale-specific? | core four policies on frozen S, B, and XL checkpoints |
| E05 | Does the intervention regress ordinary prediction? | recent vs. pose-aligned on non-revisit queries |
| E06 | When does it work? | temporal-gap, pose-distance, and yaw-difference bins |

Do not implement E02 before E01 proves that even oracle replacement helps on
CDiT/S. Do not run XL before S passes. The cluster scales a passed experiment;
it does not replace the oracle gate.

## Phase B decision

The existing hybrid branch is not deleted. Reconsider it only after Phase A is
positive and the residual error specifically motivates learned fusion. See
[`README_Hybrid_Memory.md`](../../README_Hybrid_Memory.md).
