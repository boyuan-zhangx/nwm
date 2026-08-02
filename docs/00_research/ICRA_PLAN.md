# ICRA 2027 Research Plan

## Scope decision

The plan is feasible only with a narrow claim and hard go/no-go gates. At the
planning date of 2026-08-02, approximately 44 days remained before the official
submission deadline of 2026-09-15 at 23:59 PST.

Target claim:

> Short-context NWM produces long-horizon spatial inconsistencies in
> revisit-like navigation. External memory, correct retrieval, and learned
> selective fusion reduce this failure mode without materially degrading
> short-horizon generation quality.

Official deadline:
<https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/>

Out of scope for this submission: a new backbone, full MPC, multi-sensor
fusion, online learning, object-centric learning, and a broad SOTA sweep.

## Required evidence chain

1. **Failure existence:** reproduce baseline failures on fixed revisit,
   turn-back, and loop cases using fixed data, checkpoint, and seeds.
2. **Mechanism validity:** use Recall@K and retrieval visualizations to show
   that the method retrieves the relevant history, not only similar images.
3. **Causal evidence:** show that correct memory outperforms no memory,
   random memory, temporally wrong memory, and visually similar wrong memory.
4. **Non-regression:** show that standard LPIPS, DreamSim, and related quality
   metrics do not materially regress.

## Hard go/no-go gates

| Date | Required deliverable | Action if the gate fails |
| --- | --- | --- |
| 2026-08-06 | Setup, doctor, dataset validator, and baseline smoke all pass | Stop model work and repair infrastructure |
| 2026-08-12 | Tiny subset overfits; memory gate and attention receive finite non-zero gradients | Drop full training and switch to the retrieval/benchmark fallback |
| 2026-08-20 | Correct memory clearly outperforms random and wrong memory | Do not claim an effective memory mechanism |
| 2026-08-30 | Main runs and three seeds complete; raw outputs frozen | Fill only critical gaps; add no new module |
| 2026-09-05 | Tables, main figures, and qualitative video frozen | Freeze experiment code |
| 2026-09-09 | Complete paper draft and accompanying video | Perform verification and writing edits only |
| 2026-09-15 | Submission | Start no new training in the final 48 hours |

## Minimum experiment matrix

| ID | Question | Required comparison |
| --- | --- | --- |
| E00 | Does the engineering loop work? | doctor, tests, one-batch baseline |
| E01 | Where does baseline NWM fail? | turn-return, full rotation, revisit, loop |
| E02 | Is retrieval valid? | pose-only, action-only, pose+action, random |
| E03 | Is the memory branch learnable? | tiny subset, frozen backbone, memory branch only |
| E04 | Is the effect causal? | no/correct/random/temporal-wrong/visual-wrong memory |
| E05 | Does the complete method help? | baseline vs. LT-NWM with matched data, compute, and seeds |
| E06 | Which minimal design choice matters? | top-k and memory layer location; at most two axes |

Answer the mechanism question before scaling. The cluster amplifies only an
experiment that already passed its tiny local gate.
