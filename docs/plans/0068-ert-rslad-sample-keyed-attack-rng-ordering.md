# ERT / RSLAD Sample-Keyed Attack RNG and First History Ordering

## Status

- Owner: Codex root
- Base SHA: `b4757cbe86a38402bc9f33e5d7186493fbd57c05`
- Current milestone: pre-production source freeze and detached campaign launch

## Goal

Introduce a reviewed sample-keyed training-PGD random-start contract, prove
order/batch/rank invariance, and then run the first margin-history ordering
intervention from the exact I100 epoch-99 parents on dev-1/dev-2 only.

## Non-goals

No changes to loss, teacher, augmentation, optimizer, scheduler, attack
budget/steps, exposure count, confirmation seeds, official test, AutoAttack,
additional ordering policies, or ImageNet scientific training.

## Frozen scientific identity

- I100: CropShift epochs 0--99, IDBH_WEAK epochs 100--199.
- Minimal history predictor: H2, `margin_ema` only, frozen dev Ridge alpha 1.
- Ordering: HIGH/MID/LOW = 20/60/20 by low margin EMA risk, interleaving
  HIGH/MID/MID/LOW/MID, every train sample exactly once per epoch.
- New training attack randomness: key only by attack seed, epoch, source ID,
  stream tag, and restart index; never by batch position, global step, rank,
  worker, or GPU.

## Milestones

- [x] M0 reconcile and freeze plan/manifest
- [x] M1 implement reference and optimized sample-keyed random starts
- [x] M2 pass invariance/scalability tests and rank-equivalence gate
- [ ] M3 freeze source and run parent/control/history canaries
- [ ] M4 launch four suffixes via detached marker DAG
- [ ] M5 chain endpoints, aggregate, and report
- [ ] M6 commit/push and stop before confirmation seeds

## Fail-closed gates

- Any invariance failure blocks production.
- Any unexplained control drift is reported separately from ordering effect.
- Exact epoch-99 parent hashes must remain
  `360910a8...` (dev-1) and `bb0c7c1a...` (dev-2).
- Training attack identity is intentionally new; historical I100 is diagnostic,
  not the primary control.

## Progress log

- 2026-08-31: previous audit showed batch-position coupling. Exact I100
  parents remain available and hash-verified. New contract prompt accepted for
  implementation review; no GPU production launched yet.
- 2026-08-31: sample-keyed contract, history-balanced sampler, rank gate, and
  invariance tests implemented. The bounded CUDA benchmark reports >5%
  random-start-only overhead; this is retained as a documented limitation.
