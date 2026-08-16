# 0046 — Clean-Wrong Teacher reliability proxy and safety analysis

Status: in progress

## Objective

Audit the existing reliability metric semantics, replay exact epoch-79
Teacher reliability under the training KL-PGD10 attack, compare it with the
existing CE-PGD20 oracle replay, and test whether reliability is a safety
modifier rather than a robust-benefit selector. This is read-only: no new
training, threshold tuning, route selection, official test, or AutoAttack.

## Frozen inputs

- Chen ERT L2/seed 1 and L4/seed 2.
- Exact Clean-Wrong epoch-79 masks and C0/C10/C12/C13 epoch-84 endpoints.
- KL-PGD10: teacher-clean target, pixel `[0,1]`, Linf `8/255`, step `2/255`,
  random start.
- Existing CE-PGD20 replay is the oracle diagnostic.
- Reliability split is descriptive (`mT_adv > 0`), and quintile boundaries
  use pre-treatment features only.

## Execution checklist

- [x] Reconcile endpoint, mask, parent, and CE attack lineage.
- [x] Add explicit accuracy/margin effect semantics and regression.
- [x] Implement KL-PGD10 feature replay and proxy/safety report.
- [ ] Run focused tests and lint.
- [ ] Run L2/L4 KL replay on Hamster GPUs.
- [ ] Generate JSON/Markdown report and verify metric identities.
- [ ] Stop without new training or automatic route selection.

## Acceptance

- `accuracy_delta = rescue_rate - harm_rate` is asserted.
- `margin_delta` is a separate paired margin quantity.
- KL/CE Pearson, Spearman, sign/correctness agreement, confusion counts, and
  pre-treatment quintile effects are hash-bound to the exact inputs.
- C10 safety interpretation is based on monotonicity and cross-seed direction,
  not a single best bin.
