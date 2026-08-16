# 0046 — Clean-Wrong Teacher reliability proxy and safety analysis

Status: completed (read-only proxy/safety analysis; no follow-up training)

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
- [x] Run focused tests and lint.
- [x] Run L2/L4 KL replay on Hamster GPUs.
- [x] Generate JSON/Markdown report and verify metric identities.
- [x] Stop without new training or automatic route selection.

## Acceptance

- `accuracy_delta = rescue_rate - harm_rate` is asserted.
- `margin_delta` is a separate paired margin quantity.
- KL/CE Pearson, Spearman, sign/correctness agreement, confusion counts, and
  pre-treatment quintile effects are hash-bound to the exact inputs.
- C10 safety interpretation is based on monotonicity and cross-seed direction,
  not a single best bin.

## Execution record

- KL-PGD10 replay completed on Hamster GPUs for L2/L4, using the exact epoch-79
  parents and fixed Clean-Wrong masks. Rows: L2 `8,623`, L4 `8,925`.
- CE20/KL10 Teacher margin agreement: L2 Pearson `0.9374`, Spearman `0.9203`,
  sign/correctness agreement `0.9114`; L4 Pearson `0.9355`, Spearman `0.9149`,
  agreement `0.9095`.
- C10 CE20 quintile robust accuracy deltas increase from Q1 to Q5 in both
  seeds: L2 `+0.12, +0.12, +0.93, +1.51, +3.88` percentage points; L4
  `-0.17, -0.11, +0.39, +0.78, +2.07` points. Robust harm does not decrease
  monotonically; it also rises in the upper bins, so this is benefit/net-rescue
  evidence rather than a pure safety selector.
- The earlier reliability report had mislabeled margin deltas as accuracy
  deltas. `_effect` now stores separate `accuracy_delta` and `margin_delta`,
  with `accuracy_delta = rescue_rate - harm_rate`; the corrected report is
  regenerated.
- Machine report:
  `docs/experiments/ert_clean_wrong_reliability_proxy_safety_v1.json`, content
  hash `05a288ad38b0d71181b166360aab9e2a0f76fb99d596525c638e60284077e82d`.
- No new training, threshold tuning, route selection, official test, or
  AutoAttack was started.
