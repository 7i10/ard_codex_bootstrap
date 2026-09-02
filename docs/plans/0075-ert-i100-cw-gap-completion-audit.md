# 0075 — I100 Clean-Wrong gap completion and canonical S2 audit

## Status

- Owner: Codex
- Status: in progress
- Scope: read-only feature replay and endpoint aggregation; no training or intervention
- Source parent: exact I100 epoch-99 parents for dev-1/dev-2

## Frozen design

Use the registered canonical q10 Student/Teacher state contract:

- Student `S1`: adversarial-correct and outside the lowest positive
  adversarial-margin q10 cohort.
- Student `S2`: adversarial-correct and in the lowest positive
  adversarial-margin q10 cohort.
- Student `S3`: adversarial-wrong.
- Teacher `T1/T2/T3`: the analogous positive-margin q10 partition, with
  `T3` adversarial-wrong.

All state labels are formed from epoch-99 pre-treatment rows.  Validation
non-CW harm is localized within fixed pre-treatment states and train/validation
feature distributions are compared without p-value hunting or threshold
tuning.  Existing e129/e149/e169/e189/e199 endpoint rows are reused.

## Execution

- [x] Reconcile HEAD, parent hashes, contracts, and existing endpoint inventory.
- [x] Recover or generate complete epoch-99 train feature rows by no-update replay.
- [x] Compute train-CW versus validation-CW state/feature shift and TPFM regimes.
- [x] Localize validation non-CW harm by canonical S1/S2/S3 and Teacher cells.
- [x] Compute temporal boundary trade-off and mechanism classifications.
- [x] Write machine artifacts and human report; close Clean-Wrong action exploration.
- [x] Run focused verification and commit the audit artifacts.

## Boundaries

No new training, intervention, threshold/coefficient tuning, seed, official test,
AutoAttack, or dynamic/History routing is allowed.  Missing endpoint rows remain
explicitly unavailable; no values are imputed.
