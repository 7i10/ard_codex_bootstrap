# 0045 — ERT Clean-Wrong sample-wise rescue subtype analysis

Status: completed (descriptive analysis; no follow-up training)

## Objective

Use only the completed C0/C10/C12/C13 epoch-84 endpoints and a read-only
common CE-PGD20 feature replay to explain why clean recovery does not always
become robust recovery. No training, coefficient tuning, threshold selection,
official test, or AutoAttack is permitted.

## Frozen contract

- Chen ERT L2/seed1 and L4/seed2, exact epoch-79 Clean-Wrong masks.
- Outcome: fixed C0 versus C10, C12, and C13 epoch-84 train endpoints.
- Feature replay: C0 epoch-84 checkpoint, full train ordering, independent
  eval-mode CE-PGD20 with pixel `[0,1]`, Linf epsilon `8/255`, step `2/255`,
  20 steps, random start, hard-label CE. Only registered Clean-Wrong IDs are
  retained, but the full ordering is replayed to preserve the attack RNG
  contract.
- Baseline feature fields: Student clean/adv margin and true probability;
  Teacher clean/adv correctness, margin, true probability, and
  `DeltaT=mT_clean-mT_adv`.

## Group definitions

For each treatment arm, compare the same sample ID with C0:

- `clean_and_robust_rescue`: clean rescue and robust rescue;
- `clean_only_rescue`: clean rescue but not robust rescue;
- `robust_only_rescue`: robust rescue but not clean rescue;
- `neither_or_harm`: neither rescue, including treatment harm.

The groups are descriptive endpoint transitions, not causal claims about the
anchor features. C10/C13 clean-rescue and robust-rescue ID overlap is reported
with intersection, union, and Jaccard; it is not used to construct a new arm.

## Execution checklist

- [x] Reconcile clean `master` at `478b8a3` and verify broad-screen inputs.
- [x] Implement hash-bound C0 feature replay and subtype aggregator.
- [x] Run focused tests and one real sparse-ID smoke.
- [x] Run L2/L4 feature replay on Hamster GPUs.
- [x] Generate point report and immutable JSON/Markdown outputs.
- [x] Review joins, attack identity, and group denominators; then stop.

## Risks / interpretation limits

- The feature replay is an epoch-84 C0 observation, while the outcome is the
  paired C0-to-treatment transition at the same endpoint. It is an association
  analysis, not prospective prediction.
- Student/Teacher probabilities and margins come from the common feature
  replay; endpoint transition labels come only from previously saved endpoint
  artifacts. No outcome is used to alter the feature replay.
- If any checkpoint, attack identity, stable-ID universe, or source SHA does
  not match, the CLI must fail closed.

## Execution record

- Source commit for replay: `9109d3624100329e2da830598c808ad252ead568`.
- L2 feature replay: 8,623 rows; L4 feature replay: 8,925 rows.
- Both replays used the full train ordering and matched saved C0 Student
  clean/adv correctness and margins exactly (maximum absolute margin
  difference `0.0`).
- Feature/replay attack identity matched the 64 existing endpoint outputs:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- Report source hash: `9f999504ee9e474c93e11741775f72b43ccee6cd5523167da88d0b08bd96c9b9`.
- The clean-only versus robust-only contrast shows that Teacher adversarial
  correctness and positive adversarial margin, rather than `DeltaT` alone,
  distinguish robust rescue. C10/C13 robust-rescue Jaccard is 0.250 (L2) and
  0.181 (L4).
- No additional training or automatic intervention was started.
