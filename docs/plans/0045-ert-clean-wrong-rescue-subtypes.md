# 0045 — ERT Clean-Wrong sample-wise rescue subtype analysis

Status: completed (pre-treatment reliability stratification; no follow-up training)

## Objective

Use only the completed C0/C10/C12/C13 epoch-84 endpoints and a read-only
common CE-PGD20 feature replay to explain why clean recovery does not always
become robust recovery. No training, coefficient tuning, threshold selection,
official test, or AutoAttack is permitted.

## Frozen contract

- Chen ERT L2/seed1 and L4/seed2, exact epoch-79 Clean-Wrong masks.
- Outcome: fixed C0 versus C10, C12, and C13 epoch-84 train endpoints.
- Feature replay: C0 epoch-79 parent checkpoint, full train ordering, independent
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

- The historical subtype block uses an epoch-84 C0 observation, while the
  reliability follow-up uses an epoch-79 pre-treatment observation. Both are
  association analyses, not prospective prediction or causal estimates.
- Student/Teacher probabilities and margins come from the common feature
  replay; endpoint transition labels come only from previously saved endpoint
  artifacts. No outcome is used to alter the feature replay.
- If any checkpoint, attack identity, stable-ID universe, or source SHA does
  not match, the CLI must fail closed.

## Execution record

- Source commit for the historical endpoint-conditioned replay:
  `9109d3624100329e2da830598c808ad252ead568`.
- L2 feature replay: 8,623 rows; L4 feature replay: 8,925 rows.
- Both replays used the full train ordering and matched saved C0 Student
  clean/adv correctness and margins exactly (maximum absolute margin
  difference `0.0`).
- Feature/replay attack identity matched the 64 existing endpoint outputs:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- The initial endpoint-conditioned report is retained as the historical
  subtype analysis; its feature replay used the older epoch-84 observation
  contract and is not used for the pre-treatment selector claim.
- The clean-only versus robust-only contrast shows that Teacher adversarial
  correctness and positive adversarial margin, rather than `DeltaT` alone,
  distinguish robust rescue. C10/C13 robust-rescue Jaccard is 0.250 (L2) and
  0.181 (L4).
- No additional training or automatic intervention was started.

## Reliability-stratified follow-up

- Epoch-79 parent replay was run for both seeds using the same full-train
  ordering and CE-PGD20 identity as the epoch-84 endpoint.
- CW-R was fixed as `mT_adv > 0`; CW-U as `mT_adv <= 0`. No outcome-derived
  threshold or tuning was used.
- L2/L4 feature rows exactly matched the registered sparse masks and endpoint
  attack identity. A lineage audit found that the first L4 replay used a
  different epoch-79 checkpoint than the broad-screen fork; it was discarded.
  L4 was rerun with the exact fork parent `026a36d3…`, and the report now
  fails closed on this mismatch. L2 was also rerun from the exact fork parent
  `ad43d72d…` under the final clean source.
- The current machine report is
  `docs/experiments/ert_clean_wrong_reliability_stratified_v1.json`; its final
  content hash is
  `c01b5090efe2fd7701452c109817bc9b2a7bb8c80e164bfbe7ecc11c36065d02`.
- The previous report's `delta_accuracy` field was found to contain margin
  deltas. It was corrected to separate `accuracy_delta` and `margin_delta`,
  so the reliability rows in the machine and Markdown reports were
  regenerated before the proxy/safety follow-up.
- Final reliability replay provenance is L2 source
  `5196df3d4618d7e9183e14e4a9a40a462f9fef17` with parent
  `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`, and
  L4 source `4a81f40f2c1265d966baac26f08b167949d8a5db` with parent
  `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`.
- C13 had nearly equal L2 robust deltas in CW-R/CW-U and was neutral in both
  L4 strata; the proposed Teacher-unreliable contamination explanation is not
  confirmed. C10 showed higher robust net-rescue in CW-R but no consistently
  larger robust accuracy delta.
