# 0047 — ERT Clean-Wrong reliability-gated CleanCE confirmatory intervention

## Frozen objective

Test a fixed epoch-79 Teacher-adversarial-reliability gate for the existing
`+0.15 CleanCE` treatment.  No threshold, coefficient, selector, or route is
changed after endpoint inspection.  The only training arms are fresh
`G0_BASE`, `G1_CW_ALL_CE015`, `G2_CW_R_CE20_CE015`, and
`G3_CW_R_KL10_CE015` for Chen L2/L4, continued to epoch 94 and evaluated at
84/89/94 with independent CE-PGD20.

## Checklist

- [x] Reconcile current HEAD, parent checkpoints, masks, and replay artifacts.
- [x] Implement hash-bound CE20/KL10 selector overlay preparation.
- [x] Add focused selector contract test.
- [x] Set W&B artifact retention to metrics/lineage by default; keep all
  checkpoints and run-bundle files local with content hashes.
- [x] Commit clean scientific source before GPU launch (`1039ff3`).
- [x] Prepare and audit L2/L4 selector counts and RR/RU/UR/UU groups.
- [x] Run 8 online-tracked continuations from exact epoch-79 parents.
- [x] Evaluate all 24 checkpoints (train and fixed validation) with CE-PGD20.
- [x] Aggregate paired direct/excluded-CW/non-CW/held-out effects.
- [x] Write result JSON/Markdown and perform one scientific contract review.

## Result record

All 8 trajectories and 48 endpoint artifacts completed at Git SHA
`8544fed4505d423cefe6e89ad789f45c52488aac`. The endpoint report is in
`docs/ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md` and the hash-bound machine
record is `docs/experiments/ert_cw_reliability_gated_ce015_v1.json`.
Gating was not confirmed: G2 was positive versus G0 at epoch 94 only for L2
(`+0.44 pp` robust validation accuracy) and negative for L4 (`-0.56 pp`);
G3 was negative for both seeds (`-1.06/-1.14 pp`). No follow-up training,
threshold tuning, official-test evaluation, or AutoAttack was launched.

## Frozen lineage and contracts

Parents and Clean-Wrong masks are the exact hashes stated in the user
protocol.  Selectors are `teacher_adv_margin > 0` at epoch 79; CE20 uses
hard-label eval PGD20, KL10 uses the teacher-clean training PGD10 replay.
Training remains baseline KL-PGD10 and only selected samples receive an
additional `0.15 * CE(clean)` under full-batch mean reduction.  W&B remains
online through the production parent config, but the default retention policy
publishes metrics, lineage, and small analysis artifacts only; checkpoints and
the run bundle remain local and hash-bound. Official test and AutoAttack are
excluded.

## Risks / completion

The existing Stage-A runtime is shared; no duplicated trainer is introduced.
The experiment is not scientifically complete until all eight runs and their
independent endpoints exist with matching checkpoint, mask, attack, and Git
lineage.  No automatic promotion follows the result.
