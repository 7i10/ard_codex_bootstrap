# ERT Dynamic S3 recovery experiment

## Status

- Owner: ARD lead
- Base: `3ce3cbe` (ERT confirmatory T1/T2/T3 result)
- Current milestone: M4 report complete; no automatic follow-up
- Date: 2026-08-15

## Objective

Test whether the harm seen with the fixed epoch-79 S3 treatment was caused by
continuing AdvCE after a sample had recovered. The experiment compares normal
RSLAD, an epoch-80-captured fixed action, and same-step current-state routing.

## Frozen scientific contract

For the current visit, the active action is exactly:

```text
student clean correct AND student adversarial wrong AND teacher adversarial correct
```

Active samples receive baseline RSLAD plus `0.075 * CE(student_adv, label)`.
All other samples receive baseline RSLAD. S1/S2 are both baseline; T3 and
clean-wrong are baseline. No hysteresis, dwell time, threshold fitting, or
one-epoch lag is allowed.

The student-crafted inner attack is unchanged KL-PGD10 in pixel `[0,1]` with
`epsilon=8/255`, step `2/255`, random start, and teacher-clean target. Every
endpoint is independent eval-mode CE-PGD20 with the same budget as the prior
confirmatory report. The exact pre-update state from the current attack is
used; no post-update reattack is allowed.

## Arms and lineage

For each exact Chen ERT epoch-79 parent (L2/seed 1 and L4/seed 2):

- `DYNBASE`: observe and log the candidate action, but always apply baseline;
- `S3FIX075`: use the epoch-80 candidate action for epoch 80 and freeze it;
- `S3DYN075`: recompute the candidate action at every visit.

Each seed continues epochs 80--94 and saves 84, 89, and 94. The fixed and
dynamic arms must receive identical actions and produce identical model,
optimizer, scheduler, and RNG state at the end of epoch 80. A mismatch stops
the campaign before production continuation is accepted.

## Required observations

Every valid train visit records stable ID, class, epoch, student clean/adv
correctness and probability margins, teacher clean/adv correctness and
probability margins, `DeltaS`, `DeltaT`, candidate action, applied action, and
capture action. The epoch-80 capture must contain each train ID exactly once,
with no duplicate or missing IDs; its ID/class hash and action mask are
immutable. Dynamic summaries include active fraction, entries, exits,
re-entries, action switches, consecutive active runs, transitions, and
recovery/relapse rates.

## Acceptance and stop criteria

- Focused truth-table, timing, no-lag, capture, fixed-immutability,
  full-batch, teacher-freeze, and no-special-route tests pass.
- One real one-epoch canary exercises active, exit, re-entry, T3, and
  clean-wrong paths without changing the coefficient.
- All six runs are W&B-online with immutable parent/config/mask/source hashes.
- All 36 endpoint jobs (3 arms × 2 seeds × 3 horizons × 2 splits) complete
  under independent CE-PGD20.
- No official test, AutoAttack, new seed, or automatic follow-up is run.

## Milestones

- [x] M0: reconcile repo, freeze config and plan.
- [x] M1: implement router, capture/state Parquet, runtime wiring, and tests.
- [x] M2: focused verification, shared-prefix canary, and no-update sanity.
- [x] M3: six continuations and independent CE-PGD20 endpoints.
- [x] M4: transition/effect report, hashes, review, and one cohesive commit.

## Risks and mitigations

- BN mutation during routing: use an eval/no-grad diagnostic clean forward with
  mode restoration; do not reuse a train-mode forward if it changes semantics.
- Capture or DDP padding errors: validate exact stable-ID/class coverage and
  exclude invalid padding rows.
- Fixed/dynamic confounding: compare epoch-80 checkpoint tensor and optimizer
  state hashes before accepting later horizons.
- Current-active cohort selection is descriptive only; no causal claim is made
  from post-treatment dynamic cohorts.

## Progress log

- 2026-08-15: Read the frozen Dynamic S3 Recovery specification and confirmed
  that existing Stage-A fixed masks cannot implement the same-step current
  state contract without a new router. Implementation is in progress; no GPU
  run has started.
- 2026-08-15: Fixed review P1s before GPU use: exact training/endpoint attack
  identity and parent binding, epoch-80 full-state peer gate, Trainer
  same-step/no-reattack/teacher-freeze/BN parity regression, and initial-active
  re-entry accounting. Focused tests and changed-path gate pass; scientific
  fix-delta review reports no P0/P1. No GPU run has started.
- 2026-08-15: The first real L2 independent-prefix attempt reached epoch 80
  and failed closed as designed: FIXED/DYNAMIC candidate counts were 9894 and
  9796, with nonmatching model/RNG state. This demonstrated that separate
  epoch-79-to-80 processes cannot establish parity under attack/data-order
  randomness. The implementation now uses one `S3CAP075` epoch-80 capture
  prefix and resumes both children from that exact checkpoint; the legacy
  peer-gate launch is rejected.
- 2026-08-15: Added fail-closed shared-prefix lineage validation. A child now
  requires the prefix checkpoint to attest the exact requested seed's epoch-79
  parent SHA, parent config hash, and prefix child config hash; a cross-seed
  prefix is rejected before training. Sixteen focused tests and the changed
  gate pass.
- 2026-08-15: The first real L2 shared-prefix capture completed at epoch 80
  (9,786 selected IDs). FIXED and DYNAMIC resumed from that exact checkpoint
  and both reached epochs 84/89/94 with exit code 0. The same procedure for L4
  completed at epoch 80 and both children reached all three horizons with exit
  code 0. The required DYNBASE controls then completed for both seeds.
- 2026-08-15: All 36 independent CE-PGD20 endpoint jobs completed with exit
  code 0. The first endpoint invocation was rejected by the schema-v1
  analysis config before any output was written; rerunning with each seed's
  schema-v2 parent config succeeded and preserved the frozen attack identity.
- 2026-08-15: Dynamic report generated at
  `.cache/analysis/ert-dynamic-s3-recovery-v1/dynamic-s3-report.json`.
  Report SHA-256 is
  `be23d725d6568ed65bacba4c02dd9631e2d509f067d7ce8870b90b9f07600cb4`.
  The report is descriptive and does not select a winner or launch a follow-up.

## Completion record

Completion details are recorded in
`docs/ERT_DYNAMIC_S3_RECOVERY_RESULTS.md` and the machine-readable summary
`docs/experiments/ert_dynamic_s3_recovery_v1.json`. The source SHA was
`95e72cd1a2a32caa21686ad76d318eab33e1807a`; config SHA was
`39691b9559d6df20baf09c9c33b7f63b8c37dd3e2937309e2aa1899cd26b8660`; the
endpoint attack identity SHA was
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
All endpoint rows were 45,000 train or 5,000 validation IDs with the same
class-universe hashes. No route was promoted automatically; official test,
AutoAttack, new seeds, and Stage-B continuation remain intentionally blocked.
