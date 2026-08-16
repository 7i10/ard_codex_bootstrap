# 0044 — ERT Clean-Wrong Broad Treatment Screen

Status: completed (screen only; human review required)

## Objective

Run a preregistered, two-seed, five-epoch screen of mechanisms for the fixed
epoch-79 Clean-Wrong cohort. The screen compares attack-budget reduction,
adversarial-KD pressure, clean-label recovery pressure, MART-inspired
adversarial CE, adaptive pressure, a fixed teacher-clean reliability gate, and
an IAD-inspired detached self-target. It is a mechanism screen, not a winner
promotion.

## Frozen scope

- Chen ERT L2/seed 1 and L4/seed 2; exact epoch-79 parent checkpoints.
- Fixed `student_clean_wrong` masks from the registered epoch-79 overlay.
- Epoch 80–84 continuation; independent CE-PGD20 endpoint at epoch 84 only.
- 16 arms (`C0`–`C15`) per seed, W&B `production`/`online` lineage.
- Training attack remains KL-PGD10, teacher-clean target, pixel `[0,1]`,
  epsilon `8/255`, step `2/255`, random start. Selected-sample epsilon
  changes are explicit mixed-budget treatment metadata; non-selected samples
  retain the baseline attack.
- No official test, AutoAttack, new seed, dynamic routing, coefficient sweep,
  or automatic follow-up.

## Implementation boundaries

- Extend the shared Trainer and PGD request with per-sample attack budgets and
  explicit Clean-Wrong treatment branches; do not duplicate a training loop.
- Freeze the C12 BCE coefficient with no-update gradient calibration before
  any GPU continuation.
- Add formula/gradient/empty-mask/mixed-epsilon/teacher-freeze tests and a
  one-batch canary before production.
- Record mask, parent, config, source, attack, calibration, W&B, checkpoint,
  endpoint and output hashes.

## Execution checklist

- [x] Reconcile HEAD, parent checkpoints, registered masks, and existing RSLAD
      contract.
- [x] Implement treatment API, mixed-budget attack, calibration, screen CLI,
      and report aggregation.
- [x] Focused unit/fixed-batch tests and `scripts/verify.py --changed`.
- [x] Commit clean scientific source before GPU launch.
- [x] Run one-batch engineering canary per unique treatment path.
- [x] Launch 32 trajectories (two seeds × 16 arms), then 64 independent
      endpoint evaluations (train/held-out validation).
- [x] Generate direct/spillover/held-out report and 2,000-replicate
      class-stratified paired bootstrap without using it to tune the screen.
- [x] Update human report and machine artifact; stop for human review.

## Acceptance and risks

Acceptance requires baseline/empty-mask numerical equivalence, unchanged
non-selected attack identity, no optimizer/state mutation during calibration,
teacher parameters with `grad is None`, fixed anchor masks, complete W&B
lineage, and independent endpoint attacks. The principal risks are accidental
attack-budget mixing, replacing rather than adding the selected CleanCE
branch, and interpreting direct train-cohort rescue as held-out improvement;
each is guarded by tests and separate report strata.

## Execution record

- Scientific source commit used for all valid runs: `cbe03a7b3be0b11fa1555b573c6f453a3d10f27b`.
- Valid runs: L2/seed1 (`ert-clean-wrong-broad-v1`) and L4/seed2
  (`ert-clean-wrong-broad-v1-l4r2`), 16 arms each, epoch 79→84.
- Endpoint: 64 independent eval-mode CE-PGD20 outputs (32 train + 32 fixed
  validation), with attack identity SHA `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- Fixed epoch-79 Clean-Wrong masks: L2 8,623 IDs; L4 8,925 IDs.
- C12 no-update BCE calibration was frozen before training:
  `beta_BCE=0.08891977369785309`, pooled target gradient ratio 0.25.
- Report command completed with 2,000 class-stratified paired bootstrap
  replicates per non-baseline cohort; result SHA is recorded in the machine
  artifact and human report.
- W&B online production tracking was enabled for the valid trajectories.
- One duplicate L4 C0 launch and an initial endpoint directory precreation
  mistake were stopped before valid evaluation; their outputs are excluded
  from the report. The valid L4 namespace was rerun from the same registered
  parent and is the only lineage included.
- No official test, AutoAttack, new seed, +15 continuation, or automatic
  winner promotion was performed.
