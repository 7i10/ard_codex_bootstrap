# 0043 — ERT history-smoothed S3 production screen

Status: in progress

## Frozen objective

Compare the same-step Instant route with Student-history smoothing and
two-correct-visit action persistence from exact Chen ERT epoch-79 parents:

```text
BASE, INST075, M3_075, M3E2_075
L2/seed1 and L4/seed2
epoch 79 -> 94; endpoint snapshots 84/89/94
```

No new selector, coefficient, seed, official test, or AutoAttack is allowed.

## Pre-GPU gate

- [x] Reconcile latest Git/source/artifact state.
- [x] Confirm the two Hamster GPUs are visible and idle.
- [ ] Correct offline replay to define S3 as
      `student_clean_correct AND NOT student_adv_correct`.
- [ ] Require full-window Majority-3 and compare Instant/M3/M3E2 without
      endpoint metrics.
- [ ] Stop if the corrected audit reverses the cross-seed recommendation.

## Training contract

- [ ] Verify exact epoch-79 parent/checkpoint/config hashes.
- [ ] Use KL-PGD10, pixel `[0,1]`, epsilon `8/255`, step `2/255`, random
      start, teacher-clean target.
- [ ] Use fixed `beta_advce=0.075`; no tuning.
- [ ] BASE/INST/M3/M3E2 use the same parent, optimizer/scheduler/RNG lineage.
- [ ] Ensure M3/M3E2 use current Student clean-correct and current Teacher
      adversarial-correct gates; clean-wrong and Teacher-wrong receive no
      AdvCE.
- [ ] Require 8 trajectories and epoch 84/89/94 checkpoints.

## Endpoint and report

- [ ] Run independent eval-mode CE-PGD20 on each arm/checkpoint for train and
      fixed internal validation only.
- [ ] Report paired rescue/harm/net effects, clean effects, transitions,
      Teacher-only versus Student-state switches, and class-stratified sample
      bootstrap CIs if the preregistered endpoint path is available.
- [ ] Write `docs/ERT_S3_HISTORY_PRODUCTION_RESULTS.md` and the machine report
      with source/config/parent/mask/attack/output hashes.
- [ ] Stop after the report; do not auto-promote Stage B or add Teacher
      smoothing.

## Acceptance / risks

The offline rule audit, runtime unit tests, clean-tree lineage checks, and
parent/resume parity must pass before launch. A mismatch in the corrected S3
semantics, parent state, attack identity, or shared baseline trajectory blocks
GPU execution. Two Hamster GPUs imply four sequential waves for eight runs;
there is no Ferret dependency for this screen.
