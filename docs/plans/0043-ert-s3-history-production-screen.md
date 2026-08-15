# 0043 — ERT history-smoothed S3 production screen

Status: completed; evidence report written and no follow-up run launched.

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
- [x] Correct offline replay to define S3 as
      `student_clean_correct AND NOT student_adv_correct`.
- [x] Require full-window Majority-3 and compare Instant/M3/M3E2 without
      endpoint metrics.
- [x] Corrected audit retained the cross-seed recommendation.

## Training contract

- [x] Verify exact epoch-79 parent/checkpoint/config hashes.
- [x] Use KL-PGD10, pixel `[0,1]`, epsilon `8/255`, step `2/255`, random
      start, teacher-clean target.
- [x] Use fixed `beta_advce=0.075`; no tuning.
- [x] BASE/INST/M3/M3E2 use the same parent, optimizer/scheduler/RNG lineage.
- [x] Ensure M3/M3E2 use current Student clean-correct and current Teacher
      adversarial-correct gates; clean-wrong and Teacher-wrong receive no
      AdvCE.
- [x] Require 8 trajectories and epoch 84/89/94 checkpoints.

## Endpoint and report

- [x] Run independent eval-mode CE-PGD20 on each arm/checkpoint for train and
      fixed internal validation only.
- [x] Report paired rescue/harm/net effects, clean effects, transitions, and
      Teacher-only versus Student-state switches. Class-stratified bootstrap
      CIs were not run; the fixed validation point estimates were sufficient
      for this screen and are not seed-uncertainty estimates.
- [x] Write `docs/ERT_S3_HISTORY_PRODUCTION_RESULTS.md` and the machine report
      with source/config/parent/mask/attack/output hashes.
- [x] Stop after the report; do not auto-promote Stage B or add Teacher
      smoothing.

## Acceptance / risks

The offline rule audit, runtime unit tests, clean-tree lineage checks, and
parent/resume parity passed before the accepted v2 endpoint launch. The first
non-deterministic attempt was rejected because BASE/M3/M3E2 epoch-80 hashes did
not match; the runtime was corrected to apply deterministic CUDA flags and the
screen was rerun. The accepted run used two Hamster GPUs in four sequential
arm waves; there was no Ferret dependency.
