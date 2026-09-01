# ERT / RSLAD attack random-start randomness characterization

## Status

- Owner: Codex `/root`
- Branch / base SHA: `master` / `121c064efc32f47d40f9942d349ebeb1e7856733`
- Current milestone: M6: completed characterization and report
- Last updated: 2026-09-01

## Goal

Characterize the effect of the training KL-PGD10 random-start seed only,
separating fixed-model direct attack sensitivity from 15-epoch accumulated
trajectory sensitivity. Use the exact accepted I100 epoch-99 parents for
development seeds 1 and 2, and stop without any attack intervention or seed
promotion.

## Non-goals

No change to attack budget, target, order, augmentation, optimizer, scheduler,
loss, sample weighting, new seed, or long-horizon continuation. No use of a
best attack seed as a method and no automatic follow-up intervention.

## Frozen contracts

- Parent checkpoints: seed 1 SHA
  `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835` and
  seed 2 SHA
  `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`.
- Training attack: KL-PGD10, epsilon `8/255`, step `2/255`, random start,
  teacher-clean target, existing FP32/sample-keyed contract.
- Only `seeds.train_attack` changes across attack-seed arms; data order,
  augmentation, model initialization, evaluation attack, and all other state
  remain fixed within a parent.
- Eight attack seeds are generated deterministically from a fixed domain string
  and frozen before any outcome is inspected.

## Milestones

- [x] M0: audit parent/config/RNG isolation and existing artifacts.
- [x] M1: freeze attack-seed registry and add isolation/parity tests.
- [x] M2: implement fixed-model direct sensitivity replay.
- [x] M3: materialize and validate the 16 short attack-seed forks.
- [x] M4: run e114 endpoint and sample-level aggregation.
- [x] M5: compare against pure-order reference and write report/artifacts.
- [x] M6: commit the completed characterization (push remains user-authorized only).

## Completion evidence

- The detached Hamster DAG completed 51/51 jobs: two fixed-model replays, 16
  attack-seed training forks, 16 CE-PGD20 endpoints, 16 checkpoint cleanups,
  and one aggregate.
- Fixed-model replay used 8,192 deterministic stratified train IDs per
  development seed; endpoint rows used the fixed 5,000-sample validation split.
- The fixed replay smoke passed on the exact epoch-99 parents before launch.
- The source tree used for the training forks was the clean registered
  `7f8a13fd1c2d8d266cd657b5fb42e9075f274655` worktree.  The main branch also
  contains a read-only replay indexing fix and manifest/preflight cleanup
  improvements made after the registry was frozen.
- Aggregate output is stored in the result artifact, separate from the frozen
  manifest; the accidental same-path overwrite was recovered and recorded as
  an orchestration issue, not a scientific change.

## Stop rules

Any order/augmentation mismatch, missing parent state, attack identity drift,
or incomplete telemetry blocks the campaign. Technical failures may retry with
the identical scientific identity; weak outcomes never trigger a retry or
extension. A second intervention is not part of this plan.
