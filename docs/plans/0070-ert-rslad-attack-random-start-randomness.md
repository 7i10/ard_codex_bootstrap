# ERT / RSLAD attack random-start randomness characterization

## Status

- Owner: Codex `/root`
- Branch / base SHA: `master` / `121c064efc32f47d40f9942d349ebeb1e7856733`
- Current milestone: M1: registry and fork implementation frozen
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
- [ ] M2: implement fixed-model direct sensitivity replay.
- [ ] M3: materialize and validate the 16 short attack-seed forks.
- [ ] M4: run e114 endpoint and sample-level aggregation.
- [ ] M5: compare against pure-order reference and write report/artifacts.
- [ ] M6: commit and push the completed characterization.

## Stop rules

Any order/augmentation mismatch, missing parent state, attack identity drift,
or incomplete telemetry blocks the campaign. Technical failures may retry with
the identical scientific identity; weak outcomes never trigger a retry or
extension. A second intervention is not part of this plan.
