# 0072 — ERT/RSLAD I100 historical-action transfer screen

Status: `complete`
Date: 2026-09-01

## Objective

Run the preregistered short continuation from the exact accepted I100 epoch-99
parents to test whether the previously useful BASE-era action families transfer
to the frozen I100 trajectory.  The screen has four arms per development seed:
ordinary I100 control, pilot-S3×Teacher-T1 weak AdvCE, fixed Clean-Wrong plain
AdvCE, and fixed Clean-Wrong Teacher positive-floor margin (A7, margin-only).

## Frozen gates

- Exact parent bytes: L2 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`,
  L4 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`.
- Continuation only: epochs 100–114 inclusive; no prefix retraining or e199
  extension.
- Training attack remains I100 KL-PGD10 with teacher-clean target.  Common
  random-start identity is `sample_keyed_v1`; order, augmentation, and attack
  RNG are paired within each parent.
- Historical pilot-S3 is clean-correct AND strong CE-PGD20 adversarial-wrong;
  Teacher-T1 is adversarial-correct outside the registered positive-margin q10.
  Clean-Wrong is clean incorrect and is never relabeled canonical S2.
- Coefficients/floor/cap are determined by a pooled no-update I100 epoch-99
  calibration before any outcome is inspected, then hash-frozen.
- Primary endpoint is held-out CE-PGD20 robust accuracy at epoch 114; no
  threshold, coefficient, arm, seed, official-test, or AutoAttack expansion.

## Execution order

1. Audit historical pilot/A2/A7 implementation and exact attack identities.
2. Replay epoch-99 parents once for fixed pre-treatment observations/masks.
3. Calibrate the three action coefficients/parameters without optimizer,
   scheduler, state, or checkpoint mutation; freeze machine artifacts.
4. Add the minimal resume-epoch-99 runtime support and focused parity tests.
5. Run `scripts/verify.py --changed`, then an orchestrator dry-run/canary.
6. Launch eight detached training jobs through
   `multi-gpu-experiment-orchestrator`; chain held-out/train endpoints and
   aggregation using completion markers, metrics-only W&B.
7. Produce direct/spillover/held-out and runtime reports, commit and push, then
   stop.  No automatic e199 continuation.

## Fail-closed conditions

Stop before GPU production if any parent/mask/attack identity is unresolved,
the replay schema cannot be joined by stable ID, calibration mutates training
state, nonselected baseline parity fails, or the source tree is dirty.

## Progress

- [x] Repo reconciliation and required contract documents read.
- [x] Exact I100 epoch-99 parent files re-hashed and payload lineage inspected.
- [x] Historical pilot-S3/T1 and A2/A7 implementation audit started.
- [x] I100 epoch-99 pre-treatment replay and masks.
- [x] Pooled no-update calibration and artifact freeze.
- [x] Runtime parity/canary and orchestration manifest.
- [x] Eight continuations, endpoints, aggregation, and report.

## Completion record

The production manifest was frozen at source SHA
`2522bc9a7a58b30135d85dfdeb33fdad0c23a313` (the inclusive epoch-114 launcher
fix). All 8 training and 8 endpoint jobs completed with valid markers. The
aggregated endpoint artifact is
`docs/experiments/ert_rslad_i100_action_transfer_results_v1.json`, and the
human-facing results are in
`docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md`.
