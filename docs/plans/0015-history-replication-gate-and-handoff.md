# Student-history replication gate and one-run handoff

## Status

- Owner: main thread; Terra owns the two bounded scientific code deltas
- Branch / base SHA: `master` / `7a278df`
- Current milestone: M1 final-SHA CUDA parity and artifact replay
- Last updated: 2026-07-31

## Goal

Close the frozen outcome-informed mask experiment and the Chen/Bartoldson
common-trajectory replay, freeze a confirmatory student-history specification,
and start exactly one Bartoldson `rslad_logging_only` seed-1 production run only
if every pre-registered gate passes.

## Non-goals

- Do not start Chen logging-only seed 1, any seed 2 run, a v2 intervention,
  MobileNetV2, full SAAD, or full AutoAttack for the new logging-only run.
- Do not tune predictors, features, anchor epochs, outcomes, or thresholds after
  inspecting seed 1.
- Do not call the existing mixed future-error mask a treatment oracle or infer
  which of its two strata caused an aggregate effect.

## Existing state

- Frozen-mask oracle/control-1/control-2 training is complete; control-3 is
  training on Ferret. Saved-checkpoint evaluations are terminal for controls
  1/2, and the oracle evaluation is running.
- The frozen mask contains 3,566 Bartoldson/RSLAD train IDs: 404 robustly
  correct at epoch 99 and wrong at epoch 199, plus 3,162 wrong at both
  endpoints. It is therefore an outcome-informed failure mask, not an
  improvability oracle.
- Both canonical seed-0 RSLAD run bundles contain the exact 40 periodic `last`
  checkpoints required by the common-trajectory replay.
- `rslad_logging_only` exists and has bounded parity evidence, but its stored
  state and replay comparison need the additional frozen primitives and model
  matrix specified below.

## Scientific contracts affected

- Replay uses raw CIFAR-10 train IDs only, KL PGD-10 in pixel space with
  epsilon `8/255`, step `2/255`, and independent feature/outcome seed domains.
- Logging-only observations are detached FP32 diagnostics. They must not alter
  loss, gradients, optimizer/scheduler state, attack RNG, data order, or saved
  model bytes relative to RSLAD.
- Anchor epoch is 99; prospective outcomes use epochs 100--199. Official-test
  metrics are not predictor inputs.
- Sample-level bootstrap intervals are conditional on the seed-0 trajectory and
  are not training-seed uncertainty.

## Decisions

- The primary History gate compares a frozen history-only model against the
  best of current correctness, instantaneous margin, and their main-effects
  model on the same held-out split.
- History Go requires held-out delta AUROC at least `0.02`, paired
  class-stratified bootstrap lower bound greater than zero, and lower held-out
  log-loss. Failure to improve log-loss is No-Go; remaining ambiguous cases are
  inconclusive and do not launch training.
- Secondary frozen comparisons are teacher versus history, history plus
  teacher versus history, and teacher-interaction versus teacher main effects.
- Seed roles are fixed before launch: seed 0 discovery, seed 1 replication,
  seed 2 conditional confirmation. This plan launches seed 1 only.
- Frozen-mask Go/No-Go remains the previously registered allocation rule and is
  reported independently from the History gate. A History Go can justify the
  logging-only replication even if uniform target softening is No-Go.

## Milestones

- [ ] M0: reconcile frozen-mask jobs, artifacts, and mask strata.
  - Acceptance: all four train/evaluation terminals, exact arm hashes and
    best/last clean/PGD/AA table, and unchanged pre-registered decision.
  - Commit boundary: combined with M1--M3 because no source code depends on job
    waiting.
- [ ] M1: complete and run common-trajectory replay.
  - Files: `src/ard/analysis/rslad_signal_replay.py`, focused tests, two
    immutable execution configs.
  - Tests: focused CPU unit tests, then one checkpoint/one bounded-batch CUDA
    smoke before the full replay.
  - Acceptance: 45,000 stable IDs per checkpoint, exact checkpoint panel and
    lineage, complete frozen model table and paired intervals for both teachers.
- [x] M2: complete exact logging-only state and parity contracts.
  - Files: sample state, trainer diagnostics, exact state analyzer, focused
    regression/integration tests.
  - Acceptance: first learned epoch, correctness frequency/forgetting/streaks,
    margin EMA/mean/variance/slope, teacher correctness/confidence and
    clean-to-adversarial response are checkpointed and resume exactly; padded
    IDs are excluded.
- [ ] M3: freeze the confirmatory specification and launch preflight.
  - Acceptance: immutable design bytes and SHA bind predictor/features,
    anchor/outcome, split/bootstrap seeds, thresholds, and seed roles; exact
    RSLAD/logging-only CUDA parity passes at the final code revision.
- [ ] M4: conditionally start one Bartoldson logging-only seed-1 run.
  - Acceptance: fixed Git/config/teacher/design hashes, unique output/W&B ID,
    correct GPU, process and first progress, finite metrics, observation-only
    state present in checkpoint.
- [ ] M5: one consolidated scientific review, docs, commit, and handoff.

## Agent and review budget

One read-only gap audit and two non-overlapping Terra implementation tasks are
used. GPU jobs are shell processes. One consolidated scientific review occurs
after code and evidence stabilize; a second review is permitted only for an
actual P0/P1 fix delta.

## Test plan

- Focused replay/state/analyzer unit and regression tests.
- Exact CPU and CUDA logging-only parity, including random-start PGD.
- One real teacher/checkpoint/batch replay smoke.
- `scripts/verify.py --changed` once after the complete delta is stable; reuse
  cached passes and do not run full production training as a test.
- Full common replay and the four-arm evaluation are scientific analyses, not
  cached unit-test results.

## Risks and mitigations

- Hamster CUDA is hidden only in the ordinary isolated Codex shell. Use
  non-isolated execution for `nvidia-smi`, CUDA tests, replay, and training;
  do not misclassify the missing isolated `/dev/nvidia*` view as driver failure.
- Replay history sampled every five epochs can miss transitions. Name it
  checkpoint-panel history and reserve exact online claims for logging-only.
- Large Bartoldson forwards can underutilize a GPU at small batches. Benchmark
  one bounded batch, then choose the largest memory-safe replay batch without
  changing scientific identity.
- Existing jobs may terminate while code work proceeds. Reconcile process,
  output, W&B, and marker state before any recovery; never duplicate a terminal
  run.
- Seed-1 inspection could leak into method design. Hash-bind the complete
  confirmatory design before launch and fail closed if it changes.

## Progress log

- 2026-07-31: user fixed the conditional launch target to Bartoldson,
  `rslad_logging_only`, seed 1, 200 epochs, one GPU, batch 128. Chen seed 1 and
  all later runs remain human-gated.
- 2026-07-31: control-1 best clean/PGD/AA is
  `84.61/49.82/46.35`; control-2 is `83.56/50.83/47.37`. Oracle PGD is
  `83.66/50.77` at best; its AutoAttack evaluation and control-3 remain active.
- 2026-07-31: replay predictive-model implementation completed its focused
  unit gate (`30 passed`).
- 2026-07-31: logging state format v3 now records exact correctness/forgetting
  history, streaks, margin moments/slope, and stable teacher clean-to-adversarial
  responses. Legacy state is explicitly marked incomplete and rejected by the
  frozen analyzer. The analyzer requires 100/200 observations per ID, terminal
  run-bundle lineage, exact checkpoint artifact hashes, and a tracked-clean
  implementation identity.
- 2026-07-31: the confirmatory design is frozen at
  `configs/analysis/logging_only_history_confirmatory_v1.yaml` and the training
  preflight permits only Bartoldson seed 1, 200 epochs, world size 1, batch
  128. Launch remains blocked until the History Go result itself is available
  and hash-bound.
- 2026-07-31: impact-selected non-scientific gate passed under non-isolated
  execution. Its commands reported 21, 1, 6, 33, 16, 1-skip, 3, 4, 24, 4, 3
  plus 1 skip, 4, 26, 2, 9, 2, 7, 4, 31, 49, and 32 passing tests. Final
  focused analyzer delta reported `9 passed`; Ruff and focused mypy passed.
  Consolidated scientific review found and closed JS underflow/rounding,
  replay-threshold, prospective-window, legacy-migration, terminal-lineage,
  trajectory-completeness, and launch-allocation findings; no P0/P1 remains
  in the implementation delta.
- 2026-07-31: non-isolated Hamster preflight confirmed two RTX 4090 devices
  and PyTorch `cuda_available=True`; final-SHA CUDA parity is still pending the
  tracked-clean commit.

## Completion report

Pending.
