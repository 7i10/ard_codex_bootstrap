# ERT / RSLAD stage-wise augmentation schedule

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `03e478db2d9415b6cb6706205d40d1e52a8a3977`
- Current milestone: M6 complete; results recorded and stopped
- Last updated: 2026-08-29

## Goal

Evaluate the preregistered CropShift-prefix schedules (switch 100 and 150, with
CROP_RE or IDBH_WEAK as the late policy) from exact CROPSHIFT continuation
states, and produce hash-bound parent, trajectory, endpoint, AUC, shock, and
promotion-gate reports without adding a third timing or a new training method.

## Non-goals

No full CROPSHIFT rerun, schedule sweep, ramp, new augmentation, history/state
intervention, official test, AutoAttack, or unseen-seed confirmation.

## Existing state

The accepted CROPSHIFT controls are complete local runs from source
`ffc217dd635462e1f14c93720561208db2d70254`:

- seed 1: `.../cropshift-s1-r2`, run `ert-rslad-trajstab-cropshift-s1-r2-20260827`
- seed 2: `.../cropshift-s2-r1`, run `ert-rslad-trajstab-cropshift-s2-r1-20260827`

Both have complete run bundles, epoch metrics, sample state, and sparse
checkpoints. Their configured scheduler is MultiStepLR `[100, 150]`, gamma
`0.1`, stepped at epoch end. The first late-policy minibatch is therefore
scientific epoch 100 (parent payload epoch 99), and the second is epoch 150
(parent payload epoch 149). Sparse files `epoch-099.pt` and `epoch-149.pt`
contain payload epochs 98 and 148, so each exact boundary parent requires one
deterministic CROPSHIFT epoch continuation. No historical model artifacts are
available in W&B for these runs (metrics-only retention); local checkpoints are
the authoritative bytes.

## Scientific contracts affected

- Augmentation policy changes only after the frozen LR boundary.
- Model, objective, threat model, optimizer, scheduler, sampler, and all RNG
  ownership remain unchanged except for the named late augmentation substreams.
- Resume restores complete state and preserves the source config identity.
- Endpoint evaluation remains independent CE-PGD20 in pixel space.
- W&B receives metrics/metadata only; model and run-bundle artifacts remain
  local.

## Decisions

- Accepted: use local sparse CROPSHIFT checkpoints plus deterministic one-epoch
  materialization for the exact S100/S150 end-boundary parents.
- Accepted: reuse existing CROPSHIFT prefix metrics/endpoints only when metric,
  checkpoint, and lineage hashes match.
- Rejected: substituting a nearest checkpoint, resetting RNG at the switch, or
  changing the scheduler to make a boundary convenient.
- Rejected: repeating W&B searches after the single hash-bound inventory is
  created.

## Milestones

- [x] M0 parent inventory, scheduler/off-by-one audit, and source compatibility
- [x] M1 deterministic parent materialization and parity evidence
- [x] M2 stage-wise transform/config implementation and focused tests
- [x] M3 one-boundary canary on Hamster
- [x] M4 eight registered continuations and endpoint evaluation
- [x] M5 hybrid/post-switch AUC, shock/recovery, promotion gates, and report
- [x] M6 consolidated scientific review, cohesive commit, and stop

## Agent and review budget

One owner (root) is sufficient; no subagent is needed. Use one consolidated
scientific review after implementation, canary, and stable results. Do not add
review cycles for unchanged evidence.

## Test plan

- Static preflight: Git, CUDA/GPU identity, disk, dataset, teacher, W&B login,
  and user linger.
- Focused transform/RNG tests plus `scripts/verify.py --changed`.
- Deterministic parent materialization parity against historical boundary rows.
- One-boundary canary before production; no full training in automated tests.

## Risks and mitigations

- Off-by-one boundary: bind serialized payload epoch, completed count, next LR,
  and first late-policy epoch in the inventory.
- Resume/RNG drift: require all checkpoint state components and compare the next
  sample order and known CROPSHIFT metrics.
- Prefix RNG drift: named spatial-prefix and late-layer substreams; no reseed.
- Storage: local metrics-only outputs and no model/run-bundle W&B uploads.
- Host confound: run all eight jobs on Hamster with explicit GPU UUID metadata.

## Progress log

- 2026-08-28: Reconciled clean `master` at `9322eea`; verified both local
  CROPSHIFT run bundles, scheduler `[100,150]`, and exact sparse checkpoint
  hashes. W&B relevant runs contain metrics/sample-state artifacts but no model
  artifacts. Exact S100/S150 parents still require deterministic materialization.

- 2026-08-28: Materialized and SHA-verified exact S100/S150 parents for both
  seeds. Implemented the stage-wise transform, protocol, configs, and fork
  lineage; focused data/protocol tests passed. Four Hamster canaries (both
  boundaries and both late policies) passed with correct next epoch/LR,
  prefix identity, late activation, and finite resumed gradients. Production
  source is frozen at `bb68afc0ff505248f84c0263179ec24f0b346bcd`; the eight-arm
  preregistration is `docs/experiments/ert_rslad_stagewise_augmentation_v1.json`.

- 2026-08-29: Completed all eight Hamster continuations (R100/I100/R150/I150
  × seeds 1/2) through canonical epoch 199. Independent CE-PGD20 endpoint
  evaluation completed for all 24 scheduled train/validation endpoints. The
  endpoint launcher had a shell `set -u` local-variable ordering defect; it was
  corrected before endpoint execution and did not affect training artifacts.

- 2026-08-29: Generated the hash-bound result artifact and report. The final
  endpoint deltas are computed from independent CE-PGD20 endpoint values (not
  dense trajectory metrics); dense `val_pgd_accuracy` deltas are retained only
  for AUC and shock analysis. All four schedules pass the preregistered
  descriptive gates in both seeds, so the protocol requires human review and
  makes no automatic promotion; CROPSHIFT remains the incumbent pending that
  review. No official test, AutoAttack, extra timing, or new training was run.

- 2026-08-29: `ruff` and the focused config/intervention tests pass. The broad
  `scripts/verify.py --changed` sweep was attempted; it still reports an
  unrelated environment-sensitive CUDA RNG test failure in
  `tests/unit/test_schedule_control_fork.py` (and had exposed two stale test
  expectations, which were updated). This does not invalidate the completed
  stage-wise result aggregation or endpoint artifacts.

## Completion report

- Source SHA for all eight production continuations: `bb68afc0ff505248f84c0263179ec24f0b346bcd`.
- Result artifact: `docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json`,
  SHA-256 `d7ae5afce02792faf95c58e753d509c40a0ca19f8732d1bf829fe64ca0631faf`.
- Human report: `docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md`.
- Independent endpoint attack identity:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- All schedules have positive final CE-PGD20 robustness deltas in both seeds:
  R100 `+0.70/+0.84 pp`, I100 `+1.20/+1.04 pp`, R150 `+0.28/+0.64 pp`,
  I150 `+0.76/+0.90 pp` (seeds 1/2). Full and post-switch AUC deltas are
  positive for every schedule and seed, and no clean guardrail is violated.
- Because four schedules qualify descriptively and the preregistration forbids
  automatic promotion, the freeze decision is `human_review_required`; no
  follow-up experiment was started.
- Remaining uncertainty: two development seeds, internal validation only, no
  official test or AutoAttack, and the broad verify sweep retains the unrelated
  CUDA RNG test failure noted above.
