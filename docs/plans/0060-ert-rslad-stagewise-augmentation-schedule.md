# ERT / RSLAD stage-wise augmentation schedule

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `9322eea50f4c5eed873b808beca48819d914d249`
- Current milestone: M0 parent inventory and boundary audit
- Last updated: 2026-08-28

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

- [ ] M0 parent inventory, scheduler/off-by-one audit, and source compatibility
- [ ] M1 deterministic parent materialization and parity evidence
- [ ] M2 stage-wise transform/config implementation and focused tests
- [ ] M3 one-boundary canary on Hamster
- [ ] M4 eight registered continuations and endpoint evaluation
- [ ] M5 hybrid/post-switch AUC, shock/recovery, promotion gates, and report
- [ ] M6 consolidated scientific review, cohesive commit, and stop

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

## Completion report

To be filled after M6 with commands, hashes, test results, review findings, and
remaining uncertainty.
