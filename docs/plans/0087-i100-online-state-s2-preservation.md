# 0087 — I100 Online-State S2×T1 Preservation Screen

## Status

- Owner: Codex
- Base SHA: `fcbebc6`
- Current milestone: source freeze and launch-gate preparation
- Last updated: 2026-09-04

## Goal

Run the preregistered two-development-seed online-state screen from the exact
I100 epoch-99 parents.  Compare only I100 control, Online-State Pair-Margin
Preservation (OS-PMP), and Online-State Detached Boundary-Distance Preservation
(OS-DBDP).  The intervention may act only on the current pre-update
Online-S2×T1 branch during epochs 101–114.

## Frozen scientific contract

- Parents: dev-1 `360910a8…7630835`; dev-2 `bb0c7c1a…f7aaf7`.
- Teacher SHA: `fc398a48…c383983` and frozen parameters.
- Training attack: registered sample-keyed KL-PGD10 identity
  `97a41870…9623d4d`.
- Held-out endpoint: registered CE-PGD20 identity
  `70811016…dcc4f2` at e104/e109/e114; e114 is primary.
- Shared baseline e100 prefix once per seed.  Freeze seed-specific positive
  global-margin q10 thresholds from e100 observations before all six forks.
- Reuse the registered pair-margin and detached-boundary-distance coefficients
  only: `0.05380932585058825` and `31.649566509850324`.
- No Clean-Wrong or S3 action, S-BDP, threshold/coefficient tuning, e115–199,
  new seed, official test, or AutoAttack.

## Milestones

- [x] M0: reconcile source/lineage/inputs and classify the runtime as
  FAST_EXISTING_RUNTIME or FULL_NEW_INTEGRATION.
- [x] M1: implement only the minimal online S2×T1 router if existing runtime
  cannot express the exact same-step contract; add fail-closed assertions and
  focused tests.
- [ ] M2: materialize two e100 common prefixes, derive and freeze q10
  thresholds, and verify shared-prefix parity.
- [ ] M3: freeze the source and an immutable DAG manifest; run static and
  exact scientific smoke coverage for the three scientific branches and needed
  host execution classes.
- [ ] M4: execute the two prefixes, six e101–114 continuations, e104/e109/e114
  endpoints, e114 train audit, collection, aggregation, and reporting via the
  detached DAG.
- [ ] M5: conduct one consolidated scientific review, write report/artifact,
  verify, commit, push, and stop.

## Required runtime invariants

- State is pre-update and derived from the baseline I100 adversarial example;
  the hard router/rival/gate is detached.
- Branch priority is Clean-Wrong, S3-non-Clean-Wrong, Online-S2, Online-S1.
- Treatment is exactly current Online-S2×T1; an assertion rejects every other
  active sample.
- Action pair uses the Student’s detached current strongest non-true rival and
  reuses that pair for Teacher action margins.  Router global margins remain
  distinct from action pair margins.
- Full-batch mean only; no selected-count normalization; no extra PGD; DBDP
  uses first-order input gradients and a detached Student denominator.

## Operational contract

- Use Hamster GPU0/GPU1 only.  Ferret GPU0 preflight passed, but the current
  immutable gate cannot SHA-bind fresh external outputs before collection;
  keeping the whole parent→child→endpoint chain local avoids weakening the
  lineage contract for this first online-state runtime.
- The one initial manifest includes prefixes, threshold artifact, forks,
  endpoints, e114 train audit, collection, aggregation, review bundle, and
  report nodes.
- Launch only through the production launch gate and immutable multi-GPU DAG;
  stable jobs are not actively polled by Codex.
- Manifest records work unit, host/GPU UUID, estimated work, transfer cost,
  parent/source/config/attack lineage, W&B metrics-only metadata, and all
  completion conditions.

## Launch blockers

- Any mismatch in parent, Teacher, attack, calibration artifact, source,
  state semantics, reduction, or shared-prefix parity.
- Online branch/action cannot be proven pre-update and detached.
- An extra loss can be active outside Online-S2×T1.
- Host profile, GPU reservation, required data/Teacher paths, or output
  ownership fails preflight.

## Progress log

- 2026-09-04: user authorized execution after the control-plane optimization
  milestone.  Began exact-prompt and current-runtime reconciliation.
- 2026-09-04: implemented the public online-state runtime, source/attack and
  threshold lineage bindings, separate state/action persistence telemetry,
  gradient boundary telemetry, and bounded public-runtime canary.
- 2026-09-04: consolidated scientific review found and resolved six P1
  issues (including public-canary `SampleRef` handling); final verdict was
  APPROVED pending frozen-source public canary.
- 2026-09-04: first immutable-manifest preflight correctly rejected a generic
  gate assumption that every training node must share the campaign's e114
  bound.  Added identity-bound job-local epoch contracts so the shared e100
  prefix remains exactly one epoch while children remain exactly e101–114;
  no GPU job was launched from the rejected manifest.
- 2026-09-04: bounded canary static validation rejected an incomplete
  `static_cli` entry before any GPU reservation.  Bound the check to the
  representative prefix job and added a manifest-schema regression test;
  re-freeze is required because this changes the production source SHA.

## Completion report

Pending.  Record the final source SHA, parent/prefix/threshold/calibration
lineage, all endpoint and state telemetry, the two-seed decision, review
verdict, and no-follow-on stop.
