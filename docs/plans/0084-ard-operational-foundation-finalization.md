# 0084 — ARD operational foundation finalization

## Status

- Owner: Codex
- Branch / base SHA: master / `67eff8ab8de1bb5fad497c5dd3e22f5fa515e6f1`
- Current milestone: bounded real-host verification and closure
- Last updated: 2026-09-04

## Goal

Standardize future ARD runtime writes under one canonical workspace contract;
harden local/remote orchestration ownership, identity, smoke, retry, and
collection paths; safely inventory eligible legacy paths; and leave a compact
task-context workflow that avoids repeated rediscovery.  The end condition is
an operational-only, tested, committed, and pushed foundation with a closure
artifact.

## Non-goals

- No training, optimizer/scheduler mutation, endpoint evaluation, seed,
  intervention, threshold, coefficient, or scientific-result change.
- No deletion or migration of an `UNKNOWN`, `ACTIVE_CANONICAL`, or
  `HISTORICAL_REFERENCED` path.
- No new scientific campaign, GPU benchmark, W&B policy change, or remote
  source edit.

## Existing state

- The I100 S2 forensic audit is complete at `a8b9d4e`; its plan is complete,
  reports/artifacts are committed, local and Ferret scientific processes are
  absent, and its final controller states are `completed`.
- Existing generic execution owners are the multi-GPU orchestrator,
  production-launch gate, and `run-on-ferret` skill.  The sidecar fix and
  stale-result protections are already documented in debugging note 0025.
- Historical ARD roots may still be referenced by frozen manifests and are
  read-only until the inventory proves otherwise.

## Operational contracts affected

- Workspace paths and future runtime-write ownership only; historical reads
  remain legal.
- Controller sidecars must remain physically outside scientific output paths.
- Remote origin/identity, attempt isolation, atomic collection, smoke binding,
  launch SLO, and task context are operational lineage metadata.  They must
  not alter scientific identity or configuration.

## Decisions

- The canonical repository/dataset/runtime paths are specified by a tracked
  registry and resolved through one library rather than scattered absolute
  literals in future-active runtime code.
- Cleanup is inventory-first and evidence-bound.  Ambiguity is a keep decision.
- Existing reliable remote lifecycle scripts are extended/reused rather than
  replaced by a second SSH implementation.
- R29–R35 are added only where existing tests do not already prove their
  behavior; equivalence with an existing regression is recorded rather than
  duplicated.
- The public-CLI representative smoke uses a bounded no-training dummy/known
  read-only interface and does not create new scientific data.

## Milestones

- [x] M0 — Reconcile final forensic closure, audit existing skills/code/tests,
  and create the workspace cleanup inventory in dry-run form.
  - Acceptance: no active scientific process; each requested external path has
    classification and evidence; no deletion.
- [x] M1 — Add workspace registry/resolver, canonical runtime directory
  contract, and compact `workspace_doctor --json`.
  - Acceptance: active code resolves paths from registry; unknown writes fail;
    doctor reports compact local/remote-neutral state without scientific checks.
- [x] M2 — Implement the missing R23–R35 operational guards and exact-command
  smoke binding while preserving existing sidecar and retry behavior.
  - Acceptance: remote-observed identity/origin, manifest/attempt isolation,
    atomic collection, static CLI gate, source-invalidated smoke, and fan-out
    gate are regression-tested.
- [x] M3 — Add task-context and research-status navigation updates, complete
  bounded Hamster/Ferret operational verification, and carry out only
  inventory-proven cleanup.
  - Acceptance: no scientific command; all actual deletions have pre/post
    evidence in the cleanup artifact.
- [x] M4 — Produce closure docs/artifacts, run impact-selected non-scientific
  verification, validate changed skills, commit/push, and stop.

## Agent and review budget

This is an operational fast-lane task with one writer.  No scientific reviewer
is needed unless an implementation would alter a scientific identity, resume,
or result lineage contract; in that case work stops for targeted review.

## Test plan

- Existing orchestrator, launch-gate, remote-lifecycle, and artifact-inventory
  focused tests, expanded only for missing R23–R35 behavior.
- New workspace registry/doctor/cleanup/task-context unit tests and CPU-only
  dummy public-CLI integration tests.
- Skill validation and `scripts/verify.py --changed --non-scientific --dry-run`
  before selecting the smallest full gate.
- Bounded Hamster/Ferret read/write/identity/collection checks only; no GPU
  scientific workload.

## Risks and mitigations

- Historical data loss: classification defaults to `UNKNOWN`; cleanup requires
  references, process/cwd/open-file/lock/worktree evidence and explicit safe
  action.
- Remote path drift: registry and remote preflight compare observed paths and
  hashes; no remote edit is needed.
- Overbroad refactor: preserve current skill APIs and add small compatibility
  layers/tests around them.
- Test latency: preview changed selection, run focused tests first, and do not
  start duplicate verification processes.

## Progress log

- 2026-09-04: Start gate passed after forensic final commit: local/Ferret
  scientific process checks were empty and completed controller states were
  inspected.  No cleanup or hardening began before this check.
- 2026-09-04: M0/M1/M2 passed focused CPU-only tests.  The cleanup inventory
  retained every historical referenced root, then removed only the clean,
  unreferenced stage-wise worktree, six untracked wrapper scripts, and empty
  legacy GPU lock root after process/cwd/open-file/worktree checks.
- 2026-09-04: M3 completed with bounded CPU-only Hamster/Ferret lifecycle
  checks against the frozen operational source.  Ferret collection was
  hash-equal before atomic local promotion; no scientific CLI was invoked.
- 2026-09-04: M4 completed with the R1–R35 registry closure, focused suite,
  one changed non-scientific verification gate, closure artifact, and
  human-facing operational report.

## Completion report

Completed operational-only.  The result record `a8b9d4e` preceded this work;
the foundation does not change its science or start a successor campaign.
