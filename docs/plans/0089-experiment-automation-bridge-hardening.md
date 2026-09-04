# Experiment Automation Bridge Hardening

## Status

- Owner: Codex
- Branch / base SHA: detached operational worktree / `0b553d531409f0215546ee0391d3215c58ca9949`
- Current milestone: complete
- Last updated: 2026-09-04

## Goal

Connect the production launch gate, multi-job orchestrator, scheduled
reconciler, and terminal-result publisher without touching scientific runtime
or active campaigns. A running campaign must reconcile in O(1) registered
state reads; terminal processing must have one owner and terminal GitHub
events must be idempotent.

## Non-goals

No training, evaluation, active-campaign restart, scientific source/config
change, new scheduler, W&B policy change, or automatic scientific decision.

## Existing state

`reconcile_experiment.py` currently assumes one local PID and schema v1.
Launch Gate freezes manifests consumed by the orchestrator but does not emit a
canonical campaign state or runtime-signature decision. There is no terminal
event publisher.

## Scientific contracts affected

None. Changes are limited to operational state, leases, manifest lineage,
runtime classification, and result notification. Scientific identity is
copied and validated, never inferred or mutated.

## Decisions

- The orchestrator state is authoritative for `orchestrator_campaign`; local
  PIDs are never used to infer remote campaign truth.
- Existing schema-v1 reconciler states remain `single_process` compatible.
- Fast launch requires a validated runtime signature and exact smoke; unknown
  or changed topology is Full.
- Terminal events are compact notification pointers on `experiment-results`,
  not scientific result storage.

## Milestones

- [x] Add campaign-state bridge and multi-job reconciliation.
- [x] Add registered retry delegation and postprocess ownership checks.
- [x] Add validated runtime-signature registry and gate enforcement.
- [x] Add idempotent terminal-result publisher.
- [x] Add focused CPU tests, documentation, closure artifact, and commit.

## Agent and review budget

One owner; no subagent is needed. This is an operational control-plane change
with no scientific model/loss/attack/evaluation change. One consolidated
review pass is sufficient after tests.

## Test plan

Focused unit tests for campaign state, failure delegation, postprocess
ownership, runtime signatures, and publisher idempotency; existing reconciler,
orchestrator, and launch-gate CPU suites; dummy DAG only.

## Risks and mitigations

State schema migration is read-compatible and fail-closed on identity
mismatch. Registered commands are the only delegated commands. Atomic files,
leases, and a publisher worktree prevent duplicate postprocessing or event
publication. No live scientific or GPU process is inspected beyond the
initial read-only campaign check.

## Progress log

- 2026-09-04: Isolated operational worktree while Online-State S2 recovery14
  remains active; baseline and failure evidence inventoried.
- 2026-09-04: Implemented schema-v2 bridge, orchestrator-authoritative
  reconciliation, bounded technical recovery, runtime-signature gate,
  idempotent terminal publisher, and CPU-only regression coverage.

## Completion report

Focused verification passed: 84 tests across bridge, reconciler, launch gate,
and orchestrator suites; Ruff check passed for all changed Python files; the
changed-test preview was inspected without launching its unrelated broad
integration selection. `make lint` remains red on 78 pre-existing formatting
violations outside this milestone. No GPU or live scientific campaign was
started.
