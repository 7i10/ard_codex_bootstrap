# ARD Experiment Execution Fast Path

## Status

- Owner: Codex
- Branch / base SHA: `master` / `db7cca90f252aeaaad1fc20c2de1ee2427a55fcb`
- Current milestone: M3 — complete
- Last updated: 2026-09-04

## Goal

Provide a thin Fast mode inside the existing Production Launch Gate for an
already-integrated runtime.  It must resolve one complete input and host/job
matrix, execute only representative exact public-CLI smokes, freeze one
immutable manifest, and launch the existing detached controller without
weakening R1–R35.

## Non-goals

- No scientific training, endpoint evaluation, seed, objective, attack,
  scheduler, checkpoint, or result change.
- No new gate, orchestrator, remote executor, workspace layer, R-number, or
  campaign-specific shell wrapper.
- No active monitoring loop or generic remote lifecycle canary when an exact
  remote smoke already proves the stronger lifecycle.

## Existing state

- `production-launch-gate` already resolves source, inputs, host paths,
  parent/config/Teacher hashes, exact smoke, freeze, and detached
  `multi-gpu-experiment-orchestrator` handoff.
- `multi-gpu-experiment-orchestrator` already owns reservations, DAG
  dependencies, detached controller lifetime, retries, collection, and
  aggregation chaining.
- `launch_ledger.py` already records the request-to-controller interval and
  strict prelaunch evidence, but does not yet summarize preparation-stage
  durations.

## Scientific contracts affected

None.  This is an operational-only delta.  Source/parent/config/Teacher,
dataset/split, attack, seed, output ownership, W&B, retry, and endpoint
contracts remain fail-closed.

## Decisions

- Add exactly two operational profile values:
  `FAST_EXISTING_RUNTIME` and `FULL_NEW_INTEGRATION`.
- Existing runtime defaults to Fast; explicit integration-change indicators
  force Full.  Fast launch rejects a Full profile rather than silently
  weakening validation.
- Represent equivalent smoke coverage using an author-declared
  `smoke_group` plus an equality-checked equivalence descriptor.  A group may
  cover seed variants only when their execution class, public-CLI form, output
  semantics, config schema, parent-load path, and treatment branch agree.
- An exact external smoke that declares lifecycle coverage suppresses only
  the duplicate generic lifecycle canary for that host; it does not suppress
  source, host, identity, collection, or controller checks.
- Keep the user-facing command as a thin `--fast-launch` mode on
  `launch_gate.py`; do not add another executable framework.

## Milestones

- [x] M0 — Record baseline workflow and input evidence.
  - Files: this plan, task context, baseline benchmark artifact.
  - Tests: existing CPU dummy invocation only.
  - Acceptance: baseline command/check/freeze counts are recorded without a
    scientific run.
  - Rollback: no source changes.

- [x] M1 — Implement Fast/Full selection, smoke groups, and timing summary.
  - Files: launch gate, launch ledger, existing skill documentation.
  - Tests: F1–F10 focused tests.
  - Acceptance: Fast preserves exact identity and one manifest freeze;
    unsupported/new integration is routed to Full.
  - Rollback: ordinary `--launch` behavior remains unchanged.

- [x] M2 — Verify CPU and bounded existing Ferret dummy paths.
  - Files: tests and operational result artifact only.
  - Tests: one-command CPU dummy, one bounded existing remote dummy;
    no GPU scientific work.
  - Acceptance: remote exact-smoke proof is used once and collection succeeds.
  - Rollback: artifacts are outside scientific output roots.

- [x] M3 — Document, self-audit, commit, and push.
  - Files: launch discipline / one focused Fast Path document, `AGENTS.md`,
    experiment artifact, plan.
  - Tests: focused suite; one changed-test selection preview.
  - Acceptance: before/after table and A–J self-audit are evidence-backed.
  - Rollback: cohesive operational-only commit.

## Agent and review budget

One owner.  No subagent is needed: this is a bounded operational fast-lane
change and does not alter metric semantics, resume semantics, scientific
lineage, or results.

## Test plan

- Focused production-launch-gate and orchestrator tests, including F1–F10.
- CPU-only one-command dummy DAG test.
- One bounded existing Ferret dummy lifecycle integration, using the existing
  `run-on-ferret` path and no generic duplicate lifecycle canary.
- `scripts/verify.py --changed --non-scientific --dry-run` once before any
  optional broad selected gate.

## Risks and mitigations

- Fast could accidentally mean incomplete validation: require full resolved
  identity, complete host/job matrix, static CLI, exact smoke, and freeze.
- Smoke grouping could hide a distinct treatment path: reject unequal
  equivalence descriptors or mixed execution classes.
- Faster launch could create source drift: recompute exact smoke binding and
  revalidate frozen source immediately before detached handoff.
- Operational test artifacts could pollute scientific outputs: use temporary
  CPU/remote dummy paths and existing controller sidecars.

## Progress log

- 2026-09-04: Reconciled baseline `db7cca9`; current tree was clean and no
  scientific controller/job was active.  Audited existing gate, controller,
  ledger, workspace contract, and prior operational incident documentation.
- 2026-09-04: Implemented the thin `--fast-launch` mode at `3352c41`, then
  corrected timestamp leakage from the immutable manifest at `46329e0`.
  The regression proves that the ordinary serial gate stages can reuse one
  unchanged scientific freeze.
- 2026-09-04: The final runtime SHA passed the CPU dummy benchmark and bounded
  Hamster/Ferret public-CLI checks.  Ferret proved fixed-SHA worktree, live
  process/GPU identity, completion, collected bytes, and cleanup.  No
  scientific process was started.

## Completion report

Completed.  The machine-readable result is
`docs/experiments/ard_experiment_execution_fast_path_v1.json`; the human
contract is `docs/EXPERIMENT_FAST_PATH.md`.  A final documentation-only commit
will record the result after focused verification.
