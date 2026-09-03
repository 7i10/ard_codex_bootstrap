# Orchestration Parallel Preflight and Critical-Path Telemetry

## Status

- Owner: Codex
- Branch / base SHA: `master` / `39a6e7c`
- Current milestone: complete
- Last updated: 2026-09-04

## Goal

Reduce safe control-plane serial delay for existing-runtime campaigns without
changing scientific execution.  External-host preflights and explicitly
independent static/exact smoke checks may run concurrently; the controller
records enough timing evidence to quantify parent-to-child delay and declared
work-rate estimates after a campaign.

## Non-goals

- No model, loss, attack, dataset, sampler, checkpoint, W&B, or scientific
  configuration change.
- No change to the controller's default polling interval.
- No new training, GPU benchmark, or active monitoring loop.
- No automatic scientific follow-on launch or retry policy expansion.

## Existing state

- `FAST_EXISTING_RUNTIME` already removes duplicate preparation, uses one
  immutable manifest, and supports marker-driven endpoint/aggregate/report
  DAGs.
- Remote preflights and gate checks are currently serial.
- The controller already records lifecycle events, but does not summarize
  dependency-ready-to-launch delay or elapsed job execution in an explicit
  campaign timing report.

## Scientific contracts affected

None.  These are control-plane changes only.  Exact smoke bindings, source and
input hashes, technical-only retry identity, output ownership, and fail-closed
behavior remain unchanged.

## Decisions

- Parallelism is opt-in per static/exact-smoke entry using
  `parallel_safe: true`; unmarked entries remain serial.  This avoids
  concurrent access to an unknown shared output, GPU, or mutable external
  resource.
- Distinct external-host preflights run concurrently because they are
  independently host-scoped and remain bounded by their existing timeout.
- Results are emitted in manifest order so parallel completion order cannot
  change provenance or test expectations.
- Timing records use declared `estimated_work` divided by observed elapsed
  time and label the result as an estimate-derived rate.  The generic
  orchestrator will not claim image/s or another physical unit unless the
  manifest explicitly declares that unit.
- Next production manifests must include all known terminal DAG nodes
  (endpoints, collection, aggregation, report) from first launch; this is an
  operational convention, not a new universal schema requirement.

## Milestones

- [x] M0: audit current serial paths and existing lifecycle events.
- [x] M1: add bounded parallel-safe gate check execution and focused regression
  coverage.
  - Files: `launch_gate.py`, gate tests.
  - Acceptance: only explicitly safe checks overlap; report order and failure
    semantics stay deterministic.
- [x] M2: add controller timing summary and manifest/skill documentation.
  - Files: `orchestrate.py`, orchestrator tests, skill/reference/docs.
  - Acceptance: completed dummy DAG reports job elapsed time and dependency
    launch delay without polling or scientific-output mutation.
- [x] M3: focused CPU tests, skill validation, diff inspection, cohesive local
  commit.

## Agent and review budget

One owner.  This is bounded operational work with no scientific-contract delta;
no scientific reviewer or additional writer is required.

## Test plan

- Focused launch-gate and orchestrator pytest modules.
- New CPU-only timing/DAG and bounded parallel-check fixtures.
- `ruff`/`py_compile` for modified scripts.
- `scripts/verify.py --changed --non-scientific --dry-run` before deciding
  whether the selected broad gate is appropriate.

## Risks and mitigations

- Shared smoke resource collision: parallelism is opt-in only and still uses
  isolated gate output directories.
- Nondeterministic report ordering: collect futures, then render in declared
  order.
- Misleading throughput claim: record declared-work rate with explicit unit;
  do not infer model throughput.
- DAG state compatibility: add fields rather than changing existing event or
  completion-marker validity semantics.

## Progress log

- 2026-09-04: created after Fast Path completion.  The next safe gains are
  independent control-plane concurrency and measured critical-path evidence;
  controller tick tuning is deferred pending measurement.
- 2026-09-04: implemented bounded concurrency for distinct external-host
  preflights and explicitly isolated static/exact smoke entries.  Exact smoke
  overlap additionally requires a unique declared resource key and unique
  fixed host/GPU binding.
- 2026-09-04: added terminal controller timing summaries for parent-ready to
  child-launch delay, worker elapsed time, and manifest-declared work rate.
  This records evidence needed before changing the controller tick interval.
- 2026-09-04: focused CPU verification passed: `65 passed` in
  `tests/skills/test_production_launch_gate.py` and
  `tests/skills/test_multi_gpu_orchestrator.py`; `ruff`, `py_compile`, and
  both skill validators passed.  `scripts/verify.py --changed --non-scientific
  --dry-run` selected T0/T1 only.  No GPU job, scientific source, or W&B
  policy was changed.

## Completion report

Implemented control-plane-only improvements:

- Host-aware placement and complete terminal DAG declaration are now required
  manifest practices for future campaigns.
- Distinct remote preflights run concurrently with deterministic report order.
- Static checks and exact public-CLI smokes run concurrently only after an
  explicit isolation declaration; all other gate work remains serial.
- The detached controller writes an additive timing summary after a terminal
  campaign without adding active polling.
- Controller tick tuning remains intentionally deferred until these summaries
  show that it is a material source of critical-path delay.
