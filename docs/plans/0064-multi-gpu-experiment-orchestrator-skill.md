# Reusable multi-GPU experiment orchestration skill

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `f4f8592290fae61f15a75bb4eed6c5244c5a690e`
- Current milestone: M4 complete
- Last updated: 2026-08-30

## Goal

Add a generic manifest-driven skill that safely schedules independent GPU jobs,
advances parent/child DAGs from validated completion markers, chains endpoint
and aggregation jobs, supports technical-only retries, and resumes idempotently
without tying Codex to a long-running process.

## Non-goals

- No changes to scientific training, attack, data, model, sampler, or W&B code.
- No launch, restart, monitoring, or evaluation of the active unseen-seed jobs.
- No hard-coded Hamster/Ferret throughput or dataset-specific semantics.

## Existing state

`run-on-ferret` already provides fixed-SHA remote preparation, launch, status,
collection, cancellation, and cleanup. The new skill will wrap it by contract,
not duplicate its SSH/rsync implementation. The prior unseen campaign audit
records path failures, GPU double-booking, a multi-hour parent-to-child gap, and
manual endpoint delay. Current Ferret production jobs remain untouched.

## Scientific contracts affected

None. The skill validates and preserves source/config/seed/parent/attack
identity, but never interprets scientific metrics or changes a job command.

## Decisions

- Use a small Python standard-library controller with JSON manifests and atomic
  state/event files; JSON keeps the dummy integration dependency-free.
- Keep long-lived scheduling in a detached controller/data plane, not in Codex.
- Require an explicit technical failure marker before retry; nonzero scientific
  exits remain terminal.
- Support local commands directly and expose a generic executor hook for
  existing remote wrappers; tests exercise the local backend only.
- Keep dry-run read-only and use one campaign state lock to prevent duplicate
  claims.

## Milestones

- [x] M0 audit existing skills, scripts, and orchestration evidence.
- [x] M1 implement manifest validation, identity hashing, host-aware dry-run,
  marker/state transitions, retries, and resume.
- [x] M2 write focused unit tests and CPU-only success/failure DAG fixtures.
- [x] M3 validate skill structure, run changed tests, document invocation and
  known limitations.
- [x] M4 commit and push one cohesive operational milestone.

## Agent and review budget

One owning writer (root) is sufficient. No subagent or scientific reviewer is
needed: this is an operational skill with no metric, resume artifact, or
scientific-source change. A focused local test pass replaces a review cycle.

## Test plan

- Skill quick validator.
- Shell/python syntax and manifest unit tests.
- Dummy success DAG with independent roots, fork children, endpoint,
  aggregation, and report.
- Dummy technical failure followed by same-identity retry and dependency
  unblocking.
- Dry-run, cycle/missing-dependency, GPU reservation, marker, and idempotency
  tests.
- `scripts/verify.py --changed --non-scientific` after implementation.

## Risks and mitigations

- Scientific identity drift: hash-bound immutable fields and preserve them over
  retries.
- Duplicate GPU launch: atomic campaign resource reservations plus recorded GPU
  UUIDs; external processes are never killed.
- Orphaned controller: record PID/state/log paths and expose bounded resume;
  remote lifecycle remains owned by `run-on-ferret`.
- Marker forgery or stale output: validate campaign/job/attempt/identity and
  require atomic marker writes.
- Host/path mismatch: host profiles are resolved and checked before launch.

## Progress log

- 2026-08-30: Read skill-creator guidance, repo docs, `run-on-ferret`, and the
  unseen orchestration audit. Active Ferret jobs were not queried or changed.
- 2026-08-30: Chose a JSON/std-lib controller plus local dummy backend; remote
  execution remains delegated to existing `run-on-ferret` wrappers.
- 2026-08-30: Implemented the controller, manifest reference, human-facing
  report, and CPU-only tests. Skill validation, ruff, and the changed
  non-scientific gate passed; no active Ferret job was touched.
- 2026-08-30: Added cross-campaign GPU reservation locks, pending-marker
  recovery, identity-aware technical failure markers, bounded external probes,
  and local job path/environment preflight. Nine focused tests now pass.

## Completion report

Completed as an operational-only milestone. The skill, manifest reference,
CPU-only success/retry/idempotency tests, documentation, commit, and push are
complete; active scientific runs remain outside this milestone.
