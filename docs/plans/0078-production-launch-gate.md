# Reusable production launch gate

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `4bc7adcef692edf197c2d7c7f240e74cfb30a5c0`
- Current milestone: implementation and CPU-only validation
- Last updated: 2026-09-03

## Goal

Add a generic, fail-closed launch gate that resolves a human campaign spec into
an immutable manifest for the existing multi-GPU orchestrator, validates
scientific identity and host-local inputs before any job starts, runs a bounded
canary, and validates required outputs after the DAG completes.

## Non-goals

- No scientific training, endpoint evaluation, or GPU benchmark.
- No changes to model, loss, attack, dataset, sampler, or augmentation code.
- No replacement of the existing orchestrator or remote `run-on-ferret` logic.

## Existing state

`multi-gpu-experiment-orchestrator` already owns JSON manifest scheduling,
resource reservations, detached execution, completion-marker dependencies, and
technical retries. The new gate will be a thin resolver/validator wrapper and
will emit the orchestrator's schema-v1 manifest.

## Scientific contracts affected

None by design. The gate checks and hashes scientific identity; it never
interprets metrics or edits a scientific command.

## Decisions

- Keep the gate JSON-first and stdlib-only; accept YAML when PyYAML is present.
- Separate human campaign specs from resolved manifests; production launch
  always delegates a previously frozen resolved manifest to the orchestrator.
- Hash scientific identity independently from host/GPU/attempt execution data.
- Treat dependency-produced files as deferred inputs and validate them after
  their producer completes.
- Keep W&B metrics-only and generate attempt-specific run IDs without changing
  scientific identity.
- Add regression fixtures for the twelve launch failures named in the request.

## Milestones

- [x] M0 Audit existing skill, manifest schema, tests, and repository contracts.
- [x] M1 Implement resolver, strict preflight, freeze, canary, and post-run validator.
- [x] M2 Add regression and CPU-only DAG integration tests.
- [x] M3 Document the workflow and update skill routing.
- [x] M4 Run focused verification, commit, and push.

## Agent and review budget

One owning writer (root) is sufficient. No scientific reviewer is needed because
this milestone is operational-only and does not change scientific code or
results.

## Test plan

- Launch-gate schema/resolver/unit tests, including R1-R12 regressions.
- CPU-only known-good canary and post-run validator tests.
- Existing orchestrator skill tests.
- `scripts/verify.py --changed --non-scientific` and skill quick validation.

## Risks and mitigations

- False acceptance of a stale parent: require exact SHA and metadata/alias checks.
- Path mismatch across hosts: resolve logical dataset identities through host profiles.
- Retry drift: hash only scientific fields and reject changed retry identity.
- False completion: validate markers plus expected outputs and final epochs.
- Partial launch: run campaign-wide preflight before invoking the orchestrator.

## Progress log

- 2026-09-03: Audited existing orchestrator and skill-creator guidance; no GPU or
  active scientific process was touched.
- 2026-09-03: Added the JSON-first launch gate, host/path/artifact/epoch/attack
  validation, scientific identity freeze, bounded canary, post-run validator,
  and attempt-aware W&B execution IDs. Extended the orchestrator only at its
  execution boundary; it remains the scheduler and DAG owner.
- 2026-09-03: Added twelve individually named regression cases plus CPU-only
  detached success and technical-retry DAG tests. Focused gate/orchestrator
  tests pass (30 passed); skill quick validation, Ruff, and mypy pass. The
  broad changed-test selector was previewed without GPU; its full suite is
  dominated by pre-existing CUDA/integration coverage and is not a launch-gate
  acceptance gate.

## Completion report

Implementation and focused verification completed. No scientific training,
endpoint evaluation, live W&B call, or GPU job was started. The source tree's
pre-existing `make lint` target still reports unrelated Ruff-format drift in
69 legacy files; changed launch-gate files pass targeted Ruff checks. The
production gate is ready for future manifests, but must be exercised with a
real campaign spec only after its scientific inputs and host profiles are
independently frozen.
