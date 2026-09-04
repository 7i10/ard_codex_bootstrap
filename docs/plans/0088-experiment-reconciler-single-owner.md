# Experiment reconciler and single-owner postprocessing

## Status

- Owner: Codex
- Branch / base SHA: `master` / `3fe3c2528ce053c28dbdb59ea739b2d3840e3771`
- Current milestone: complete
- Last updated: 2026-09-04

## Goal

Provide a small, idempotent scheduled wake that reconciles one canonical
`experiment-state.json` and hands valid terminal training to an existing
postprocess DAG under one durable lease.

## Non-goals

No scientific training, evaluation implementation, repository scanning,
W&B access, new orchestrator, parameter repair, or long-running polling.

## Existing state and decisions

The multi-GPU orchestrator remains authoritative for campaign execution and
dependency chaining. The new helper only reads registered state/evidence,
uses an OS `flock` plus an expiring lease, and delegates postprocessing via an
existing argv command. Missing or mismatched terminal evidence fails closed;
technical postprocess launch retries preserve scientific identity and are
bounded at two attempts. Scientific failures stop at
`NEEDS_RESEARCH_DECISION`.

## Milestones

- [x] Implement reconciler and canonical state transitions.
- [x] Add focused CPU tests, including concurrent ownership and lease expiry.
- [x] Document the scheduled-wake boundary in the operational orchestration doc.
- [x] Run focused verification and prepare one operational commit.

## Agent and review budget

One owner; no subagent or scientific reviewer was needed. This is an
operational control-plane change with no model, loss, attack, dataset,
checkpoint, or evaluation contract change.

## Test plan

Focused unit tests and existing CPU orchestration/launch-gate suites. GPU,
W&B, and scientific training runs are intentionally deferred and untouched.

## Risks and mitigations

The state is atomically written and identity-bound. A dead PID without exit,
marker, and expected-output evidence is never success. Active leases are
honored until expiry, and terminal markers validate experiment, source, and
scientific identity before state advancement.

## Progress log

- 2026-09-04: Implemented helper, added nine focused tests, and passed focused
  reconciler plus existing orchestration suites.

## Completion report

Implementation and tests are complete. Production jobs were not inspected,
restarted, reassigned, or modified by this task.
