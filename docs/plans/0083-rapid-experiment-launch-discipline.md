# 0083 — Rapid experiment launch discipline

## Status

- Owner: Codex
- Branch / base SHA: master / `d53dbc040c822200b2e5fc155a639ef7aff3f212`
- Current milestone: complete; reusable launch ledger and evidence-backed postmortem accepted
- Last updated: 2026-09-04

## Goal

Make short, already-specified GPU screens launch promptly without weakening
scientific or lineage gates.  Record actual request-to-launch time and make a
host-path/config mismatch fail before any GPU reservation.

## Non-goals

- No change to an active scientific job, its source SHA, checkpoint, config,
  mask, attack, or tracking.
- No new training, endpoint, benchmark, or scientific result.
- No generic scheduler rewrite or host-specific throughput hard-code.

## Decisions

- A ready short campaign has a 30-minute target from a recorded request time
  to controller launch.  A newly implemented runtime may take longer, but the
  launch estimate and concrete blocker must be reported before that target is
  exceeded.
- The prompt author supplies science; Codex owns host path resolution,
  frozen source/manifest, timing ledger, and resource placement.
- After one host/config mismatch, validate the complete host×job config matrix
  before any further retry.  Do not repair one cell at a time.
- Infrastructure improvement is a post-launch or separate operational task;
  it must not serially delay an otherwise launch-ready scientific campaign.

## Milestones

- [x] Reconstruct the Dynamic-BDD and forensic-audit timing evidence without
      inventing missing request timestamps.
- [x] Add a small reusable timing ledger and host-config matrix rule to the
      orchestration skill.
- [x] Add focused CPU-only regression tests.
- [x] Record the current forensic-audit delay and resulting protocol in a
      human-facing operational note.
- [x] Run changed verification, commit, push, and stop.

## Test plan

- CPU-only ledger init/mark/summary test.
- Existing orchestrator dummy DAG suite.
- `scripts/verify.py --changed --non-scientific`.

## Completion report

This is an operational safeguard only.  It must not alter the ongoing
forensic audit's scientific lineage.
