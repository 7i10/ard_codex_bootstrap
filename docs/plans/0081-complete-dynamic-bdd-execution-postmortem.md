# 0081 — Complete Dynamic BDD execution postmortem

## Status

- Owner: Codex
- Branch / base SHA: master / `8ddb7add56957dea27d1581021ff4a50c3cf70dc`
- Current milestone: complete
- Last updated: 2026-09-04

## Goal

Close the Dynamic BDD remote-launch failures with generic, machine-enforced
Production Launch Gate and orchestrator contracts.  Add R13–R22 regression
coverage, bounded Hamster/Ferret lifecycle verification, and an evidence-based
postmortem closure record without changing a scientific campaign.

## Non-goals

- No training, endpoint evaluation, method/configuration/seed/parent change,
  scientific retry, W&B production run, or result reinterpretation.
- No replacement remote scheduler; `run-on-ferret` remains the SSH lifecycle
  authority and the existing orchestrator remains the DAG controller.

## Decisions

- Remote evidence is supplied by an explicit bounded external preflight/status
  command and validated locally; the launch gate does not implement SSH.
- An external wrapper spawn is distinct from a host-confirmed start.  A valid
  remote status payload is required before a completion probe can run.
- Remote artifacts become aggregation inputs only after canonical local copy,
  exact SHA verification, and a complete identity-bound inventory.
- The S-BDD non-finite result remains a scientific outcome, outside technical
  retry and launch-regression mechanisms.

## Milestones

- [x] Audit current Gate/orchestrator/Ferret contracts and record the R13–R22 design.
- [x] Implement remote preflight, source-freeze, wrapper, and host-confirmation contracts.
- [x] Implement canonical artifact collection/inventory validation and lock regression coverage.
- [x] Update skill/protocol/gate/postmortem documentation and closure states.
- [x] Run CPU-only unit/integration tests and bounded Hamster/Ferret verification.
- [x] Commit, push, and stop.

## Test plan

- `pytest -q tests/skills/test_production_launch_gate.py`
- `pytest -q tests/skills/test_multi_gpu_orchestrator.py`
- `pytest -q tests/remote/test_ferret_scripts.py`
- `python scripts/verify.py --changed --non-scientific`
- `bash -n` for changed Ferret scripts and a bounded no-training Ferret lifecycle run.

## Completion record

- Infrastructure source: `4b3aad3f436920202e1df371b110dcc122d4266a`
  (`d8193ce1a4653126f52004f504a8080b9c340b23` corrects the emitted Ferret
  status payload used by host confirmation).
- Focused unit/integration regressions: `67 passed`.
- Diff-selected non-scientific verification: `24 passed`.
- Bounded Ferret lifecycle: fixed-SHA worktree, `/bin/sleep 6`, identity-bound
  live host confirmation on GPU 0, terminal completion, collection, and
  matching manifest SHA. No scientific training, endpoint evaluation, or W&B
  run was performed.
