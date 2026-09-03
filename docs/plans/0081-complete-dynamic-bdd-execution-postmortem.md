# 0081 — Complete Dynamic BDD execution postmortem

## Status

- Owner: Codex
- Branch / base SHA: master / `8ddb7add56957dea27d1581021ff4a50c3cf70dc`
- Current milestone: in progress
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

- [ ] Audit current Gate/orchestrator/Ferret contracts and record the R13–R22 design.
- [ ] Implement remote preflight, source-freeze, wrapper, and host-confirmation contracts.
- [ ] Implement canonical artifact collection/inventory validation and lock regression coverage.
- [ ] Update skill/protocol/gate/postmortem documentation and closure states.
- [ ] Run CPU-only unit/integration tests and bounded Hamster/Ferret verification.
- [ ] Commit, push, and stop.

## Test plan

- `pytest -q tests/skills/test_production_launch_gate.py`
- `pytest -q tests/skills/test_multi_gpu_orchestrator.py`
- `pytest -q tests/remote/test_ferret_scripts.py`
- `python scripts/verify.py --changed --non-scientific`
- `bash -n` for changed Ferret scripts and a bounded `/bin/true` Ferret lifecycle run.
