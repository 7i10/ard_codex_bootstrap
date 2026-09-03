# 0080 — I100 Dynamic BDD execution postmortem

## Status

- Owner: Codex
- Branch / base SHA: master / `5a45a41c3a9925c995ea52ec45fc44228cf2d41c`
- Current milestone: complete
- Last updated: 2026-09-04

## Goal

Record the evidence-backed operational delays in the completed I100 Dynamic
BDD screen, add one narrow regression preventing the repeated wrapper launch
failure, and provide a reusable prompt/launch checklist without changing any
scientific result.

## Non-goals

- No training, endpoint rerun, result reinterpretation, or scientific retry.
- No change to model, attack, loss, calibration, seed, parent, or W&B policy.
- No host-specific hard-coded scheduling policy.

## Decisions

- Treat user-reported request time as an approximate timestamp and distinguish
  it from artifact-backed controller/commit times.
- Treat S-BDD non-finiteness as a scientific outcome, not an orchestration
  retry opportunity.
- Reject only a locally present, directly invoked non-executable `*.sh` argv;
  do not claim to validate a remote-only executable path.

## Milestones

- [x] Reconstruct timeline and technical failures from manifests, logs, and
      Git history without inventing missing timestamps.
- [x] Add the shell-wrapper manifest-validation regression and focused test.
- [x] Record agent-controlled improvements and prompt authoring guidance.
- [x] Run focused tests and changed verification, commit, push, and stop.

## Test plan

- `pytest -q tests/skills/test_multi_gpu_orchestrator.py`
- `python scripts/verify.py --changed`

## Completion report

The postmortem is operational only. It records the request-to-controller
launch delay, technical recovery events, and the remaining evidence limits.
