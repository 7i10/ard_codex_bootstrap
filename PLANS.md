# Execution plan template

Create one file per substantial task under `docs/plans/NNNN-short-title.md`.

```markdown
# <Plan title>

## Status

- Owner:
- Branch / base SHA:
- Current milestone:
- Last updated:

## Goal

State the observable end condition.

## Non-goals

List work that is intentionally excluded.

## Existing state

Summarize relevant files, current behavior, upstream dependencies, and uncommitted changes.

## Scientific contracts affected

List threat-model, gradient, normalization, checkpoint, evaluation, or tracking contracts that can change.

## Decisions

Record decisions and rejected alternatives with concise rationale.

## Milestones

- [ ] M0 ...
- [ ] M1 ...

For each milestone include:

- files/modules
- implementation owner agent and why delegation is needed
- tests selected by impact
- acceptance criteria
- rollback point
- one planned commit boundary/message

## Agent and review budget

State the minimum role usage for the task. Default to one planner pass for genuinely complex work, one owning writer, one batched Luna synchronization after API freeze, and one consolidated scientific review. Subsequent review should cover only the changed delta and requires a remaining P0/P1 or new evidence. Record when no subagent is needed.

## Test plan

List the minimum tests required. Mark tests that are cached, newly required, GPU-bound, or intentionally deferred.

## Risks and mitigations

Include scientific correctness, numerical parity, licensing, runtime, memory, DDP, W&B, and resume risks.

## Progress log

Append dated, concise entries. Record decisions and exact result summaries, not raw terminal output or repeated closed findings.

## Completion report

Summarize implementation, actual commands run, cached passes, deferred scientific runs, reviewer findings, and remaining uncertainty.
```
