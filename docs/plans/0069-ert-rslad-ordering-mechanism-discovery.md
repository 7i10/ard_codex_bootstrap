# ERT / RSLAD ordering mechanism discovery and second intervention

## Status

- Owner: Codex `/root`
- Branch / base SHA: `master` / `33fa2458cd645d0f101c5b543a98c130cf9352a9`
- Current milestone: M0 audit and sampler contract correction
- Last updated: 2026-08-31

## Goal

Use existing RSLAD trajectories and frozen I100 parents to determine whether a
simple batch-level ordering mechanism explains the failed history-balanced
ordering intervention.  Run at most one second history-conditioned ordering
intervention, and stop with hash-bound reports.

## Non-goals

No retraining of the prior ordering policy, no fresh confirmation seeds, no
teacher/history intervention, and no post-hoc policy or threshold sweep.

## Existing state

The accepted history-ordering development campaign is complete but showed no
consistent gain.  `HistoryBalancedSampler` currently labels the highest-margin
samples as HIGH despite documenting HIGH as high-risk; this is a direction
contract bug.  Existing artifacts and trajectories remain immutable.

## Scientific contracts affected

Only the sampler's HIGH/LOW label direction is corrected for future runs.  The
prior campaign is not rerun.  Exact-once exposure, stable-ID tie breaking,
sample-keyed attack RNG, full-batch semantics, and frozen teacher/attack
contracts remain unchanged.

## Decisions

- Correct HIGH risk to mean lowest margin EMA and add a regression test.
- Audit existing batch-level descriptors and gradient geometry before any GPU
  probe or second intervention.
- If mechanism gates fail, produce read-only reports and stop.
- If gates pass, use one pre-registered order-only probe family and at most one
  holdout intervention through the multi-GPU orchestrator.

## Milestones

- [ ] M0: fix sampler direction and focused tests.
- [ ] M1: inventory existing trajectories and compute preregistered D1--D6
  descriptors plus gradient geometry where artifacts permit.
- [ ] M2: freeze and execute the 16 pure-order probes only if M1 permits.
- [ ] M3: select one mechanism/policy or stop if no mechanism is identified.
- [ ] M4: conditionally execute confirm-a/confirm-b holdout intervention.
- [ ] M5: write immutable artifacts/report, review, commit, and push.

## Agent and review budget

No subagent is needed: this is one dependency chain with a single owning
writer.  Use one consolidated scientific review after evidence is stable.

## Test plan

Run the sampler unit tests, direction regression, ruff, and
`python scripts/verify.py --changed`.  GPU probes and production runs are
explicitly deferred until their scientific gates pass.

## Risks and mitigations

- Existing old batch-keyed attack RNG limits causal interpretation; record this
  and do not call old shuffle forks pure ordering evidence.
- Missing batch/order artifacts may block mechanism identification; report
  unavailable cells rather than infer them.
- Parent or endpoint lineage ambiguity is fail-closed.
- All new runs, if any, use immutable manifests, metrics-only W&B, and bounded
  detached orchestration.

## Progress log

- 2026-08-31: reconciled clean `33fa245`; confirmed HIGH/LOW sampler direction
  mismatch and prior history-ordering result is complete.

## Completion report

Pending.  Record exact commands, artifact availability, gate decisions, and
remaining uncertainty here before the final commit.
