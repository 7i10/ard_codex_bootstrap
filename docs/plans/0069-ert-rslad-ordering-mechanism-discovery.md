# ERT / RSLAD ordering mechanism discovery and second intervention

## Status

- Owner: Codex `/root`
- Branch / base SHA: `master` / `3592a57a9c70951d796bcf2d18c7edffc6263f59`
- Current milestone: M5 complete; mechanism gate failed closed
- Last updated: 2026-09-01

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

- [x] M0: fix sampler direction and focused tests.
- [x] M1: inventory existing trajectories and assess preregistered D1--D6
  descriptor availability.
- [x] M2: e99 gradient geometry and eight-schedule pure-order probes.
- [x] M3: apply the preregistered mechanism association gate (failed closed).
- [x] M4: no second intervention because M3 failed.
- [x] M5: write immutable artifacts/report and commit.

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
- 2026-08-31: corrected the sampler in commit `8111f23` and pushed it. Existing
  NEW_HISTORY artifacts expose only epoch-level permutation hashes and final
  sample state; no batch risk/order telemetry exists. Exact e99 I100 parents
  matched. Phase A therefore failed closed and no GPU work was started.
- 2026-08-31: added the corrected pure-order telemetry path, e99 gradient
  geometry runner, immutable eight-offset schedule registry, and detached-probe
  aggregation. Production launch remains gated on focused tests, clean source,
  and the e99 geometry result.
- 2026-09-01: all 16 corrected pure-order forks completed with telemetry. The
  detached aggregate hit an AUC helper bug (`zip(..., strict=True)` on
  adjacent lists); the helper was fixed and the result recomputed from the
  completed artifacts. All D1--D6 mechanism associations failed the frozen
  cross-seed gate, so no second intervention was launched.

## Completion report

The corrected direction regression and existing-run audit passed. The e99
geometry and all 16 corrected pure-order probes completed, but the frozen
mechanism association gate failed for every descriptor. The second intervention
is therefore blocked and the campaign stops with the hash-bound result/report.
