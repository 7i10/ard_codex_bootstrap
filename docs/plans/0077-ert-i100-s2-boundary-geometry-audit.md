# 0077 — I100 canonical S2×T1 boundary geometry audit

## Status

- Owner: Codex
- Status: complete
- Branch / base SHA: `3367c72cf9e20337bb94bec20ddb6052f180c7c2`
- Current milestone: complete read-only replay, analysis, and report
- Last updated: 2026-09-03

## Goal

Use the exact I100 epoch-99 parents and the fixed canonical S2×T1 cohort to
measure scalar Student/Teacher input-boundary geometry at the registered
Student CE-PGD20 point, then test whether the geometry adds cross-seed
predictive information for future control robust failure.  This is a read-only
analysis; no training or intervention is allowed.

## Non-goals

- No new training, intervention, threshold/coefficient search, seed, History
  routing, official test, or AutoAttack.
- No raw gradient-vector artifact or geometry-based selector.

## Existing state

The exact e99 I100 parents, fixed S2×T1 masks, and e104/e109/e114 control
CE-PGD20 row artifacts are available from the completed boundary-preservation
screen.  The registered validation feature replay is the anchor source; its
metadata is audited before replay because its human-readable random-start
label must not override the machine replay protocol.

## Scientific contracts affected

The replay uses eval-mode frozen Student/Teacher models, pixel-space input
gradients, Student strongest rival shared by both models, and first-order
L∞ distance proxies.  It must not update model parameters, optimizer,
scheduler, sample state, or tracking state.  Future labels come only from
existing I100_CONTROL endpoint rows.

## Decisions

- Primary scope is validation S2×T1; train S2×T1 is a secondary diagnostic if
  the exact full-train replay can be completed without changing the contract.
- The registered CE-PGD20 replay JSON is authoritative for the existing
  validation anchor protocol; any conflicting descriptive label is recorded,
  not silently “fixed.”
- Cross-seed predictors use fixed ridge logistic regression (`alpha=1.0`),
  train-seed standardization only, and no pooled fitting or tuning.
- Geometry results are mechanism-consistent predictive evidence only and never
  trigger a new loss or route.

## Milestones

- [x] M0 audit parents, masks, replay protocol, endpoint rows, and source.
- [x] M1 implement scalar-only geometry replay, canary, and fixed-contract
  tests.
- [x] M2 run validation (and secondary train where feasible) replay on Hamster
  GPUs; merge shards by stable ID.
- [x] M3 compute future-failure joins, univariate/cross-seed analyses, cells,
  and machine artifacts.
- [x] M4 write report, run focused verification, commit, and stop.

## Agent and review budget

One owning writer (Codex) and one consolidated self-review are sufficient;
the task is bounded analysis implementation and does not need additional
agents.  GPU replay is analysis-only and uses the existing orchestrator when
long-running shards are required.

## Test plan

- New pure-metric and contract unit tests for pair selection, distance/cosine,
  joins, fixed logistic feature sets, and non-finite handling.
- One real checkpoint/real sparse validation canary through the public replay
  CLI, including batched-vs-single input-gradient equality.
- Orchestrator validate/preflight/plan before GPU replay; production training
  remains outside tests.

## Risks and mitigations

- Wrong parent, attack, rival pair, or endpoint identity: fail closed on SHA,
  epoch, split, and complete attack identity.
- BatchNorm cross-sample gradient contamination: eval mode plus batched/single
  canary.
- Hidden parameter updates: use `autograd.grad` on inputs only and assert
  frozen Teacher parameters have no gradients.
- Large raw gradients: reduce to scalars immediately and discard tensors.
- Sample-level uncertainty being mistaken for seed inference: report the two
  cross-seed directions descriptively only.

## Progress log

- 2026-09-03: exact parent bytes, masks, endpoint rows, and existing feature
  replay metadata inventoried; no training started.
- 2026-09-03: validation and secondary train scalar replay completed on Hamster
  GPU 0/1 with the fixed e99 parents; validation and e114 train endpoint joins
  produced the required reports.  Distance-gap geometry was classified as
  BG2 descriptively; no intervention was launched.

## Completion report

Completed in `docs/ERT_RSLAD_I100_CANONICAL_S2_BOUNDARY_GEOMETRY_AUDIT.md` and
the versioned `docs/experiments/ert_rslad_i100_s2_geometry_*.json` artifacts.
