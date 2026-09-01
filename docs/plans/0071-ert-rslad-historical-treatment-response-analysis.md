# ERT / RSLAD Historical Treatment-Response Analysis

## Status

- Owner: Codex / root
- Branch / base SHA: `master` / `bab6791f7116bf177bcd501d1a9588e4e7c190cb`
- Current milestone: M4 verification / commit
- Last updated: 2026-09-01

## Goal

Produce a read-only, lineage-bound retrospective analysis of available Student-State,
Teacher-State, and History-conditioned interventions, with rescue/harm, direct/spillover/
held-out, temporal, response-prediction, and action-map summaries. Do not train or alter
scientific artifacts.

## Non-goals

- No training, endpoint attack regeneration, new seed, threshold/coefficient tuning, or new intervention.
- No official test, AutoAttack, dynamic routing, augmentation, ordering, or loss changes.
- No relabeling of historical masks or claims of deployable response selection.

## Existing state

The repository is clean at `bab6791`. Historical JSON summaries and several local Parquet/
sample-stat artifacts exist, but availability and lineage differ by campaign. The analysis
must inventory those artifacts before deciding whether any row-level join is possible.

## Scientific contracts affected

Read-only reporting only. Preserve attack identities, parent/source hashes, stable IDs,
historical versus canonical state namespaces, and rescue/harm metric semantics. No model,
optimizer, scheduler, RNG, W&B, or checkpoint state is modified.

## Decisions

- Use existing machine artifacts as source of truth; do not infer missing sample rows.
- Freeze the union response schema before any replay. Missing feature/outcome fields remain unavailable.
- Keep large unified sample-level tables outside Git as compressed Parquet/Arrow with hashes.
- Fit only preregistered low-capacity cross-seed diagnostics when the required rows and lineage exist.
- Treat direct/spillover/held-out and historical/online namespaces as separate estimands.

## Milestones

- [x] M0: inventory artifacts, paths, hashes, lineage, namespaces, and row availability.
- [x] M1: freeze response schema; no replay was required because registered rows were available.
- [x] M2: compute point estimates and response heterogeneity/prediction/temporal/action-map summaries.
- [x] M3: write machine artifacts and human report with limitations and final R1–R5 decision.
- [x] M4: run impact-selected verification, review diff, and commit the read-only milestone.

## Agent and review budget

No subagent is needed: this is a single-owner read-only analysis with no overlapping writes.
One consolidated scientific review is sufficient after evidence and report are stable.

## Test plan

- Targeted JSON/Parquet schema and stable-ID join checks before analysis.
- Public-CLI real-checkpoint smoke only if a missing artifact makes replay necessary; otherwise no GPU work.
- `scripts/verify.py --changed` for changed analysis/report files.
- No full training, live W&B, or AutoAttack tests.

## Risks and mitigations

- Missing row-level content: report aggregate-only/partial cells and do not fabricate joins.
- Historical state-name ambiguity: retain explicit namespace and dual annotation.
- Leakage: freeze feature families and use cross-seed fit/eval without pooled primary fitting.
- Disk pressure: keep large tables in `.cache/analysis`, commit only summaries/hashes.
- Lineage drift: hash every source artifact and fail closed on mismatched IDs/attacks/parents.

## Progress log

- 2026-09-01: reconciled repository at `bab6791`; required docs read; started artifact/path inventory.
- 2026-09-01: registered endpoint inventory found 356 row artifacts; built 5,100,000-row local Parquet response table and seven compact JSON reports without training or endpoint regeneration.
- 2026-09-01: `scripts/verify.py --changed` reached the changed-file test tiers but reported one pre-existing environment-sensitive failure in `tests/unit/test_schedule_control_fork.py` (CUDA RNG state size mismatch); no analysis test failed.
- 2026-09-01: committed as `ab66718` and pushed to `origin/master`; `.cache/analysis/ert-rslad-historical-treatment-response-v1/outputs/response_rows.parquet` remains local-only.

## Completion report

Completed in `ab66718`. Commands: `python scripts/analysis/ert_rslad_historical_treatment_response.py`, `python -m py_compile scripts/analysis/ert_rslad_historical_treatment_response.py`, and `scripts/verify.py --changed`. Coverage: Stage A, broad Clean-Wrong C0–C15, confirmatory T1/T2/T3, dynamic/history S3 endpoints, and registered aggregate-only sources. Limitations: no row-level data for some gated/ordering historical cells; no pooled prediction; direct-to-held-out associations are descriptive. Final decision is `RESPONSE_NOT_PREDICTABLE` for a deployable universal selector.
