# Experiment Dashboard and W&B Curation

## Status

- Owner: main thread
- Branch / base SHA: `master` / `515d7cc`
- Current milestone: complete
- Last updated: 2026-07-29

## Goal

Provide one human-readable page that explains the campaign purpose, immutable conditions, outputs, completed/running/
queued work, current results, and the role of every W&B run. The page must distinguish paper-candidate evidence from
pilots, incompatible reference runs, and failed development attempts.

## Non-goals

- Do not delete, stop, restart, or modify a W&B run or experiment.
- Do not change training, attack, evaluation, checkpoint, or tracking code.
- Do not draw efficacy conclusions from seed 0 or compare validation metrics with official test metrics.

## Existing state

- The seed-0 campaign runs at scientific SHA `2d54b82` on Hamster and Ferret with a fixed single-GPU execution profile.
- Results and current state are distributed across campaign JSON, metric streams, evaluation manifests, W&B, and the
  existing handoff document.
- The user reports 26 W&B runs. They have not yet been classified in a single checked-in index.
- The worktree was clean at base SHA `515d7cc`.

## Scientific contracts affected

No scientific runtime contract changes. Reporting must preserve checkpoint (`best`/`last`), dataset split,
evaluation attack, sample count, world size, BatchNorm profile, seed, Git SHA, and `job_type` distinctions.

## Decisions

- Query W&B read-only. Retain all runs during the active campaign; recommend later deletion only for proven
  development failures with no unique artifact or lineage value.
- Treat W&B train and evaluation runs as linked records, not duplicates.
- Keep the older world-size-two run as a reference only; never pool it with the world-size-one campaign.
- Use no subagent or scientific re-review because this is a bounded documentation/inventory delta with no scientific
  code change.

## Milestones

- [x] D0 — Capture current Hamster, Ferret, and W&B inventories.
  - Files: read-only runtime manifests and W&B metadata.
  - Tests: schema/consistency checks in the inventory command.
  - Acceptance: every visible run is assigned a status and evidence class.
  - Rollback: no external mutation.
  - Commit: included with D2.
- [x] D1 — Write the human experiment dashboard.
  - Files: `docs/EXPERIMENT_DASHBOARD.md`.
  - Tests: manual link/table consistency and docs-only impact selection.
  - Acceptance: overview, purpose, conditions, outputs, status, results, and limitations are explicit.
  - Rollback: additive documentation file.
  - Commit: included with D2.
- [x] D2 — Add navigation, verify, commit, and push.
  - Files: `docs/README.md`, this plan.
  - Tests: `python scripts/verify.py --changed`; no GPU/scientific tests.
  - Acceptance: dashboard matches the captured evidence and the worktree is clean after the cohesive commit.
  - Rollback: revert the documentation commit.
  - Commit: `Document experiment status and W&B run roles`.
- [x] D3 — Add paper-reference targets and classify reproduction fidelity.
  - Files: `docs/EXPERIMENT_DASHBOARD.md`, this plan.
  - Tests: primary-paper table/source cross-check and docs-only changed-path gate.
  - Acceptance: original RSLAD, SAAD RSLAD analysis, full SAAD, and entropy-only references are not conflated;
    Student/Joint are explicitly identified as new ablations without published target values.
  - Rollback: revert the reference-only documentation delta.
  - Commit: `Compare current results with published baselines`.

## Agent and review budget

No subagent is needed. This avoids repeating already-reviewed scientific context and does not spend a reasoning-heavy
review on a read-only reporting delta.

## Test plan

- Validate that completed official evaluations state checkpoint, attack, and sample count.
- Verify that W&B API count matches the inventory table.
- Run the docs-only changed-path gate once; do not run GPU, training, or numerical tests.

## Risks and mitigations

- Stale live status: include an observation timestamp and regenerate from manifests/W&B rather than presenting it as
  permanent state.
- Misleading comparison: separate validation, official PGD-20, and AutoAttack; mark seed-0 results exploratory.
- Destructive cleanup: perform no W&B deletion. Preserve train/evaluation lineage and artifact-bearing runs.
- W&B/local lag: prefer local terminal manifests for phase truth and W&B for tracking identity; disclose conflicts.

## Progress log

- 2026-07-29: Started from clean `515d7cc`; documentation and W&B protocol reviewed. No external mutation authorized
  or performed.
- 2026-07-29: Read-only W&B API inventory found 26 runs: 14 train/12 evaluation, 23 finished/3 running, and artifacts
  on every run. Classified 13 canonical single-GPU production records, 2 world-size-two references, 6 accepted pilot
  records, and 5 superseded/legacy pilot records.
- 2026-07-29: Hamster and Ferret manifests were reconciled at 14:18 JST. The dashboard records 3 cells with all
  planned phases complete, 1 with PGD complete and AutoAttack queued, 3 training, and 1 queued. Official 10,000-example
  best/last PGD and available AutoAttack results were copied from evaluation result files.
- 2026-07-29: `git diff --check` passed. `/home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify.py --changed`
  selected T0 and reported `no impacted tests`; no GPU or scientific test was run for the docs-only delta.
- 2026-07-29: Reopened the dashboard for a primary-source comparison after the user correctly noted that current
  controlled reproduction results need their paper targets and method provenance. GPU reassignment remains a separate
  operational decision because the current runner fixes host/GPU in its immutable campaign identity.
- 2026-07-29: D3 added the original RSLAD best result, SAAD Chen/Bartoldson RSLAD records, full-SAAD Bartoldson
  result, and the Gowal entropy-weighting compatibility result. The dashboard now labels the active campaign as a
  controlled SAAD-analysis reproduction plus new Student/Joint ablations, not an exact RSLAD or full-SAAD reproduction.

## Completion report

Added `docs/EXPERIMENT_DASHBOARD.md` and linked it from the documentation index. The dashboard separates live status,
official results, execution profiles, output locations, W&B roles, and current scientific limitations. A reporting
audit also corrected stale student/joint fallback wording in two contract documents to match the already-implemented
schema-v2 target-softening semantics. No training, evaluation, W&B metadata, artifact, or run was changed or deleted.
