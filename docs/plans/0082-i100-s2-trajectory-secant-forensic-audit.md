# 0082 — I100 S2 trajectory and secant boundary-distance forensic audit

## Status

- Owner: Codex
- Branch / base SHA: master / `e4166b0897f3394c491f2f486e102e392ef9963d`
- Current milestone: complete; reports, artifacts, verification, and delta review accepted
- Last updated: 2026-09-04

## Goal

Perform a read-only prerequisite for Online-State S2 preservation. Reconstruct
the registered CE-PGD20 e99/e104/e109/e114 state trajectories for the fixed
e99 train S2×T1 cohort and diagnose the corrected Secant Boundary-Distance
(historical S-BDD) formulation with no optimizer/scheduler/model/checkpoint
mutation.

## Frozen boundaries

- No training, optimizer/scheduler step, scientific retry, new seed, tuning,
  stabilization variant, endpoint protocol change, official test, or
  AutoAttack.
- Student state is mutually exclusive: Clean-Wrong; clean-correct S3 non-CW;
  adversarial-correct S2 lower-q10; adversarial-correct non-S2 S1. Teacher
  T1/T2/T3 is recorded separately.
- The historical e104/e109/e114 state replays use an augmented/batch-keyed
  observation contract and therefore cannot be joined to e99 as a primary
  trajectory.  The primary trajectory is regenerated from saved checkpoints
  under e99's raw, unaugmented, sample-keyed CE-PGD20 contract with a frozen
  epoch-99 attack key.  Historical rows remain secondary diagnostics only.
- Any KL-PGD10 runtime replay is separately named a checkpoint no-update
  training-proxy.  It cannot recover historical per-visit activity and must
  not be conflated with the canonical CE-PGD20 branch.
- Historical DPM/D-BDD/S-BDD names remain in provenance; reports use Fixed-
  Cohort Dynamic Pair-Margin, Detached Boundary-Distance, and Secant
  Boundary-Distance Preservation. Here “dynamic” means dynamic teaching target,
  not online mask selection.

## Milestones

- [x] Reconcile source, launch-gate regressions, lineage, and existing artifact availability.
- [x] Implement/test read-only longitudinal state aggregation and fixed-mask runtime-activity proxy.
- [x] Implement/test secant source/formula, scalar and parameter directional finite-difference forensic.
- [x] Run only the missing checkpoint replays/forensics through the frozen host-aware DAG.
- [x] Aggregate both seeds and write provisional reports/machine artifacts; final review identified P1 reporting, finite-difference, and lineage gaps.
- [x] Correct observed-membership/entrant accounting, no-update derivative safety, and replay/proxy lineage validation.
- [x] Replace the one dev-2 DPM proxy whose execution-config hash did not match the registered host rebase; regenerate reports and obtain one delta review.
- [x] Run impact-selected verification, commit/push, and stop.

## Evidence and decision rules

- State paths use only observed endpoints; no “continuous” state claim with
  sparse endpoints.  P1–P6 are explicitly overlapping indicators; the
  membership-pattern table is the mutually exclusive partition.
- The Secant forensic freezes the Student-selected rival, adversarial tensor,
  gate, abs sign, and hinge region for finite differences.  It snapshots and
  restores parameters and buffers bitwise.
- S-BDD classification is one of SBPF1–SBPF5 in the request. Scalar or
  directional finite-difference disagreement is implementation evidence;
  agreement plus small-q/tail coupling supports numerical-formula instability.
- Sensitivity values are diagnostic only and cannot select a new epsilon or
  coefficient.
- An online-state action recommendation is discussion-only; no successor
  experiment is launched automatically.

## Verification plan

- Targeted state aggregation and secant forensic unit/finite-difference tests.
- Existing Dynamic-BDD runtime/aggregation tests.
- Production Launch Gate R1–R22 focused regression suite.
- `scripts/verify.py --changed --non-scientific` after the implementation.
