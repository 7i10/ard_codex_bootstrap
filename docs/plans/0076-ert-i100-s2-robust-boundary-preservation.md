# 0076 — I100 canonical S2 robust-boundary preservation screen

## Status

- Owner: Codex
- Status: complete
- Branch / base SHA: `ada40aac96b1832fd14ca0ab046b71a0c7bb255a`
- Current milestone: completed; scientific decision recorded
- Last updated: 2026-09-03

## Goal

Run the preregistered short e100–114 screen from the exact I100 e99 parents,
comparing a Student-only canonical S2 margin floor with the historical
Teacher-positive-floor margin mechanism on fixed Train S2×T1 IDs, then evaluate
fixed Validation S2×T1 retention at e104/e109/e114 and produce lineage-bound
reports.  Stop after the decision; do not extend to e199 or add History/Clean-
Wrong stacking.

## Non-goals

- No e115–199 extension, new seed, dynamic/History routing, Clean-Wrong TPFM,
  coefficient or threshold sweep, official test, or AutoAttack.
- No changes to I100 augmentation, RSLAD objective, KL-PGD10, Teacher, sampler,
  scheduler, or RNG contract.

## Existing state

The exact epoch-99 I100 checkpoints are present and hash-verified.  Existing
sample-keyed CE-PGD20/KL-PGD10 e99 replay rows provide the canonical state
features.  The shared Stage A runtime already supports fixed and Teacher-floor
margin targets; this milestone adds only S2×T1 mask/calibration preparation and
the campaign-specific aggregation.

## Scientific contracts affected

The screen consumes the frozen positive-margin q10 S1/S2 and T1/T2/T3 state
contract, uses fixed epoch-99 masks, and calibrates one coefficient per margin
mechanism from pooled no-update gradients.  Training/evaluation attacks,
full-batch reduction, Teacher freeze, deterministic resume, and metrics-only
W&B retention remain unchanged.

## Decisions

- Use the existing e99 replay rows rather than re-running an equivalent replay;
  parent, rows, attack identities, and source lineage are hash checked.
- Derive Student floor from each seed's positive Student CE-PGD20 q10 boundary.
- Derive Teacher floor/cap from pooled positive Teacher margins in the selected
  S2×T1 cohort using the registered q25/q75 procedure; do not reuse outcome
  values to select them.
- Use one pooled coefficient for SBF and one pooled coefficient for TPFM, with
  the registered 0.25 gradient-ratio target.
- Run all six arms on Hamster GPU0/GPU1 through the reusable orchestrator; GPU2
  and remote Ferret are unnecessary for this two-slot campaign.

## Milestones

- [x] M0 audit parents, rows, config, attack, and canonical state counts.
- [x] M1 generate fixed train/validation S2×T1 masks and pooled no-update
  calibration artifact.
- [x] M2 add/execute focused contract tests and CPU/fixed-batch canary; commit
  the immutable source before GPU launch.
- [x] M3 validate/preflight/plan and launch six detached training jobs.
- [x] M4 chain e104/e109/e114 validation and e114 train CE-PGD20 endpoints.
- [x] M5 aggregate state effects/runtime, write human and machine reports,
  review, commit, push, and stop.

## Agent and review budget

One planning pass and one consolidated scientific review are sufficient.  A
read-only planner is already assigned for the contract audit; the main agent
owns implementation and integration.  No additional writer is needed because
all new files share one API and one campaign identity.

## Test plan

- Existing impact-selected runtime/calibration/margin tests (cached where
  unchanged).
- New fixed-mask/schema/calibration contract tests and a no-update gradient
  smoke on the real e99 parent and sparse selected IDs.
- Orchestrator `validate`, `preflight`, and `plan --dry-run` before launch.
- Production training and CE-PGD20 endpoints remain outside automated tests.

## Risks and mitigations

- Missing/foreign parents or replay rows: fail closed on SHA, epoch, config,
  split, and attack identity.
- Incorrect q10 tie handling: sort `(margin, stable_id)` and assert exact
  positive-correct counts.
- Calibration mutation: run without optimizer/scheduler/state updates and
  snapshot model buffers.
- Wrong resume boundary: require payload epoch 99/end, `--resume-epoch 99`,
  and `--epochs 115` with horizons 104/109/114.
- Endpoint or W&B storage cost: use local checkpoints and metrics-only W&B;
  no model/run-bundle uploads.

## Progress log

- 2026-09-03: HEAD and exact source/replay contracts inspected; both e99 parent
  checkpoint hashes are present locally.
- 2026-09-03: Fixed masks and pooled no-update calibration completed; focused
  tests and Hamster GPU0/GPU1 canaries passed.
- 2026-09-03: Recovery6 completed 5 treatment/control continuations, six
  CE-PGD20 endpoint jobs, and runtime aggregation. A prior W&B-ID collision was
  corrected at the tracking layer without changing scientific identity.
- 2026-09-03: Fixed validation S2×T1 boundary effects and train direct effects
  were aggregated. Both mechanisms were classified MIXED/descriptive; neither
  was promoted and no extension was started.

## Completion report

Completion report: `docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md`.
Machine result: `docs/experiments/ert_rslad_i100_s2_rbp_results_v1.json`.
