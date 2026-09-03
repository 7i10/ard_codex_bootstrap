# 0079 — I100 S2×T1 dynamic boundary-distance intervention screen

## Objective

Run the preregistered eight-trajectory, e99-parent screen comparing dynamic
pair-margin, detached boundary-distance, and indirect secant boundary-distance
interventions on the fixed canonical S2×T1 cohort.  The intervention must keep
the KL-PGD10 inner attack, full-batch RSLAD reduction, frozen Teacher, and all
paired RNG/order contracts unchanged.

## Recovery status (2026-09-03)

The first production attempts are under recovery rather than being blindly
rerun.  The recovery contract is:

- inventory every seed/arm from exact files before launching any GPU work;
- retain only `VALID_COMPLETE` checkpoints plus their hash-bound endpoints;
- repair the S-BDD parameter-gradient formula, recalibrate only S-BDD, and
  treat prior S-BDD output as an invalid detached-secant variant;
- repair host-local parent materialization and launch-state consistency before
  launching only the jobs classified `NOT_STARTED`, `TECHNICALLY_INTERRUPTED`,
  or `SCIENTIFICALLY_INVALID`;
- add fixed-e99-mask longitudinal state-transition diagnostics from existing
  and recovered endpoints without retraining merely to add logging.

The recovery is bounded to the originally registered four arms, two
development seeds, epoch-114 horizon, and CE-PGD20 endpoints.  It does not
introduce an arm, coefficient sweep, threshold, seed, e199 continuation,
official test, or AutoAttack.

## Gates and milestones

- [ ] Reconcile exact e99 parents, masks, Teacher, attack identities, and host
      profiles; fail closed on any hash or path mismatch.
- [ ] Implement the three dynamic pair/geometry loss branches in the shared
      Trainer without changing non-selected baseline samples.
- [ ] Add formula, gradient, detach, teacher-freeze, zero-radius, and
      full-batch-denominator tests.
- [ ] Run pooled no-update calibration and freeze one coefficient per branch
      plus the numerical epsilon in a hash-bound artifact.
- [ ] Run production-launch-gate validation and bounded canaries on both hosts.
- [ ] Launch exactly four arms × two development seeds through the detached
      completion-marker DAG; run only the registered e104/e109/e114 endpoints.
- [ ] Aggregate state effects, runtime cost, lineage, and the preregistered
      BDI1–BDI7 decision; do not extend the experiment.

## Recovery milestones

- [x] R0 — Write the per-job recovery inventory table and classify each job as
      `VALID_COMPLETE`, `VALID_TRAIN_ENDPOINT_MISSING`, `NOT_STARTED`,
      `TECHNICALLY_INTERRUPTED`, or `SCIENTIFICALLY_INVALID` from its files,
      not its exit code alone.
- [x] R1 — Correct the S-BDD Student secant parameter graph, add the
      detached-vs-non-detached gradient regression, and calibrate only the
      corrected S-BDD branch with the original no-update procedure.
- [x] R2 — Prove control/DPM/D-BDD one-batch parity across the S-BDD-only
      source delta and harden launch-gate/Ferret parent materialization,
      completion-probe, and source-freeze checks.
- [x] R3 — Commit and push the recovery source, freeze one recovery manifest,
      and run gate preflight, dry-run, and bounded canary before any recovery
      job is launched.
- [x] R4 — Recover only the non-valid jobs, recover endpoints without
      retraining where permitted, then validate all final completion markers
      and endpoint lineage.
- [x] R5 — Reconstruct fixed e99 S2×T1 state transitions at e104/e109/e114,
      aggregate the registered BDI decision, write the human/machine reports,
      commit, push, and stop.

R5 uses a new read-only CE-PGD20 state replay because the original endpoint
rows retain Student predictions/margins but not the Student adversarial images
or Teacher outputs, and train-split rows were registered only at epoch 114.
The replay is checkpoint-only: it neither resumes nor modifies training.

The corrected `student_parameter_graph_v2` S-BDD implementation was run once
per development seed with its hash-bound v2 calibration and became non-finite
in both runs.  It is therefore frozen as `NUMERICALLY_UNSUPPORTED`: no further
technical retry or in-place floor/cap/smoothed-reciprocal change is permitted
in this screen.  R5 reports Control, DPM, and D-BDD as the causal comparison;
any stabilization proposal requires a separate contract, calibration, and
experiment.

## Scientific decisions frozen for this milestone

- Fixed train/validation S2×T1 masks are anchored at e99; no dynamic selector.
- Student and Teacher share the current Student strongest non-true logit pair.
- DPM uses `0.5*relu(mT-mS)^2`; D-BDD and S-BDD use the exact formulas in the
  experiment contract, with detached Teacher quantities and no second-order
  graph. Geometry uses the preregistered numerical epsilon `1e-12`.
- One pooled calibration coefficient per treatment targets median intervention
  gradient ratio 0.25; no outcome-driven retuning or threshold sweep.
- Production uses metrics-only W&B and local hash-bound checkpoints/artifacts.

## Risk / fail-closed conditions

Do not launch GPU work if parent/mask/Teacher/attack lineage, input-gradient
detach semantics, scheduler/RNG/order parity, or calibration artifact hashing
cannot be proven.  No e199 extension, new seed, official test, AutoAttack,
dynamic routing, or additional treatment is in scope.
