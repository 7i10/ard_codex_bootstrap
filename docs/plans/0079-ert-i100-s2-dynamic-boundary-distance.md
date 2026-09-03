# 0079 — I100 S2×T1 dynamic boundary-distance intervention screen

## Objective

Run the preregistered eight-trajectory, e99-parent screen comparing dynamic
pair-margin, detached boundary-distance, and indirect secant boundary-distance
interventions on the fixed canonical S2×T1 cohort.  The intervention must keep
the KL-PGD10 inner attack, full-batch RSLAD reduction, frozen Teacher, and all
paired RNG/order contracts unchanged.

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
