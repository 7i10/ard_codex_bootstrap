# Post-decay floor calibration

## Status

- Owner: human (approval to spend GPU), Claude Code (execution)
- Branch / base SHA: master, see progress log
- Current milestone: M0, not started
- Last updated: 2026-09-05

## Goal

Measure how far apart two *identical* continuations land.

This is instrument calibration, not a scientific experiment.  It tests no
hypothesis about adversarial robustness distillation.  It measures one number:
the standard deviation of the difference between two untreated forks of the same
epoch-99 parent, run for fourteen epochs past the learning-rate decay, evaluated
at the registered CE-PGD20 endpoint.

That number decides what every future screen can conclude, and it has never been
measured.  `docs/MEASUREMENT_DESIGN.md` brackets it at 0.25 to 0.50 pp from three
indirect estimates that disagree by a factor of two, and notes that the exact
quantity — two untreated forks from a common epoch-99 parent over fourteen epochs
— is the shortest fork of the three, so it could plausibly sit below the whole
bracket.  The bracket is not academic: it moves the number of paired blocks
needed to resolve a 0.17 pp effect from seventeen to sixty-eight.

## Non-goals

- No treatment.  Every arm is an untreated control.  Nothing is being screened.
- No new parent, seed, teacher, attack or schedule.  The design copies plan 0087
  exactly except that all arms are controls.
- No conclusion about PMP, ST1W, TPFM or any other intervention.  This plan
  supplies the denominator those questions need; it does not answer them.

## Existing state

`docs/MEASUREMENT_DESIGN.md` establishes the surrounding picture and is the
reason this plan exists:

- Before the learning-rate decay the floor is 1.14 to 1.25 pp and flat in
  horizon.  Training longer does not help; the decay does.
- After the decay the per-run spread collapses about fourfold within four epochs.
- A shared 79-epoch prefix removes only about six per cent of the variance
  before the decay, so pairing is much weaker protection than it looks.
- Two nominally identical controls in different campaigns differ by 0.94 and
  1.78 pp (`docs/COEFFICIENT_AUDIT.md`).  That is the failure this plan prevents.

`docs/EVIDENCE_RECLASSIFICATION.md` ranks this measurement first among all open
work, as the prerequisite for the other four.

## Design

Fork each of the two I100 epoch-99 parents into **three untreated control
replicates**: identical config, identical schedule, differing only in the
post-fork random stream.  Run epochs 101 to 114 exactly as plan 0087 did, sharing
the same epoch-100 no-action prefix per seed.  Evaluate the registered CE-PGD20
endpoint at epochs 104, 109 and 114.

That yields `2 seeds x C(3,2) = 6` control-versus-control comparisons at each of
three post-decay horizons: the missing cell in the floor table, measured on
exactly the design this project keeps using.

Parents are the plan-0087 lineage: dev-1 `360910a8...7630835`, dev-2
`bb0c7c1a...f7aaf7`, teacher `fc398a48...c383983`, training attack
`97a41870...9623d4d`, endpoint `70811016...dcc4f2`.

## The one open question M0 must settle

"Differing only in the post-fork random stream" needs a precise definition in
this runtime before anything launches.  The attack stream is derived per step
from the training-attack seed, and the data order and augmentation views have
their own seeds, but all seven seeds are part of the recorded scientific
identity.  M0 must decide which seed or seeds a replicate varies, justify the
choice against `docs/SCIENTIFIC_INVARIANTS.md`, and record it in the frozen
contract, because the answer defines what "the same experiment twice" means for
every future screen.  Varying the wrong thing would measure a different and
larger quantity, which is precisely the confusion this plan exists to end.

If the runtime cannot express a same-identity replicate at all, that is itself
the finding, and the plan stops at M0 and reports it.

## Scientific contracts affected

None are changed.  The arms are untreated, so no threat model, normalization,
objective, checkpoint or evaluation contract moves.  The only new recorded
quantity is the replicate index.

## Preregistered analysis

Report, per horizon and per seed, the six pairwise absolute differences in
held-out CE-PGD20 accuracy and their standard deviation.  Report the pooled
estimate across seeds.  State the resulting minimum detectable effect for a
two-arm, k-block design at k = 2, 5 and 10, so a future screen can size itself.

There is no pass or fail.  A large floor and a small floor are equally
informative; only the number matters.  It will be reported with its own
uncertainty, from six comparisons, and not presented as precise.

## Milestones

- [ ] M0: define and freeze what a replicate varies; confirm the runtime can
  express it; confirm the epoch-100 prefixes and frozen thresholds from plan
  0087 are reusable, or rebuild them.
- [ ] M1: launch six continuations from a pinned worktree, epochs 101 to 114.
- [ ] M2: evaluate eighteen endpoints at e104, e109, e114.
- [ ] M3: compute the floor, write the record and report, commit.
- [ ] M4: fold the number into `docs/MEASUREMENT_DESIGN.md` and restate the
  minimum detectable effects there.

## Test plan

`PYTHONPATH=src python scripts/verify.py --changed` for any config or script
addition.  A one-epoch canary on one replicate before the fan-out, as plan 0087
did, to prove the lineage checks pass.

## Risks and mitigations

- **The replicate definition is contestable.** Mitigated by settling it in M0,
  in writing, before any GPU time, and by recording it in the contract.
- **Six comparisons give a noisy estimate of a standard deviation.** True and
  unavoidable at this cost.  The estimate will carry an interval and the report
  will not round it into false precision.
- **The result may widen rather than narrow the bracket.** That is a legitimate
  outcome and does not make the measurement wasted; it would mean short screens
  are even less usable than currently thought.

## Cost

Fourteen epochs at 92.2 s (dev-1) and 95.2 s (dev-2) per epoch is about 22
minutes per run.  Six runs is **2.2 GPU-hours**, plus eighteen endpoint
evaluations.  On two idle GPUs that is under two hours of wall clock.  Less than
the three hours budgeted for option A in decision packet 0001, and a prerequisite
for interpreting option A rather than an alternative to it.

## Progress log

- 2026-09-05: plan authored from the prescription in `docs/MEASUREMENT_DESIGN.md`
  section 3.5 and the ranking in `docs/EVIDENCE_RECLASSIFICATION.md` Part 3(c).
  Not launched; awaiting approval to spend GPU.

## Completion report

Pending.
