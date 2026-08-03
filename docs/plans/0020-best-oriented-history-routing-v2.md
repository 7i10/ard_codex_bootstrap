# Best-oriented student-history routing v2

## Status

- Owner: main thread; one planning pass, one core implementation owner, one
  consolidated scientific review after focused evidence is stable
- Base scientific SHA: `cd56b729948782e82d446b186a58bd89a1817303`
- Current milestone: M0 design frozen; implementation not started
- Last updated: 2026-08-03

## Goal

Test whether exact online student history can select samples for a bounded
true-label-anchored RSLAD intervention that improves **Best validation PGD**,
while separating selector value from a generic intervention effect.  Do not
use official CIFAR-10 test or AutoAttack for development decisions.

## Evidence and fixed decisions

- H5-Early passed prospectively on both Bartoldson development trajectories.
  The earliest passing anchor is epoch 39 for both anchor-correct peak failure
  and anchor-wrong non-recovery.
- The deployable score is computed once from the epoch-39 exact online
  `SampleStateStore`:

  `0.5 * percentile_rank(1 - robust_correct_frequency) +`
  `0.5 * percentile_rank(-margin_ema)`.

  It includes observations through epoch 39, uses stable sample ID for ties,
  and does not use a frozen external predictor or future labels.
- Selection budget is top `q=10%` **within each disjoint anchor state**.  This
  is a run-local percentile, not a fitted probability threshold.  The budget,
  formula, and anchor cannot be tuned after arm results.
- Bartoldson teacher-wrong-only routing has insufficient mass for the primary
  Best objective.  Teacher correctness, wrong-confidence, entropy, clean/adv
  response, and selected-group prevalence are logged as moderators only.
- `B*` is ordinary RSLAD with delayed milestones `[120,170]`.  It is chosen as
  the stronger matched control because it consistently improved Best, Last,
  and RO gap over normal RSLAD, although its mean Best gain was only
  `+0.12 pp` and is not itself a material result.
- The intervention changes only the adversarial RSLAD teacher target for
  selected samples:

  `target = 0.5 * p_teacher(clean) + 0.5 * one_hot(y)`.

  Clean KD, attack, temperature, branch coefficient, loss reduction, training
  schedule, and non-selected samples remain unchanged.  Hard-label replacement
  is a separate untested method and is not silently combined with this arm.

## Development design

Use L1/seed 1 and L3/seed 2 epoch-39 checkpoints.  Their trajectories are
identical to the delayed schedule before the first delayed milestone.  Reuse
the completed SC-L1/SC-L3 runs as no-routing `B*` controls.

| Arm | Eligible state | Selection | Intervention |
|---|---|---|---|
| `C` | all | none | delayed ordinary RSLAD; reuse SC-L1/SC-L3 |
| `PF-TA` | epoch-39 robust-correct | top 10% history score | true-label anchor |
| `PF-R` | same | class/state/count-matched random | true-label anchor |
| `NR-TA` | epoch-39 robust-wrong | top 10% history score | true-label anchor |
| `NR-R` | same | class/state/count-matched random | true-label anchor |

One deterministic random mask is frozen per parent and route before training.
Across L1/L3 this gives two independent training seeds and two matched random
masks.  Random arms are diagnostic controls, not extra method candidates: they
prevent a later rerun merely to determine whether selection or generic
true-label regularization caused the result.

All children resume the exact model, optimizer, scaler, RNG, sampler, sample
state, and global step from epoch 39; intervene from epoch 40 through 199; and
use delayed milestones `[120,170]`.  PF and NR risk sets must remain disjoint.

## Preregistered development gates

Evaluate Best/Last clean and validation PGD, Best epoch, RO gap, and post-anchor
validation-PGD AUC.  Candidate selection uses validation only.

A history arm is eligible only if all conditions hold:

- two-seed mean `history - C` Best PGD is at least `+0.50 pp`;
- each seed has non-negative `history - C` Best PGD;
- two-seed mean `history - matched-random` Best PGD is positive and each seed
  is non-negative;
- mean Best clean degradation versus C is at most `0.50 pp`;
- mean RO gap does not worsen versus C.

If both PF and NR pass, select the larger mean Best-PGD gain; if their gains
differ by less than `0.10 pp`, select NR because it changes fewer samples.  If
neither passes, stop this true-label-anchor family.  Do not tune `q`, anchor,
mixing coefficient, or score on the same L1/L3 outcomes.

## Milestones

- [x] M0 -- freeze evidence, formula, controls, gates, and official-test seal.
- [ ] M1 -- implement the minimal reusable route.
  - Add `teacher_target_true_label_mix@1` behind the existing target-policy
    boundary; do not duplicate the trainer.
  - Generate PF/NR history and matched-random masks once at epoch 39 with
    checkpoint/state/config/source hashes, selected IDs, class counts, and
    anchor-state counts.
  - Reuse fixed-mask loading, stable IDs, DDP broadcast, checkpoint/resume,
    RSLAD attack/objective, and tracker lineage.
  - Keep teacher moderator observations detached from loss.
- [ ] M2 -- focused verification and one real-checkpoint branch smoke.
  - Formula and selected-only target application; finite unreduced loss and
    gradients; teacher parameter gradients remain `None`.
  - Attack/threat identity, clean branch, non-selected samples, and loss scale
    remain unchanged.
  - Epoch-39-only input, exact 10%, PF/NR disjointness, deterministic ties,
    class/state/count-matched random, DDP agreement, checkpoint/resume parity.
  - Run one synthetic GPU smoke and one real epoch-39 single-branch smoke,
    then one focused changed-path gate and one consolidated scientific review.
- [ ] M3 -- launch the L1/L3 development block using longest-job-first host
  placement and hash-verified transfer where cheaper than recomputation.
  Reuse C; run the eight new history/random continuations.  Do not add an
  operational campaign framework to the scientific core.
- [ ] M4 -- apply the frozen development gate and select at most one route.
  No official test/AA and no post-result hyperparameter adjustment.
- [ ] M5 -- only after Development Go, run the selected frozen route and its
  matched delayed control on unseen Bartoldson seeds 3 and 4.
- [ ] M6 -- only after both Bartoldson confirmations pass, run Chen seeds 3
  and 4 as no-harm checks; then freeze and open official evaluation once.

## Confirmation and stop rules

For Bartoldson seeds 3/4, Confirmation Go requires:

- mean Best validation-PGD gain over matched delayed C at least `+0.50 pp`;
- both seed differences non-negative;
- mean Best clean degradation at most `0.50 pp`;
- mean RO gap non-worsening.

Only then run Chen seeds 3/4.  Chen no-harm requires mean Best-PGD difference
non-negative, each seed at least `-0.50 pp`, mean clean degradation at most
`0.50 pp`, and no anomalous routing prevalence.  Entropy-only remains a
secondary RO/stability baseline and is added on matching unseen seeds only
after Bartoldson confirmation; it does not choose the candidate.

Stop this route family if Development fails, either Bartoldson confirmation
seed is negative and the two-seed mean misses `+0.50 pp`, or Chen no-harm
fails.  A failed route does not invalidate the already confirmed predictive
student-history result; it means the prescriptive intervention failed.

## Risks

- A random arm can improve through generic regularization.  Matched random
  controls separate that from history selection without claiming a full random
  mask distribution.
- Teacher-correctness moderation may differ sharply between Bartoldson and
  Chen.  It is reported, not used to retune the Bartoldson route.
- Development L1/L3 are not confirmation seeds.  No final method claim is
  made from them.
- The intervention begins before the original peak; Best gains cannot be
  attributed solely to delayed scheduling because C uses the same schedule.
- Official test remains sealed until the route and confirmation results are
  frozen; subsequent methods require fresh seeds or datasets.
