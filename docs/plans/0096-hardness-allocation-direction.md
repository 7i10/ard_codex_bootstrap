# Which samples should get the richer augmentation?

## Status

- Owner: human (approved 2026-09-07 to run), Claude Code (execution)
- Current milestone: M0
- Blocked on: the six environment-v2 parents (`runs/parents-v2`, 1/6 finished).
  M0 cannot start on its own even once they finish: the materialiser looks for
  `<out>/checkpoints/epoch-99.pt` and the trainer writes `<out>/epoch-099.pt`, so
  it will time out at 0/6. Diagnosis and the proposed fix are in plan 0093's
  Progress log (2026-09-07).

## Goal

I100 — weak augmentation to epoch 99, then a richer one from epoch 100 — is this
project's only reproduced result.  It gives every image the same policy.

**This asks which images should get the richer one.**  Two published accounts
answer differently, neither tests it, and the answer is cheap.

## The two accounts

**AROID says hard samples.**  Its policy-learning objective is "Vulnerability,
Affinity and Diversity", where Vulnerability targets the instances most
susceptible to adversarial attack (VERIFIED, arXiv 2306.07197).  The one
published instance-wise augmentation method for adversarial training therefore
sends more augmentation towards the samples the model handles worst.

**IDBH's own account says the opposite.**  Hardness helps only by reducing robust
overfitting and costs clean accuracy in a way that depends on capacity, while
diversity helps unconditionally.  On that reading the budget belongs where there
is margin to spend.  Three instance-wise methods in ordinary training allocate
the same way.

Neither paper tests the direction, and AROID has no random-allocation control.

## Why this is not AROID

AROID learns a policy from a per-instance hardness reward.  This plan does not
learn a policy and does not claim to beat AROID.  It asks a question AROID never
poses: **does the direction of allocation matter, and is a hand-specified
allocation better or worse than giving everyone the same thing?**  A reviewer
will accept that as distinct only if the random-allocation arm is present, so it
is not optional.

## Design

All arms fork from the same epoch-99 parent and share the epoch-100 no-action
prefix.  Before epoch 100 nothing differs.  From epoch 100 the late policy is
IDBH_WEAK for the selected images and CropShift for the rest.

| arm | receives IDBH_WEAK from e100 | role |
| --- | --- | --- |
| `I100` | every image | the existing result, and the comparator |
| **`ALLOC_S1`** | S1 only: adversarially correct with margin above the frozen 10th percentile | **primary** |
| `ALLOC_FRAGILE` | S2 and S3: fragile or adversarially wrong | **declared secondary, same analysis rule** |
| `ALLOC_RANDOM` | a size- and class-matched random draw | the control that makes the question answerable |

`ALLOC_RANDOM` is matched in size and class composition to `ALLOC_S1`.  Its draw
seeds are declared here before anything runs: **`2026090801`, `2026090802`,
`2026090803`** if replicated, otherwise `2026090801`.

Six parents, treated and control branches paired on a shared `continuation_seed`,
two replicates per parent, to **epoch 199**.

**Judged at e199, recorded at e149 and e114.**  Not at e114 alone: I100's own
effect is absent there, gains +0.24 pp by e149 and +0.69 pp by e199
(`docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:51,55`), so a screen that stops
at e114 cannot see an effect of this shape however precise it is.

Endpoint: held-out CE-PGD20 at all three horizons; official test with AutoAttack
on the primary and `I100` only, seen once.

## Why the arms are student-only, with the arithmetic

A teacher term was considered and dropped.  At epoch 100 on dev-1, **94.7 % of S1
images are also T1**: conditioning on the teacher inside S1 changes almost
nothing, and a reviewer can compute that from the released state file in a
minute.  The teacher earns its place only on the disagreement set — 23.9 % of S3
is T1 — and that is a separate plan, deliberately gated on this one.

VERIFIED from `prefix/dev-1/training/online-state/epoch-100.parquet` and the
frozen thresholds; the table is in `docs/METHOD_DIRECTIONS_V2.md`.

## Preregistered rule

Effects are parent-level paired differences against `I100`, at e199, on the
held-out endpoint.  The floor for a paired fork at e199 is not yet measured; plan
0093's twelve controls produce it (that plan's section 8), and this plan's
threshold is set from that number **before** its own results are read.  If plan
0093 has not produced it, the threshold is the e114 value inflated to the widest
end of the pre-registered bracket, 0.25 pp, and that choice is recorded here now.

- **`ALLOC_S1` at least 0.3 pp below `I100`** refutes the capacity account.
- **`ALLOC_FRAGILE` above `ALLOC_S1`** refutes the memorisation account.
- **All three within the floor of each other and of `I100`** closes per-sample
  augmentation allocation in both directions at once.  That is a real result and
  is reported as one, not as a failure.

The primary is `ALLOC_S1` because the evidence points there, not because it is
preferred.  `ALLOC_FRAGILE` is analysed under the identical rule and is reported
whatever it does; it is the direction this project has been reaching for and it
does not get a lower bar or a higher one.

**Clean accuracy is reported alongside** and a drop beyond 0.5 pp disqualifies an
arm regardless of its robust accuracy, as elsewhere in this project.

## What has to be built

Every augmentation transform is keyed on the sample id, so restricting a late
policy to a mask leaves the untreated images' views **bit-identical across
arms**: the comparison is exactly paired everywhere it is not treated.  The
fixed-mask machinery and the class-matched random selector already exist.

What is new: a stagewise late policy that consults a frozen mask.  This touches
the scientific core and goes through review before a source SHA is frozen.

## Cost

123 GPU-hours as scoped; 92 if plan 0093's controls are shared rather than
duplicated, which is the intention.

## Milestones

- [ ] M0: freeze the S1 and fragile masks from each parent's epoch-100 state;
  record their sizes, class composition and hashes
- [ ] M1: implement the mask-gated late policy; review; freeze a source SHA
- [ ] M2: run the four arms on six parents to e199
- [ ] M3: endpoints at e114, e149, e199; aggregate; close

## What this cannot settle

It tests one richer policy against one weaker one at one switch epoch.  A null
does not show that augmentation allocation never matters, only that it does not
matter for this pair at this point in this schedule.

And it says nothing about distillation.  **A result here would be a result about
adversarial training**, which is worth having and is not by itself an argument
for a teacher.  That argument, if it exists, is the disagreement-set plan.
