# Was it the state, or was it the size of the group?

## Status

- Owner: human (approval to spend GPU), Claude Code (execution)
- Created 2026-09-07
- Current milestone: M0

## Goal

Every state-conditional treatment in this project selects a cohort of images by
their current state — adversarially fragile, clean-wrong, and so on — and applies
an extra loss to it.  Nothing has ever tested whether **the selection is doing
the work**.

This plan applies the same treatment, at the same coefficient, for the same
number of epochs, to a cohort of the **same size and the same class composition**
drawn at random.  If the random cohort moves the held-out result as much as the
state-selected one, then the state was never the mechanism, and an entire family
of results closes.

It also produces a number that is missing and that several past verdicts needed:
**the floor for a treated-versus-control contrast that shares a random stream.**

## Why the existing floor does not answer this

Plan 0092 measured 0.092 pp at epoch 114.  That is the spread between untreated
runs whose `continuation_seed` differs — different random streams.

Every historical treated-versus-control contrast is a different animal.  Plan
0087's `control`, `pmp` and `dbdp` arms all record `continuation_seed: null`
(VERIFIED in their `arm-summary.json`): one stream, shared.  Two such runs with a
truly inert treatment would be bit-identical, so 0.092 pp is not their floor and
their floor has never been measured.

`docs/POST_DECAY_FLOOR_RECLASSIFICATION.md` judged nine contrasts against
0.092 pp.  It therefore used a threshold of the wrong kind.  It errs toward
leaving verdicts underpowered rather than toward inventing findings, so nothing
it concluded is unsafe — but it is not the right test, and this plan supplies the
right one.

A placebo that merely sets the coefficient to zero would not do it.  The control
arm already runs the state router (plan 0092's control records 40,113 state
switches with zero actions taken), so a zero-coefficient treatment consumes the
same randomness and would come back bit-identical to the control by construction.
The cohort must change, not the coefficient.

## Target

`CLEAN_WRONG_PLAIN_ADVCE` (alias `PLAIN_ADVCE`), row E6 of
`docs/EVIDENCE_RECLASSIFICATION.md`.  Observed `+0.44 / +0.10 pp` against the
I100 control at e114.

It is the target for three reasons.  It has the largest observed effect among the
e114 arms that the measured floor made eligible.  It is one of the two arms that
passed the exploratory paired test in
`docs/POST_DECAY_FLOOR_RECLASSIFICATION.md` (`+0.270 pp`, t = 4.13) — a result
that document is explicit about not being a finding, because its threshold was
chosen after the numbers were visible.  And its cohort rule is `/fixed@e99`, so
the mask can be swapped without touching the online router, which has no
random-cohort mode.

## Design

Both parents, from the shared epoch-100 no-action prefix to epoch 114, on
`continuation_seed = 1` for dev-1 and `2` for dev-2, matching plan 0092's rep1
and rep2 so that the comparison reproduces the historical single-stream design.

| arm | cohort | runs |
| --- | --- | --- |
| `AdvCE(β) @CW-matched-random /fixed@e99` draw 1 | class-matched random, same size | 2 |
| draw 2 | a different random draw, same size | 2 |
| draw 3 | a third draw | 2 |
| untreated control, `continuation_seed = 4` | none | 2 |

Eight runs of about 0.36 GPU-hours: **2.9 GPU-hours**.

Endpoint: the registered held-out CE-PGD20 at e114, whole split and the
Clean-Wrong subgroup, per-sample rows kept.

## Preregistered rule

Let `P` be the mean over the three placebo draws of (placebo − control) on a
parent, and `T` the recorded treated effect on that parent.

- **`|P| < 0.15 pp` on both parents** → the selection carries the effect. The
  state-conditional family survives this test and Stage 2 is justified.
- **`P > 0.25 pp` on either parent** → the effect is not about the state. The
  family closes, and E1 through E7 are recorded as explained rather than
  underpowered.
- **In between** → inconclusive, and the number of draws was too small. Report
  it as such; do not reinterpret.

The spread of the three placebo draws around their own mean is the missing
single-stream floor.  It is reported whatever the verdict, because it is what
several past verdicts needed and never had.

**The rule is fixed before the runs.  It does not change once the numbers are
visible, including the 0.15 and 0.25 pp boundaries.**

## What has to be built

- `src/ard/analysis/intervention_selector.py` has `_random_selection`, which
  draws a class-matched cohort, but it is keyed on one fixed constant
  (`CLASS_MATCHED_RANDOM_SEED`). Three independent draws need that seed to become
  an argument. The draw seeds are declared here, before any run:
  **`2026090701`, `2026090702`, `2026090703`.**
- The mask has to be wired into the action-transfer runner and the arm name
  registered.
- Nothing else. The fixed-mask CW arms, the runner and the e100 prefix exist.

Both changes touch the scientific core, so they go through review before a
source SHA is frozen.

## Milestones

- [ ] M0: read the target arm's coefficient and cohort sizes from the record
  rather than from prose; confirm the e100 prefix and its thresholds are reusable
- [ ] M1: parameterise the draw seed, wire the random mask, register the arm;
  review; freeze a source SHA
- [ ] M2: eight runs from a pinned worktree
- [ ] M3: endpoints, aggregate, record the single-stream floor, close the plan

## What this cannot settle

It tests one arm, at one horizon, with one cohort rule. A null here does not
close the `/online` family, whose router has no random-cohort mode; that placebo
is deliberately not proposed until plan 0093 reports.

It also cannot distinguish "the state does not matter" from "this particular
state definition does not matter". The Clean-Wrong rule is one of several.
