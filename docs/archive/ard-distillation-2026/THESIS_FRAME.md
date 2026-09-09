# Thesis frame: the recommendation

Date: 2026-09-08. Read-only analysis of five proposed framings and their sixteen adversarial
reviews. This is the only file written. No GPU job was run. It is a recommendation, not a decision.

Every number below was re-checked against a record in this repository or against a document that
cites one. Where a framing asserted a number that does not exist, that is said in section 7.

---

## 1. The recommendation

**Write framing 1: the matched comparison of adversarial distillation against teacher-free
adversarial training on CIFAR-10 ResNet-18.**

Change the title. The proposed title, "What a Robust Teacher Is Worth", claims a quantity the
design cannot estimate. Use instead:

> **A matched, seeded comparison of adversarial distillation and teacher-free adversarial
> training on CIFAR-10 ResNet-18: one teacher, fixed student data, unmatched compute.**

Change the thesis sentence from an answer to a question. The proposed sentence asserts that the
teacher's contribution is of the same order as run-to-run variation. That sentence is false if the
comparison lands at +1.2 pp, and it would have to be rewritten in December under calendar
pressure. State the question in the thesis sentence and keep the expected answer in the
preregistration.

---

## 2. Why it beat the others, on the evidence

The four axes were scored by four independent judges each. Framing 1 totals 26.75 of 40; the
runner-up totals 25.25. The composition matters more than the total.

| framing | novelty | defensibility | sufficiency | robustness | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. Matched comparison | **5.25** | 6.50 | **8.00** | 7.00 | **26.75** |
| 2. Noise floor | 4.25 | **6.75** | 7.00 | **7.25** | 25.25 |
| 3. Negative on per-sample conditioning | 4.50 | 5.25 | 7.75 | 6.50 | 24.00 |
| 4. Hardness after the decay | 4.25 | 5.25 | 7.50 | 6.50 | 23.50 |
| 5. When and where hardness pays | 4.25 | 6.00 | 7.50 | 5.25 | 23.00 |

Framing 1 does not win because it is safer. It loses to framing 2 on defensibility by 0.25 and on
robustness by 0.25. It wins on novelty by 1.00 and on sufficiency by 1.00. It is the bigger and
newer thesis, not the safer one.

Three pieces of evidence decide it, and none of them is a matter of taste.

**First: framing 1 is the only one whose central question has a guaranteed answer.** The
comparison produces an interval on a difference whichever way the difference falls. The other four
stake their headline on contrasts whose existing measurements already point at nulls their own
designs cannot resolve.

- Framings 4 and 5 both rest on the schedule beating IDBH applied throughout. That contrast has
  been run. `EVIDENCE_RECLASSIFICATION.md` row A3 gives I100 minus I0 at +0.08 and +0.58 pp on two
  seeds, mean +0.33 pp, against a k=2 minimum detectable effect of 0.79 pp. Framings 4 and 5 both
  call this contrast one that has never been run, and both size it at five seeds, which resolves
  about 0.6 pp on official-test AutoAttack. They are designed to miss.
- Framing 3 rests on plan 0096. That plan has four of twenty-four forks finished, all on one
  parent, with no endpoint evaluated on any arm. Its one completed parent gives validation
  best-checkpoint PGD of 0.6006 for ALLOC_SAFE, 0.6006 for ALLOC_FRAGILE and 0.6004 for
  ALLOC_RANDOM, with the full-dose I100 reference at 0.6120. Three allocation arms within 0.02 pp
  and a 1.14 pp dose effect is a three-way null in the making.
- Framing 2 concedes in its own strongest-objection section that the floor it measures speaks to no
  contrast this project has ever run, and that the field's own published spreads, not this floor,
  carry its indictment of the literature.

**Second: the absence framing 1 exploits was checked twice by this project and holds.**
`ARD_VERSUS_AT_ASSESSMENT.md` section 1.4 records that no single paper compares the best ARD and
the best AT on ResNet-18, that ARD baselines are two to eight years older than the papers citing
them, and that the two most recent ARD papers report those baselines at the last checkpoint where
robust overfitting has removed six to seven points. `SMALL_MODEL_AT_2024_2026.md` section 4 then
re-checked fifteen 2025-2026 distillation papers and found none comparing against a teacher-free
method newer than TRADES. The nearest citation, CAT 2023 Table 5, is disclosed by framing 1 itself
without shading, and it does not close the gap: its distillation rows are copied from the RSLAD
paper, every row is a single run, and its teacher-free side is collaborative training rather than
DAJAT, IDBH or DAT.

**Third: this repository's own assessment already reached the same rank, independently.**
`ARD_VERSUS_AT_ASSESSMENT.md` section 4 ranks the reframe first and says "The absence is the
consequential finding." `METHOD_DIRECTIONS_V2.md` section 7 calls it "the thesis spine and the
largest expected effect (0.5-1.5 pp), so the surest positive result". Two documents written a day
apart, from different starting questions, converge on it. That convergence is evidence about the
option set, not about the authors.

---

## 3. The margin, and what breaks the tie

The lead is 1.5 points of 40. That is a clear lead, not a decisive one, and it rests on two axes
where framing 1 is ahead against two where it is marginally behind.

The tie is broken by a measurement, and the measurement can be taken in the first three weeks.

**Framing 1 degrades into framing 2 if the teacher-free side cannot be built.** The AT side of the
comparison currently consists of PGD-AT at one seed and corrected TRADES at one seed. PGD-AT is
anchored: 47.63 official-test AutoAttack at the best checkpoint against DAT's seven-run
47.63 ± 0.08. Corrected TRADES is not: 47.87 at the best checkpoint against a literature range of
49.0 to 49.4, so it sits 1.2 to 1.5 pp low with no explanation. AWP and SWA, the two ingredients
that carry every teacher-free record on this architecture, are not implemented.

So the tie-breaker is this. **If corrected TRADES reaches 49.0 to 49.4 across five seeds, and AWP
plus SWA clears scientific review by the end of week five, write framing 1.** If either fails,
framing 1 has no valid teacher-free side, its comparison chapter is a plan rather than a result,
and framing 2's higher defensibility becomes decisive. Decide this in week five, not in November.

Note that the TRADES stop-and-debug condition has already tripped once, at seed 0. That run is as
much a diagnosis as an anchor, and section 7 says what must be measured first.

---

## 4. What is grafted, and from where

Every judge named an element worth taking. Six fit. Two do not, and are named with the reason.

**(a) Screen validity, from the reviews of framing 2 (named by four separate judges).** A screen at
horizon H may gate a claim at horizon H' only when H differences are shown to predict H'
differences. This project has the counterexample: I100's gain is +0.18 and −0.04 pp in the
e100-e114 window where the instrument is sharpest, +0.24 pp at e149, and +0.69 to +1.12 pp at e199.
The horizon where the floor is smallest is the horizon where the one effect known to be real is
invisible. This costs no GPU hours, it survives every null, it transfers to any adversarial-training
paper, and it is the honest explanation for why four months produced twenty-three underpowered
rows. One judge suggested presenting it as Prentice's surrogate-endpoint criterion transplanted
from clinical trials to training horizons; that gives it a pedigree and a validation procedure
instead of leaving it as a house rule. Take that suggestion. It becomes the measurement chapter's
load-bearing claim, ahead of the floor number itself.

**(b) The decay-indexed floor table, from framing 2, with its arithmetic corrected.** Report the
floor as a table indexed by split, attack, horizon and design, never as one number. The honest
same-design statement is a 4.8-fold collapse of independent-seed per-run standard deviation across
the first decay: 1.088 pp at e99, 0.467 at e100, 0.226 at e104. The "twelvefold" figure divides an
independent-seed training-log quantity by a paired within-parent held-out quantity, which is the
regime splice `NUMERIC_CONSISTENCY_AUDIT.md` pattern 2 forbids. Do not write it.

**(c) The implementation-identity chapter, from the reviews of framing 1 (top-ranked graft by two
judges).** Promote it out of the fairness plumbing and state the general claim: a documented,
deliberate, test-pinned deviation from a published baseline is indistinguishable from a defect
until its cost is measured. The evidence is unusually hard. Gradient norms 1.91 against 3.23 on a
fixed synthetic batch, a 58 per cent relative difference. A cost of 2.73 pp of official-test
AutoAttack, 45.14 to 47.87 at the best checkpoint. The deviation was never hidden; it was written
into `UPSTREAM_BASELINES.md` section 8 as intentional and pinned by two regression tests whose job
was to assert that the local and official gradients differ. Tests defending the defect. Pair it
with what the survey found on the other side: DGAD copied AdaAD's baseline rows digit for digit and
MMARD copied RSLAD's, and the same RSLAD baseline appears at 51.49, 51.03 and 48.45 across three
re-implementations. The chapter's claim is then that the ARD literature's comparator numbers are
unaudited artifacts, evidenced from inside this repository. It needs no new GPU hours and it
survives every null.

**(d) The size- and class-matched random-allocation control, from the reviews of framing 3 (named
by three judges).** State it as a demand on the field rather than as one experiment. Every
instance-wise claim in either literature asserts that a selection rule is smart, and not one of
them runs a dose-matched, class-matched random selection. Without that arm, "who is treated" cannot
be separated from "how many are treated", and AROID's +0.10 pp over IDBH cannot be read as an
allocation effect at all. This is a two-page methods contribution that costs nothing and applies
whether or not plan 0096 finishes.

**(e) The direct/held-out transfer ratio, from the reviews of framing 3, with its denominator
fixed.** Reporting the treated training cohort's own effect beside the held-out effect from the
same run is a zero-cost diagnostic any per-sample paper can be asked for, and row E8 is the worked
example across four campaigns and three unrelated families. But the 13x-to-33x headline divides
sub-cohort effects by the whole-train 0.134 pp two-run gap over 45,000 rows. Row D6 already scales
a train floor to a sub-cohort by the square root of the row ratio and says the scaling is an
assumption. Do the same at E8. On a 15,317-image cohort the factor is 1.71, so the ratio is roughly
8x to 19x. Still a real gap. Not the number currently written.

**(f) The teacher-label diagnostic, from the reviews of framing 4 (called the best
evidence-per-GPU-hour item in the set).** One inference pass of the frozen Chen LTD teacher over
the 45,000 training images under CropShift views and under IDBH_WEAK views. Accuracy, margin
quantiles, student agreement. Under half a GPU-hour, no training. It answers a question nobody has
asked: does a frozen robust teacher's soft label stay valid off the augmentation distribution it
was trained on? If the margins degrade under IDBH views, the thesis has a distillation-specific
mechanism for why fifteen 2025-2026 distillation papers are still on crop-and-flip, and the
augmentation schedule becomes a result about teachers rather than a result about augmentation. Run
it in week zero regardless of anything else.

**Two grafts rejected, with the reason.**

The decay-alignment 2x2 from framing 4 — fork four arms from one shared epoch-79 parent, crossing
first-decay epoch with augmentation-switch epoch — is the best experimental design in the whole set
and three judges named it. It does not fit. It costs about 102 GPU-hours, it changes the
learning-rate schedule and therefore needs a preregistered human decision under rule 7, and it
answers a timing question that framing 1 does not ask. Record it as the follow-up a successor
should run.

Golatkar, Achille and Soatto, "Time Matters in Regularizing Deep Networks" (NeurIPS 2019), was
named by a framing-4 judge as a paper that asks in which phase augmentation acts and answers
"early" for standard training. It appears nowhere in this corpus. It is not a graft under framing
1, because framing 1 makes no timing claim. It must still be checked before the augmentation
chapter writes any sentence about when hardness acts.

---

## 5. The objection that remains after grafting, and how the thesis answers it

Three of the four serious objections to framing 1 are answerable.

The **missing champion arm** is answerable by experiment. DAT Table 3 puts TRADES + AWP + SWA at
51.14 ± 0.13, which is 0.7 to 1.6 pp below the 52.3-52.8 band, and the recipe that actually holds
the record is IDBH-throughout with AWP and SWA at 52.31 ± 0.26. Framing 1's arm list contains
neither. Run the published record recipe unmodified as the champion, and carry the I100 variant
beside it as the matched arm. Five runs, about 25 GPU-hours. Without this, RSLAD + I100 at 53.0-53.2
beats a straw comparator by 1.5 to 2 pp and the first examiner who opens DAT Table 3 will say so.

The **design confound** — fixing I100, an augmentation schedule developed and tuned entirely inside
RSLAD over four months, across all arms and calling that fairness — is answered by the same runs.
Two cells, record recipe and I100 variant, on both sides.

The **compute axis** is answerable by scoping plus one arm. State in numbers that the ARD arm
consumes a WRN-34-10 whose training cost is several times the student's plus a teacher forward pass
at every step. Preregister the fixed-data axis before any run. Add one compute-matched teacher-free
arm at the same total budget so that an examiner who prefers the other axis has a number rather than
a hedge.

**The objection that cannot be answered is the teacher.** It is n=1, the student chose it, and its
swing is several times larger than the effect being measured. In this repository the same RSLAD
moves from 51.90 AutoAttack with the Chen LTD WRN-34-10 teacher to 47.1 best and 43.1 last with the
Bartoldson teacher. In SAAD's own table the same method moves from 44.42 to 40.57 across two
teachers. That is a 3.5 to 4 pp swing from a free parameter that exists on one side of the
comparison only, against an expected difference under 1 pp. Five seeds per arm do nothing about it:
the variance component that matters is between teachers, and the thesis will have one, or with a
sensitivity arm two.

There is a second, structural version of the same objection that is not among the red team's
thirteen and that an examiner will find by reading two chapters in order. The Chen LTD teacher is a
RobustBench checkpoint trained on all 50,000 CIFAR-10 training images. Every student arm trains on
45,000. So the distilled arm has indirect access, through the teacher's weights, to the 5,000
images the teacher-free arms never see, and those are the very images used to select the best
checkpoint on both sides. Calling that axis "fixed data" is wrong; it is fixed student data with an
unfixed teacher. The partial answer is to run the comparison at 50,000 images with a 1,000-image
validation split, as DAJAT does, so that best-checkpoint selection survives and the asymmetry
shrinks to the teacher's 1,000. The residue must be admitted.

**Draft the admission into the text, in these words:**

> The number this thesis reports is what one teacher was worth. The same RSLAD student moves by
> 3.5 to 4 percentage points between the two teachers measured here and in SAAD's own table, from
> 51.90 with the Chen LTD WRN-34-10 to 47.1 with Bartoldson in this engine, and from 44.42 to
> 40.57 across SAAD's two. The free parameter on the distillation side therefore has a spread
> several times wider than the difference being measured, and fourteen weeks does not buy an
> estimate of it. Every result in this chapter is stated as "with the Chen LTD teacher, at fixed
> student data, with compute unmatched", and no claim is made about robust teachers in general.
> The teacher is also trained on all 50,000 CIFAR-10 images while the students are not, so the
> distillation arm retains indirect access to 5,000 images the teacher-free arms never see; the
> 50,000-image arms of chapter 3 bound that residue at the 1,000 images held out there, and do not
> remove it.

Put that paragraph in the comparison chapter, not in the limitations chapter. An objection admitted
where the number is reported is a scope condition. The same objection admitted forty pages later is
a concession.

---

## 6. The direct answer: method contribution, or measurement contribution?

**Framing 1 carries the measurement and mechanism contribution securely. It carries a method
contribution weakly, conditionally, and only in one shape.**

Secure, and there are three of them.

1. A screening standard for adversarial training: the decay-indexed floor table, the four labels on
   every floor and every effect, and screen validity with a worked counterexample from this
   project's own best result.
2. The measured cost of an undeclared implementation deviation, 2.73 pp, in the direction that
   flattered the hypothesis under test, with a mechanism for the 3.04 pp spread across published
   RSLAD re-implementations.
3. The comparison itself, reported as an interval with a stated minimum detectable effect,
   whichever sign it takes.

Weak, and there is only one.

The method-shaped element is giving a distilled student adversarial weight perturbation and
stochastic weight averaging under a matched, seeded protocol. It is a stacking result, not a new
objective. Its novelty claim must also be narrowed: SAAD applies SWA to its student from epoch 95
and never ablates it, so the defensible sentence is "no ARD paper gives the student AWP, and the
one that gives it SWA never ablates it". The only published data point is CAT's single-run
RSLAD + AWP at 51.62 against RSLAD 51.49, which is +0.13 pp with no spread. Both outcomes carry
content. If they stack, seven years of distillation papers have been leaving points on the table by
benchmarking against 2019, and the recipe is the deliverable. If they do not stack, a robust teacher
and flatness control supply the same thing, which is a mechanism claim about what a teacher does and
tells the next distillation paper what to stop doing. Write chapter 5 as a preregistered mechanism
test with both outcomes stated in advance, not as "does the recipe stack".

**What that costs, plainly.** A committee that requires a novel training rule will not find one in
this thesis. The student has asked for a method contribution and this framing does not supply one
in the sense the student means.

**Why no framing on the table supplies one.** This is the useful part of the answer. All five
framings say so in their own words. Framing 2's method contribution is a protocol and it says so.
Framing 3's is contingent on plan 0096, which is four forks of twenty-four. Framings 4 and 5 both
admit their method halves are start epochs on a published pipeline that tie the incumbent at
+0.33 pp with an MDE of 0.79 pp. The two candidates that would supply a real method are hardness
allocation and the teacher-relation gate on the disagreement set. Hardness allocation is, by its own
plan's closing section, a result about adversarial training and not an argument for a teacher, and
its one completed parent shows three arms within 0.02 pp. The teacher-relation gate acts on a cohort
where 94.7 per cent of S1 images are already T1, so the teacher condition is nearly vacuous where it
would act, and it earns a place only on the disagreement set where 23.9 per cent of S3 is T1. Both
need a screen plus an official-test confirmation on fresh seeds. That is two campaign cycles, and
the calendar has room for about two total with no room for a negative first cycle.

So the choice is not between a method thesis and a measurement thesis. It is between measurement
theses, and framing 1 is the one with the best-attached method-shaped experiment and the largest
question. Say that in the introduction now, while it is a choice, rather than in the defence, where
it will read as a retreat.

---

## 7. Before a word is written or a job is launched

Six items. Four cost nothing. Two cost under three GPU-hours between them.

1. ~~**Delete the DAJAT anchor, or make it true.**~~ **WITHDRAWN 2026-09-08. The anchor is
   true and the run exists.** `runs/dajat-eval-v1/result/result.json`, 2026-09-07 05:47:
   clean **85.71 %** and AutoAttack **52.45 %** against a published 85.71 and 52.48, a
   difference of **-0.03 pp**, an order of magnitude inside the 0.3 pp calibration band.
   This section searched the repository and concluded from an empty search that the run did
   not exist. Results live in the runtime tree until rule 5's import step moves them, and
   that step had been skipped; the run's own log also read `nan%` because of a wrong key
   name. Both are fixed and the record is now at
   `docs/experiments/dajat_rn18_foreign_checkpoint_official_test_v1.json`. The framings that
   asserted the anchor were right. **The lesson stands but inverts: an absent record in
   `docs/` is not evidence of an absent measurement, and a read-only analysis that greps
   only the repository will mistake an un-imported result for a fabricated one.** The
   original item, kept for the record:
    Framing 1, framing 2, framing 4 and framing 5 all
   state as established that DAJAT's released ResNet-18 checkpoint scores clean 85.71 and AutoAttack
   52.45 through this pipeline against a published 85.71 and 52.48. That run does not exist. A
   search of the entire repository for "52.45" returns nothing, and there is no `runs/` tree. The
   only foreign-checkpoint measurement on record is commit `914b10a`: 87.3 per cent clean on 512
   images, no AutoAttack, and `METHOD_DIRECTIONS_V2.md` line 107 still lists the full-test
   evaluation as a future one-hour job at "expected 52.48". The framing converted an expectation
   into a measurement and rounded 87.3 into an exact match with 85.71. This is finding 1 of this
   project's own numeric audit, in the direction of a stronger claim, in a document whose own
   chapter is about that failure pattern. It is load-bearing: it is the single stated bound on the
   one outcome framing 1 names as the outcome that actually hurts. Run it. Two GPU-hours.
2. **Measure the attack-initialisation difference on a fixed batch, at zero GPU cost.** Framing 1's
   chapter 3 says the only stated difference between this TRADES and the official one is 45,000
   training images against 50,000. `UPSTREAM_BASELINES.md` section 8, after the detach entry was
   struck, still lists local attack initialisation as uniform in [-eps, eps] against official
   Gaussian at 0.001 scale, plus normalisation and the learning-rate path, none with a measured
   cost. Worse, the split is the least likely explanation, not the leading one: `debugging/0028`
   says in this project's own words that "a split effect would move both numbers", and PGD-AT is
   exact. Measure the initialisation difference the same way the detach defect was measured, before
   spending nine runs on a hypothesis the repository already argues against.
3. **Correct the SAAD claim.** "No ARD paper gives the student AWP or SWA" is false on the SWA half.
4. **Correct the power arithmetic.** Framing 1 declares a 0.5 pp minimum detectable effect and
   supports it with a standard error of 0.13 to 0.32 pp. The standard error of a difference of two
   five-seed means is 0.632 times the per-run standard deviation. At the recorded e199 held-out
   per-run SD of 0.339 pp that is 0.214 pp, and the MDE is about 0.60 pp, above the declared
   minimum. Declare 0.6 pp, or add seeds, and preregister "inconclusive" as a third reportable
   outcome distinct from a null.
5. **Close decision packets 0003, 0004, 0005 and 0006.** All four are `chosen: null`, and
   `.claude/rules/results-records.md` forbids a scientific launch while any is. The fourteen-week
   clock is currently running against a closed gate. Note also that the corrected TRADES official
   AutoAttack evaluation, the number section 3 turns on, was produced while packet 0005 was pending.
6. **Render the live RobustBench leaderboard in a browser once.** It is the one overturn condition
   the survey set and could not verify. A seed-reported, no-generated-data ResNet-18 result at or
   above 54 pp published since March 2025 would make the premise of chapter 1 false. Two minutes.

---

## 8. The schedule

Fourteen weeks to mid-December for experiments, then writing to mid-January.

Rates used throughout, all from this repository: 92.2 s per epoch for a teacher-bearing ResNet-18
on Hamster, so 5.1 GPU-h per 200-epoch run and 2.56 GPU-h for an e100-to-e199 fork; teacher-free
runs drop a WRN-34-10 forward pass and are cheaper, but that rate is nowhere on record, so measure
it in the first run and re-cost; about 1 GPU-h per AutoAttack evaluation, which the audit lists as
REPORTED and untraced.

**Compute is not the constraint and must not be scheduled as if it were.** The whole plan is under
400 GPU-hours, which is under a week of wall clock on five 4090s. The constraints are the closed
decision gate, the campaign cycle, one attention-bound build, and infrastructure failure. Over the
last fortnight this project lost time to a materialiser run from a worktree pinned before the commit
that extended it, a frozen config that saves none of one of its three declared horizons, a mask that
selected the wrong margin tail, and a run-id collision. Budget two to three of the fourteen weeks
for that class of failure. The buffer in week twelve is not slack.

**Week 0.** Section 7, all six items. Nothing else launches until packets 0003-0006 are closed.

**Weeks 1-3, in parallel.**
- *Campaign A, the AT-side anchor.* Corrected TRADES at five seeds and PGD-AT at four more seeds,
  official test, best and last, AutoAttack on all. About 45 GPU-h plus 18 evaluations. Its
  stop-and-debug rule is preregistered: corrected TRADES must land in 49.0 to 49.4 at the best
  checkpoint, or the residual is a second defect and must be found before the comparison runs.
- *Build, the only serial blocker.* AWP and SWA in the training step, about 200 lines plus tests,
  through the scientific reviewer before a source SHA is pinned. AWP changes the optimiser step. SWA
  changes what "best checkpoint" means and needs a batch-norm recalibration pass whose
  adversarial-versus-clean choice is a scientific decision nobody here has made. Budget three weeks
  of calendar, not one week of coding.

**Weeks 2-5, in parallel with the above.** *Campaign B, the floors the thesis actually needs.*
Twelve untreated control forks to e199, evaluated at e114, e129, e149, e174 and e199, plus
AutoAttack on the twelve e199 checkpoints. About 43 GPU-h. This produces the first paired floor at
the horizon where effects live and the first paired floor on the endpoint the field reads, and it
produces the screen-validity table as a by-product from runs already being paid for. Every threshold
in every later chapter is currently sized against a number that does not exist.

**Week 5. The decision named in section 3.** If corrected TRADES is anchored and AWP plus SWA has
cleared review, proceed. If not, retitle now and write framing 2.

**Weeks 5-9. Campaign C, the comparison and the stacking test — one campaign, not two.** Arms, with
the champion defect fixed:

| arm | status |
| --- | --- |
| RSLAD(Chen) + I100 | exists, three confirmation seeds with official AutoAttack |
| RSLAD(Chen) + I100 + AWP + SWA | the method-shaped arm |
| PGD-AT + I100 | config-only |
| corrected TRADES + I100 | config-only |
| TRADES + AWP + SWA + I100 | needs the build |
| **the published record recipe unmodified: IDBH-throughout + AWP + SWA** | **the champion** |

Five seeds per arm with model-init, data-order and augmentation seeds shared across arms. About 25
runs and 50 AutoAttack evaluations, roughly 125 plus 50 GPU-hours. Before launching twenty-five
runs, spend one seed-0 sanity run on the champion: it must land near 52.3 to 52.5, and that is the
single highest-value hour in the plan.

**Weeks 8-11. Campaign D, the fairness axis.** The headline arms retrained on 50,000 images with a
1,000-image validation split held out, as DAJAT does, so best-checkpoint selection survives and the
result is commensurable with a literature number that is a best-checkpoint number. Three seeds each.
About 40 GPU-h. This is both the holdout-tax measurement, which the field lacks, and the partial
answer to the teacher's 50,000-image asymmetry in section 5.

**Weeks 11-13.** Aggregate, import records, close plans, write decision packets. No new science
after mid-December.

**Weeks 13-14.** Reserved for one re-run of whichever campaign missed. It will be needed.

**The critical path** is: close the packets, then anchor the AT side and clear AWP review, then
campaign C. Campaign B runs beside it and blocks nothing but interpretation. Campaign D can slip.

---

## 9. What gets cut, in order

Decide this order now, in writing, so that a slip is a cut rather than a discovery.

1. **The two extra I100 confirmation seeds.** About 28 GPU-h. The best-checkpoint contrast already
   stands at +0.39 pp with SD 0.079 and t(2) = 8.5. The fix for the one-in-eight sign rule is a
   sentence that prints the null rate beside the word CONFIRMED, not twenty GPU-hours.
2. **Plan 0096.** Twenty forks, a materialiser fix, a re-pinned worktree, and a horizon that is
   unrecoverable on the arms already run. Its own plan says a result there would be a result about
   adversarial training and not an argument for a teacher. Report it as an unfinished screen with
   its p1 numbers and their scope, or drop it. Nothing in chapters 1 to 5 may depend on it.
3. **Campaign D, the 50,000-image arms**, as a separate chapter. If it must go, fold the axis into a
   scope condition and label every project row "45,000 train, Chen LTD teacher, 200 epochs" in every
   cross-paper table.
4. **AWP and SWA, if they miss the week-five gate.** This is the load-bearing cut. It removes the
   champion arm from campaign C and it removes the only method-shaped element from the thesis at
   the same time. Then the comparison chapter is titled for the comparators it has — "against PGD-AT
   and corrected TRADES under a shared protocol, five seeds" — which is still a comparison no
   adversarial-distillation paper has run with seeds, and the thesis carries a measurement
   contribution alone. Write that title and that abstract in week five if the gate fails, not in
   December.

Everything above item 4 is expendable without changing what the thesis argues. Item 4 changes it.
That is why it has a date.
