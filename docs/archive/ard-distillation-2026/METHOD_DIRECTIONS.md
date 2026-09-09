# Method directions after the post-decay floor measurement

Date: 2026-09-07. Read-only analysis; this is the only file written. No training, no GPU, no
checkpoint touched. Every number is marked **VERIFIED** (read from a hash-bound record, a run
artifact, or a report that the numeric audit found clean, with the path), **DERIVED** (computed
here from verified numbers, arithmetic shown), or **SPECULATIVE** (an estimate the records do
not yet support, used only to size a design and never as a finding).

Nothing below rests on a number that `docs/NUMERIC_CONSISTENCY_AUDIT.md` flags. In particular the
pooled `+0.864 pp` I100 headline, the `0.25 pp` type-(c) floor derived from it, the `0.35 pp`
provisional floor, the `0.25-0.50 pp` bracket, and the `3.5 GPU-hours per parent` figure are not
used; where a cost per parent is needed the on-record Hamster figure is used instead.

These are options for the human to choose between. None of them is a decision.

---

## 0. Terms

| term | meaning here |
| --- | --- |
| **pp** | percentage point. One image of the 5,000-image validation split is 0.02 pp. |
| **held-out split** | the 5,000 training images held out from the 45,000 used for training; every screen is scored on it with the registered CE-PGD20 attack (20-step PGD on cross-entropy). The 10,000-image official test set and AutoAttack are used only for confirmation. |
| **parent** | a saved epoch-99 checkpoint. A screen forks copies of it and trains each copy for a few more epochs under a different rule. |
| **fork, replicate** | one continuation from a parent. Two replicates differ only in `continuation_seed`, which re-seeds the attack random start and the global random streams after the fork; data order and augmentation view stay identical (plan 0092 §"mechanism"). |
| **floor, σ_d** | the standard deviation of the difference between two replicates that differ in nothing but randomness. An effect smaller than it cannot be told from nothing. It carries four labels: split, attack, horizon, design. Two floors may be compared only when all four match. |
| **MDE** | minimum detectable effect: the smallest effect a design finds with 80% power at the 5% level. With k paired blocks and a known floor, `MDE ≈ 2.8 σ_d / √k`; with the floor estimated on four degrees of freedom the factor is 3.49 (`docs/POST_DECAY_FLOOR.md`). |
| **block** | one treated run and one control run from the same parent. |
| **horizon** | the epoch at which a number is measured. e114 is fourteen epochs past the first learning-rate decay at e100; the second decay is at e150; runs end at e199. |
| **state** | how a training sample stands under the current student: S1 safe-correct, S2 fragile-correct (adversarially correct with a small margin), S3 wrong, CW clean-wrong. T1 marks samples the teacher gets right with a comfortable margin. Definitions differ between the Stage-A lineage and the I100 lineage (`docs/ARM_REGISTRY.md`, Hazard 1); this document stays inside the I100 lineage. |
| **state-conditional treatment** | an extra loss term applied only to samples in one state. |
| **placebo arm** | the same extra loss applied to a class- and count-matched random set of samples instead of the selected state. If the state matters, the placebo does nothing. |
| **V-CW, V-nonCW** | the held-out images that the parent got clean-wrong at e99 (about 1,140 of 5,000), and the rest. Fixed before treatment, so they are a legitimate subgroup. |
| **common random numbers** | two runs that share every random stream, so any difference between them is caused by the treatment, though its size still includes the chaotic amplification of that cause. |

---

## 1. What the records show that the prose has not absorbed

The brief asks what the sharper instrument makes testable. Four things in the records change what is
worth testing, and none of them is stated anywhere in the documentation. All were computed here.

### 1.1 The historical controls are inside plan 0092's replicate sets

Plan 0092's replicate with `continuation_seed = 1` on dev-1 scores 56.06 / 56.94 / 57.32 % at
e104 / e109 / e114, and the replicate with `continuation_seed = 2` on dev-2 scores 55.80 / 56.40 /
56.96 % (VERIFIED, `docs/experiments/ard_post_decay_floor_v1.json` → `replicates[].horizons`).
Those are, to the last digit at all three horizons, the `I100_CONTROL` values that every e114 screen
compared against (VERIFIED, `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md` held-out table).
Plan 0094 shows why: training is bit-deterministic, and the old runs used the default streams.

Two consequences follow.

**First, the nine historical e114 arms can be re-baselined against the mean of three controls
instead of one.** DERIVED from the record and the three source reports:

| parent | three-replicate mean at e114 | historical control | shift applied to that parent's effects |
| --- | ---: | ---: | ---: |
| dev-1 | 57.320 % (57.32 / 57.38 / 57.26) | 57.32 % | 0.000 pp |
| dev-2 | 57.033 % (57.10 / 56.96 / 57.04) | 56.96 % | **−0.073 pp** |

| arm (registry name) | reported dev-1 / dev-2 (pp) | re-baselined dev-1 / dev-2 (pp) | re-baselined two-seed mean |
| --- | --- | --- | ---: |
| OS_PMP = `PM(0.054) @S2×T1 /online` | +0.14 / +0.20 | +0.14 / +0.13 | +0.133 |
| OS_DBDP | +0.14 / +0.06 | +0.14 / −0.01 | +0.063 |
| DPM = `PM(0.054) @S2×T1 /fixed@e99` | +0.08 / +0.12 | +0.08 / +0.05 | +0.063 |
| D-BDD | +0.04 / +0.20 | +0.04 / +0.13 | +0.083 |
| SBF | +0.16 / +0.04 | +0.16 / −0.03 | +0.063 |
| TPFM @S2T1 | +0.22 / +0.04 | +0.22 / −0.03 | +0.093 |
| PILOT_S3_T1_WEAK_ADVCE | +0.22 / +0.04 | +0.22 / −0.03 | +0.093 |
| CLEAN_WRONG_PLAIN_ADVCE | +0.44 / +0.10 | +0.44 / +0.03 | +0.233 |
| CLEAN_WRONG_A7_MARGIN_ONLY | +0.44 / +0.08 | +0.44 / +0.01 | +0.223 |

The "12 of 12 positive at e114" pattern that `docs/MEASUREMENT_DESIGN.md` §3.4 attributed to a low
control draw splits in two. On dev-2 that explanation holds: the control was 0.073 pp low and five of
nine arms go to zero or below once corrected. On dev-1 it does not: the control sits exactly on its
replicate mean, and all nine arms remain positive. Seven of the nine dev-1 arms exceed the highest
of the three dev-1 control replicates (57.38 %). A single treated run compared with a three-replicate
mean has a standard error of `0.065 × √(1 + 1/3) = 0.075 pp` (DERIVED from the record's pooled
within-parent SD of 0.0653 pp), so dev-1's +0.44 pp values are 5.8 standard errors and its +0.22 pp
values are 2.9. Whatever this is, it is a property of parent dev-1, not of the control.

**Second, every historical treated-versus-control contrast was a common-random-numbers contrast.**
The plan 0087 record does not contain the string `continuation_seed` at all (VERIFIED,
`docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json`), so treated and control arms
shared attack starts, data order and augmentation view. That is a legitimate design, but its floor is
not automatically the replicate floor: it is the floor of "how far does one small perturbation to the
weights carry the trajectory by e114". Pre-decay, every perturbation type saturated at the same
1.2 pp (`docs/MEASUREMENT_DESIGN.md` §2.4), and it is reasonable to expect the same saturation
post-decay, but nobody has measured it. A placebo arm on the same streams measures exactly this.
Direction 1 uses that.

### 1.2 What a 0.09 pp floor is made of

Between two control replicates at e114, about thirty held-out images turn robust-correct and about
thirty turn robust-wrong; the two counts nearly cancel. DERIVED from the six e114
`endpoint-sample-stats.parquet` files under `runs/post-decay-floor-v1/endpoints/`:

| pair | images gained / lost | net | net in pp |
| --- | ---: | ---: | ---: |
| dev-1 rep1 vs rep2 | +34 / −31 | +3 | +0.06 |
| dev-1 rep1 vs rep3 | +30 / −33 | −3 | −0.06 |
| dev-1 rep2 vs rep3 | +27 / −33 | −6 | −0.12 |
| dev-2 rep1 vs rep2 | +27 / −34 | −7 | −0.14 |
| dev-2 rep1 vs rep3 | +28 / −31 | −3 | −0.06 |
| dev-2 rep2 vs rep3 | +32 / −28 | +4 | +0.08 |

The full-split floor is small because roughly sixty flips balance to within a handful. A subgroup
does not inherit that balance, so **subgroup floors are larger than square-root scaling predicts**.
For random subgroups of the same 5,000 images (200 draws each, pooled within-parent variance over the
same six replicates, converted to a two-run difference):

| subgroup size | empirical σ_d (median over draws) | √n scaling would give |
| ---: | ---: | ---: |
| 5,000 (full) | 0.092 pp | — |
| 3,860 (≈ V-nonCW) | 0.121 pp | 0.105 pp |
| 1,140 (≈ V-CW) | 0.286 pp | 0.193 pp |
| 500 | 0.432 pp | 0.291 pp |
| 165 (≈ held-out S2 at e99) | 0.782 pp | 0.506 pp |

These are floors for *random* subgroups. A subgroup defined by difficulty may flip more; Direction 1
measures the real one before using it.

Two further floors fall out of the same rows, both DERIVED and both previously unstated:

- **Clean accuracy** at e114: σ_d = **0.114 pp** (0.136 at e104, 0.112 at e109). Plan 0093's clean
  guardrail of 0.5 pp is 4.4 floors wide.
- **Mean adversarial logit margin** over the 5,000 images: 0.327 (dev-1) / 0.325 (dev-2), σ_d =
  **0.0010**. Per image, two replicates disagree with SD 0.045 (correlation 0.9992). A margin-based
  endpoint is not sharper than accuracy in relative terms (0.3 % of its mean versus 0.16 % for
  accuracy), which is why no direction below proposes replacing accuracy with margin.

### 1.3 The one known effect is not present at e114

The I100 result — CropShift for epochs 0–99, then IDBH_WEAK — is the project's only replicated
improvement. Its size at e199 is +1.20 / +1.04 pp on the two development seeds against CropShift
continued, and +0.78 / +0.68 / +0.62 pp on the three confirmation seeds against CropShift-then-
RandomErasing (VERIFIED, `docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md` endpoint table;
`docs/NUMERIC_CONSISTENCY_AUDIT.md` verification §2). The audit is right that these are two different
questions; both are used below with their comparator named.

The same runs' training logs give the paired difference at every epoch. DERIVED from
`epoch-metrics.jsonl` (`val_pgd_accuracy`, the training-time validation attack, not the registered
endpoint) in `ard-runs/.../ert-rslad-stagewise-v1/idbh-s100-s{1,2}`,
`ert-rslad-static-trajstab-v1/cropshift-s1-r2`, `cropshift-s2-r1`, and
`unseen-confirm-b-{i100,crop}-suffix/outputs/student`; window means of the I100-minus-comparator
difference:

| window | dev-1 (vs CropShift) | dev-2 (vs CropShift) | confirm-b (vs CropShift→CROP_RE) |
| --- | ---: | ---: | ---: |
| e100–114 | +0.18 | −0.04 | +0.01 |
| e115–149 | +0.67 | +0.42 | +0.23 |
| e150–174 | +1.32 | +0.86 | +0.49 |
| e175–199 | +1.66 | +1.10 | +0.29 |

The registered endpoint agrees where it exists. At e149 the per-seed differences are +0.82 / +0.04
(dev, vs CropShift) and +0.42 / +0.10 / +0.20 (confirm, vs CROP_RE late); at e199 they are
+1.20 / +1.04 and +0.78 / +0.68 / +0.62 (VERIFIED, five-seed cache values reproduced in
`docs/MEASUREMENT_DESIGN.md` §2.5, comparator per the audit). About a third of the final gain exists at
e149; at e114 the window means are all below the k = 2 MDE of 0.243 pp.

The gain is not CropShift declining. Both arms keep improving after the second decay; I100 improves
faster. Window means of absolute validation robust accuracy (same files):

| run | e155–159 | e195–199 | change |
| --- | ---: | ---: | ---: |
| dev-1 CropShift | 58.93 | 59.20 | +0.27 |
| dev-1 I100 | 60.06 | 60.80 | +0.74 |
| dev-2 CropShift | 58.87 | 59.13 | +0.26 |
| dev-2 I100 | 59.54 | 60.48 | +0.94 |

The single-switch timing family says the same thing from the other side (VERIFIED,
`docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`; e199 endpoint, difference from CropShift, seeds 1 / 2):
IDBH from e0 +1.12 / +0.46; from e50 +0.68 / +0.92; from e75 +0.96 / +1.02; from e100 +1.20 / +1.04;
from e125 **+1.20 / +1.04** (identical to e100 on both seeds); from e150 +0.76 / +0.90. Exposure to
IDBH before e100 adds nothing; exposure during e100–125 adds nothing; roughly three quarters of the
gain is earned after e150.

The consequence for this brief is uncomfortable and has to be stated first: **the instrument is
sharpest at e114, and the one effect the project knows to be real is absent there.** Every e114
verdict, including the ones the sharper floor now makes readable, is a verdict about early post-decay
behaviour, which for the augmentation effect is unrelated to the final result. This is the strongest
possible argument for the measurement standard's rule that a screen is a gate, not a verdict, and it
means the e199 fork floor — which plan 0093's twelve control runs will produce as a by-product — is
the number every direction below ultimately needs.

### 1.4 The state-conditional null has structure inside it

`docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md` decomposes the held-out effect of the two
Clean-Wrong transfer arms into the images that were clean-wrong under the parent at e99 (V-CW, 1,138 /
1,143 images) and the rest (V-nonCW). Over two seeds, two arms and five horizons (e129 … e199), the
twenty V-CW effects have mean **+0.736 pp**, SD 0.440, **18 of 20 positive**; the twenty V-nonCW
effects have mean **−0.041 pp**, SD 0.392, 9 of 20 positive (values VERIFIED from the report's
table; summary DERIVED). Split by arm: Plain AdvCE V-CW +0.79 with V-nonCW **−0.28**; TPFM V-CW +0.68
with V-nonCW +0.20.

The reclassification (row E7) judged only the overall column and called it noise, correctly at the
floor it had. The subgroup columns were never judged, because no subgroup floor existed. Section 1.2
now supplies one: a random 1,140-image subgroup has σ_d ≈ 0.29 pp at e114, so the k = 2 MDE is about
0.57 pp and +0.74 is above it. That is an exploratory reading of a post-hoc split, not a result; it is
also the best-supported unexplained pattern in the records, and it is cheap to test properly.

---

## 2. Arithmetic and costs used throughout

- Blocks needed for effect δ: `k = 7.85 σ_d² / δ²`. MDE for k blocks: `2.8 σ_d / √k` (normal) or
  `3.49 σ_d / √k` (floor known on four degrees of freedom). Both are quoted where they differ.
- Measured floors (VERIFIED, `ard_post_decay_floor_v1.json`): σ_d = 0.159 / 0.124 / **0.092 pp** at
  e104 / e109 / e114; held-out split, CE-PGD20, two dev parents, fourteen-epoch fork. k = 2 resolves
  0.243 pp at e114; k = 6 resolves 0.106 pp (normal) or 0.132 pp (t on 4 df).
- Old working floor: 0.40 pp, k = 2 MDE 0.79 pp, k = 5 MDE 0.50 pp (`docs/MEASUREMENT_DESIGN.md`
  Rule 5). "Why now" below always compares these two.
- **The e199 paired fork floor is not measured.** The only hint is the within-comparator SD of the
  I100 contrasts at e199: 0.11 pp over the two dev seeds and 0.08 pp over the three confirmation
  seeds (`docs/NUMERIC_CONSISTENCY_AUDIT.md` finding #2). Two and three observations are not a floor.
  Where it is needed it is written as **SPECULATIVE, 0.08–0.25 pp**, and the design is sized at both
  ends.
- Cost basis: the control arm trains at **92.2 s / epoch** on Hamster (VERIFIED,
  `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md` runtime table). A 14-epoch fork is 21.5 min =
  **0.36 GPU-h**; a 29-epoch fork to e129 is 0.74 GPU-h; a 100-epoch fork to e199 is **2.56 GPU-h**.
  Endpoint evaluations take minutes and are not counted. Generating one e99 parent is 100 epochs, so
  about 2.6 GPU-h at this rate; plan 0093's stated 3.5 GPU-h is untraced (audit) and the difference
  does not change any ranking here.
- Budget: five RTX 4090s, experiments stop mid-December; call it about 90 days. The directions below
  sum to well under 200 GPU-h, so budget is not the constraint. Calendar and the human's attention are.
- Plan 0093 will generate six e99 parents under environment v2 and run twelve control forks to e199.
  Any direction that forks the same parents can share those controls only if it runs inside the same
  plan, which plan 0093 forbids. Costs below therefore include the direction's own controls, and note
  the saving if the human chooses to fold a direction in.

---

## 3. Directions, ranked by evidence per GPU-hour

### Direction 1 — Does a state-conditional loss move the held-out images in its own state, and pay for it elsewhere?

**Revives** rows E6 and E7 of `docs/EVIDENCE_RECLASSIFICATION.md` in decomposed form, and rows E1,
E2, E4, E5 as secondary. Observed effects against the new floor: E6's two Clean-Wrong arms have
re-baselined two-seed means of +0.233 and +0.223 pp against a k = 2 MDE of 0.243 pp — just under, at
the whole-split level; their V-CW subgroup effects average +0.74 pp against a subgroup k = 2 MDE of
about 0.57 pp — over. The S2 arms have re-baselined means of +0.06 to +0.13 pp, all under 0.243.

**Question.** A state-conditional loss produces a large effect on the training images it treats
(+3.7 to +4.1 pp on the fixed Clean-Wrong cohort; row E8, STANDS) and no resolvable effect on the
held-out split. Two explanations are open. (a) Nothing transfers: the loss memorises the treated
images. (b) The effect transfers to held-out images in the same state and is cancelled by a loss on
the other images. Only (b) is a mechanism a method can build on.

**Prediction that would be wrong if the idea is wrong.** Under (a), the held-out V-CW effect is
inside its floor and indistinguishable from the V-nonCW effect. Under (b), V-CW is positive by at
least the subgroup MDE and V-nonCW is zero or negative. The records lean to (b) — 18 of 20 V-CW values
positive, mean +0.74 pp; V-nonCW mean −0.04 pp, and −0.28 pp for the Plain AdvCE arm — but on a
post-hoc split with no preregistered floor. If a preregistered six-parent screen returns a V-CW effect
under 0.3 pp, (b) is dead and so is the last argument for state-conditional losses in this project.

A second prediction attaches to the placebo. If the state matters, the same loss on a class- and
count-matched random cohort produces neither the V-CW gain nor the whole-split shift. If the placebo
reproduces dev-1's +0.2 to +0.4 pp, then what the e114 screens have been seeing is "any extra loss on
twenty percent of the data perturbs this parent's trajectory upward", and the state is irrelevant.

**Why now, with the arithmetic.** Three things were impossible at the 0.40 pp working floor and are
possible now.

1. Subgroup floors did not exist, because no control replicates existed. They exist now (§1.2) and
   can be computed on the real V-CW subgroup from the same eighteen row files.
2. At a whole-split floor of 0.40 pp, the V-CW subgroup floor would scale to about 1.26 pp
   (0.40 × 0.286 / 0.092) and detecting +0.74 pp would need `7.85 × 1.26² / 0.74² = 23` blocks. At the
   measured 0.29 pp it needs `7.85 × 0.29² / 0.74² = 1.2`, so two blocks, and six parents resolve
   `2.8 × (0.29/√2) / √6 = 0.23 pp` with two replicates per parent (0.27 pp with a t-factor on 5 df).
3. The placebo contrast on one parent, three placebo replicates against the three existing control
   replicates, has a standard error of `0.065 × √(2/3) = 0.053 pp` and an MDE of 0.15 pp. Dev-1's
   +0.44 pp is three times that. At the old floor the same design would have resolved 0.65 pp and
   said nothing.

**Smallest experiment that settles it.** Three stages; each can stop the direction.

- *Stage 0, no GPU.* Recover the V-CW ID lists (hash `415756e8…`, n = 1,138 for dev-1; the e99
  validation replay lives under `.cache/analysis/ert-i100-cw-gap-e99`, or is regenerated by one
  validation pass of each parent in minutes). Compute the V-CW and V-nonCW robust accuracy of each of
  plan 0092's eighteen endpoints; that gives the real subgroup floors at e104 / e109 / e114. Re-judge
  the twenty existing V-CW values against them. If the real V-CW floor is above about 0.5 pp, the
  existing pattern is unreadable and the direction needs six parents before anything else is said.
- *Stage 1, the placebo, about 3 GPU-h.* On dev-1 and dev-2, from the e100 no-action prefix to e114,
  using the historical streams (`continuation_seed = 1` on dev-1, `2` on dev-2) so that the contrast
  reproduces the historical design exactly: three arms `AdvCE(β) @CW-matched-random /fixed@e99`, each
  with a different random draw of a class-matched cohort of 9,263 / 8,709 images, at the Plain AdvCE
  coefficient read from the action-transfer run's resolved config (the registry lists it as
  `AdvCE(?)`; the plan must state it). Plus one fresh untreated control per parent with
  `continuation_seed = 4`, so the plan owns a control. Eight runs × 0.36 GPU-h = 2.9 GPU-h. Endpoint:
  whole-split and V-CW robust accuracy at e114, per-sample rows kept. Readout on dev-1: placebo mean
  within 0.15 pp of the control mean means the +0.44 pp was cohort-specific; placebo mean above
  +0.25 pp means it was not, and Stage 2 is cancelled. The three placebo replicates also give the
  common-random-numbers floor of §1.1, which the project has never measured.
- *Stage 2, the confirmation, about 13 GPU-h.* Six parents (plan 0093's, once they exist), two
  replicates each, arms: `I100_CONTROL`, `CLEAN_WRONG_PLAIN_ADVCE` (largest V-CW gain, largest
  collateral), `AdvCE @CW-matched-random` (placebo). Thirty-six runs × 0.36 GPU-h = 13 GPU-h to e114
  (26 GPU-h if the human prefers e129, where the existing subgroup rows live; the e114 checkpoints of
  the original arms do not survive locally, so nothing older can be re-evaluated). Preregistered
  primary endpoint: parent-level mean V-CW effect, declared minimum 0.3 pp. Secondary: V-nonCW
  effect, whole-split effect, placebo V-CW effect. Adoption rule per the measurement standard §5.4. A
  parent-by-arm table is reported so that the dev-1 / dev-2 asymmetry of §1.1 is either reproduced
  across six parents or shown to be a dev-1 accident.
- The S2 family gets the same test for free from plan 0093's rows: define the held-out fragile
  subgroup from the parent at e99 (about 3.3 % of images, so about 165; the frozen thresholds already
  exist per parent), and read the PM effect on it. Prediction if PM works as its mechanism says: the
  +0.13 pp whole-split effect is concentrated there, which would be about +4 pp on 165 images against a
  subgroup MDE of about 0.6 pp at six parents. Prediction if PM is diffuse or null: nothing visible.

**Built versus to build.** The fixed-mask CW arms, the action-transfer runner and the e100 prefix
exist. The class-matched random selector exists (`src/ard/analysis/intervention_selector.py`,
`_random_selection`, seed `2026080201`); wiring its mask into the action-transfer runner and
registering the placebo arm name is mechanical. The stable-ID subgroup analysis exists in the CW gap
analysis code. The online router has no random-cohort mode; the placebo for `/online` arms would need
one, and is not proposed until plan 0093 has reported.

**What it is worth if null.** If Stage 1's placebo matches the treated arms, the entire
state-conditional programme (E1–E7, and by extension the Stage-A lineage) is closed for about 3 GPU-h
with the cleanest possible reason: the effect was not about the state. If Stage 2 finds no V-CW
effect at six parents, the thesis can say "the treated-cohort gain does not reach held-out images even
in the same state", which is a stronger negative than the current one. If (b) is confirmed, the
thesis's central negative result becomes a mechanism — transfer minus collateral — and the next method
question is defined and small: whether a lower coefficient or a gate keeps the transfer and drops the
collateral.

---

### Direction 2 — Is the augmentation-switch gain earned in the low-learning-rate phase?

**Revives** row A3 (switch timing; UNDERPOWERED at a k = 2 MDE of 0.79 pp) and the robust half of
row A4. Observed against the new floor: I125 versus I100 is +0.00 / +0.00 pp on both seeds; I150
versus I100 is −0.44 / −0.14 pp; both at e199, where no fork floor is measured.

**Question.** The I100 gain over CropShift is invisible at e114, about a third present at e149,
and complete at e199, with most of it arriving after the second decay (§1.3). Is the gain a property
of training under rich augmentation at a low learning rate, so that it does not matter when before
e125 the switch happens and it matters a great deal whether the low-rate phase is spent under the rich
policy? Or is it a slow accumulation that any exposure contributes to?

**Prediction that would be wrong if the idea is wrong.** The low-rate hypothesis predicts three
things, two of which the records already test. (i) A switch at e125 loses nothing against e100:
observed +1.20 / +1.04 versus +1.20 / +1.04 (VERIFIED, identical). (ii) IDBH from e0 or e50 gains
nothing over e100, or loses: observed +1.12 / +0.46 and +0.68 / +0.92 versus +1.20 / +1.04 (VERIFIED,
consistent, underpowered). (iii) A switch at e150 keeps most but not all of the gain, the remainder
being what e125–150 contributes: observed +0.76 / +0.90, so a deficit of 0.44 / 0.14 pp (VERIFIED,
unresolved). The accumulation hypothesis predicts instead that the gain scales with epochs of
exposure, so I50 > I100 > I150 monotonically; the records contradict it at I50 already. The idea is
wrong if, at six parents, I150 equals I100 within 0.1 pp (nothing before e150 matters at all, so it is
purely a final-phase effect, which is a different and simpler claim) or if I150 falls short by 0.6 pp
or more (exposure length matters as much as rate).

**Why now, with the arithmetic.** The claim "absent at e114" is now a claim: the k = 2 MDE at e114
is 0.243 pp, and the window means of +0.18 / −0.04 pp (and +0.01 pp on confirm-b against CROP_RE)
sit under it; at the old floor the same numbers sat under 0.79 pp, which could not distinguish "absent"
from "one third present". The timing differences that matter — 0 to 0.44 pp between I100, I125 and
I150 at e199 — need an e199 fork floor. At the SPECULATIVE 0.08–0.25 pp, six parents with two
replicates resolve `2.8 × (σ/√2) / √6` = 0.09 to 0.29 pp; at the old 0.40 pp they resolved 0.46 pp,
which is why row A3 was never answerable. The e199 fork floor is a by-product of plan 0093 (§4), so the
GPU part of this direction should wait two to three weeks for it and then be sized honestly.

**Smallest experiment that settles it.**

- *Stage 0, no GPU.* Put the §1.3 curves on record properly: per-seed paired difference at every
  epoch for all five seeds where logs exist, the registered endpoint at e149 and e199 (already in the
  five-seed cache), and the absolute post-e150 slopes. This is a results document generated from the
  run artifacts, and it is enough to write the thesis paragraph "the gain is earned after the decays".
- *Stage 1, about 31 GPU-h, conditional on the e199 floor.* One arm on plan 0093's six parents:
  `Aug(CropShift→IDBH_WEAK@e150) @all /none` (registry name I150), two replicates per parent, to
  e199. Twelve runs × 2.56 GPU-h = 31 GPU-h. Comparator: plan 0093's control (IDBH from e100), if run
  inside that plan; otherwise twelve controls of its own, doubling the cost. Endpoint: e199 held-out
  CE-PGD20, best and last checkpoints both kept; clean accuracy alongside, because I100 costs clean
  accuracy early (−0.5 / −1.0 pp at e110–114 in the training log, vanishing by e199; DERIVED from the
  same files, and real against the clean floor of 0.114 pp). Declared minimum effect 0.3 pp.
  Optionally a `CropShift @all` no-switch arm as the positive control on the new parents (another
  31 GPU-h); the effect it would confirm is already confirmed on five seeds, so this is a luxury.

**Built versus to build.** Everything exists: the stagewise transform takes any switch epoch, I150
is registered, the parents will exist. Nothing to build.

**What it is worth if null.** If I150 equals I100, the thesis result sharpens to "rich augmentation
during the final low-rate phase is sufficient", which halves the epochs under the slower policy and is
a cleaner mechanism statement than the current one. If I150 falls well short, exposure length matters
and the mechanism is accumulation; also a clear statement. Either way the Stage 0 curves are a thesis
figure at zero cost, and they are the reason the measurement standard's "screen is a gate" rule is
right.

---

### Direction 3 — What in the late policy carries the gain: the change, the photometric operations, or the erasing?

**Revives** nothing directly; it follows from the audit's separation of the two I100 contrasts.

**Question.** IDBH_WEAK is CropShift plus one random photometric or geometric operation
(colour, brightness, contrast, sharpness, autocontrast, equalize, shear, rotate) plus RandomErasing
(VERIFIED, `src/ard/data/datasets.py`, `EpochIdbhWeakTransform`, `_idbh_color_with_generator`).
Switching to CropShift-plus-RandomErasing at e100 gains +0.70 / +0.84 pp over no switch on the dev
seeds; switching to IDBH_WEAK gains +1.20 / +1.04 pp; IDBH_WEAK over CropShift-plus-RandomErasing is
+0.78 / +0.68 / +0.62 pp on the confirmation seeds (all VERIFIED). Three readings fit: the components
add; the photometric operation is the active ingredient and erasing is redundant once it is present;
or any change of distribution at the decay helps and the policy matters less than the switch.

**Prediction that would be wrong if the idea is wrong.** Two discriminating arms, each with a
prediction that can fail.

- `Aug(CropShift→CropColor@e100)` — CropShift plus the photometric operation, no erasing. If the
  operation is the ingredient, this equals IDBH_WEAK within 0.2 pp. If the components add, it falls
  short of IDBH_WEAK by roughly the erasing contribution, about 0.7 pp.
- `Aug(CropShift→canonical@e100)` — a switch to the *weaker* canonical RSLAD augmentation. If "any
  change at the decay" contributes, this beats no-switch by 0.3 pp or more. If only enrichment
  matters, it is at or below no-switch.

**Why now, with the arithmetic — honestly, the weakest of the three.** The component contrasts of
about 0.7 pp were within reach of a five-seed design at the old floor (MDE 0.50 pp), and one of them
was in fact measured. What was not within reach is the *difference between readings*: "equals IDBH
within 0.2" versus "short by 0.7" is a 0.5 pp gap, and "any change" is a 0.3 pp prediction. At the old
floor six parents resolved 0.46 pp; at the SPECULATIVE e199 floor of 0.08–0.25 pp they resolve 0.09
to 0.29 pp. So this direction is answerable now only if the e199 fork floor comes in at the low end of
its range, which plan 0093 will tell.

**Smallest experiment that settles it.** The CropColor arm first: twelve runs (six parents, two
replicates) to e199, 31 GPU-h, against plan 0093's IDBH controls if inside the plan. The canonical-
switch arm needs a no-switch CropShift reference on the same parents, so it costs 61 GPU-h; run it
only if the CropColor result leaves the "any change" reading alive. Endpoint e199 held-out CE-PGD20,
best and last; official test and AutoAttack only for a confirmed winner, once.

**Built versus to build.** A `CropColor` transform is about thirty lines composing the existing
`_apply_cropshift` and `_idbh_color_with_generator` layers, plus the schema `Literal`, a registry
entry and unit tests: mechanical, but a new augmentation policy is a scientific-surface change and
should go through the scientific reviewer before a source SHA is pinned. The canonical policy exists;
allowing it as a stagewise late policy is a two-line schema change with the same review requirement.

**What it is worth if null.** If CropColor equals IDBH_WEAK, RandomErasing is redundant and the
mechanism is photometric diversity at low learning rate, a specific and defensible thesis claim. If
CropColor falls short, erasing (occlusion) is essential. If the canonical switch also helps, the
project has been crediting a policy for a change, which would matter for how the I100 result is
described. Each outcome changes a sentence in the thesis; none changes the headline.

---

## 4. Ranking, and what to run first

| rank | direction | GPU-h to a verdict | what one GPU-hour buys |
| ---: | --- | ---: | --- |
| 1 | Direction 1, state-conditional specificity | 0 (Stage 0) + 3 (Stage 1) + 13 (Stage 2) | Stage 1 alone can close a two-year programme or turn its null into a mechanism. Nothing else in the project has that ratio. |
| 2 | Direction 2, low-rate mechanism of the switch gain | 0 (Stage 0) + 31 (conditional) | Stage 0 is a thesis figure for free. Stage 1 sharpens the main result but does not change it. |
| 3 | Direction 3, what carries the gain | 31 to 92 (conditional) | Answers a reviewer's question about the main result; needs the e199 floor first. |

**Run first: Direction 1, Stage 0 today and Stage 1 as soon as a decision packet permits it.**
Stage 0 is an afternoon of computation on files that already exist. Stage 1 is eight fourteen-epoch
runs on the two existing parents and answers, for about 3 GPU-h, whether the only positive-looking
e114 numbers the project has are about the state at all. It uses no new parent, no new code path
beyond a mask, and the measured floor at exactly its horizon. It can run on Hamster while plan 0093's
parents generate on Ferret. Its readout is binary and preregistrable: placebo within 0.15 pp of
control on dev-1, or not.

Directions 2 and 3 wait on the e199 fork floor, which plan 0093 produces if its controls are
evaluated at the right horizons (next section). Their Stage 0 parts do not wait.

Two things gate any launch and are the human's: decision packets 0001 and 0002 are still
`chosen: null`, and plan 0093's declared minimum effect of 0.30 pp is larger than the effect it exists
to confirm (audit finding #1, still open in the plan text).

---

## 5. Free by-products to request from plan 0093 (not directions)

Each costs minutes of evaluation, no training, and removes a SPECULATIVE label above.

1. Evaluate the twelve control forks with the registered endpoint at **e114, e129, e149, e154, e174
   and e199**, not only at the plan's two declared horizons, and keep the per-sample rows. Twelve
   controls, two per parent, give the fork floor at every horizon on five degrees of freedom, including
   across the second decay. This is the e199 paired floor that Directions 2 and 3 need and the only way
   to learn whether the floor stays near 0.09 pp or regrows at e150.
2. Keep the e149 checkpoints.
3. Record the `continuation_seed` of every arm in the record, so that no future re-baselining has to
   be inferred from matching digits.
4. Compute the clean-accuracy floor at each horizon from the same runs; the guardrail then has a
   measured denominator at e199.
5. From the treated and control rows, compute the held-out fragile-subgroup effect of Direction 1's
   S2 half; the frozen per-parent thresholds define the subgroup.

---

## 6. Considered and rejected

One line each. Rejections are ordered roughly by how tempting they looked.

- **PM(online) itself, or any of its coefficient neighbours.** Plan 0093.
- **Online versus frozen cohort for the same loss (E1 vs E1, E2 vs E2).** The re-baselined
  differences are 0.02 to 0.07 pp; at σ_d = 0.092 that needs `7.85 × 0.092² / 0.06² = 18` blocks at
  e114, for an answer that would not change a thesis sentence, at a horizon §1.3 says may be the wrong
  one.
- **Switch timing between e75 and e125 (A3).** I125 equals I100 to the second decimal on both
  seeds; there is no residual question worth 31 GPU-h. The e150 question is inside Direction 2.
- **A stronger IDBH policy, or other late policies as a family.** Effects of that kind were
  testable at the old floor with five seeds; the sharper floor changes nothing about them.
- **Batch ordering (B7, B8).** The eight pure-order schedules at e114 span 0.31 and 0.24 pp on the
  train split (VERIFIED, `docs/ERT_RSLAD_ORDERING_MECHANISM_DISCOVERY.md`); the expected range of eight
  draws at the train floor (per-run SD 0.134/√2 = 0.095 pp, range ≈ 2.85 SD = 0.27 pp) is the same
  number. There is nothing to revive.
- **Whether the treated-cohort direct effect is zero-sum on the training split (E8's mechanism).**
  Free from `sample-stats-train.parquet` and worth an afternoon, but the train floor was already known
  (0.134 pp); it was testable before and does not answer this brief.
- **A schedule without the second decay, to test Direction 2 causally.** Changes the schedule,
  which is a hard-rule scientific decision, and confounds rate with the number of low-rate epochs;
  propose only after Direction 2's Stage 0 is on record.
- **Held-out margin as a primary endpoint.** Relative replicate precision is no better than
  accuracy (§1.2); keep margin shifts as the mechanism gate they already are.
- **TPFM on Clean-Wrong moved post-decay (reclassification's rank 2).** Already run, as
  `CLEAN_WRONG_A7_MARGIN_ONLY` (+0.44 / +0.01 re-baselined); it is inside Direction 1, not separate.
- **Coefficient neighbours for the Clean-Wrong arms.** After Direction 1 confirms an effect, not
  before; a dose sweep around an unconfirmed effect is the pattern `docs/COEFFICIENT_AUDIT.md` warns
  about.
- **Extending plan 0092's six replicates from e114 to e199 (about 13 GPU-h).** Superseded by §5
  item 1 if plan 0093's controls are evaluated at the extra horizons; a two-parent floor is worse than a
  six-parent one.
- **Pooling old post-decay results across campaigns.** Plan 0094 shows identity holds, but the old
  arms are environment v1 with continuation seeds inferred rather than recorded; this is decision
  packet 0003 territory, not an experiment.
- **The clean-accuracy cost of the switch at e114.** Real against the clean floor (about −0.5 to
  −1.0 pp in the training log, gone by e199) and worth one sentence inside Direction 2; not a direction.
- **Attack-seed and evaluation noise (G5).** Per-sample margins reproduce to 1e-3 for the same
  weights and to 0.045 across replicates; neither limits any accuracy claim here.
- **Teacher selection, learnability measures and the other proposals in
  `docs/ARD_RESEARCH_ISSEUES_AND_PROPOSALS.md`.** Outside what the floor changed, and each would need
  a new lineage with its own floor.

---

## Verification of the two claims that change what we do

These directions were produced by a separate pass.  Two of their supporting
claims would change decisions already taken, so they were checked again from the
records by the session that commissioned them.  **Both hold.**

### The historical treated-versus-control contrasts used common random numbers

`arm-summary.json` for plan 0087's `control`, `pmp` and `dbdp` arms on dev-1 all
record `continuation_seed: null` and `rng_source_seeds: null`.  They ran on one
random stream.

That matters because the floor measured by plan 0092 is the spread between runs
with **different** `continuation_seed` values.  A treated arm and a control that
share the stream are not that kind of pair: with a treatment of exactly zero
effect they would be bit-identical, so their floor is not 0.092 pp and is not
known.  **`docs/POST_DECAY_FLOOR_RECLASSIFICATION.md` therefore applied a floor
of the wrong kind to those nine contrasts.**  Its conclusion — that no verdict
changes — was reached with a threshold that is probably too large, so it is
conservative in the direction of leaving verdicts as UNDERPOWERED, but it is not
the right test.  A placebo arm that computes the treatment and multiplies it by
zero would measure the correct floor directly.

### The only known-real effect is not present at the horizon the floor was measured at

`docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:51,55` gives I100 minus
`CROP_SUFFIX` as **+0.24 pp at e149** and **+0.69 pp at e199**.  The effect grows
with horizon rather than being present from the switch.

Plan 0092 measured the floor at e104, e109 and e114, where it is smallest and the
instrument is sharpest.  It is also where this effect has not yet appeared.
**A screen at e114 is therefore sharp about a horizon at which the one
intervention this project has reproduced would look like nothing.**  That is not
an argument for a longer intervention; it is an argument that some effects need
epochs after the second decay to express, and a screen that stops at e114 cannot
see them however precise it is.

Both of these point the same way: the floor is measured, but *which* floor
applies to *which* contrast at *which* horizon is still partly open, and the
cheapest way to close it is the placebo arm and the extra endpoint evaluations
that Direction 1 and section 5 already propose.
