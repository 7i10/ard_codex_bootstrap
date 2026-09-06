# Method directions, second derivation

Date: 2026-09-07. Read-only analysis; this is the only file written. No GPU job was run. It replaces
`docs/METHOD_DIRECTIONS.md`, which was written before five things were established (section 1) and
is now out of date. The earlier file is left in place as history.

Every factual claim carries one mark.

- **VERIFIED**: read from a hash-bound record, a run artifact, source code, a commit, or a paper's own
  text, with the path or table named.
- **REPORTED**: relayed from a summary, an abstract, or a second-hand table without checking the
  primary table myself.
- **INFERRED**: my reasoning from the other two, with the arithmetic shown where there is any.

No number below is quoted from prose alone. Section 8 lists where each one was read. None of the
fourteen defects in `docs/NUMERIC_CONSISTENCY_AUDIT.md` is reused: the pooled `+0.864 pp`, the
`0.25 pp` type-(c) floor, the `0.35 pp` provisional floor, the `0.25-0.50 pp` bracket and the
`3.5 GPU-hours per parent` figure do not appear as inputs.

These are options for the human. None is a decision.

---

## 0. Terms

| term | meaning here |
| --- | --- |
| **AT** | adversarial training: training on inputs an attacker has perturbed. |
| **ARD** | adversarial robustness distillation: AT where a large robust teacher's outputs replace or supplement the labels. RSLAD is the form this project uses. |
| **pp** | percentage point. One image of the 5,000-image held-out split is 0.02 pp; one image of the 10,000-image official test set is 0.01 pp. |
| **held-out split** | 5,000 of the 50,000 CIFAR-10 training images, held out from the 45,000 used for training. Every screen is scored on it with **CE-PGD20**, a fixed 20-step attack on cross-entropy. The official test set and **AutoAttack** (a fixed four-attack ensemble) are used only for confirmation. |
| **parent** | a saved epoch-99 checkpoint of a CropShift run. A campaign forks copies of it at epoch 100 and trains each under a different rule. |
| **fork, replicate** | one continuation from a parent. Two replicates differ only in `continuation_seed`, which re-seeds the attack random start and the global random streams after the fork. |
| **floor, σ_d** | the standard deviation of the difference between two runs that differ in nothing but randomness. An effect smaller than it cannot be told from nothing. A floor carries four labels: split, attack, horizon, design. Two floors may be compared only when all four match. |
| **MDE** | minimum detectable effect at 80% power and the 5% level. With k paired blocks and a known floor, `MDE ≈ 2.8 σ_d / √k`; with a floor estimated on four or five degrees of freedom the factor is about 3.5. |
| **horizon** | the epoch at which a number is measured. The learning rate falls by ten at e100 and again at e150; runs end at e199. |
| **decay** | one of those two learning-rate drops. "Post-decay" means after e100. |
| **robust overfitting** | test robustness falling late in training while training robustness keeps rising. It starts at the first decay. |
| **CropShift, IDBH_WEAK** | two augmentation pipelines. CropShift is flip plus a random crop-and-shift. IDBH_WEAK is CropShift plus one random photometric or geometric operation plus RandomErasing with probability 0.5. IDBH_STRONG differs only in RandomErasing probability 1.0. |
| **I100** | the project's one replicated method-agnostic result: CropShift for epochs 0-99, then IDBH_WEAK from e100. Canonical name `Aug(CropShift→IDBH_WEAK@e100) @all /none`. |
| **hardness** | in IDBH's sense, how much an augmentation makes an image easier to attack for a fixed robust model. IDBH_WEAK is harder than CropShift. |
| **state** | how one training image stands under the current student, in the canonical contract: **S1** safe-correct (correct under the training attack, margin above the tenth percentile of positive margins), **S2** fragile-correct (correct, margin below it), **S3** wrong under attack, **CW** clean-wrong. Teacher states: **T1** teacher correct under attack with margin above its tenth percentile, **T2** teacher correct but fragile, **T3** teacher wrong under attack. `docs/ARM_REGISTRY.md` Hazard 1 warns that the Stage-A lineage used different S1/S2/S3 definitions; this document uses only the canonical ones. |
| **allocation** | which training images receive the richer augmentation. `@all` is I100. |
| **matched random allocation** | the same number of images, with the same class counts, drawn at random instead of by state. If the state matters, this arm does nothing that the state-selected arm does. |
| **fixed@e100** | membership computed once from the shared, treatment-free epoch-100 prefix and never recomputed. Plan 0087 froze its thresholds this way. |

---

## 1. What is established now that was not when the earlier analysis was written

Each item changes what is worth testing. Each is stated with what it rules in or out.

### 1.1 Conditioning augmentation on the student's own state is not new

AROID (Li, Qiu and Spratling, IJCV 2024, arXiv 2306.07197) learns an online, instance-wise augmentation
policy inside adversarial training from a per-instance signal. VERIFIED by the prior-art report from the
paper text (`docs/PRIOR_ART_STATE_CONDITIONED_AUGMENTATION.md`, verdict and section 1). What its signal
does matters for the designs below. The paper defines "Vulnerability" as the loss under attack minus the
clean loss on the *augmented* image, says "a larger Vulnerability indicates that x becomes more vulnerable
to adversarial attack after DA", and motivates it with "a common belief ... that AT benefits from
adversarially hard samples" (REPORTED: verbatim quotes returned from the arXiv HTML, not the PDF I would
have to read myself). So AROID rewards **hardness on every instance**, bounded by an "Affinity" term that
keeps augmentations recognisable to a separately trained standard model. It never asks whether a fixed
increment of hardness should go to the images with margin to spare or to the fragile ones, it has no
size-matched random allocation as a control, and it uses no teacher. On PreActResNet-18 it reports 50.57
AutoAttack against 50.47 for IDBH in its own Table 4, three runs (REPORTED, same fetch).

Consequence: a student-state gate is admissible only as a **test of allocation direction against a random
allocation**, which AROID does not contain, and never as "instance-wise augmentation for AT".

Conditioning on the student-teacher relation is unoccupied but narrow: TST (Shao et al., 2023) and
TeachAugment (CVPR 2022) use a teacher to choose augmentation without an adversary; IAD, DGAD and SAAD use
the teacher to gate losses and attacks, never augmentation. VERIFIED in the same report, section 2.

### 1.2 The IDBH paper says nothing about *when*, and its own account points the opposite way

IDBH applies one pipeline from epoch 0 to 200 and has no schedule argument (VERIFIED, prior-art report
section 4, from `.external/DA-Alone-Improves-AT/src/data/idbh.py`). Its causal account is: hardness costs
clean accuracy and fitting at all times, pays only by reducing robust overfitting, and the right hardness
"is very sensitive to the capacity of the model" (VERIFIED quotes, same section). Read per sample, capacity
to spare is margin to spare, so the account puts the richer augmentation on **S1**, not on the fragile
images. EntAugment (ECCV 2024), MADAug (ICCV 2023) and a 2025 influence-based method all arrive at the same
allocation in standard training: stronger augmentation on easy or stable samples (VERIFIED for EntAugment
and MADAug abstracts, REPORTED for the third; prior-art report table 1).

There is a second account that predicts the reverse. Robust overfitting is memorisation of hard,
atypical samples (Rice et al. 2020, REPORTED), and the ICML 2026 paper on why robust teachers fail locates
the damage on a student-specific unlearnable subset (VERIFIED in `docs/ARD_VERSUS_AT_ASSESSMENT.md`,
section 1.1 row). Under that reading augmentation should land where memorisation happens, on the hard
images. Both accounts are coherent. Neither has been tested under an adversary with the other as an arm.
That is why direction 1 runs both.

### 1.3 The teacher-free frontier is a band, and the field left the small-model regime

The strongest no-extra-data ResNet-18 AutoAttack result is DAT + AWP + SWA, 52.76 ± 0.14 over seven runs
(NeurIPS 2024, Table 3), with IDBH + AWP + SWA at 52.31 ± 0.26 and DAJAT at 51.85 ± 0.26 re-measured in the
same table. VERIFIED (`docs/SMALL_MODEL_AT_2024_2026.md`, section 0, from the arXiv PDF text). Every
CIFAR-10 entry added to RobustBench in 2023-2024 is a WideResNet with generated data; the leaderboard is
dormant since March 2025 (VERIFIED, same file, section 1). Fifteen distillation papers from 2025-2026
compare against nothing newer than TRADES (2019) (VERIFIED, section 4).

Two same-engine anchors exist. This repository's PGD-AT scores 47.63 AutoAttack on the official test set;
DAT's Table 1 reports PGD-AT at 47.63 ± 0.08 over seven runs (VERIFIED both; commit `1d73bb5`). The
repository's TRADES was four points low because the clean-side KL target was detached; the defect is fixed
in commit `7666d77`, the 45.14 result is marked superseded, and **no corrected TRADES run exists yet**
(VERIFIED, `docs/debugging/0028-trades-clean-target-detached.md`; `docs/EXPERIMENT_DASHBOARD.md` note).
A script to evaluate a foreign checkpoint through this pipeline exists; DAJAT's ResNet-18 scored 87.3 %
clean on 512 images and has not yet been run through AutoAttack (VERIFIED, commit `914b10a`). AWP and SWA
are not implemented (VERIFIED absence, `docs/ARD_VERSUS_AT_ASSESSMENT.md` section 2.1).

### 1.4 The measured floor is the wrong kind for the historical contrasts

Plan 0092 measured σ_d = 0.092 pp at e114 (0.159 at e104, 0.124 at e109) between untreated forks with
**different** `continuation_seed` (VERIFIED, `docs/POST_DECAY_FLOOR.md`; the audit found every number in
that file matches `docs/experiments/ard_post_decay_floor_v1.json`). Every historical treated-versus-control
pair ran on **one shared stream** (`continuation_seed: null`, VERIFIED in plan 0095 from the plan-0087
`arm-summary.json`). A shared-stream pair with an inert treatment would be bit-identical, so its floor is
not 0.092 pp and has never been measured. Plan 0095 (preregistered, M0 open) measures it with three
class-matched random cohorts on the Clean-Wrong arm.

Consequence for this document: every design below uses **different** `continuation_seed` per replicate
and pairs treated and control forks on the same seed, which is the design the 0.092 pp floor describes.

### 1.5 The one real effect is absent at the horizon where the floor is sharp

I100 against CropShift continued: +1.20 / +1.04 pp at e199 on the two development seeds (VERIFIED,
`docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md`, endpoint table). I100 against CropShift-then-RandomErasing:
+0.78 / +0.68 / +0.62 pp at e199 and a mean of +0.24 pp at e149 on the three confirmation seeds (VERIFIED,
`docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md`). On the official test set under AutoAttack, last checkpoint:
+0.41 / +0.06 / +0.46 pp (VERIFIED, `docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md` and the record's
`decision.per_seed_pp`). At e149 the five-seed endpoint differences are +0.42 / +0.10 / +0.20 / +0.82 /
+0.04 pp, the first three against CropShift-then-RandomErasing and the last two against CropShift
(VERIFIED per-seed values in `docs/MEASUREMENT_DESIGN.md` section 2.5 from the five-seed cache; comparator
split per `docs/NUMERIC_CONSISTENCY_AUDIT.md` finding 2). Switching at e125 gives exactly the same e199
result as switching at e100 on both seeds, +1.20 / +1.04 (VERIFIED,
`docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`), so nothing between e100 and e125 contributes, and switching at
e150 keeps +0.76 / +0.90 (VERIFIED, stagewise report). Roughly three quarters of the gain is earned after
the second decay. INFERRED from those rows.

Consequence: **any augmentation direction is judged at e199, with e149 recorded, and e114 recorded only
to build the screen-validity pairs the measurement standard asks for** (§3.1). No e199 fork floor exists
yet (VERIFIED absence, `docs/POST_DECAY_FLOOR.md` last section). Plan 0093 §8 preregisters evaluating its
twelve control forks at e114/e129/e149/e154/e174/e199 with per-sample rows, which will supply it on five
degrees of freedom (VERIFIED plan text). Until then designs are sized at both ends of a stated range.

### 1.6 Where the apparatus stands today

- Plan 0093's six new parents: `parents-v2/seed1` and `seed2` have passed epoch 99 on Hamster and are at
  about epoch 170 at 04:29 JST; the materialiser is "waiting for six epoch-99 checkpoints" (VERIFIED,
  `runs/parents-v2/logs/materialize.log`, `epoch-metrics.jsonl` line counts). The parents run the full 200
  epochs, so they also produce six CropShift-throughout e199 references (VERIFIED, `parent.sh` comment).
- Cost basis: 92.2 s per epoch for the I100-lineage control on Hamster (VERIFIED,
  `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`, runtime table). A fork from e100 to e199 is 100
  epochs, 2.56 GPU-h; twelve such forks are 31 GPU-h; a fork to e149 is 1.28 GPU-h (DERIVED). A full
  200-epoch teacher-bearing run is about 4 GPU-h by the brief's figure and 5.1 GPU-h at 92.2 s/epoch; the
  larger figure is used for budgets (INFERRED).
- Every augmentation transform is keyed by the immutable sample id and epoch (`source_id_keyed = True`
  for CropShift, CROP_RE, IDBH_WEAK and the stagewise switch; VERIFIED, `src/ard/data/datasets.py` lines
  30, 75, 296). A per-sample gate therefore leaves the view of every untreated image bit-identical across
  arms, which is what makes a paired allocation contrast clean. The RandomErasing probability is a
  parameter, default 0.5 (VERIFIED, line 141). Fixed masks with SHA-256 manifests, and a class-matched
  random selector, already exist (`src/ard/policies/fixed_mask.py`,
  `src/ard/analysis/intervention_selector.py`; VERIFIED). The per-sample store holds each image's margin,
  robust correctness and the teacher's adversarial correctness (VERIFIED, `src/ard/state/sample_store.py`).
- Teacher-free configs (`cifar10_r18_pgd_at.yaml`, `cifar10_r18_trades.yaml`) share the dataset schema
  whose `augmentation_policy` field carries the stagewise switch, and use the same 200-epoch schedule,
  milestones [100, 150], weight decay 5e-4, batch 128 and the 45,000/5,000 split (VERIFIED, config text).
  "PGD-AT + I100" and "TRADES + I100" are config-only arms.
- Training-split cohort sizes at e99 on the development parents: Clean-Wrong 9,263 / 8,709 of 45,000
  (about 20 %), pilot S3×T1 7,898 / 8,907 (about 18-20 %) (VERIFIED,
  `docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md` line 37). The online S2×T1 state occupies about 3.3 % of
  sample-epochs over e100-e114 (VERIFIED, online-state report, mechanics table). S1 is the majority state
  after the decay; its exact size is a Stage-0 readout from the e100 prefix (INFERRED).

---

## 2. Rules every design below obeys

1. Judged at **e199** on the held-out CE-PGD20 endpoint, best and last checkpoints both kept, clean
   accuracy alongside; e149 and e114 recorded; official test and AutoAttack once, from saved checkpoints,
   only for an adopted primary arm. Rule 6 of `CLAUDE.md` is untouched: no attack, epsilon, step,
   schedule or checkpoint-selection change anywhere.
2. **Six parents, treated and control forked in pairs on the same `continuation_seed`, two replicates per
   parent, parent-level paired test on five degrees of freedom**, per `docs/MEASUREMENT_STANDARD.md` §2.
   Controls run inside the same plan. Plan 0093 forbids extension, so no design borrows its controls; the
   saving is noted where a human could choose to waive that.
3. Canonical arm names, registered in `docs/ARM_REGISTRY.md` before a run; one plan, one family, one
   primary arm; minimum effect declared before launch.
4. **Both allocation directions are preregistered arms, and every state-selected allocation has a
   size- and class-matched random allocation as its control.** Without the random arm, "who is treated"
   cannot be separated from "how many are treated".
5. Sizing: at e199 the fork floor is unmeasured; designs are sized at σ_d = 0.10 and 0.25 pp. Six parents
   with two replicates resolve `2.8 × (σ_d/√2)/√6 = 0.08-0.20 pp` with a known floor and about
   0.10-0.25 pp with a floor on five degrees of freedom (DERIVED). A declared minimum of 0.30 pp is
   reachable at both ends.
6. Any comparison with a published number states that this project trains on 45,000 images, not 50,000.

---

## 3. Direction 1 — Where should a fixed increment of augmentation hardness go: to the images with margin to spare, or to the fragile ones?

This is the direction the evidence points to, and it contains the direction the student hoped for as a
named arm with equal standing in the analysis.

**Question.** I100 gives every image the richer policy from e100. Two accounts (section 1.2) disagree about
which images earn the gain. The capacity account says the richer policy pays where the student has margin
to absorb the hardness (S1) and costs accuracy where it does not. The memorisation account says it pays
where robust overfitting happens, on the fragile and wrong images (S2 ∪ S3 ∪ CW). A third possibility is
that only the *amount* of rich augmentation matters and its location does not. The question can come out
any of the three ways.

**Arms** (six parents, forks e100→e199, all in the canonical contract, membership `fixed@e100` from the
shared treatment-free prefix, as plan 0087 froze its thresholds):

| role | canonical name | what it does |
| --- | --- | --- |
| control | `Aug(CropShift→IDBH_WEAK@e100) @all /none` (= I100) | rich on every image |
| **primary** | `Aug(CropShift→IDBH_WEAK@e100) @S1 /fixed@e100` | rich on safe-correct images; CropShift continues on the rest |
| secondary, the reverse | `Aug(CropShift→IDBH_WEAK@e100) @¬S1 /fixed@e100` | rich on fragile, wrong and clean-wrong images; CropShift on S1 |
| control for location | `Aug(CropShift→IDBH_WEAK@e100) @rand(\|S1\|, class-matched) /fixed@e100` | rich on a random set of the same size and class counts as S1 |
| reference, free | `Aug(CropShift) @all /none` | the parents themselves, to e199 |

The primary is S1 because three standard-training results and IDBH's own text put it there (section 1.2);
the reverse arm carries the student's original direction and is analysed with the same rule. The random arm
is drawn with a declared seed; with S1 the majority state its complement is small, so a second random arm
matched to |¬S1| is not needed to read the reverse arm against dose, because the reverse and the random
arm bracket the treated fraction from both sides (INFERRED; if the Stage-0 S1 fraction is below 55 %, add
the second random arm).

**Predictions that would be wrong if the idea is wrong.** All at e199, held-out CE-PGD20, parent-level
paired differences.

- Capacity account: S1-rich ≥ I100 − 0.1 pp on robust accuracy **and** S1-rich has a smaller clean cost
  than I100 (I100's clean cost against CropShift is −0.30 / −0.18 pp on the dev seeds and −0.06 / −0.22 /
  −0.46 pp against CropShift-then-RandomErasing; VERIFIED); reverse ≤ I100 − 0.3 pp; random sits between
  them in proportion to its fraction. The account is wrong if S1-rich falls 0.3 pp or more below I100.
- Memorisation account: reverse ≥ S1-rich + 0.3 pp. Wrong if reverse ≤ S1-rich.
- Dose-only: S1-rich, reverse and random agree within the floor and all sit between CropShift and I100.
  This outcome is a null for both accounts and is the result that closes per-sample augmentation in this
  lineage in both directions at once.

**Relationship to prior art, one sentence.** AROID learns each instance's augmentation by rewarding
per-instance hardness; nobody has tested whether *where* a fixed hardness increment lands, chosen by
student margin and checked against a size-matched random allocation, changes robustness, and IDBH's own
capacity account predicts the opposite allocation from AROID's reward. Against AROID specifically: this is a
two-policy rule with no policy network and no extra affinity model; if S1-rich wins it is a zero-cost method,
and if location does not matter AROID's own +0.10 pp over IDBH (REPORTED, its Table 4) reads as a dose
effect. Against the project's own history: the loss-lever treatments on S3 and CW produced large direct
effects and no held-out effect (`docs/EVIDENCE_RECLASSIFICATION.md` row E8, STANDS); this moves the lever
to augmentation, whose global form is the one thing that did transfer.

**Smallest experiment that settles it.**

- *Stage 0, no GPU, one afternoon.* From the two existing epoch-100 prefixes (dev-1, dev-2) and, when they
  exist, the six v2 parents: read the S1 fraction and class counts from the per-sample store; write the
  three masks per parent with SHA-256 manifests; confirm that in a one-epoch dry run the tensor for any
  untreated image is bit-identical between the `@S1` and `@all` arms (the transforms are id-keyed, so it
  should be; VERIFIED design, INFERRED outcome). Record the state-transition rate S1→¬S1 over e100-e114 from
  plan 0092's controls so the drift of a fixed mask is known before the runs.
- *Build, two to three days of attention.* A mask-gated stagewise transform: same prefix, late policy chosen
  per `source_id` from a fixed mask; a schema field pointing at the mask and its digest; registry entries.
  It touches the data path, so it goes through the scientific reviewer before a SHA is frozen.
- *Campaign, 123 GPU-h.* Four arms × six parents × two replicates = 48 forks × 2.56 GPU-h. About 25 hours
  of wall clock on five GPUs (INFERRED). If the human waives rule 2 and lets the design share plan 0093's
  twelve I100 controls, 92 GPU-h. Endpoints at e114, e149, e199, per-sample rows kept; official test and
  AutoAttack for S1-rich and I100 only, once, if S1-rich is adopted (12 evaluations, about 12 GPU-h,
  REPORTED cost from plan 0091).
- Declared minimum effect 0.30 pp; adoption rule per the measurement standard §5.4.

**What it is worth if null.** A three-way null (dose-only) ends per-sample augmentation in this project
with the cleanest possible reason, and it does so for the student's direction and the evidence's direction
in one campaign. It also gives the dose response nobody has measured: rich on about 60 % of the data
versus 100 %. If the capacity account holds, the thesis has a method: margin-allocated augmentation, no
teacher required, and a mechanism statement about I100. If the memorisation account holds, the student's
original intuition is confirmed under a design a reviewer will accept.

---

## 4. Direction 2 — On the images the student gets wrong and the teacher gets right, which account of augmentation wins?

This is the student-teacher relation, the one slot the prior-art check left open. It is worth running only
after Direction 1 has shown that location matters; if location does not matter, this question has no
content.

**Question.** TST's principle in ordinary distillation is to augment "what the teacher is good at but the
student is not" (VERIFIED quote, prior-art report section 2). Under an adversary, IDBH's capacity account
predicts the opposite: an image the student gets wrong under attack has no margin to spend, so hardness
there costs accuracy without buying anything. The disagreement set S3×T1 is where the two principles
collide, and it is about a fifth of the training data at e99 (VERIFIED sizes, section 1.6).

**Arms** (six parents, e100→e199, on top of Direction 1's primary):

| role | canonical name |
| --- | --- |
| control | `Aug(CropShift→IDBH_WEAK@e100) @S1 /fixed@e100` (Direction 1 primary) |
| **primary** | `Aug(CropShift→IDBH_WEAK@e100) @S1∪(S3×T1) /fixed@e100` |
| control for location | `Aug(CropShift→IDBH_WEAK@e100) @S1∪rand_in_S3(\|S3×T1\|) /fixed@e100` |

The random arm draws the same number of images from S3 without regard to the teacher, so the only
difference between primary and random is the teacher's verdict.

**Prediction that would be wrong if the idea is wrong.** TST direction: primary ≥ control + 0.3 pp and
primary > random by at least the floor. Capacity direction: primary ≤ control, and random ≤ control by a
similar amount (the teacher adds nothing because hardness on S3 hurts regardless). The idea "the teacher
relation carries information for augmentation" is wrong if primary and random agree within the floor,
whatever their sign against the control.

**Relationship to prior art, one sentence.** The teacher-student relation has chosen what to teach (TST)
and which loss to apply (IAD, DGAD, SAAD) but never which augmentation to apply under an Lp adversary, and
the two published principles predict opposite signs on exactly this set.

**Smallest experiment.** Two arms × twelve forks × 2.56 GPU-h = 61 GPU-h, plus the Direction 1 primary as
control if run as a separate plan (92 GPU-h total) or nothing extra if run as declared secondaries inside
Direction 1's plan (the standard allows secondaries in the same family; the primary stays S1-rich). The
mask needs the teacher's adversarial correctness at e100, which the per-sample store already records
(VERIFIED). Nothing else to build beyond Direction 1's transform.

**What it is worth if null.** It closes the last narrowly novel slot for augmentation in ARD at 61 GPU-h,
and the thesis can say that the augmentation lever does not read the teacher, which is a clean statement
next to the finding that the loss lever's teacher gates never transferred either.

---

## 5. Direction 3 — What does a robust teacher add to a ResNet-18 once the strongest teacher-free ingredients are present, and do those ingredients stack with the teacher?

This is the comparison `docs/ARD_VERSUS_AT_ASSESSMENT.md` says nobody has made. It is measurement-shaped
by nature; the method-shaped question inside it is whether AWP and SWA, which carry the teacher-free record,
add anything to a distilled student. The student has said a measurement alone is a fallback. It is listed
here because its first two steps cost almost nothing and every other direction depends on them, and
because its result is the spine any method result will be judged against.

**Question.** Under one engine, one 200-epoch schedule, one augmentation schedule (I100), one evaluation
(official test, best and last, CE-PGD20 and AutoAttack from saved weights) and five seeds per arm, how far
does RSLAD with the Chen teacher sit from TRADES + AWP + SWA + I100, and does adding AWP + SWA to RSLAD move
it? The literature assembled across papers puts the distilled student 0.5-1.5 pp above the teacher-free
band with no measured spread on the published side (INFERRED in the assessment from VERIFIED numbers). This
repository's RSLAD + I100 is 53.08 / 53.02 / 53.19 last-checkpoint AutoAttack on three seeds against a
teacher-free band of 52.3-52.8 measured on 50,000 images (VERIFIED both; the split caveat applies).

**Arms.**

| arm | status |
| --- | --- |
| RSLAD(Chen) + I100 | exists: two dev seeds, three confirmation seeds with official AutoAttack |
| PGD-AT + I100 | config-only |
| TRADES + I100 | config-only, after the corrected TRADES is validated |
| TRADES + AWP + SWA + I100 | needs AWP and SWA |
| RSLAD(Chen) + I100 + AWP + SWA | needs AWP and SWA |

**Predictions that would be wrong if the idea is wrong.** (i) The teacher is worth at least 0.5 pp over
TRADES + AWP + SWA + I100 at the same budget; wrong if RSLAD + I100 ≤ that arm + 0.2 pp, in which case the
thesis's answer flips to "AT suffices on ResNet-18" and the reframed thesis still stands. (ii) AWP + SWA
stack with the teacher by at least 0.5 pp; the one published data point is CAT's single-run RSLAD + AWP
51.62 against RSLAD 51.49, +0.13 pp (VERIFIED in `docs/SMALL_MODEL_AT_2024_2026.md` from the CAT PDF), and
SAAD applies SWA from epoch 95 without ablating it (VERIFIED in the assessment). Wrong if the stack is
under 0.2 pp, which would say the teacher already supplies what AWP and SWA supply, itself a mechanism
statement.

**Relationship to prior art, one sentence.** No paper puts a distilled ResNet-18 against the 2022-2024
teacher-free recipes on one engine with seeds, and no paper tests whether AWP or SWA gains are additive to
a robust teacher's.

**Smallest experiment.**

- *Step 0, this week, no decision needed, 5 GPU-h.* One corrected TRADES run, seed 0, config-only, expected
  49.0-49.4 AutoAttack at the best checkpoint (VERIFIED literature range, debugging note 0028); if it lands
  elsewhere, stop and debug before anything uses TRADES. DAJAT's checkpoint through AutoAttack on the full
  test set, about one hour, expected 52.48 ± evaluation noise (VERIFIED published value); if it does not
  reproduce, the evaluation stack is wrong and every comparison waits.
- *Step 1, about one week of attention.* AWP and SWA in the training step, through the scientific reviewer.
  AWP changes the optimiser step; SWA is small. Neither touches the attack.
- *Step 2, 80 GPU-h training plus about 40 GPU-h of AutoAttack.* Four new arms × five seeds × 4 GPU-h, with
  common data order across arms; forty checkpoints (best and last) through AutoAttack at about one hour each
  (REPORTED cost, plan 0091).
- Power: independent seeds. The e199 held-out two-run SDs on record are 0.29 pp (BASE), 0.48 pp (I100) and
  0.72 pp (CropShift) (DERIVED in `docs/MEASUREMENT_DESIGN.md` §2.5 from the five-seed cache); official-test
  AutoAttack SD for I100 over three seeds is 0.218 pp last, 0.079 pp best (VERIFIED, audit finding 3). With
  per-run SD s, the standard error of a difference of two five-seed means is 0.63 s, so 0.13-0.32 pp; a
  0.5 pp effect is resolvable if s is at or below about 0.28 pp (DERIVED). Declare 0.5 pp as the minimum
  and report the interval either way.

**What it is worth if null.** A null on (i) is the thesis: a teacher adds less than half a point to a
ResNet-18 when augmentation, AWP and SWA are already there. A null on (ii) says what the teacher is doing.
Both are publishable inside the thesis, and both are needed to interpret any positive from Directions 1, 2
or 4, because a 0.3 pp allocation gain matters differently on top of 53.1 than on top of 52.3.

---

## 6. Direction 4 — Is a second step of hardness at the second decay worth taking?

The global, teacher-free-compatible form of the IDBH account, and the cheapest method-shaped test of it.

**Question.** The switch at e125 equals the switch at e100 to the second decimal on both seeds, and about
three quarters of I100's gain arrives after e150 (section 1.5). If hardness pays only where robust
overfitting opens a gap, and the gap opens at each decay, then a second hardness step at e150 should pay,
and the same strong policy from e100 should pay less because it spends fifty epochs paying the accuracy
cost before the benefit exists. If instead IDBH_WEAK already saturates what a ResNet-18 can absorb, neither
arm moves.

**Arms** (six parents, e100→e199):

| role | canonical name |
| --- | --- |
| control | `Aug(CropShift→IDBH_WEAK@e100) @all /none` (= I100) |
| **primary** | `Aug(CropShift→IDBH_WEAK@e100→IDBH_STRONG@e150) @all /none` |
| secondary | `Aug(CropShift→IDBH_STRONG@e100) @all /none` |

IDBH_STRONG is the upstream strong pipeline: RandomErasing probability 1.0 instead of 0.5 (VERIFIED,
prior-art report from `idbh.py`; the local transform exposes the probability as a parameter).

**Prediction that would be wrong if the idea is wrong.** Ladder ≥ I100 + 0.3 pp and ladder > strong-from-
e100 by at least the floor. Wrong if ladder ≤ I100 (a second step does not pay) or if strong-from-e100 ≥
ladder (more hardness for longer is simply better, and the alignment with the decay is decoration).

**Relationship to prior art, one sentence.** IDBH fixes one policy for all 200 epochs and its text says
nothing about timing (VERIFIED); a hardness schedule stepped at the learning-rate decays is untested, and
the only scheduling precedent in adversarial settings anneals the other way (DYNACL, REPORTED). The novelty
is modest and is stated as such: it is a schedule, not a new principle.

**Smallest experiment.** Two arms × twelve forks × 2.56 GPU-h = 61 GPU-h; 92 GPU-h with its own I100
controls; nothing extra if run as declared secondaries inside Direction 1's plan, which is the same Aug
family and the same parents. Build: a two-step stagewise schedule and an `idbh_strong` late policy, about
forty lines and a schema change, reviewed. Endpoints and declared minimum as in Direction 1.

**What it is worth if null.** It pins the hardness dose for this student, which Direction 3's comparison
needs anyway (the teacher-free record uses IDBH[weak] + AWP + SWA), and it removes "why not a stronger late
policy" from the reviewer's list.

---

## 7. Ranking, and what to run first

Evidence per GPU-hour, with attention cost stated because it is the scarcer resource.

| rank | direction | GPU-h to a verdict | attention | what one GPU-hour buys |
| ---: | --- | ---: | --- | --- |
| 1 | Direction 1, allocation of hardness by margin | 123 (92 if controls shared) | 3 days build + review | Settles the direction of per-sample augmentation both ways, gives a dose point, and yields a method if either account holds. Nothing else closes as much per hour. |
| 2 | Direction 3, step 0 only: corrected TRADES and DAJAT through AutoAttack | 5 | none | Validates the comparator and the evaluation stack that every other direction is interpreted against. A prerequisite, not a direction, but the best ratio in the table. |
| 3 | Direction 4, hardness ladder | 61-92 | 1 day build + review | Tests the IDBH account globally; method-shaped; cheapest if folded into Direction 1's plan. |
| 4 | Direction 3, full comparison | 120 | 1 week for AWP/SWA + review | The thesis spine and the largest expected effect (0.5-1.5 pp), so the surest positive result; measurement-shaped. |
| 5 | Direction 2, teacher relation on the disagreement set | 61-92 | none beyond Direction 1 | Conditional on Direction 1 showing that location matters; before that it has no content. |

**Run first: Direction 1, with Direction 3's step 0 alongside because it needs no decision.** Concretely,
this week: Stage 0 of Direction 1 on the two existing prefixes; the corrected TRADES seed-0 run and the
DAJAT AutoAttack evaluation on one GPU; the mask-gated transform through review. Launch Direction 1 as soon
as six v2 parents exist, which the materialiser is already waiting for, with Direction 4's two arms declared
as secondaries in the same plan if the human accepts a six-arm campaign (184 GPU-h, about 37 hours of wall
clock on five GPUs, INFERRED). AWP and SWA are built in parallel because they are attention-bound, not
GPU-bound, and Direction 3's step 2 runs in the second fortnight. Direction 2 waits for Direction 1's
verdict.

Two things gate any launch and are the human's: plan 0093 is running and its by-products (the e199 fork
floor at six horizons) are the number every design above is sized against; and decision packet 0002 is
still `chosen: null`.

---

## 8. Numbers and where each was read

| number | where read | mark |
| --- | --- | --- |
| σ_d 0.092 / 0.124 / 0.159 pp at e114 / e109 / e104 | `docs/POST_DECAY_FLOOR.md`; audit says it matches `docs/experiments/ard_post_decay_floor_v1.json` | VERIFIED |
| I100 − CropShift +1.20 / +1.04 pp, clean −0.30 / −0.18; I150 +0.76 / +0.90 | `docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md`, endpoint table | VERIFIED |
| I125 +1.20 / +1.04 | `docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`, fresh endpoint table | VERIFIED |
| I100 − CROP_SUFFIX +0.78 / +0.68 / +0.62 at e199, mean +0.24 at e149, clean −0.06 / −0.22 / −0.46 | `docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md` | VERIFIED |
| official-test AutoAttack +0.41 / +0.06 / +0.46 (last), 53.08 / 53.02 / 53.19 | `docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md`; record `decision.per_seed_pp` read directly | VERIFIED |
| e149 five-seed differences +0.42 / +0.10 / +0.20 / +0.82 / +0.04 | per-seed values in `docs/MEASUREMENT_DESIGN.md` §2.5, quoted from the five-seed cache CSV; comparator split per audit finding 2 | VERIFIED values, DERIVED differences |
| e199 two-run SDs 0.287 / 0.479 / 0.721 pp (BASE / I100 / CropShift) | `docs/MEASUREMENT_DESIGN.md` §2.5 | DERIVED there |
| 92.2 s / epoch | `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`, runtime table, dev-1 CONTROL row | VERIFIED |
| S2×T1 state fraction 3.3 % | same report, mechanics table | VERIFIED |
| Clean-Wrong 9,263 / 8,709; pilot S3×T1 7,898 / 8,907 | `docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md` line 37 | VERIFIED |
| `continuation_seed: null` on plan 0087 arms | `docs/plans/0095-cohort-placebo-floor.md`, citing `arm-summary.json` | VERIFIED there |
| DAT 52.76 ± 0.14; IDBH 52.31 ± 0.26; DAJAT 51.85 ± 0.26; PGD-AT 47.63 ± 0.08 | `docs/SMALL_MODEL_AT_2024_2026.md` §0, from the DAT PDF | VERIFIED there |
| in-house PGD-AT 47.63; TRADES 45.14 (superseded) | `docs/EXPERIMENT_DASHBOARD.md` baseline table | VERIFIED |
| TRADES defect and fix; no rerun yet | commits `ef61b69`, `7666d77`, `3ccd333`; `docs/debugging/0028-...md` | VERIFIED |
| DAJAT checkpoint 87.3 % clean on 512 images, no AutoAttack yet | commit `914b10a` message | VERIFIED |
| CAT: RSLAD + AWP 51.62 vs RSLAD 51.49 | `docs/SMALL_MODEL_AT_2024_2026.md` §4, from the CAT PDF | VERIFIED there |
| AROID Vulnerability definition, motivation, PRN-18 50.57 vs IDBH 50.47 | arXiv HTML 2306.07197v2, quotes returned by the fetch tool | REPORTED |
| parents-v2 progress | `runs/parents-v2/logs/materialize.log`, `epoch-metrics.jsonl` line counts at 04:29 JST | VERIFIED |
| transforms id-keyed; erasing probability a parameter | `src/ard/data/datasets.py` | VERIFIED |
| AutoAttack about one hour per checkpoint | plan 0091 prose; the audit lists it as untraced | REPORTED |

---

## 9. Considered and rejected

One line each.

- **`PM(online)` and its coefficient neighbours.** Plan 0093, running.
- **The cohort placebo on the Clean-Wrong loss arm.** Plan 0095, preregistered; it is the right test and is
  not repeated here.
- **Any further state-conditioned *loss* arm** (TPFM, AdvCE, BD, SBF, KDScale, softening). Twenty-three
  UNDERPOWERED rows, all judged at e114 or before, no transfer mechanism shown; nothing new until plan
  0095 reports.
- **A per-step online augmentation gate on the student's state, learned or hand-set.** AROID.
- **Held-out margin as an endpoint.** Reproducible only to 1e-3 per sample (plan 0094), and relative
  precision no better than accuracy.
- **Batch ordering and history routing.** Rows B2, B3, B7, B8: refuted at 0.5 pp or inside the floor.
- **Teacher selection, learnability benchmark, directional margin distillation, asynchronous peer
  (July proposals).** Each needs a new lineage with its own floor and more than six weeks; the benchmark is
  a measurement.
- **A generated-data arm.** Outside the data contract, a human decision, and the published small-model
  regime there (58.6, Gowal 2021) is a hundred million images away.
- **Switching to AT outright.** Loses the comparison nobody has made; IDBH on PRN-18 is already published
  with seeds.
- **A schedule without the second decay to test Direction 4 causally.** A schedule change is a hard-rule
  scientific decision and confounds rate with epochs at low rate.
- **Re-running the switch-timing grid between e75 and e125.** I125 equals I100 to the second decimal on
  both seeds.
- **Any augmentation screen judged at e114.** Section 1.5: the effect is not there.
- **Per-sample attack-strength gating (FAT-style).** Published, and it changes the attack, which rule 6
  forbids.
- **Pooling old post-decay arms across campaigns.** Decision-packet territory (plan 0094), not an
  experiment; the old arms have inferred, not recorded, continuation seeds.
- **A stronger late policy as a family of its own.** Direction 4 keeps the one member with a mechanism
  prediction and drops the rest.

---

## The tension this rests on, checked at both ends

Direction 1 asks which way hardness should be allocated.  It is worth stating
that the two published accounts point in opposite directions, because that is
the whole reason the question is open.

**AROID says hard.**  Its abstract names its policy objective as "Vulnerability,
Affinity and Diversity", and Vulnerability targets the instances most
susceptible to adversarial attack.  VERIFIED by reading arXiv 2306.07197.  So the
one published instance-wise augmentation method for adversarial training gives
*more* augmentation to the samples that are *least* robust.

**IDBH's account says the opposite.**  Its causal story is that hardness helps
only through reduced robust overfitting and costs accuracy in a way that depends
on capacity, while diversity helps unconditionally.  On that account the budget
belongs where there is margin to spend, not where the model is already failing.
Three instance-wise methods in standard training allocate the same way.

Neither paper tests the allocation direction, and AROID has no random-allocation
control.  **The question is therefore open, cheap and falsifiable in both
directions**, and it does not depend on the teacher, which means a null result
still says something about adversarial training generally rather than only about
distillation.

That last property is also its weakness, and the thesis has to face it: a
direction that works without a teacher is not by itself an argument for
distillation.  Direction 5 exists to attach a teacher to whichever answer comes
out, and it is deliberately gated on direction 1 rather than run beside it.
