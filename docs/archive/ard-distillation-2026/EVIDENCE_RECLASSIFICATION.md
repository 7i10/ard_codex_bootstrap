# Evidence reclassification: which verdicts refuted a hypothesis, and which failed to test one

Status: read-only analysis. No training, no GPU, no checkpoint, no endpoint regeneration. One new
file; nothing else in the repository is touched.

This document is built on three audits committed at `a027a93` and does not re-derive what they
already establish:

- `docs/MEASUREMENT_DESIGN.md` — how big a difference these screens can resolve.
- `docs/COEFFICIENT_AUDIT.md` — how each coefficient was set, and whether any neighbour was tested.
- `docs/ARM_REGISTRY.md` + `docs/experiments/ard_arm_registry_v1.json` — 118 arms under canonical
  names, and which arms are the same treatment under different names.

Every number is marked **VERIFIED** (read from a file, with `file:line`) or **DERIVED** (computed
here, with the arithmetic shown).

---

## 0. Reader's guide

### 0.1 The problem in one paragraph

Over roughly forty preregistered screens this project returned "No-Go", "not supported", "no
promotion" or "MIXED" almost every time, and `docs/ERT_RESEARCH_STATUS_SUMMARY.md` now treats those
verdicts as settled. The measurement audit shows the gap between two runs that differ **only in
their random number stream** is 1.14–1.25 pp before the learning-rate decay and 0.25–0.50 pp after
it, while the effects being judged were typically 0.02–2 pp. A verdict reached below its own floor
did not refute a hypothesis; it failed to test one. This document says, screen by screen, which is
which.

### 0.2 Terms, defined on first use

| term | meaning |
| --- | --- |
| **arm** | one named training configuration: the usual recipe plus one extra rule |
| **block** | one matched training lineage that produced a treatment-minus-control number, e.g. `L2-R1`. Two seeds with one control each is `k = 2` blocks |
| **pp** | percentage point. On the 5,000-image held-out split, one image = 0.02 pp |
| **floor (`s`)** | the standard deviation of the difference between two runs that differ only in randomness. This is the quantity an effect must be compared against |
| **MDE** | minimum detectable effect: the smallest effect `k` blocks can find at 80% power and the 5% level. `MDE = 2.8 * s / sqrt(k)` (`docs/MEASUREMENT_DESIGN.md:77`) |
| **blocks needed** | `k = 7.849 * s^2 / d^2` for a target effect `d` (`docs/MEASUREMENT_DESIGN.md:75`) |
| **pre-decay** | measured while the learning rate is still 0.1 — every screen from an epoch-79 parent to epochs 84/89/94 |
| **post-decay** | measured after the final learning-rate decay at epoch 100 |
| **direct effect** | measured on the same training samples the treatment was applied to. Optimistic by construction |
| **held-out effect** | measured on the fixed 5,000-image validation split the treatment never touched |

### 0.3 The floor table used throughout

Reproduced from `docs/MEASUREMENT_DESIGN.md:367-388` (VERIFIED there; all figures DERIVED in that
document from the cited campaigns).

| regime and horizon | floor `s` (pp) | how many comparisons it rests on |
| --- | ---: | --- |
| pre-decay, e79 parent → e84 | **1.14** | 84 untreated pairwise comparisons |
| pre-decay, e79 parent → e89 | **1.20** | 84 |
| pre-decay, e79 parent → e94 | **1.25** | 84 |
| pre-decay, **clean** metric, e94 | 1.40 | 84 |
| pre-decay, pooled working value | 1.21 | five independent campaign families agree |
| post-decay, optimistic (paired, 5 seeds, e199) | 0.25 | 5 seeds, 4 degrees of freedom |
| **post-decay, working value** | **0.40** | midpoint of a bracket, not a measurement |
| post-decay, placebo fork measured (e199) | 0.50 | 6 comparisons |
| post-decay, pessimistic (between seeds, CROPSHIFT e199) | 0.72 | 10 |

Two rules from that audit are load-bearing here and are applied without exception:

- **The post-decay floor for the design the project keeps using has never been measured.** No two
  *untreated* forks from a common post-decay parent were ever run and compared
  (`docs/MEASUREMENT_DESIGN.md:854-860`). The 0.25–0.50 pp bracket is an interpolation between two
  adjacent designs. Where a row's classification would change across that bracket, this document
  says so.
- **A bootstrap interval over the 5,000 held-out images is not an interval over training runs.**
  `ST1W`'s `[+0.680, +2.320]` is the former, and the source report says so itself
  (VERIFIED, `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:29-31`). Sample-level intervals appear in
  this document only when explicitly labelled "over samples".

---

## Part 2 — The classification rubric

Stated before the table, as required, so a reader can check the work. The five classes are applied
**in this order**; the first one that fits wins.

### 1. NOT-AN-EFFECT-CLAIM

The verdict is not about a difference in accuracy at all. It is a numerical-divergence finding, a
mechanism identification, a correlation or AUROC claim, or a feasibility/inventory result. The
accuracy floor does not apply to these and they stand or fall on their own terms.

Examples in this project: `S-BDD`'s `NUMERICALLY_UNSUPPORTED` (the observable is finite versus
non-finite, not an accuracy difference) and the ordering probe's `MECHANISM NOT IDENTIFIED`.

### 2. STANDS

The observed effect is comfortably above the floor for its regime — taken here as **at least twice
the floor** — and the verdict follows from it. Where the post-decay bracket matters, the row states
the margin at each end of the bracket rather than picking the flattering one.

### 3. REFUTED

The study was able to see the effect size its plan **preregistered as interesting**, and did not see
it. Judged against the preregistered target, never against the tiny observed value. Only two screens
in the whole project stated a numeric target, so this class is small.

A note on how "able to see it" is applied. The strict reading is 80% power at the target. Where a
study falls short of that but the observed value nonetheless lands well *below* the gate, the
document records the one-sided arithmetic and says exactly what is refuted: an effect *as large as
the target* is ruled out; a smaller effect is not. That distinction is stated in the row, never
glossed.

### 4. UNDERPOWERED

The floor for that regime exceeds the effect the study would have needed to detect, so the verdict
carries no information about the hypothesis. Each such row states the minimum detectable effect for
the number of blocks actually run, and the number of paired blocks that would have been required,
using the arithmetic in `docs/MEASUREMENT_DESIGN.md:72-79`.

An UNDERPOWERED verdict is **not** evidence that the treatment works. It is the absence of evidence
either way. Nothing in this document argues that any shelved treatment should be revived on the
strength of its own null result.

### 5. COEFFICIENT-UNTESTED

An additional flag, **not exclusive** with the others. The treatment ran at a single calibrated
coefficient with no neighbour value ever trained, so even a clean negative bounds only that one
point in coefficient space. Twenty of the project's twenty-three calibrated scalars are in this
state (VERIFIED, `docs/COEFFICIENT_AUDIT.md:494-502`).

Where a row is both UNDERPOWERED and COEFFICIENT-UNTESTED, both are recorded. They are different
failures. The first says the experiment could not see the effect it was looking for. The second says
that even a perfect experiment would have described one point on a curve.

### UNRESOLVED

Used where the records do not permit a classification. The row states the **single fact** that would
settle it.

---

## Part 1 — The table

One row per distinct scientific question, not per arm and not per report. Duplicate arms are merged
using the registry and every alias is named, so a reader of the old reports can find the row.

"Effect" is the held-out treatment-minus-control difference at the primary endpoint, per seed,
unless the cell says otherwise. "Target" is the preregistered numeric effect the plan stated, or `—`
if the plan stated only a sign or a direction. "Nbr?" is whether a different value of the same
coefficient was ever trained under a matched design.

### Group A — global augmentation schedule (full 200-epoch runs)

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect (pp, per seed) | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| A1 | Does CropShift augmentation beat the canonical RSLAD augmentation? | `CROPSHIFT` vs `BASE(aug)`; registry warns `BASE` names three unrelated things | held-out CE-PGD20, e199 | post | +1.32 / +1.16 (2 seeds, VERIFIED); within-seed over 5 seeds +2.18 / +1.30 / +2.10 / +1.32 / +1.16, mean **+1.612**, SD 0.487 (DERIVED) | 2, later 5 | 0.25–0.50; 0.72 pessimistic | — | n/a (not a coefficient) | **STANDS** — +1.612 pp is 3.2x the 0.50 pp placebo-fork floor and 2.2x the 0.72 pp pessimistic floor | `ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md:71-74`; five-seed CSV via `MEASUREMENT_DESIGN.md:289-296` |
| A2 | Does switching to IDBH_WEAK at epoch 100 beat CropShift alone? | `I100`; alias `I100_SUFFIX` (confirmation re-run). vs `CROPSHIFT` / `CROP_SUFFIX` | held-out CE-PGD20, e199 | post | +0.78 / +0.68 / +0.62 / +1.20 / +1.04; mean **+0.864**, SD 0.247, t(4) = 7.82, 95% interval **over training runs** [+0.557, +1.171] | 5 | 0.25 (type c, this exact contrast) | — | n/a | **STANDS** — 3.5x the 0.25 pp paired floor, 2.2x the 0.40 pp working value, 1.7x the 0.50 pp placebo floor; 5 of 5 seeds positive. Margin is comfortable at the working value and adequate at the pessimistic end | `ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:11`; `ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:46-53`; `MEASUREMENT_DESIGN.md:341-353` |
| A3 | Does the epoch at which the augmentation switch happens matter? | `I0`, `I50`, `I75`, `I125`, `I150` vs `I100` | held-out CE-PGD20, e199 | post | I0 -0.08 / -0.58; I50 -0.52 / -0.12; I75 -0.24 / -0.02; I125 +0.00 / +0.00; I150 -0.44 / -0.14 (DERIVED) | 2 | 0.40 working | — | n/a | **UNDERPOWERED** — MDE at k=2 is **0.79 pp** (0.99 at s=0.50); every observed difference is smaller. Detecting 0.5 pp needs **k=5** blocks (k=8 at s=0.50). "I100 is the best timing" is not supported; "no tested timing differs from I100 by more than this screen can see" is | `ERT_RSLAD_SINGLE_SWITCH_TIMING.md:18-23, 31-42` |
| A4 | Do stronger static policies applied throughout training help? | `CROP_RE`, `IDBH_WEAK` vs `CROPSHIFT` | last robust e199 **and** normalized trajectory AUC | post | last robust: CROP_RE +0.86 / +0.80; IDBH_WEAK +0.74 / +0.76. AUC: CROP_RE -0.509 / -0.794; IDBH_WEAK -0.557 / -0.635 | 2 | robust 0.40 working; **AUC floor never measured** | — | n/a | **UNRESOLVED.** The verdict was decided by the AUC gate, and the run-to-run standard deviation of the normalized trajectory-AUC statistic has never been measured anywhere in this project. Settling fact: the AUC spread between two untreated full runs. On the robust side the effect (+0.74 to +0.86) sits right at the k=2 MDE of 0.79 pp, so that half is UNDERPOWERED | `ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md:43-46, 75-90` |

### Group B — student history: prediction, and interventions built on it

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| B1 | Does student margin-history predict future failure better than current state does? | no training arm — a read-only replay | held-out delta AUROC and Spearman | n/a | Chen +0.06220 (paired 95% CI [+0.05470, +0.07001]); Bartoldson +0.05250 ([+0.04549, +0.06027]); L1–L4 +0.0499 / +0.0686 / +0.0513 / +0.0663; P4-over-P2 Spearman +0.1560 / +0.1551 / +0.1587 on three unseen seeds | 4 lineages, 3 unseen | n/a | AUROC delta >= 0.02 | n/a | **NOT-AN-EFFECT-CLAIM** — a discrimination claim, not an accuracy difference. Unaffected by the floor; stands on its own terms, and it cleared a preregistered numeric gate by 2.5x | `EXPERIMENT_DASHBOARD.md:420-435`; `plans/0015-...:60`; `ERT_RSLAD_STUDENT_HISTORY_PREDICTIVE_VALIDITY.md:14-20` |
| B2 | Does target-softening on a frozen outcome-informed failure mask beat class-matched random masks? (upper-bound experiment) | `bart-oracle-soft-s0` vs `bart-rand{1,2,3}-soft-s0` | best AutoAttack accuracy, full 200-epoch run | post | mask 47.36; randoms 46.35 / 47.37 / 47.04; random mean 46.92; **mask minus random mean = +0.44** | 1 seed, 4 arms | **0.601** (DERIVED here as the SE of mask-minus-random-mean, from the three random arms themselves: SD 0.5205, times sqrt(1 + 1/3)) | **+0.50 pp over the random mean, and above every random** | n/a (mixing rho = 0.5 is a design constant) | **UNDERPOWERED.** z = 0.44 / 0.601 = **0.73**. MDE at k=1 is **1.68 pp**; detecting the +0.50 pp gate needs **k = 11** replicates of the four-arm block. The recorded verdict `inconclusive` is the right word for the wrong reason: the design could never have resolved its own gate. The one random arm that beat the mask (47.37 vs 47.36) is a coin toss at this spread | `EXPERIMENT_DASHBOARD.md:411-416`; `plans/0015-history-replication-gate-and-handoff.md:28-32, 195-198` |
| B3 | Does true-label-anchored target mixing on history-selected samples improve Best validation PGD? | `PF_TA`, `NR_TA` vs `C`, and vs matched-random `PF_R`, `NR_R` | Best validation CE-PGD20, e40→e199 | post | PF-TA minus C = **-0.18** (two-seed mean); NR-TA minus C = **-0.17**; NR-TA minus NR-R = +0.15 | 2 | **0.426** (Best PGD, placebo forks in this same campaign, DERIVED) | **two-seed mean >= +0.50 pp, each seed non-negative** | no (0.5/0.5 mix never varied; not one of the 23 audited scalars but structurally the same gap) | **REFUTED at the preregistered target**, + **COEFFICIENT-UNTESTED.** SE of the two-seed mean = 0.426/sqrt(2) = 0.301 pp; the observed -0.18 sits **2.26 SE below the +0.50 gate** (DERIVED), one-sided p ~ 0.012. Honest caveat: a-priori power to detect exactly +0.50 pp was only **38%** two-sided (DERIVED), so this refutes "a gain as large as +0.5 pp" and does **not** refute a smaller gain. MDE at k=2 was 0.84 pp; the gate needed k=6 | `HISTORY_ROUTING_V2_RESULTS.md:46-64`; `plans/0020-...:79-91, 177-187`; `MEASUREMENT_DESIGN.md:253-257` |
| B4 | Does history-smoothed online S3 routing improve held-out robustness? | `INST075` (= `S3DYN075`), `M3_075` (alias `M3E2`), `M3E2_075` vs `BASE(S3hist)` | held-out CE-PGD20, e94 | pre | INST075 -0.18 / -0.16; M3_075 -0.88 / -0.46; M3E2_075 -1.22 / -0.98 | 2 | 1.25 | — | no (beta = 0.075 frozen; no neighbour in this family) | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is **2.47 pp**; the largest observed effect is 1.22 pp. Detecting the biggest of them needs k=8; detecting INST075's -0.18 needs **k > 380** | `ERT_S3_HISTORY_PRODUCTION_RESULTS.md:28-35, 83-90` |
| B5 | Does majority-3 smoothing reduce routing churn? | same arms as B4 | switch and re-entry counts on the train split | pre | switches 149,468 → 70,797 (L2) and 149,556 → 71,101 (L4), about -52.6%; re-entry -66% | 2 | n/a | — | n/a | **NOT-AN-EFFECT-CLAIM** — a count of routing events, not an accuracy difference. Stands on its own terms. This is the half of the ledger sentence that survives | `ERT_S3_HISTORY_PRODUCTION_RESULTS.md:46-53` |
| B6 | Does freezing the S3 cohort beat recomputing it every step? | `S3FIX075` vs `S3DYN075`, both vs `DYNBASE` | held-out CE-PGD20, e94 | pre | S3FIX075 **+1.82 / +0.62**; S3DYN075 -0.34 / +0.00 | 2 | 1.25 | — | no | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** +1.82 pp is the largest held-out number in any pre-decay family, and it is still below the k=1 MDE of 3.50 pp. For scale, this project has directly observed two **untreated** runs differing by 3.30 pp at e84. Detecting +1.22 (the two-seed mean) needs k=9 | `ERT_DYNAMIC_S3_RECOVERY_RESULTS.md:47-59`; `MEASUREMENT_DESIGN.md:193` |
| B7 | Does history-balanced batch ordering improve robustness? | `NEW_HISTORY` vs `NEW_CONTROL` (the earlier `HISTORY_BALANCED` / `CONTROL(order-v1)` pair was blocked and never trained) | held-out CE-PGD20, e199 | post | e199 +0.020 / -0.360; e149 -0.620 / -0.260; final robust +0.060 / -0.100 | 2 | 0.40 working | — | n/a | **UNDERPOWERED.** MDE at k=2 is 0.79 pp; every observed value is smaller. Detecting 0.36 pp needs **k = 10** | `ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md:7-10` |
| B8 | Can a batch-order descriptor explain the ordering effect? | `SHUFFLE_PLUS_0` … `SHUFFLE_PLUS_7` | descriptor-to-probe-AUC association | post | no reproducible association across two seeds | 2 | n/a | — | n/a | **NOT-AN-EFFECT-CLAIM** — `MECHANISM NOT IDENTIFIED` is a statement about a descriptor, not about accuracy. Stands on its own terms | `ERT_RSLAD_ORDERING_MECHANISM_DISCOVERY.md:88-92` |

### Group C — Stage A and the Confirmatory re-run (epoch-79 parent, pre-decay)

The registry establishes that the Confirmatory campaign re-ran three Stage A arms under new names.
`ST3K05 = T3LP05CONF` is an exact coefficient match; `ST1W` / `T1WCONF` and `ST2W` / `T2WCONF` are
near-twins at a rounded coefficient (+5.7%). **Each pair is one line of evidence, not two.**

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| C1 | Does weak extra adversarial CE on the S3xT1 cohort transfer to held-out data? | `ST1W` (alias `ST1W-rerun`) = `T1WCONF` at rounded beta 0.075; the overlapping-cohort screen `S3FIX075` is the same idea on a superset cohort (row B6) | held-out CE-PGD20, e84 (Stage A) and e84/89/94 (Confirmatory) | pre | ST1W **+1.500** (sample bootstrap [+0.680, +2.320], **over images, not runs**) / **+0.040**. T1WCONF e84 -0.14 / -0.04; e94 -0.34 / -0.92 | 2 per campaign | 1.14 (e84) | — | **yes** — 2x (`ST1M`) and +5.7% (`T1WCONF`) | **UNDERPOWERED.** Over training runs, z = 1.500 / 1.136 = **1.32**, a one-in-five event under the null. MDE at k=1 is 3.18 pp; at k=2, 2.25 pp. Detecting +1.5 pp needs **k = 5**; detecting +0.5 pp needs **k = 40**. The L2/L4 spread of 1.46 pp is almost exactly the predicted floor. Not flagged COEFFICIENT-UNTESTED — but the neighbours reverse sign by seed, and nothing **below** the calibrated 0.070959 was ever trained | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52, 69, 86-90`; `ERT_CONFIRMATORY_T123_RESULTS.md:79, 85, 88, 94`; `MEASUREMENT_DESIGN.md:781-848`; `COEFFICIENT_AUDIT.md:434-453` |
| C2 | Does weak extra adversarial CE on the S3xT2 cohort transfer? | `ST2W` = `T2WCONF` (rounded, +5.7%) | held-out CE-PGD20, e84 / e94 | pre | ST2W **+1.420 / +1.320** (positive on both seeds, with direct effects near zero: -0.046 / -0.281). T2WCONF e84 +0.32 / -0.18; e94 -0.54 / -0.32 | 2 per campaign | 1.14 (e84) | — | yes (`ST2M`, 2x) | **UNDERPOWERED.** Two-seed mean +1.37 pp against a k=2 MDE of **2.25 pp**; needs k=6. Note the pattern: a held-out gain with no direct gain is spillover, which is what a control anomaly looks like. The re-run at +5.7% is negative on both seeds at e94 | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:55, 72`; `ERT_CONFIRMATORY_T123_RESULTS.md:80, 86, 89, 95` |
| C3 | Does halving the adversarial-KD pressure on the S3xT3 cohort help? | `ST3K05` = `T3LP05CONF` (**exact** coefficient match, mult 0.5) | held-out CE-PGD20, e84 / e94 | pre | ST3K05 **+0.980** [+0.120, +1.820] over images **/ +1.280**. T3LP05CONF non-positive in all six seed-horizon cells: -0.22 / -1.42 / -1.80 / -0.10 / -2.14 / -0.96 | 2 per campaign | 1.14 (e84) | — | yes — a genuine 3-point design grid `ST3K1`/`ST3K05`/`ST3K0` | **UNDERPOWERED.** The identical arm is **positive on both seeds in one campaign and negative on both seeds in the other**, and the flip size (L2 1.20 pp, L4 1.38 pp; DERIVED) is almost exactly the shift between the two campaigns' controls (`C79CONF` minus `C79` = +0.94 / +1.78 pp; VERIFIED in the coefficient audit). This is a control artefact, not a treatment result. The project's own Rule 6 cites `T3LP05CONF` as a legitimate "ruling things out" case; that citation does not survive this comparison | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:59, 76`; `ERT_CONFIRMATORY_T123_RESULTS.md:81, 84, 87, 90, 93, 96`; `COEFFICIENT_AUDIT.md:414-427`; `MEASUREMENT_DESIGN.md:693` |
| C4 | Does softening the teacher target on the S3 cohorts help? | `ST1S`, `ST2S` (alias `ST2S-rerun`), `alpha_soft` = 1.2523 | held-out CE-PGD20, e84 | pre | ST1S -0.240 / **+2.180**; ST2S -0.040 / **+1.960** | 2 | 1.14 | — | **no** — audit #1, nothing above or below ever trained | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 2.25 pp. The two L4 values sit inside the Stage A control anomaly: 11 of 12 L4 arms are positive with mean +1.010 pp, which is what one low control draw looks like, not twelve working treatments | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:54, 57, 71, 74`; `MEASUREMENT_DESIGN.md:526-532`; `COEFFICIENT_AUDIT.md:462` |
| C5 | Does doubling the AdvCE coefficient change the outcome? (the project's only designed dose contrast) | `ST1M` / `ST2M` at 2x `ST1W` / `ST2W` | held-out CE-PGD20, e84, treatment-minus-treatment | pre | ST1: L2 **-3.88**, L4 **+0.40**; mean -1.74, between-seed SD **3.03**. ST2: L2 -2.40, L4 +0.10; mean -1.15, SD 1.77 (all DERIVED) | 2 | 1.14 | — | yes, by construction | **UNDERPOWERED.** The headline "doubling flips the sign" holds on one seed and vanishes on the other, for both cohorts. The observed between-seed SD of 3.03 pp is nearly three times the assumed floor, so with k=2 the t-statistic is 0.81 on 1 degree of freedom. A dose response is not established in either direction | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52-56, 69-73`; `COEFFICIENT_AUDIT.md:532-561` |
| C6 | Does extra clean CE at ~0.078 on the Clean-Wrong cohort help held-out robustness? | `CW1`, `CW2`, `CW3` (identical coefficient, differing only in a sub-mode); near-duplicates `C4`, `C5` at 0.075 | held-out CE-PGD20, e84 | pre | CW1/2/3 L2 -1.300 / -1.460 / -1.440; L4 +0.440 / +0.100 / -0.580. C4 -0.32 / -0.62; C5 -0.52 / -2.10 | 2 | 1.14 | — | **no** — audit #4, all three share one value | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 2.25 pp; the largest effect is 1.46 pp. The registry warns these must not be counted as three independent confirmations of "CleanCE at 0.075–0.078 does not help" — they are one | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:61-63, 78-80`; `ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md:15-16, 31-32`; `ARM_REGISTRY.md:280-284` |

### Group D — the Clean-Wrong family (epoch-79 parent, pre-decay)

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| D1 | Does extra clean CE at 0.15 on the whole fixed Clean-Wrong cohort help? | **`A1` = `C10` = `G1_CW_ALL_CE015` (alias `G1`) = `F1`** — the project's largest duplicate cluster; `A1` and `G1` are byte-identical numbers, `F1` is documented as reused | held-out CE-PGD20, e94 (and e84 for `C10`) | pre | e94 **-1.64 / -0.64**; e84 +0.04 / -0.56 | 2 (**one** trained run per seed, reported four times) | 1.25 (e94) | — | yes — 0.075 (`C4`/`C5`) and 0.078 (`CW1`) on the same cohort | **UNDERPOWERED.** MDE at k=2 is 2.47 pp. The four-way reporting is the single most misleading feature of the old records: a reader counting reports sees four independent screens where there is one run per seed | `ERT_CW_MARGIN_GENERALIZATION_SCREEN.md:36, 63`; `ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md:31, 34`; `ARM_REGISTRY.md:262-278` |
| D2 | Does gating that CleanCE by teacher reliability improve it? | `G2_CW_R_CE20_CE015` (alias `G2`), `G3_CW_R_KL10_CE015` (alias `G3`) vs `G0_BASE` and vs `G1` | held-out CE-PGD20, e94 | pre | G2 minus G0 **+0.44 / -0.56**; G3 minus G0 -1.06 / -1.14; G2 minus G1 +2.08 / +0.08 | 2 | 1.25 | — | n/a (the gate is a selector; the 0.15 coefficient is shared with D1) | **UNDERPOWERED.** MDE at k=2 is 2.47 pp. G2-minus-G1 reaches +2.08 pp on L2 but +0.08 pp on L4 — the same one-seed pattern as C1 and C5. Detecting the two-seed mean of -0.06 pp would need **k > 10,000** | `ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md:29-41, 57-60` |
| D3 | Does the teacher-probability-floor margin treatment on Clean-Wrong help? | **`A7` = `F2` = `L2_CAL` = `A100`** (aliases `CW-TPFM`, "A7 margin-only"; `A100` is a genuine fresh replicate on different hardware, not a byte-reuse) | held-out CE-PGD20, e94 | pre | A7 +0.82 / +1.06; A100 (4 matched blocks) +1.340 / -0.200 / +0.660 / -0.060. **Six blocks: mean +0.603 pp, between-block SD 0.614** (DERIVED) | 6 blocks across 2 campaigns | 1.25 | — | **yes — the only densely swept coefficient in the project** (0, 0.10, +/-5%, +/-10%, 0.25, 0.50) | **UNDERPOWERED**, and the closest near-miss in the pre-decay records. MDE at k=6 is **1.43 pp** against an observed +0.603. Using the within-study SD instead gives t(5) = 2.41 (p ~ 0.06), but that SD's own 95% interval is [0.383, 1.506] pp (DERIVED, chi-square, df 5) and contains the campaign-wide floor, and two of the six blocks come from a different campaign with a different control. Detecting +0.603 pp at the campaign floor needs **k = 34** | `ERT_CW_MARGIN_GENERALIZATION_SCREEN.md:42, 69`; `ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`; `ARM_REGISTRY.md:324-329` |
| D4 | Does the margin coefficient lambda change the outcome? | `L1_010`, `L3_025`, `L4_050`, `N90`, `N95`, `N105`, `N110` around the calibrated 0.2388 | held-out CE-PGD20, e94 | pre | arm means over 4 blocks: N90 +0.580, N95 +0.130, A100 +0.435, N105 +0.015, N110 -0.715; **spread of arm means 1.295 pp** (DERIVED). The famous "+5% flips the sign" step is -1.90 pp in one block, -0.82, **+1.10**, -0.06 in the other three | 4 matched blocks x 5 arms | 1.25 | — | yes | **UNDERPOWERED** for the lambda-effect question. MDE for an arm contrast at k=4 is **1.75 pp** against a 1.295 pp observed spread; F(4,12) = 1.69, permutation p ~ 0.19. Separately, the same data establish a **measurement** finding that does stand: changing nothing at all moves the e94 effect by up to **2.06 pp**, the same size as the +5% coefficient step (1.90 pp) | `ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`; `ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md:7-36`; `COEFFICIENT_AUDIT.md:341-391` |
| D5 | Do any of fifteen Clean-Wrong mechanism variants beat the control? | `C1`–`C9`, `C11`–`C15` vs `C0` (MART-style BCE, adaptive pressure, teacher gate, IAD-inspired, KD rescaling, reduced attack budget) | held-out CE-PGD20, e84 | pre | L2 range -2.70 to +0.34; L4 range -2.10 to 0.00. **Zero of fifteen arms are positive on both seeds** (DERIVED) | 2, **one control per seed** | 1.14 | — | **no** — audit #7, `beta_bce` single point from a 4-batch probe | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 2.25 pp; only one of thirty arm-seed values exceeds it. "Zero of fifteen positive on both seeds" is not fifteen independent negatives: all fifteen sit above the same two control endpoints, so one high control draw per seed produces exactly this picture — the mirror image of Stage A's "11 of 12 L4 arms positive" | `ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md:11-42`; `MEASUREMENT_DESIGN.md:526-532, 641-649` |
| D6 | Does extra clean CE raise clean accuracy on the treated Clean-Wrong **training** cohort? | `CW2`, `CW3`, `G1`/`G2`/`G3` | direct clean accuracy on the fixed train cohort (n ~ 8,600–8,900), e84 / e94 | pre, **train split** | CW2 +5.868 (L2) / +2.790 (L4); CW3 +5.439 / +4.874; G1 CW clean net rescue +5.64% / +8.67% | 2 | ~0.24 pp (DERIVED: the whole-train two-run clean gap is 0.085–0.104 pp over 45,000 rows; scaled to n = 8,623 by sqrt(45000/8623) = 2.28) | — | see C6 / D1 | **STANDS** as a train-cohort claim — 12x to 25x the scaled train floor, replicated across two seeds and two campaigns. It licenses **nothing** about held-out accuracy (Rule 8), and the scaling of the train floor to a subcohort is an assumption, flagged | `ERT_STAGE_A_EFFECT_DECOMPOSITION.md:62-63, 79-80`; `ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md:50-55`; `MEASUREMENT_DESIGN.md:150-166, 701-706` |

### Group E — the I100 post-decay S2 family (epoch-99 parent, epochs 100–114)

Every row in this group is measured against **the same two control endpoints**. dev-1 e114 control
robust is `57.32%` in all three source reports (VERIFIED,
`docs/MEASUREMENT_DESIGN.md:562-565`). Twelve-plus arm-seed values resting on two control runs are
not twelve pieces of evidence, and all twelve are positive at e114 with mean +0.120 pp — which one
low control draw explains entirely.

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| E1 | Does pair-margin preservation on the S2xT1 cohort improve held-out robustness? | **`DPM` = `OS_PMP`** (aliases `PMP`, `OS-PMP`) — identical mechanism and identical coefficient 0.05380932585058825, differing only in whether the cohort is frozen at e99 or recomputed each step | held-out CE-PGD20, e114 | post | DPM +0.08 / +0.12; OS-PMP **+0.14 / +0.20**. Pooled +0.170 pp = 17 net images out of 5,000 | 2 | 0.25–0.50 (bracket); 0.40 working | — | **no** — audit #21; 95% sampling band on the coefficient 0.0520–0.0609 | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** Not significant even **over images**: 1.61 sample-level SE pooled, short of 1.96. Over training runs the effect is under one SE at every candidate floor. Blocks needed: **17 (s=0.25) to 141 (s=0.72); 44 at the working value.** The screen ran two. "Both seeds positive" has null probability 0.25, not 0.05. **The `SUPPORTED` label should read `DIRECTIONAL, UNRESOLVED`** | `ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:3-15, 45-56`; `ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:34-37`; `MEASUREMENT_DESIGN.md:708-779`; `COEFFICIENT_AUDIT.md:482, 636` |
| E2 | Does detached boundary-distance preservation on the same cohort improve held-out robustness? | **`D-BDD` = `OS_DBDP`** (aliases `DBDD`, `DBDP`, `OS-DBDP`, `D-BDP`) — identical coefficient 31.649566509850324 | held-out CE-PGD20, e114 | post | D-BDD +0.04 / +0.20; OS-DBDP +0.14 / +0.06. D-BDD minus DPM -0.04 / +0.08 | 2 | 0.25–0.50 | — | **no** — audit #22; sampling band 29.16–38.99, a 8.5% bootstrap SE on the median | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** Same arithmetic as E1. Detecting +0.10 pp needs **k = 126** at the working floor. The `NOT_SUPPORTED` verdict on D-BDP-specific superiority is equally uninformative: it compares two arms whose difference is 0.04–0.14 pp | `ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:11-37`; `ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:45-56`; `COEFFICIENT_AUDIT.md:483, 639` |
| E3 | Is the secant boundary-distance formulation numerically usable? | `S-BDD` (v1 coefficient 3.487, v2 1.522 — two different formulas, not a coefficient pair) | finite versus non-finite training loss | post | dev-1 non-finite at e106 (last finite loss 8.58e+10, weight max 3.2e+27); dev-2 no valid checkpoint after e101. Both hosts, same identity | 2 | n/a | — | n/a | **NOT-AN-EFFECT-CLAIM.** The observable is divergence, not accuracy, so the floor is irrelevant and `NUMERICALLY_UNSUPPORTED` stands on its own terms. Worth adding: the v1 calibration artifact already showed a 126x achieved-ratio spread while both siblings sat at ~2x, so this was visible **before launch** | `ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:84-91`; `COEFFICIENT_AUDIT.md:259-288` |
| E4 | Does a student-boundary-floor margin improve held-out robustness? | `SBF` | held-out CE-PGD20, e114 | post | +0.160 / +0.040 | 2 | 0.25–0.50 | — | **no** — audit #16/#17 | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 0.79 pp; detecting +0.10 pp needs k = 126. A further defect specific to this arm: its floor is **per seed** (0.04178 for dev-1, 0.03348 for dev-2, a 25% difference), so the two seeds were trained against different targets and "both seeds positive" is not a replication of the same treatment | `ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:8, 19, 23-30`; `COEFFICIENT_AUDIT.md:477-478, 166-171` |
| E5 | Does a teacher-probability-floor margin on the S2xT1 cohort improve held-out robustness? | `TPFM` @S2T1 (registry hazard: **three** unrelated TPFMs exist; never cite a bare "TPFM result") | held-out CE-PGD20, e114 | post | +0.220 / +0.040 | 2 | 0.25–0.50 | — | **no** — audit #18–20 | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** Identical arithmetic to E4. Direct fixed-train net rescue was +1.266 / +1.074 pp, roughly ten times the held-out effect — Rule 8 again | `ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:8-10, 23-30`; `ARM_REGISTRY.md:296-312` |
| E6 | Do the Stage-A-derived actions transfer to the I100 lineage? | `PILOT_S3_T1_WEAK_ADVCE`, `CLEAN_WRONG_PLAIN_ADVCE` (alias `PLAIN_ADVCE`), `CLEAN_WRONG_A7_MARGIN_ONLY` (alias `TPFM(action-transfer)`) | held-out CE-PGD20, e114 | post | +0.22 / +0.04; +0.44 / +0.10; +0.44 / +0.08 (DERIVED from the absolute endpoint table). Direct train robust effects +3.520 / +1.663 / +1.684 | 2 | 0.25–0.50 | — | **no** — audit #12–15, four single-point scalars | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 0.79 pp; the largest effect is 0.44 pp and its own replicate is 0.10 pp. Detecting +0.26 pp (the best two-seed mean) needs **k = 19** | `ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md:56-79, 103-114` |
| E7 | Does the treated-cohort rescue survive to epoch 199? | `CLEAN_WRONG_PLAIN_ADVCE` and `CLEAN_WRONG_A7_MARGIN_ONLY` continued to e129/149/169/189/199 | held-out CE-PGD20, five long horizons | post | 20 values, mean **+0.136**, SD **0.316**; Plain AdvCE mean -0.036, TPFM mean +0.308; at e199 TPFM +0.040 / +0.200. Sign flips between consecutive horizons in **7 of 16** opportunities | 2 (the five horizons are not independent) | 0.25–0.50 | — | **no** | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** The observed SD of 0.316 pp **is** the post-decay floor; five horizons buy measurement occasions, not power. The honest reading is not "the effect settles" but "there is no effect here larger than the floor, and what moves is noise" | `ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:34-53`; `MEASUREMENT_DESIGN.md:470-497` |
| E8 | Is there a real gap between direct train-cohort rescue and held-out accuracy? | cross-family: `ST1W`, `ST2M`, `CLEAN_WRONG_PLAIN_ADVCE`, `A7`-transfer, `SBF`, `TPFM`@S2T1 | direct train effect versus held-out effect, same run, same endpoint | both | direct +4.12 (Plain AdvCE dev-1), +4.409 / +2.210 (ST1W), +3.654 (ST2M L2), +1.684 (A7-transfer) versus held-out +0.220, +1.500 / +0.040, **-0.980**, +0.44 | many, 3 unrelated families | train two-run gap **0.134 pp** over 45,000 rows | — | n/a | **STANDS.** The direct effects are 13x to 33x the measured train two-run gap (DERIVED), while every held-out counterpart sits at or below the held-out floor. The gap reproduces in Stage A, the Confirmatory screen, the action-transfer screen and the long-horizon audit. **This is the project's most robust finding and it is a genuine negative result** | `MEASUREMENT_DESIGN.md:150-166, 701-706`; `ERT_CONFIRMATORY_T123_RESULTS.md:116-118`; `ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:34` |

### Group F — FF/NR causal routing (epoch-79 parent, pre-decay)

| # | question | canonical arm(s) and aliases | endpoint and horizon | regime | observed effect | seeds | floor `s` | target | Nbr? | classification | sources |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| F1 | Does routing on predicted future failure beat class/state/count-matched random routing? | `RA` vs `RAR`; `RB` vs `RBR` | validation-wide CE-PGD20, e84/89/94 | pre | RA minus RAR: -0.72 / +1.84 / +0.42 (L2), -0.44 / +0.26 / +0.44 (L4). RB minus RBR: -0.64 / +1.24 / +0.28 (L2), -1.10 / -0.84 / +1.74 (L4). All DERIVED | 2 | 1.14–1.25 | — | **no** for the CE coefficient; Route A and B differ in the KD weight (0.5 vs 1.0) **and** in cohort, so they are not a matched neighbour pair | **UNDERPOWERED** + **COEFFICIENT-UNTESTED.** MDE at k=2 is 2.25–2.47 pp; only one of twelve values exceeds it, and it reverses at the next horizon. The report's "Route A selected-minus-random is negative at every horizon" is a **selected-cohort** statement; validation-wide the sign is mixed, which the ledger does not record | `FFNR_CAUSAL_HORIZON_CE20_RESULTS.md:97-113`; `FFNR_CAUSAL_PILOT_RESULTS.md:35-36` |

### Group G — measurement and mechanism findings

These verdicts are not about accuracy differences. They are unaffected by the floor and stand on
their own terms. They are listed so the reader can see that the project's genuinely secure results
are concentrated here.

| # | question | canonical arm(s) | classification | note | sources |
| ---: | --- | --- | --- | --- | --- |
| G1 | What causes run-to-run divergence? | `REF1/2`, `ATTACK1/2`, `DATA1/2`, `BOTH1/2`; `SHUF1/2`, `AUG1/2` | **NOT-AN-EFFECT-CLAIM** | Stands. `REF2 - REF1 = 0.0000` exactly, so the pipeline is bitwise deterministic and every gap is caused by the deliberate RNG change. This is also the strongest single input to the floor table | `ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md:26, 34, 46-63`; `ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md:27, 35, 47-64` |
| G2 | How much do independent training seeds differ? | `BASE`, `CROPSHIFT`, `I100` x 5 seeds | **NOT-AN-EFFECT-CLAIM** | Stands. Per-run SD falls from 1.088 pp at e99 to 0.226 pp at e104 at the learning-rate decay. The single most useful measurement in the project | `ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:58-62`; `MEASUREMENT_DESIGN.md:307-312` |
| G3 | Is action utility heterogeneous by teacher margin? | read-only re-analysis of `C0`/`C10`/`C12`/`C13` | **NOT-AN-EFFECT-CLAIM** | Stands as descriptive. The ledger already says it "does not validate a margin threshold" — correct, and unchanged by this document | `ERT_CW_MARGIN_ACTION_MAP.md`; `ERT_RESEARCH_STATUS_SUMMARY.md:33` |
| G4 | Is the KL-PGD10 online proxy a safe stand-in for the CE-PGD20 endpoint? | read-only replay | **NOT-AN-EFFECT-CLAIM** | Stands. Pearson 0.9374, Spearman 0.9203, sign agreement 0.911 on L2 | `ERT_CLEAN_WRONG_RELIABILITY_PROXY_SAFETY.md` (KL10-vs-CE20 table) |
| G5 | Does attack random-start choice materially move an endpoint? | fixed-model attack-seed probes | **NOT-AN-EFFECT-CLAIM** | Stands as a characterization of the evaluation, not of a treatment | `ERT_RSLAD_ATTACK_RANDOMNESS_CHARACTERIZATION.md` |

### Class counts

| class | rows |
| --- | ---: |
| NOT-AN-EFFECT-CLAIM | 9 (B1, B5, B8, E3, G1–G5) |
| STANDS | 4 (A1, A2, D6, E8) |
| REFUTED | 1 (B3) |
| UNDERPOWERED | 23 (A3, B2, B4, B6, B7, C1–C6, D1–D5, E1, E2, E4–E7, F1) |
| UNRESOLVED | 1 (A4) |
| **total rows** | **38** |
| of which also flagged COEFFICIENT-UNTESTED | 13 (B3, B4, B6, C4, C6, D5, E1, E2, E4, E5, E6, E7, F1) |

Counting note: row A4 is split in its own cell (UNRESOLVED on the AUC gate that produced the
verdict, UNDERPOWERED on the robust side) and is counted once, as UNRESOLVED. Row B3 is counted as
REFUTED and is also flagged COEFFICIENT-UNTESTED.

---

## Part 3 — What this changes

**(a) Sentences in `docs/ERT_RESEARCH_STATUS_SUMMARY.md` no longer supported as written.**

| line | quoted sentence | why it fails | replacement wording |
| ---: | --- | --- | --- |
| 36 | "Effects are below the documented 1--2 pp RNG floor." / "First SUPPORTED arm-vs-control label" | The 1–2 pp figure is a **pre-decay** number quoted at a **post-decay** e114 endpoint, overstating the floor 3–4x; and `SUPPORTED` sits on an effect that is not significant even over images | "e114 held-out CE-PGD20 vs Control: PMP `+0.14/+0.20 pp`, DBDP `+0.14/+0.06 pp`. The post-decay floor for this design has never been measured; it is bracketed at 0.25–0.50 pp, needing 17–141 paired blocks. Two were run. Classification: **DIRECTIONAL, UNRESOLVED**, at a single untested coefficient." |
| 31 | "T1 weak AdvCE is positive on some selected training cohorts, while held-out effects are not durable" | "not durable" claims the held-out effect was measured and faded. At the e84 floor of 1.14 pp with one block per seed the MDE is 3.18 pp, so it was never resolvable | "T1 weak AdvCE produces a large, replicated direct effect on its selected training cohort (`+4.409/+2.210 pp`). Its held-out effect was never measurable at this design: 5 paired blocks are needed to see +1.5 pp and 40 to see +0.5 pp; one block per seed was run." |
| 32 | "reliability-gated variants fail to beat BASE consistently at epoch 94" | "fail to" implies a test was passed or failed. The e94 MDE at k=2 is 2.47 pp; the largest gate effect is 1.14 pp | "reliability-gated variants were screened at epoch 94 under a floor of 1.25 pp that exceeds every effect observed; the screen does not distinguish the gate from no gate in either direction." |

Boundary 58–59 ("descriptive two-development-seed evidence") needs a second clause: **the screens were also not powered for the effects they sought, so a negative reading is not evidence of absence.** Boundary 43–44 (direct rescue is no substitute for held-out accuracy) is the one line this analysis strengthens: it is row E8, and it STANDS.

**(b) How the thesis's negative result should now be phrased.** Not "we tested many sample-level interventions and none transferred" — that claims a test that did not happen. Not "our screens could not have detected what we sought" — true of most rows, but it discards four real results. The honest third phrasing:

> Two global augmentation results are real and replicated (`+1.612 pp` for CropShift over five seeds; `+0.864 pp`, `SD 0.247`, `t(4) = 7.82` for the epoch-100 switch). Against them, every sample-level intervention produced a large, reproducible effect on the samples it treated — 13x to 33x the train-split noise — and no held-out effect our screens could resolve. That contrast is the finding. At learning rate 0.1 a forked control differs from its sibling as much as a fresh seed does, so a five-epoch screen cannot see below about 2.5 pp while the interventions aimed at 0.1–2 pp. We therefore report a **measured ceiling on what this apparatus can detect**, not a refutation of sample-level intervention. One hypothesis was genuinely refuted at its own preregistered target; the rest remain open and untested.
>
> **Correction, 2026-09-07.** The `+0.864 pp` half of this sentence pools two comparators; see `docs/NUMERIC_CONSISTENCY_AUDIT.md` finding 2. The two real figures are `+1.12 pp` against no late change and `+0.69 pp` against a different late change, both positive in every seed.

**(c) Hypotheses genuinely still open, ranked by how cheaply a powered test could run.**

| rank | hypothesis | why it is open | cheapest adequate test | cost |
| ---: | --- | --- | --- | ---: |
| 1 | *(prerequisite)* the post-decay floor for a 14-epoch fork from an e99 parent | never measured; the 0.25–0.50 pp bracket moves the blocks needed for a 0.17 pp effect from 17 to 68 | 3 untreated control replicates per seed, e101–114, endpoints at e104/109/114 | **2.2 GPU-h** |
| 2 | TPFM on the Clean-Wrong cohort (D3) | six blocks give +0.603 pp, t(5) = 2.41 — the only pre-decay near-miss, and already densely swept in lambda | re-run post-decay (floor 3–4x smaller), 5 seeds, 2 controls per seed | ~1 day |
| 3 | pair-margin / boundary-distance at a **different** coefficient (E1, E2) | a clean negative would still bound one point; sampling bands are 0.0520–0.0609 and 29.16–38.99 | add one neighbour arm at each band edge to any future post-decay screen | +2 arms |
| 4 | whether a **smaller** AdvCE coefficient helps (C1) | 0.070959 is the smallest value ever trained for any coefficient except lambda | one arm at 0.5x, post-decay, 5 seeds | ~1 day |
| 5 | history-informed routing below +0.5 pp (B2, B3) | only "a gain as large as +0.5 pp" was refuted; +0.2 pp was never in reach | needs k >= 11–31 blocks; should wait on rank 1 | weeks |

---

## 4. What this analysis could not resolve

1. **The normalized trajectory-AUC statistic has no measured floor** (row A4). Two of the
   project's promotion gates were decided on it.
2. **Three of the four `STANDS` rows are post-decay**, where the floor itself is bracketed rather
   than pinned. If the true post-decay floor is at the 0.50 pp end, row A2's margin is 1.7x rather
   than 2.2x — still a result, but not a comfortable one.
3. **Row D6's train-cohort floor is scaled, not measured.** The 0.24 pp figure assumes the
   whole-train two-run gap scales as `1/sqrt(n)` to a subcohort. The effect is 12x–25x that figure,
   so the conclusion is robust to a large scaling error, but the figure itself is an assumption.
4. **The FF/NR and history-routing reports do not restate the 5,000-sample validation cardinality
   in their own text**, so the percentage-point-to-image conversion for rows B3 and F1 is assumed
   from the shared attack identity rather than checked locally.
5. **Row B2 uses AutoAttack, not CE-PGD20.** Its floor is measured within the study from the three
   random arms rather than imported from the floor table, which is the correct move but rests on
   three observations.
6. **No campaign in this project ever ran the identical control twice under two run ids.** Every
   type (a) floor estimate comes from a campaign that deliberately perturbed an RNG substream. Rank
   1 in Part 3(c) is the measurement that would fix this.

---

## 5. Sources

Audits used as the evidence base: `docs/MEASUREMENT_DESIGN.md`, `docs/COEFFICIENT_AUDIT.md`,
`docs/ARM_REGISTRY.md`, `docs/experiments/ard_arm_registry_v1.json`.

Reports and plans read directly for this document:
`docs/ERT_RESEARCH_STATUS_SUMMARY.md`, `docs/ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md`,
`docs/ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md`, `docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md`,
`docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`, `docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md`,
`docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`, `docs/ERT_RSLAD_STUDENT_HISTORY_PREDICTIVE_VALIDITY.md`,
`docs/EXPERIMENT_DASHBOARD.md`, `docs/HISTORY_ROUTING_V2_RESULTS.md`,
`docs/ERT_S3_HISTORY_PRODUCTION_RESULTS.md`, `docs/ERT_DYNAMIC_S3_RECOVERY_RESULTS.md`,
`docs/ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md`, `docs/ERT_RSLAD_ORDERING_MECHANISM_DISCOVERY.md`,
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md`, `docs/ERT_CONFIRMATORY_T123_RESULTS.md`,
`docs/ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md`, `docs/ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md`,
`docs/ERT_CW_MARGIN_GENERALIZATION_SCREEN.md`, `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md`,
`docs/ERT_CW_A7_CLEANCE_ABLATION.md`, `docs/ERT_CLEAN_WRONG_RELIABILITY_PROXY_SAFETY.md`,
`docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`,
`docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md`,
`docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md`,
`docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md`,
`docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md`,
`docs/FFNR_CAUSAL_HORIZON_CE20_RESULTS.md`,
`docs/plans/0015-history-replication-gate-and-handoff.md`,
`docs/plans/0020-best-oriented-history-routing-v2.md`,
`docs/plans/0038-ert-stage-a-effect-decomposition.md`,
`docs/plans/0072-ert-rslad-i100-action-transfer-screen.md`.

All held-out endpoints referenced here use the single registered CE-PGD20 attack identity
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`, except row B2, which uses
AutoAttack and is flagged accordingly.
