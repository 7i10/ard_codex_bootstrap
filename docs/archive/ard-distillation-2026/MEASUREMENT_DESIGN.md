# Measurement design: what a short continuation screen in this project can actually resolve

Status: read-only analysis. No training, no GPU, no checkpoint, no endpoint regeneration.
Every number below is marked **VERIFIED** (read directly from a file, with the file and line
given) or **DERIVED** (computed here from verified numbers, with the arithmetic shown).

---

## 0. Reader's guide

This document answers one question: **how big does a difference have to be before this
project's short continuation screens can tell it apart from nothing?**

A *continuation screen* here means: take a saved model checkpoint (the "parent"), make several
copies, train each copy for a few more epochs under a different rule ("arm"), and compare the
resulting accuracies. One arm is the *control*, which gets no new rule. The headline number is
usually `treatment accuracy - control accuracy` on a fixed 5,000-image held-out split, measured
by a fixed 20-step attack (CE-PGD20).

Three terms are used throughout and must not be mixed up:

| term | meaning |
| --- | --- |
| **horizon** | how many epochs of extra training the screen ran, and/or the absolute epoch number at which the endpoint was measured |
| **floor** | how far apart two runs land when *nothing* differs between them except randomness |
| **pp** | percentage point. On a 5,000-sample split, one sample = 0.02 pp |

### The one-paragraph answer

The floor for a short continuation screen in this project is **about 1.1-1.3 pp** (standard
deviation of the difference between two untreated runs) whenever training is at learning rate
0.1, which covers every screen this project has run from an epoch-79 parent to epochs 84/89/94.
That floor is essentially **flat in horizon**: it is already at 1.14 pp after 5 extra epochs and
only reaches 1.25 pp after 15. It does not shrink by training longer inside that window. It
shrinks by a factor of about three to four - to roughly **0.25-0.50 pp** - at the learning-rate
decay at epoch 100, and stays there. So the thing that stabilises a verdict is not horizon; it is
the learning-rate schedule. A two-arm, two-seed screen at epochs 84-94 can resolve differences of
about **2.5 pp and up**, and nothing smaller. The same design run past the decay can resolve
roughly **0.8 pp**, and about **0.5 pp** with five seeds - with real uncertainty on those figures,
because the exact post-decay floor for a short untreated fork has never been measured; it is
bracketed between 0.25 and 0.50 pp from two adjacent designs.

---

## 1. Definitions, and the arithmetic a reader needs

### 1.1 The held-out split and the sample quantum

The held-out split is a deterministic stratified 10% split: **5,000 samples, exactly 500 per
CIFAR-10 class, no ID overlap with the 45,000-sample train split**
(VERIFIED, `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:11-12`; the same 5,000/45,000 cardinality is
restated at `docs/ERT_DYNAMIC_S3_RECOVERY_RESULTS.md:39` and
`docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:16`).

Therefore:

- one sample changing from wrong to right = `1/5000 = 0.0002 = 0.02 pp` (DERIVED)
- a reported `+0.14 pp` effect is `0.14/0.02 = 7 net samples` (DERIVED)
- a reported `+1.500 pp` effect is `1.5/0.02 = 75 net samples` (DERIVED)

Any effect quoted to two decimal places in pp is quoted to the nearest single image.

### 1.2 Turning a set of observed gaps into a standard deviation

Most of this project's records report *absolute gaps* between two runs, not standard deviations.
For a normally distributed difference `D ~ N(0, s)`, the expected absolute value is
`E|D| = s * sqrt(2/pi) = 0.7979 * s`. So `s = mean|gap| / 0.7979` (DERIVED, standard identity).

If two independent runs each have per-run standard deviation `sigma`, the difference between
them has `s = sigma * sqrt(2)`. So `sigma = s / 1.4142` (DERIVED, standard identity).

### 1.3 How many runs you need

For a two-sided test at the 5% level with 80% power, the number of independent paired blocks is
`k = (1.96 + 0.84)^2 * s^2 / d^2 = 7.849 * s^2 / d^2`, where `d` is the effect you want to
detect and `s` is the floor standard deviation. Rearranged, the smallest effect `k` blocks can
detect is the **minimum detectable effect**, `MDE = 2.8 * s / sqrt(k)` (DERIVED, standard
formula). These are used throughout as a planning aid, not as a claim that any past screen ran a
formal test.

---

## 2. QUESTION 1 - the noise floor, and how it scales with horizon

### 2.1 Three different variances that this project has been conflating

The reports use "variance", "spread", "gap" and "stochasticity" for three quantities that are
not the same and do not have the same size.

| label | comparison | what is shared between the two runs | what differs |
| --- | --- | --- | --- |
| **(a) RNG replicate** | untreated run R1 vs untreated run R2, both forked from the same parent checkpoint | the entire training history up to the fork, the parent weights, the config, the schedule | only the random number stream after the fork (data order, augmentation draws, attack random starts) |
| **(b) training seed** | seed A vs seed B, trained from scratch | nothing but the recipe | everything random, from epoch 0 |
| **(c) paired effect** | `(treatment - control)` measured once, then measured again in a second replicate of the whole pair | within one pair: the parent and the prefix. Across pairs: nothing random | the treatment is applied in one arm; across pairs, the RNG stream |

Type (a) is what a screen's control arm actually is. Type (b) is what the five-seed study
measures. Type (c) is the quantity a screen's *conclusion* depends on, and it is the one the
project has almost never measured, because almost every campaign ran exactly **one** control.

Reading a type (b) number and applying it to a type (a) design, or reading a type (a) number and
assuming type (c) is smaller because of pairing, are both mistakes that appear in the project's
current documents. Section 2.5 says which one to use.

### 2.2 Confirming the primary published measurement

`docs/archive/ard-distillation-2026/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md` continued four blocks (L2-R1, L2-R2, L4-R1,
L4-R2) from epoch-79 parents to epochs 84/89/94, with arms N95/A100/N105 plus an untreated
`B0_BASE` (VERIFIED, `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:18`). Its "BASE gap" column
is exactly a type (a) control-versus-control difference.

| teacher | epoch | BASE gap, held-out robust (pp) | status |
| --- | ---: | ---: | --- |
| L2 | 84 | 0.840 | VERIFIED, `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:29` |
| L2 | 89 | 0.820 | VERIFIED, line 30 |
| L2 | 94 | 1.880 | VERIFIED, line 31 |
| L4 | 84 | 0.160 | VERIFIED, line 32 |
| L4 | 89 | 0.280 | VERIFIED, line 33 |
| L4 | 94 | 1.820 | VERIFIED, line 34 |

The Markdown table is confirmed against the machine artifact
`docs/experiments/ert_cw_margin_rng_stability_diagnostic_v1.json`, key
`base_and_treatment_endpoint_variance[teacher][epoch].robust_accuracy.base_gap`
(VERIFIED: 0.0084, 0.0082, 0.0188, 0.0016, 0.0028, 0.0182 as fractions).

### 2.3 What the machine artifact contains that the Markdown omits

**Clean accuracy is present in the artifact and is absent from the report.** Same key path,
`clean_accuracy.base_gap`:

| teacher | epoch | BASE gap, held-out **clean** (pp) | status |
| --- | ---: | ---: | --- |
| L2 | 84 | 0.720 | VERIFIED, artifact `L2.84.clean_accuracy.base_gap = 0.0072` |
| L2 | 89 | 0.580 | VERIFIED, `0.0058` |
| L2 | 94 | 0.500 | VERIFIED, `0.0050` |
| L4 | 84 | 0.940 | VERIFIED, `0.0094` |
| L4 | 89 | 0.220 | VERIFIED, `0.0022` |
| L4 | 94 | 0.480 | VERIFIED, `0.0048` |

Pooled mean of the six clean BASE gaps = `(0.720+0.580+0.500+0.940+0.220+0.480)/6 = 3.440/6 =
0.573 pp` (DERIVED). Pooled mean of the six robust BASE gaps = `(0.840+0.820+1.880+0.160+0.280+
1.820)/6 = 5.800/6 = 0.967 pp` (DERIVED). Clean accuracy is quieter than robust accuracy by
about 40%, but it is not quiet.

**A full epoch-80-to-94 R1/R2 gap curve is present**, under
`training_metric_curves.r1_r2_absolute_gaps`. The Markdown mentions it exists
(`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:38`) but prints none of it. For the untreated
`B0_BASE` arm, mean absolute R1-vs-R2 gap over epochs 80-94 (VERIFIED from artifact, means
DERIVED):

| logged metric | split size | L2 mean gap | L4 mean gap |
| --- | ---: | ---: | ---: |
| `learning_rate` | - | 0.0000 | 0.0000 |
| `train_loss` | 45,000 | 0.0005 (absolute loss units) | 0.0005 |
| `train_clean_accuracy` | 45,000 | 0.085 pp | 0.104 pp |
| `train_robust_accuracy` | 45,000 | 0.134 pp | 0.134 pp |
| `val_clean_accuracy` | 5,000 | 0.879 pp | 0.868 pp |
| `val_pgd_accuracy` | 5,000 | 0.760 pp | 0.664 pp |

Two things follow. First, the learning rate is bit-identical between replicates and constant at
**0.1** across epochs 80-94 (VERIFIED, artifact `training_metric_curves.absolute['L2-R1'].B0_BASE
.learning_rate` = 0.1 for every epoch 80-94). This matters for Question 2. Second, the
**training-set** metrics are roughly six times quieter than the held-out metrics
(`0.760 / 0.134 = 5.7`, DERIVED); part of that is the 3x from sample size
(`sqrt(45000/5000) = 3`, DERIVED) and the rest is that the model is fitted to the train split.
This is why "direct rescue on the selected training cohort" always looks convincing and
routinely fails to transfer.

### 2.4 Every other control-versus-control measurement in the project

The brief mentioned the RNG-stability diagnostic as the only direct measurement. It is not. Two
further campaigns each ran **seven untreated continuations per teacher** from the same epoch-79
parents, and both report the full epoch-84/89/94 held-out CE-PGD20 trajectory. Neither is
labelled as a noise-floor study, but that is exactly what they are: every arm is an untreated
continuation whose only difference from the reference is which RNG stream was perturbed.

| campaign | file | untreated arms per teacher |
| --- | --- | --- |
| RNG-source decomposition | `docs/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md:46-63` | REF1, REF2, ATTACK1, ATTACK2, DATA1, DATA2, BOTH1, BOTH2 |
| shuffle-vs-augmentation | `docs/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md:47-64` | REF1, REF2, SHUF1, SHUF2, AUG1, AUG2, BOTH1, BOTH2 |

In both, `REF2 - REF1 = 0.0000` exactly (VERIFIED,
`docs/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md:26` and `:34`;
`docs/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md:27` and `:35`). The pipeline is bitwise
deterministic given identical seeds, so every gap below is caused by the deliberate RNG change
and nothing else. REF2 is therefore dropped as a duplicate, leaving **7 distinct untreated runs
per teacher per campaign = 28 untreated runs in total**, and `2 campaigns x 2 teachers x
C(7,2) = 2 x 2 x 21 = 84` untreated pairwise comparisons at each horizon (DERIVED).

Pooling all 84 pairs at each horizon (DERIVED from the two verified trajectory tables):

| endpoint epoch | horizon from e79 parent | n pairs | mean abs gap (pp) | median | 90th pct | max | SD of a two-run difference (pp) | implied per-run SD (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 84 | +5 | 84 | 0.907 | 0.700 | 1.940 | 3.300 | **1.136** | 0.804 |
| 89 | +10 | 84 | 0.959 | 0.780 | 2.180 | 2.460 | **1.202** | 0.850 |
| 94 | +15 | 84 | 0.998 | 0.770 | 2.200 | 2.980 | **1.250** | 0.884 |

Arithmetic for the epoch-84 row: `mean = 0.907`; `SD = 0.907 / 0.7979 = 1.136`;
`per-run SD = 1.136 / 1.4142 = 0.804` (DERIVED).

This agrees closely with the RNG-stability diagnostic's six BASE gaps: `0.967 / 0.7979 = 1.212 pp`
(DERIVED). Two independent campaign families, built at different source SHAs, give a two-run
difference standard deviation of **1.14 to 1.25 pp**.

Clean accuracy over the same 84 pairs at epoch 94: mean absolute gap `1.114 pp`, implying
`1.114 / 0.7979 = 1.397 pp` (DERIVED, from `docs/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md:26-40`
and `docs/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md:26-41`). The endpoint clean floor is not
smaller than the robust floor once measured across many runs; the smaller figure in section 2.3
came from only six comparisons.

### 2.4b Three further families, using placebo arms instead of duplicated controls

Three more campaigns give control-versus-control-like numbers without ever labelling them that
way, by running a *placebo* arm: an arm that applies the treatment to a **class/state/count-matched
random** cohort instead of the selected one. If routing on random samples does nothing - which is
the premise the design rests on - then control-versus-placebo and placebo-versus-placebo are both
floor estimates. Placebo arms carry slightly more variance than a duplicated control, because the
placebo action really is applied, so these figures are upper bounds.

**FF/NR causal horizon, epoch-79 parent, horizons 84/89/94** (VERIFIED,
`docs/FFNR_CAUSAL_HORIZON_CE20_RESULTS.md:99-104`; the table is stated to be "the full validation
aggregate, not a selected-cohort estimate" at line 95). Within each seed, the three pairwise
absolute gaps among `C79` (control), `RAR` (Route-A matched random) and `RBR` (Route-B matched
random):

| horizon | n gaps | mean abs gap (pp) | implied SD (pp) |
| ---: | ---: | ---: | ---: |
| 84 | 6 | 0.907 | 1.136 |
| 89 | 6 | 1.287 | 1.613 |
| 94 | 6 | 0.267 | 0.334 |
| pooled | 18 | 0.820 | **1.028** |

All DERIVED. Restricting to the cleanest quantity - `RAR` versus `RBR`, two independent random
draws with no shared selection - gives gaps `1.26 / 1.80 / 0.04 / 0.30 / 0.94 / 0.66 pp`, mean
`0.833`, implied SD `1.044 pp` (DERIVED).

**FF/NR causal pilot, epoch 83** (VERIFIED, `docs/FFNR_CAUSAL_PILOT_RESULTS.md:35-36`; described as
"validation clean/PGD-20 accuracy at epoch 83" at lines 30-31). The same three pairwise gaps per
seed on PGD-20 accuracy: `1.56 / 0.26 / 1.82 / 0.86 / 1.04 / 0.18 pp`, mean `0.953`, implied SD
`1.195 pp` (DERIVED).

Both families independently reproduce the 1.0-1.25 pp pre-decay floor of section 2.4, from a
different teacher pipeline and a different intervention. Neither report restates the held-out split
size in-document, so the 5,000-sample convention is assumed from the shared endpoint attack
identity rather than verified locally.

**History routing v2 - the only post-decay paired-fork measurement in the project.** This campaign
forks from an epoch-39 parent, intervenes over epochs 40-199, and uses LR milestones `[120, 170]`
(VERIFIED, `docs/HISTORY_ROUTING_V2_RESULTS.md:19-21`), so its epoch-199 endpoint is well past the
final decay. `R` is defined as "class/state/count-matched random" (VERIFIED, line 28). Within each
seed, the three pairwise gaps among `C`, `PF-R` and `NR-R` (VERIFIED,
`docs/HISTORY_ROUTING_V2_RESULTS.md:46-47, 50-51, 54-55`):

| metric | six within-seed gaps (pp) | mean | implied SD (pp) |
| --- | --- | ---: | ---: |
| Last PGD (epoch 199) | 0.16 / 0.48 / 0.64 / 0.56 / 0.04 / 0.52 | 0.400 | **0.501** |
| Best PGD (best epoch) | 0.32 / 0.28 / 0.60 / 0.42 / 0.36 / 0.06 | 0.340 | 0.426 |
| Last clean | 0.48 / 0.52 / 0.04 / 0.10 / 0.28 / 0.18 | 0.267 | 0.334 |

All DERIVED. `Best PGD` is a maximum taken over ~200 epochs and so is not comparable to a
fixed-horizon endpoint; the `Last PGD` row is the one to use. The report itself flags the same
instability in prose - the `PF-R` seed-1 value is the highest of any arm and is explicitly to be
treated as "a one-off value showing random-mask variation", not as a basis for adoption (VERIFIED,
`docs/HISTORY_ROUTING_V2_RESULTS.md:69`).

**Three fresh BASE seeds at epoch 199** (VERIFIED, `docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:46-48`,
on the "fixed internal validation 5,000 samples" stated at line 27 with split identity
`16ec66...0a3cc6` at line 31): `58.22 / 58.58 / 58.18%`. Pairwise gaps `0.36 / 0.04 / 0.40 pp`,
SD across the three values `0.220 pp`, implied two-run difference SD `0.312 pp` (DERIVED). These
are the same three confirmation seeds that appear in section 2.5, reported independently, and they
agree to the second decimal - a useful cross-check that the cache CSV used there matches the
committed report.

### 2.5 Type (b): between training seeds

`docs/archive/ard-distillation-2026/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md` compares five independently seeded runs
(`dev-1`, `dev-2`, `confirm-a/b/c`) of BASE, CROPSHIFT and I100 at epochs 49/99/149/199. Its
headline table gives *training-log* standard deviations at epoch 199: BASE `0.272 pp`,
CROPSHIFT `0.532 pp`, I100 `0.118 pp` (VERIFIED,
`docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:58-60`), plus a peak BASE SD of `2.907 pp` at epoch 97
(VERIFIED, line 62).

The *endpoint* per-seed values - measured on the same 5,000-row held-out split with the same
CE-PGD20 attack used by every screen, and therefore directly comparable to section 2.4 - are in
the campaign's own cache, `.cache/analysis/ert-rslad-five-seed-stochasticity-v1/outputs/
rescue_summary.csv` (VERIFIED, read). Per-seed values and their spread:

| arm | epoch | per-seed held-out robust (%) | per-run SD (pp) | SD of a two-run difference (pp) |
| --- | ---: | --- | ---: | ---: |
| BASE | 49 | 46.54 / 45.34 / 44.46 / 46.42 / 45.96 | 0.859 | 1.214 |
| BASE | 99 | 46.26 / 46.14 / 45.52 / 47.60 / 45.54 | 0.846 | 1.197 |
| BASE | 149 | 56.84 / 56.20 / 56.26 / 56.58 / 56.40 | 0.260 | 0.367 |
| BASE | 199 | 58.22 / 58.58 / 58.18 / 58.08 / 58.10 | 0.203 | 0.287 |
| CROPSHIFT | 49 | 47.14 / 46.34 / 44.22 / 46.80 / 47.52 | 1.296 | 1.833 |
| CROPSHIFT | 99 | 45.62 / 49.00 / 47.58 / 49.16 / 45.60 | 1.739 | 2.459 |
| CROPSHIFT | 149 | 57.40 / 57.08 / 57.06 / 57.06 / 57.62 | 0.255 | 0.361 |
| CROPSHIFT | 199 | 60.40 / 59.88 / 60.28 / 59.40 / 59.26 | 0.510 | 0.721 |
| I100 | 149 | 57.82 / 57.18 / 57.26 / 57.88 / 57.66 | 0.322 | 0.455 |
| I100 | 199 | 61.18 / 60.56 / 60.90 / 60.60 / 60.30 | 0.339 | 0.479 |

Per-seed values VERIFIED from the CSV; SDs and `SD x sqrt(2)` DERIVED. Seed order is
`confirm-a / confirm-b / confirm-c / dev-1 / dev-2`. The arm means reproduce the published
sample-level table (BASE e199 mean `58.232%`, I100 e199 `60.708%`, CROPSHIFT e199 `59.844%`,
VERIFIED, `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:86-88`), so the CSV and the report agree.
I100 and CROPSHIFT share a prefix through epoch 99 and are therefore identical at epochs 49/99
by construction (VERIFIED, `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:34`).

The training-log SD curve in the same campaign's artifact
(`docs/experiments/ert_rslad_five_seed_global_stochasticity_v1.json`, key `by_epoch`) shows the
same shape at one-epoch resolution (VERIFIED): BASE per-run SD is `0.99` at epoch 84, `0.72` at
89, `1.359` at 94, `1.088` at 99, then `0.467` at 100, `0.274` at 102, `0.226` at 104, `0.272`
at 114, `0.281` at 149, `0.272` at 199. The mean robust accuracy jumps from `46.888%` at epoch
99 to `53.348%` at epoch 100 (VERIFIED), which is the learning-rate decay.

### 2.6 Type (c): the paired treatment-minus-control effect, replicated

This is the one that matters for a verdict, and there are exactly two places in the project where
a *whole treatment-versus-control pair* was run twice.

**Pre-decay.** The RNG-stability diagnostic's "effect gap" columns are `|(treat - base)_R1 -
(treat - base)_R2|` (VERIFIED as a definition,
`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:11`; the same values reappear as the "Replicate
variance" table at `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:128-141`).

| epoch | mean of 6 paired-effect gaps (pp) | implied SD (pp) | mean of 2 BASE gaps at same epoch (pp) |
| ---: | ---: | ---: | ---: |
| 84 | 0.970 | 1.216 | 0.500 |
| 89 | 1.090 | 1.366 | 0.550 |
| 94 | 1.453 | 1.821 | 1.850 |

Values from `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:29-34`, means DERIVED. **Pairing
does not reduce the variance here.** The replicated paired effect is at least as noisy as the
raw control-versus-control gap. The report states this in prose: "Treatment-effect gaps span
0.02-2.06 pp and are not uniformly larger than BASE gaps" (VERIFIED, line 103). The correct
reading is that at learning rate 0.1 the control arm and the treatment arm decorrelate almost
completely within five epochs, so subtracting the control removes essentially none of the noise.

**Post-decay.** The five-seed study gives the only replicated paired effect after the decay:
`I100 - CROPSHIFT` at epoch 199, computed within each seed, where the two arms share the same
prefix through epoch 99 and differ only in the intervention from epoch 100.

| seed | I100 (%) | CROPSHIFT (%) | difference (pp) |
| --- | ---: | ---: | ---: |
| confirm-a | 61.18 | 60.40 | +0.78 |
| confirm-b | 60.56 | 59.88 | +0.68 |
| confirm-c | 60.90 | 60.28 | +0.62 |
| dev-1 | 60.60 | 59.40 | +1.20 |
| dev-2 | 60.30 | 59.26 | +1.04 |

> **Correction, 2026-09-07.** The `0.25 pp` type (c) value below is the standard deviation
> of the five-seed `I100 - CROPSHIFT` set, and that set mixes two comparators: three of its
> five members are measured against `CROP_SUFFIX`, not against `CROPSHIFT`. Within each group
> the spread is `0.11 pp` (dev) and `0.08 pp` (confirm), so **the `0.25 pp` figure is inflated
> by the difference between the two comparators rather than being a noise floor**. It was the
> lower end of the bracket every post-decay verdict used until 2026-09-06, when the floor was
> measured directly at `0.092 pp` (`docs/POST_DECAY_FLOOR.md`). Both readings now point the
> same way: the post-decay floor is smaller than this section assumed.
> See `docs/archive/ard-distillation-2026/NUMERIC_CONSISTENCY_AUDIT.md` finding 2.

Per-seed values VERIFIED from the campaign CSV; differences DERIVED. Mean `+0.864 pp` (which
reproduces the published `+0.864 pp`, VERIFIED,
`docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:13`), **standard deviation across seeds `0.247 pp`**,
standard error `0.247/sqrt(5) = 0.111 pp`, `t(4) = 7.82`, 95% interval over training runs
`[+0.557, +1.171]` (all DERIVED).

Here pairing *does* work: `0.247 pp` is well below the `0.287-0.721 pp` unpaired two-run
difference SD at epoch 199 from section 2.5. That `0.247 pp` figure rests on five observations
(4 degrees of freedom), so its own 95% interval is wide - `[0.148, 0.710] pp` (DERIVED from the
chi-square interval `s*sqrt(df/chi2)` with `chi2(0.975,4) = 11.143` and
`chi2(0.025,4) = 0.484`).

### 2.7 The floor table

All figures are the standard deviation of the difference between two runs on the 5,000-sample
held-out split under CE-PGD20, in percentage points. This is the quantity to compare an effect
against.

| comparison type | training regime | horizon | n underlying comparisons | floor SD (pp) | status |
| --- | --- | --- | ---: | ---: | --- |
| (a) RNG replicate, untreated | LR 0.1, e79 parent | +5 epochs (e84) | 84 | **1.14** | DERIVED from `docs/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md:46-63` + `docs/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md:47-64` |
| (a) RNG replicate, untreated | LR 0.1, e79 parent | +10 epochs (e89) | 84 | **1.20** | DERIVED, same sources |
| (a) RNG replicate, untreated | LR 0.1, e79 parent | +15 epochs (e94) | 84 | **1.25** | DERIVED, same sources |
| (a) RNG replicate, untreated | LR 0.1, e79 parent | e84/89/94 pooled | 6 | 1.21 | DERIVED from `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:29-34` |
| (a) RNG replicate, untreated, **clean** metric | LR 0.1, e79 parent | +15 epochs (e94) | 84 | 1.40 | DERIVED, same sources |
| (a') placebo fork, pre-decay | LR 0.1, e79 parent | e84/89/94 | 18 | 1.03 | DERIVED from `docs/FFNR_CAUSAL_HORIZON_CE20_RESULTS.md:99-104` |
| (a') placebo fork, pre-decay | LR 0.1, e79 parent | e83 | 6 | 1.20 | DERIVED from `docs/FFNR_CAUSAL_PILOT_RESULTS.md:35-36` |
| (a') placebo fork, **post-decay** | e39 parent, milestones [120,170] | e199 | 6 | **0.50** | DERIVED from `docs/HISTORY_ROUTING_V2_RESULTS.md:46-55` |
| (a) RNG replicate, untreated, **post-decay** | LR after e100 decay | any | **0** | **never measured** | closest available is the (a') row above; see section 5 |
| (b) training seed, BASE | LR 0.1 | e49 | 10 | 1.21 | DERIVED from five-seed cache CSV |
| (b) training seed, BASE | LR 0.1 | e99 | 10 | 1.20 | DERIVED, same |
| (b) training seed, BASE | post-decay | e149 | 10 | **0.37** | DERIVED, same |
| (b) training seed, BASE | post-decay | e199 | 10 | **0.29** | DERIVED, same |
| (b) training seed, BASE, 3 fresh seeds | post-decay | e199 | 3 | 0.31 | DERIVED from `docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:46-48` |
| (b) training seed, CROPSHIFT | post-decay | e199 | 10 | 0.72 | DERIVED, same |
| (b) training seed, I100 | post-decay | e199 | 10 | 0.48 | DERIVED, same |
| (c) paired effect, replicated | LR 0.1, e79 parent | e84 | 6 | 1.22 | DERIVED from `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:29-34` |
| (c) paired effect, replicated | LR 0.1, e79 parent | e89 | 6 | 1.37 | DERIVED, same |
| (c) paired effect, replicated | LR 0.1, e79 parent | e94 | 6 | 1.82 | DERIVED, same |
| (c) paired effect, replicated | post-decay, shared e99 prefix | e199 | 5 seeds | **0.25** (95% CI 0.15-0.71) | DERIVED from five-seed cache CSV |

### 2.8 How the floor scales - the answer

**Five independent campaign families agree on the pre-decay number.** Untreated RNG forks give
1.14-1.25 pp (84 pairs); the RNG-stability BASE replicates give 1.21 pp (6 pairs); the FF/NR
causal-horizon placebo arms give 1.03 pp (18 pairs); the FF/NR pilot placebo arms give 1.20 pp
(6 pairs); and five independent training seeds give 1.20-1.21 pp at epochs 49 and 99. Different
teachers, different interventions, different source SHAs, one answer: **about 1.1-1.2 pp**.

**Within a short screen, it does not scale with horizon in any useful way.**
From 5 to 15 extra epochs the floor grows from 1.14 to 1.25 pp, a rise of 10% over a tripling of
horizon (DERIVED). Two untreated runs sharing a 79-epoch prefix reach 90% of their eventual
separation within the first five epochs.

**It is already as large as the between-seed floor.** At epochs 84-94 the RNG-replicate floor is
1.14-1.25 pp; at epochs 49-99 the between-seed floor for BASE is 1.20-1.21 pp (DERIVED, section
2.7). Ratio `1.14/1.21 = 0.94` (DERIVED). Sharing a 79-epoch training prefix removes roughly 6%
of the variance of two completely independent seeds. For practical purposes, at learning rate
0.1, **a forked control is as different from its sibling as a fresh seed is**.

**The floor is governed by the learning rate, not the horizon.** At the epoch-100 decay the
per-run training-log SD drops from `1.088` (e99) to `0.467` (e100) to `0.226` (e104) and stays
in the 0.1-0.4 band for the remaining 100 epochs (VERIFIED, five-seed `by_epoch` artifact). On
held-out endpoints the two-run difference SD drops from `1.20` at e99 to `0.37` at e149 and
`0.29` at e199 - a factor of `1.20/0.29 = 4.1` (DERIVED).

**Pairing only helps after the decay.** Pre-decay, the replicated paired effect SD (1.22-1.82 pp)
is no smaller than the raw control gap (1.14-1.25 pp). Post-decay, the paired effect SD
(0.25 pp) is roughly 0.7x-1.2x the unpaired figure and clearly below the unpaired CROPSHIFT
figure. Pairing buys you something only once the trajectories stop diverging chaotically.

### 2.9 Which floor is the right reference for a short screen

**Use type (c), the replicated paired treatment-versus-control effect, at the regime and horizon
of the screen.** That is literally the sampling distribution of the number a screen reports.

Where type (c) is unavailable, **use type (a), the untreated fork-versus-fork floor**, and do not
substitute type (b). The reasons:

1. Type (b) answers a different question ("would a different seed have given the same result?"),
   which is a *stronger* claim than a screen makes, and is estimated from runs that share nothing
   with the screen's parent.
2. Type (a) is measured on exactly the object the screen's control arm is, so it needs no
   assumption about how much a shared prefix helps.
3. Empirically (section 2.8) type (a) and type (b) coincide at LR 0.1 anyway, so nothing is lost
   pre-decay. Post-decay they are known to diverge - type (c) is `0.25 pp` while type (b) is
   `0.29-0.72 pp` - which is exactly the regime where using type (b) would be too pessimistic and
   type (a) has only one near-measurement - the history-routing-v2 placebo forks at `0.50 pp`
   (section 2.4b). Taken together the post-decay paired-fork floor is bracketed at
   **0.25 to 0.50 pp**, and `0.40 pp` is a reasonable working value.

Two things the project currently does that this rules out:

- Quoting "RNG alone moves e114 held-out by 1-2 pp"
  (VERIFIED, `docs/plans/0087-i100-online-state-s2-preservation.md:174-176`, citing
  `docs/ERT_RESEARCH_STATUS_SUMMARY.md:34`). The 1-2 pp figure is a **pre-decay, epoch-84-to-94**
  measurement; applying it at epoch 114 overstates the floor by roughly 3-4x. The conclusion it
  supports happens to survive, but the number is the wrong number.
- Treating a sample bootstrap interval as if it constrained training-run variation. Every report
  that carries one says so in its own text
  (VERIFIED, `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:29-31`;
  `docs/ERT_CONFIRMATORY_T123_RESULTS.md:48-50`), and the caveat should be honoured downstream.

---

## 3. QUESTION 2 - where does a verdict stabilise?

### 3.1 Method

For each campaign that evaluated the same arms at more than one horizon, the treatment-minus-
control difference on the full held-out split is tabulated at each horizon, and the absolute
change between consecutive horizons is computed. If verdicts stabilise, that movement should
shrink as training proceeds.

### 3.2 The long series - CW held-out generalization gap

This is the only family with five endpoints, so it carries the most weight. `V-overall` is the
treatment-minus-control difference on all 5,000 held-out samples (VERIFIED as a definition,
`docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:30`, and the weighted identity that ties it
to the subgroup columns is asserted at lines 55-61).

| seed | action | e129 | e149 | e169 | e189 | e199 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | Plain AdvCE | +0.220 | -0.160 | -0.220 | +0.020 | -0.300 |
| dev-1 | TPFM | +0.320 | +0.380 | +0.260 | +0.580 | +0.040 |
| dev-2 | Plain AdvCE | -0.280 | +0.580 | +0.020 | -0.240 | +0.000 |
| dev-2 | TPFM | +0.200 | +0.880 | -0.040 | +0.260 | +0.200 |

All twenty values VERIFIED, `docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:34-53`.

Movement between consecutive horizons (DERIVED):

| step | epochs spanned | mean abs movement (pp) | per epoch (pp) |
| --- | ---: | ---: | ---: |
| e129 -> e149 | 20 | 0.495 | 0.025 |
| e149 -> e169 | 20 | 0.415 | 0.021 |
| e169 -> e189 | 20 | 0.280 | 0.014 |
| e189 -> e199 | 10 | 0.290 | 0.029 |

Example arithmetic for the first row: the four movements are `|-0.160-0.220| = 0.380`,
`|0.380-0.320| = 0.060`, `|0.580-(-0.280)| = 0.860`, `|0.880-0.200| = 0.680`; mean `= 1.980/4 =
0.495` (DERIVED).

Two observations. The movement declines only slightly and non-monotonically, and the final step
is the largest per epoch. **The sign flips between consecutive horizons in 7 of 16 opportunities**
(DERIVED) - very close to a coin toss. The 20 effect values have mean `+0.136 pp` and standard
deviation `0.316 pp` (DERIVED), which is the size of the post-decay floor. The honest reading is
not "the effect settles"; it is "there is no effect here larger than the floor, and what you are
watching move is noise."

### 3.3 The short campaigns at epochs 84/89/94

| campaign | arms x blocks | mean abs movement e84->e89 (pp) | e89->e94 (pp) | source |
| --- | --- | ---: | ---: | --- |
| CW-margin local lambda stability | 5 arms x 4 blocks = 20 | **1.009** | **1.224** | VERIFIED, `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`; movements DERIVED |
| confirmatory T123 | 3 arms x 2 seeds = 6 | 1.230 | 0.810 | VERIFIED, `docs/ERT_CONFIRMATORY_T123_RESULTS.md:79-96`; DERIVED |
| dynamic S3 recovery | 2 arms x 2 seeds = 4 | 1.170 | 1.200 | VERIFIED, `docs/ERT_DYNAMIC_S3_RECOVERY_RESULTS.md:49-54`; DERIVED |

The lambda-stability campaign is decisive because it has four matched blocks (L2-R1, L2-R2,
L4-R1, L4-R2), i.e. every arm was run twice per teacher. Across its 20 arm-blocks, the sign of
the effect flips between consecutive horizons **23 times out of 40** (DERIVED) - more often than
chance. Worse, the two RNG replicates of the *same arm at the same epoch* agree on sign only:

| epoch | replicate sign agreement | status |
| ---: | ---: | --- |
| 84 | 1 of 10 | DERIVED from `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36` |
| 89 | 5 of 10 | DERIVED, same |
| 94 | 2 of 10 | DERIVED, same |

A verdict from a single block at epochs 84-94 does not reproduce on a second block of the same
experiment. No horizon inside that window fixes this.

The T123 campaign says the same thing in absolute terms: its common control `C79CONF` moves
`47.26 -> 46.04 -> 46.20` on L2 and `46.12 -> 47.94 -> 47.58` on L4 across the three horizons
(VERIFIED, `docs/ERT_CONFIRMATORY_T123_RESULTS.md:59-64`). A control that swings 1.8 pp between
its own endpoints cannot anchor a sub-1-pp treatment claim.

Stage A, which ran only at epoch 84, provides a further clue. Across its 12 treatment arms, the
held-out robust effects have mean `-0.247 pp` and SD `1.266 pp` on seed L2, but mean `+1.010 pp`
and SD `0.889 pp` on seed L4, with **11 of 12 L4 arms positive** (DERIVED from
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52-63` and `:69-80`). Twelve chemically different
treatments do not all help one seed by about a point. What that pattern looks like is a single
control run (`C79` for L4) that happened to land low, shifting every arm that is compared against
it. That is the failure mode a control replicate exists to catch, and Stage A had none.

### 3.4 The post-decay campaigns at epochs 104/109/114

Three campaigns forked the same two epoch-99 I100 parents and ran to epoch 114. They report six
distinct treatment arms.

| seed | arm | e104 | e109 | e114 | source |
| --- | --- | ---: | ---: | ---: | --- |
| dev-1 | PMP | -0.02 | +0.04 | +0.14 | VERIFIED, `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:39-47` |
| dev-1 | DBDP | -0.04 | +0.02 | +0.14 | VERIFIED, same |
| dev-2 | PMP | +0.00 | +0.14 | +0.20 | VERIFIED, `:48-56` |
| dev-2 | DBDP | +0.06 | +0.16 | +0.06 | VERIFIED, same |
| dev-1 | DPM | -0.04 | -0.10 | +0.08 | VERIFIED, `docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:13-21` |
| dev-1 | D-BDD | +0.02 | -0.16 | +0.04 | VERIFIED, same |
| dev-2 | DPM | +0.00 | -0.18 | +0.12 | VERIFIED, `:22-30` |
| dev-2 | D-BDD | -0.02 | -0.12 | +0.20 | VERIFIED, same |
| dev-1 | SBF | -0.08 | -0.06 | +0.16 | DERIVED from `docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:25-27` |
| dev-1 | TPFM | -0.02 | -0.26 | +0.22 | DERIVED, same |
| dev-2 | SBF | +0.08 | +0.04 | +0.04 | DERIVED from `:28-30` |
| dev-2 | TPFM | +0.00 | -0.04 | +0.04 | DERIVED, same |

Movement between horizons: `e104 -> e109` mean `0.102 pp`; `e109 -> e114` mean `0.180 pp`
(DERIVED). Per epoch that is `0.020` and `0.036 pp`, about **one tenth** of the `0.16-0.25 pp per
epoch` seen at epochs 84-94 (DERIVED). This is the regime change, visible directly in the screens
themselves.

But the same table contains a warning. At epoch 104, 3 of 12 arm-seed values are positive; at
epoch 109, 5 of 12; **at epoch 114, 12 of 12**, with mean `+0.120 pp` (DERIVED). Six unrelated
mechanisms did not all start working at epoch 114. All twelve values are computed against the
**same two control runs** - dev-1 e114 CONTROL robust is `57.32%` in all three reports (VERIFIED,
`docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:45`,
`docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:19`,
`docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:27`). Twelve results resting on
two control endpoints are not twelve pieces of evidence. If those two controls sit `0.12 pp` low
- well within any plausible floor - the entire pattern disappears.

### 3.5 The answer to Question 2

**A verdict does not stabilise as a function of horizon. It stabilises at the final
learning-rate decay, and the decay is a scheduler event, not a horizon.**

The evidence:

1. Inside the pre-decay window (LR 0.1) nothing improves with horizon. The untreated floor is
   flat (1.14 -> 1.25 pp over 5 -> 15 epochs). The between-horizon movement of a paired effect is
   1.0-1.2 pp at every step. Replicate sign agreement is 1/10, 5/10, 2/10 at epochs 84, 89, 94.
   Training longer inside this window - to epoch 99 - does not help either: the between-seed
   floor is `1.20 pp` at epoch 99, statistically identical to `1.21 pp` at epoch 49.
2. At the decay the floor falls by a factor of about four, within one to four epochs
   (per-run SD `1.088` at e99 -> `0.467` at e100 -> `0.226` at e104, VERIFIED).
3. After the decay the floor is stable for a hundred epochs (`0.37 pp` at e149, `0.29 pp` at
   e199) and the between-horizon movement of a paired effect is 0.10-0.18 pp over five epochs.

**Defensible estimate of the earliest trustworthy horizon: four to five epochs past the final
learning-rate decay - epoch 104 in the I100 200-epoch schedule.** By epoch 104 the per-run
training-log SD has fallen to `0.226 pp` and the observed effect movement per epoch has dropped
tenfold. Extending to epoch 114 or 199 buys very little further reduction in the floor; it buys
only more independent measurement occasions.

**Honest statement of the uncertainty on that estimate.** It has three real weaknesses.

- The e100-e104 transition is characterised by exactly **one** measurement family - the five-seed
  BASE curve, `n = 5` seeds - and is a *training-log* metric, not the held-out CE-PGD20 endpoint
  used for verdicts. The endpoint series only samples epochs 49/99/149/199, so no endpoint
  measurement exists between epoch 99 and epoch 149.
- The post-decay floor is bracketed, not pinned. Three estimates exist and they disagree by a
  factor of two: type (c) paired across five seeds gives `0.247 pp` (95% interval
  `[0.148, 0.710]`); type (b) between independent seeds gives `0.29-0.72 pp`; and the only
  post-decay paired-fork measurement - the history-routing-v2 placebo arms at epoch 199 - gives
  `0.50 pp` from six comparisons, on a different teacher, a 160-epoch fork and a placebo action
  rather than an untreated control. The exact quantity a plan-0087-style screen needs - **two
  untreated forks from a common epoch-99 parent, run 14 epochs** - has never been measured, and
  it is the shortest fork of the three, so it could plausibly sit below the whole bracket.
- The whole picture rests on a single learning-rate schedule. If a future campaign forks from a
  different parent under a different schedule, the epoch-104 recommendation does not carry over;
  the rule that carries over is "four to five epochs past the last decay".

**The single cheap measurement that would settle it.** Fork each of the two I100 epoch-99
parents (`dev-1` and `dev-2`) into **three untreated control replicates** - identical config,
identical schedule, differing only in the post-fork RNG stream - run them for epochs 101-114
exactly as plan 0087 did, and evaluate the registered CE-PGD20 endpoint at e104/e109/e114. That
gives `2 seeds x C(3,2) = 6` type (a) control-versus-control comparisons at three post-decay
horizons: the missing cell in the floor table, measured on exactly the design the project keeps
using.

Cost: the online-state report records control-arm training at `92.2 s/epoch` (dev-1) and
`95.2 s/epoch` (dev-2) (VERIFIED,
`docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:177-182`). Fourteen epochs is
`14 x 92.2 = 1291 s = 21.5 min` per run; six runs is `6 x ~22 min = 2.2 GPU-hours`, plus
eighteen endpoint jobs (DERIVED). That is **less than the 3 GPU-hours already budgeted for
option A in `docs/decisions/0001-online-state-s2-next-step.md:8`**, and it is a prerequisite for
interpreting option A's result rather than an alternative to it. If only two replicates per seed
are affordable, that still yields two comparisons - enough to detect a floor grossly larger than
expected, but not enough to estimate it (section 5, item 2 notes that the pre-decay floor
itself rests on only two independent parent lineages).

---

## 4. QUESTION 3 - design rules for future screens

### 4.1 The rules

**Rule 1 - Minimum horizon: end the screen at least four epochs past the final learning-rate
decay, and never end it while the learning rate is still at its highest value.**
A screen that ends at epoch 84, 89 or 94 from an epoch-79 parent at LR 0.1 is measuring a
quantity whose floor is 1.1-1.3 pp regardless of how long it runs. Justification: section 2.8,
section 3.5.

**Rule 2 - A control replicate is required, and two controls per seed is the minimum; three is
the number to plan for.**
The control is not free-standing truth; it is one draw from the same distribution the treatment
is drawn from. With one control per seed, a low control draw makes every arm look good and there
is no way to see it. Stage A's `11 of 12 positive on L4` (section 3.3) and the post-decay
`12 of 12 positive at e114` (section 3.4) are both what that looks like. Two controls per seed
detect a gross control anomaly; three allow the floor to be estimated within the screen itself.
Cost is one extra untreated run per seed - the cheapest arm in any campaign, since it needs no
new code.

**Rule 3 - Two seeds is a direction check, not a result. Five is the number that produced this
project's one clean answer.**
Two seeds agreeing in sign has probability 0.5 under the null and is therefore not evidence.
Five seeds gave `I100 - CROPSHIFT = +0.864 pp, SD 0.247, t(4) = 7.82, 5/5 positive`
(section 2.6) - a result that would survive scrutiny. Use two seeds only to decide whether to
spend five.

**Rule 4 - Compare an effect to the floor, never to a sample bootstrap interval.**
A bootstrap over the 5,000 held-out samples answers "if I had drawn a different 5,000 images from
this distribution, would this model still look better?" It says nothing about whether a second
training run of the same recipe would land in the same place. The reports already say this
(`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:29-31`,
`docs/ERT_CONFIRMATORY_T123_RESULTS.md:48-50`); the rule is to obey it downstream. When quoting
an interval, label it "over samples" or "over training runs" explicitly, every time.

**Rule 5 - Effect size worth chasing, given the floor.**
With `k` paired blocks the smallest detectable effect is `MDE = 2.8 * s / sqrt(k)` (section 1.3).

| regime | floor `s` used | k=2 | k=3 | k=4 | k=5 | k=8 | k=10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-decay (LR 0.1) | 1.21 pp (types a, a', b, c all agree) | 2.40 | 1.96 | 1.70 | 1.52 | 1.20 | 1.08 |
| post-decay, optimistic | 0.25 pp (type c, I100-CROPSHIFT, e199) | 0.49 | 0.40 | 0.35 | 0.31 | 0.24 | 0.22 |
| **post-decay, working value** | **0.40 pp** (midpoint of the bracket) | **0.79** | **0.65** | **0.56** | **0.50** | **0.40** | **0.35** |
| post-decay, placebo fork measured | 0.50 pp (type a', history routing v2, e199) | 0.99 | 0.81 | 0.70 | 0.63 | 0.50 | 0.44 |
| post-decay, pessimistic | 0.72 pp (type b, CROPSHIFT e199) | 1.43 | 1.16 | 1.01 | 0.90 | 0.71 | 0.64 |

All DERIVED. The practical thresholds: **do not launch a pre-decay screen for an effect you
believe is under about 2.5 pp**, and **do not launch a post-decay screen for an effect you
believe is under about 0.5 pp** unless you are prepared to run at least five seeded blocks and
accept that the post-decay floor itself is only known to within a factor of three. For
calibration, real effects in this project look like `CROPSHIFT - BASE = +1.612 pp` and
`I100 - BASE = +2.476 pp` at epoch 199 (DERIVED, section 2.6). Effects of that size are
comfortably resolvable. Effects of 0.1-0.2 pp are not, at any affordable cost.

**Rule 6 - A short screen is still legitimate for four purposes, none of which is an effect-size
claim.**

| legitimate use | why the floor does not block it | example in this project |
| --- | --- | --- |
| numerical-stability / does-it-run checks | the observable is finite-vs-non-finite, not an accuracy difference | S-BDD went non-finite in both dev seeds and was excluded on that basis alone (VERIFIED, `docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:5, 84-91`) |
| mechanism diagnostics on fixed cohorts | measured on the 45,000-row train split, which is ~6x quieter (section 2.3), and reported as a description, not a generalization claim | fixed-mask state transitions, rescue/harm counts, regime-disagreement rates |
| very large effects | a 4 pp direct effect clears even a 1.2 pp floor at k=1 | `ST1W` direct selected-cohort effect `+4.409 / +2.210 pp` (VERIFIED, `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52, 69`) |
| ruling things out | an arm that is negative in every block at every horizon is informative even when the floor is large | `T3LP05CONF` held-out is non-positive in all six seed-horizon cells (VERIFIED, `docs/ERT_CONFIRMATORY_T123_RESULTS.md:81, 84, 87, 90, 93, 96`) |

**Rule 7 - Report the floor next to the effect, in the same table, in the same units.**
Every screen report should carry a line of the form "effect `+X pp`; floor for this regime and
horizon `s = Y pp`; blocks `k`; MDE `2.8*Y/sqrt(k) = Z pp`." If `X < Z`, the screen's own report
should say `UNRESOLVED`, not `SUPPORTED`. A directional label applied to a value below the MDE is
a coin toss with a name.

**Rule 8 - Do not read a train-cohort effect as a held-out effect.**
Train-split metrics are about six times quieter and are fitted. The project has already
documented the failure directly: T123 point 4, "Direct selected-cohort rescue can increase while
held-out accuracy falls" (VERIFIED, `docs/ERT_CONFIRMATORY_T123_RESULTS.md:116-118`), and the CW
gap audit, where a `+4.12 pp` train direct effect became `+0.22 pp` overall held-out (VERIFIED,
`docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:34`).

### 4.2 Case (a) - plan 0087, PMP minus Control = +0.14 / +0.20 pp at epoch 114

**Verdict: not resolvable by this design. It is not resolvable at the sample level either.**

The facts. Epoch 114 held-out CE-PGD20, dev-1 `+0.14 pp`, dev-2 `+0.20 pp`, both positive
(VERIFIED, `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:46, 55`, and the paired table at
`:68, 77`). The screen used one control per seed and two seeds, i.e. `k = 2` blocks (VERIFIED,
`docs/plans/0087-i100-online-state-s2-preservation.md:26`, "Shared baseline e100 prefix once per
seed").

*Sample-level arithmetic first.* The paired table gives rescue and harm counts: dev-1 `31/24`,
dev-2 `33/23` (VERIFIED, lines 68 and 77). Net rescue is therefore `31-24 = 7` samples on dev-1
and `33-23 = 10` on dev-2, i.e. `7 x 0.02 = 0.14 pp` and `10 x 0.02 = 0.20 pp` - which is exactly
the reported effect, confirming the identity the report asserts at lines 81-83 (DERIVED). The
standard error of a paired difference of this kind is `sqrt(rescue + harm)` samples:
`sqrt(55) = 7.42` samples `= 0.148 pp` on dev-1 and `sqrt(56) = 7.48` samples `= 0.150 pp` on
dev-2 (DERIVED). So:

| seed | net (samples) | effect (pp) | sample-level SE (pp) | ratio |
| --- | ---: | ---: | ---: | ---: |
| dev-1 | 7 | +0.140 | 0.148 | 0.94 |
| dev-2 | 10 | +0.200 | 0.150 | 1.34 |
| pooled | 17 of 111 discordant | +0.170 | 0.105 | 1.61 |

All DERIVED. **Neither seed is significant even over samples**, and pooling both seeds still
leaves the effect at 1.6 standard errors, short of the 1.96 needed at the 5% level. This is
before any consideration of training-run variation.

*Training-run arithmetic.* The pooled effect is `+0.170 pp` over `k = 2` blocks. Against each
candidate post-decay floor (all DERIVED):

| floor `s` (pp) | source | SE of the 2-block mean = `s/sqrt(2)` | ratio | blocks needed for 80% power |
| ---: | --- | ---: | ---: | ---: |
| 0.25 | type (c), I100-CROPSHIFT paired, e199 | 0.177 | 0.96 | 17 |
| **0.40** | working value, section 2.9 | 0.283 | **0.60** | **44** |
| 0.50 | type (a'), history-routing placebo forks, e199 | 0.354 | 0.48 | 68 |
| 0.72 | type (b), CROPSHIFT between seeds, e199 | 0.509 | 0.33 | 141 |

Under every one of them the effect is well under one standard error, and the number of paired
blocks required is between 17 and 141. For comparison, the screen ran two.

Two further problems specific to this result:

- The "both seeds positive" criterion has a null probability of `0.5 x 0.5 = 0.25` (DERIVED). It
  is not a 5% test.
- The result is not independent of its neighbours. Twelve arm-seed values from three campaigns
  all sit above the same two control endpoints, and all twelve are positive at epoch 114
  (section 3.4). The report's own text notes the parallel `DPM - Control = +0.08 / +0.12 pp` from
  plan 0079 (VERIFIED, `docs/decisions/0001-online-state-s2-next-step.md:24`). A single low
  control draw explains all of it.

**What would make it resolvable.** Three things, in order of cost:

1. **Control replicates** - at minimum three per seed, per Rule 2, so that the shared-control
   offset can be seen and removed. Without this, no number of treatment seeds fixes the problem,
   because they all lean on the same control.
2. **Seeds** - 17 to 37 paired blocks for an 0.17 pp effect, per the arithmetic above. The three
   unused confirmation seeds proposed in `docs/decisions/0001-online-state-s2-next-step.md:8`
   take `k` from 2 to 5, which reduces the MDE from `0.49` to `0.31 pp` (DERIVED) - still roughly
   twice the observed effect. Five seeds is a good use of 3 GPU-hours, but it will not make
   `0.17 pp` significant; it will most likely return "still unresolved", which is a legitimate
   and useful outcome if the pre-registered rule says so in advance.
3. **A bigger effect.** The most honest conclusion is that `0.17 pp` - seventeen images in ten
   thousand - is below what this measurement apparatus can see for any affordable number of runs,
   and the question should be reframed rather than re-powered.

The screen's own caveat already says most of this (VERIFIED,
`docs/plans/0087-i100-online-state-s2-preservation.md:174-177`: "3-10 samples ... below the noise
floor"). This analysis corrects the floor figure it cites (0.25-0.50 pp at epoch 114, not
1-2 pp), adds the sample-level test the caveat did not run, and identifies the shared-control
dependency the caveat did not mention. **The `SUPPORTED` label should be read as `DIRECTIONAL,
UNRESOLVED`.**

### 4.3 Case (b) - ST1W, held-out +1.500 pp on L2 and +0.040 pp on L4 at epoch 84

**Verdict: the bootstrap interval establishes that the L2 effect is real *in that one run*. It
establishes nothing about ST1W as a method, and the L4 result plus the floor together make the
method claim unsupported.**

The facts. `ST1W` on seed L2: direct `+4.409 pp`, held-out `+1.500 pp`, 95% sample bootstrap
`[+0.680, +2.320]` (VERIFIED, `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52`). On seed L4:
direct `+2.210 pp`, held-out `+0.040 pp`, bootstrap `[-0.760, +0.860]` (VERIFIED, line 69). The
bootstrap is 2,000 deterministic class-stratified paired replicates and the report states it is
"not training-seed uncertainty" (VERIFIED, lines 29-31). Both are epoch-84 endpoints from
epoch-79 parents at LR 0.1 - the noisiest regime in the project.

*What the bootstrap does establish.* From the machine artifact
`docs/experiments/ert_stage_a_effect_decomposition_v1.json`, path
`seeds.L2.arms.ST1W.heldout`, the held-out rescue count is 270 and the harm count is 195, giving
net 75 (VERIFIED). `75 x 0.02 = 1.500 pp` (DERIVED), matching the report. The paired sample-level
standard error is `sqrt(270 + 195) = sqrt(465) = 21.56` samples `= 0.431 pp`, so the effect is
`1.500 / 0.431 = 3.48` sample-level standard errors (DERIVED). A `+/- 0.820 pp` bootstrap
half-width is `1.90 x 0.431`, consistent with a 95% interval (DERIVED). So: **if you re-drew the
5,000 held-out images from the same distribution, this particular L2 checkpoint pair would still
show a positive difference.** That is a genuine and correctly computed statement.

*What it does not establish.* It does not vary the thing that actually varies. Comparing against
the floor from Question 1 at exactly this regime and horizon:

| quantity | value (pp) | status |
| --- | ---: | --- |
| ST1W L2 held-out effect | +1.500 | VERIFIED |
| sample-level SE (over images) | 0.431 | DERIVED |
| **training-run floor `s`, type (a), untreated fork, e84** | **1.136** | DERIVED, section 2.7 |
| effect divided by training-run floor | **1.32** | DERIVED |
| observed L2 BASE control-vs-control gap at e84 | 0.840 | VERIFIED, `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:29` |
| largest untreated-vs-untreated gap observed at e84 | 3.300 | DERIVED, section 2.4 |
| MDE for k=1 block at this floor | 3.18 | DERIVED |

The training-run floor is **2.6 times larger** than the sample-level standard error
(`1.136 / 0.431 = 2.6`, DERIVED). The bootstrap therefore understates the real uncertainty by
that factor. At `z = 1.32` over training runs, ST1W L2 is roughly a one-in-five event under the
null - unremarkable. And this project has directly observed an untreated run beating another
untreated run by `3.300 pp` at epoch 84 with no treatment applied at all (DERIVED, section 2.4);
`+1.500 pp` is less than half of that.

*The replication.* `+0.040 pp` on L4 is two images. The report's own reading - "a promising
seed-1 generalization signal, not a two-seed confirmed generalization effect" (VERIFIED,
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:86-90`) - is correct, and this analysis strengthens it:
with a floor of `1.14 pp`, the L2 and L4 results are not even in tension. Both are what you get
by drawing twice from a distribution centred near zero with a standard deviation of about one
point. The seed-level spread of the two values, `|1.500 - 0.040| = 1.460 pp`, is almost exactly
the predicted `1.14 pp` floor plus sampling (DERIVED).

*Summary of what ST1W establishes and does not.*

| claim | supported? |
| --- | --- |
| "On the L2 run, treated and control checkpoints differ on the held-out set by more than image-sampling noise" | **Yes** - `z = 3.48` over samples |
| "ST1W improves held-out robust accuracy at epoch 84" | **No** - `z = 1.32` over training runs, `k = 1` block, MDE 3.18 pp |
| "ST1W improves held-out robust accuracy in general" | **No** - and the L4 replication is `+0.040 pp` |
| "ST1W produces a large direct effect on its selected training cohort" | **Yes** - `+4.409 / +2.210 pp`, replicated in both seeds, on the quieter 45,000-row split |
| "That direct effect transfers to held-out data" | **No** - this is precisely the gap Rule 8 describes |

*What would make it resolvable.* At the epoch-84 floor of `1.14 pp`, detecting a `1.5 pp` effect
at 80% power needs `7.849 x 1.136^2 / 1.5^2 = 4.5`, i.e. **5 paired blocks** (DERIVED). Detecting
`0.5 pp` would need `7.849 x 1.136^2 / 0.5^2 = 40` blocks (DERIVED) and is not worth attempting.
The better move, per Rule 1, is to stop screening ST1W at epoch 84 at all and re-run it as a
post-decay continuation where the floor is 3-4x smaller - which is what the report itself
proposes ("a new preregistered, longer-horizon ST1W experiment", VERIFIED,
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:118-120`).

---

## 5. What this analysis could not resolve

1. **The type (a) post-decay floor does not exist for the design that needs it.** No two
   *untreated* forks from a common post-decay parent have ever been run and compared. The nearest
   substitutes are the history-routing-v2 placebo forks (`0.50 pp`, but a different teacher, a
   160-epoch fork and a placebo action rather than nothing) and the I100-CROPSHIFT paired
   difference (`0.25 pp`, four degrees of freedom). The resulting `0.25-0.50 pp` bracket is wide
   enough to matter: it changes the required block count for a 0.17 pp effect from 17 to 68.
   Section 3.5 specifies the measurement that would close it.
2. **The pre-decay floor's original source is thinner than it looks.** The RNG-stability
   diagnostic's six BASE gaps come from only **two** independent run pairs (one per teacher) read
   at three correlated epochs. The 84-pair figures in section 2.4 are much stronger, but they too
   come from only two campaigns and two teachers; the effective number of independent parent
   lineages behind the whole pre-decay floor is **two**.
3. **No held-out endpoint exists between epoch 99 and epoch 149** in the five-seed study, so the
   sharpness of the decay transition is characterised only by training-log metrics. The
   epoch-104 recommendation is interpolated across that gap.
4. **The two metric families are not identical.** Training-log `val_pgd_accuracy` and the
   registered endpoint `CE-PGD20` differ - for example the L2 BASE R1/R2 gap at epoch 94 is
   `1.900 pp` in the training log and `1.880 pp` at the endpoint, and at epoch 84 it is
   `0.760` versus `0.840` (VERIFIED, both from the RNG-stability artifact). The discrepancies are
   small and do not change any conclusion, but curves and endpoints should not be spliced.
5. **e114 row-level endpoints are missing** from the CW long-horizon inventory (VERIFIED,
   `docs/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md:17`), so the only long series starts at
   epoch 129 and cannot be joined to the epoch-104/109/114 screens.
6. **Held-out split size is unverified in three of the eight source reports.** The FF/NR
   causal-horizon, FF/NR pilot and history-routing-v2 reports do not restate the 5,000-sample
   validation cardinality in their own text. They share the registered CE-PGD20 attack identity
   with the reports that do, so the convention is very likely the same, but the pp-to-sample
   conversion for those three tables is assumed rather than checked.
7. **No campaign ever ran the identical control twice under two run ids.** A search across
   `docs/`, `docs/experiments/`, `ard-runs/`, `ferret-results/` and `.cache/analysis/` found that
   `-r2`-style suffixes are relaunch labels applied uniformly to every arm in a campaign, not
   second draws of one control; and every "canary"/"smoke" artifact is an operational pre-launch
   check, not a scientific replicate. The one apparent exception,
   `ard-runs/ard_codex_bootstrap/ert-rslad-static-trajstab-v1/base-s1` versus `base-s1-r2`, is an
   aborted first attempt with no epoch metrics and no checkpoints, so only one usable run exists.
   All type (a) evidence in this document therefore comes from campaigns that deliberately
   perturbed an RNG substream, never from an accidental duplicate.

---

## 6. Sources

Reports read and cited:

- `docs/archive/ard-distillation-2026/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md`
- `docs/experiments/ert_cw_margin_rng_stability_diagnostic_v1.json`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`
- `docs/experiments/ert_rslad_five_seed_global_stochasticity_v1.json`
- `.cache/analysis/ert-rslad-five-seed-stochasticity-v1/outputs/rescue_summary.csv`
- `docs/archive/ard-distillation-2026/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md`
- `docs/archive/ard-distillation-2026/ERT_CONFIRMATORY_T123_RESULTS.md`
- `docs/archive/ard-distillation-2026/ERT_DYNAMIC_S3_RECOVERY_RESULTS.md`
- `docs/archive/ard-distillation-2026/ERT_STAGE_A_EFFECT_DECOMPOSITION.md`
- `docs/experiments/ert_stage_a_effect_decomposition_v1.json`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md`
- `docs/archive/ard-distillation-2026/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md`
- `docs/archive/ard-distillation-2026/FFNR_CAUSAL_HORIZON_CE20_RESULTS.md`
- `docs/archive/ard-distillation-2026/FFNR_CAUSAL_PILOT_RESULTS.md`
- `docs/archive/ard-distillation-2026/essays/HISTORY_ROUTING_V2_RESULTS.md`
- `docs/plans/0087-i100-online-state-s2-preservation.md`
- `docs/decisions/0001-online-state-s2-next-step.md`
- `docs/archive/ard-distillation-2026/ERT_RESEARCH_STATUS_SUMMARY.md`

All endpoints referenced here use the single registered CE-PGD20 attack identity
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2` (VERIFIED in every one of the
above reports that carries an identity block), so accuracies are comparable across campaigns.
