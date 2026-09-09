# Evidence Ledger v2

Status: corrected evidence ledger, 2026-09-08. Supersedes the verdict columns of
`docs/EVIDENCE_RECLASSIFICATION.md` and the executive conclusion of
`docs/ERT_RESEARCH_STATUS_SUMMARY.md` wherever the two disagree with this file.
Built from a defect audit that traced 117 candidate defects; 28 survived two
independent checks.

Authority order used throughout, and the order a reader should use:

1. `docs/experiments/*.json` — hash-bound records.
2. Run artifacts under `ard-runtime/.../runs/*/` and `ard-runs/ard_codex_bootstrap/*/` —
   `arm-summary.json`, `resolved_config.yaml`, `epoch-metrics.jsonl`, endpoint `summary.json`.
3. Prose in `docs/*.md` — never authoritative.

Every number below is marked **VERIFIED** (I read the claim and the record and can cite
the file), **REPORTED** (relayed from the audit, not re-read here), or **INFERRED** (my
own arithmetic or reasoning on verified inputs).

---

## 0. Terms, and the one distinction the project keeps losing

**Effect.** A difference in accuracy between a treated arm and a comparator, in
percentage points (pp), at a named epoch, on a named data split, under a named attack.

**Floor.** The spread you would see between two runs that differ only in something you
do not care about. An effect smaller than its floor is not evidence of anything.

**Kind of floor.** Two runs can differ in several ways, and each way has its own spread.
This project has four kinds on record:

- **(a) RNG replicate** — same recipe, different random streams (data order, augmentation
  draws, attack random starts).
- **(a') placebo fork** — one parent, two branches, an inert action applied to one.
- **(b) between training seed** — two independent runs from scratch with different seeds.
- **(c) paired effect, replicated** — the spread of a treated-minus-control difference
  across seeds.

**The distinction that matters.** A floor is only usable for a contrast that varies the
same thing the floor varied. Before the learning-rate decay, all four kinds land in the
same place — 1.03 to 1.25 pp at epochs 84–94 (VERIFIED, `docs/MEASUREMENT_DESIGN.md:378-397`)
— so substituting one kind for another costs nothing. After the decay they do not agree,
and substitution is invalid. This is the single most consequential fact in this ledger,
and the project has substituted kinds post-decay in at least four documents.

**Regime.** Before the epoch-100 learning-rate decay, or after it. The floor drops by
roughly a factor of ten across that boundary. A pre-decay number applied post-decay
overstates the floor by 3–4×; a post-decay number applied pre-decay understates it by the
same.

**Block.** One paired observation: one treated run and its matched control.

**MDE (minimum detectable effect).** The smallest true effect a design would detect 80% of
the time. This project uses `MDE = 2.8 * s / sqrt(k)` for a floor `s` treated as known, and
a t-based version when the floor itself was estimated from few runs. Both are quoted below
where they differ.

---

## 1. What is established

An entry is established only if the effect exceeds the MDE computed from a floor **of the
matching kind, regime, horizon, split and attack**. Where no such floor exists, the entry
is not established, however large or however positive the effect.

### 1.1 Augmentation policy — the project's real results

| # | Claim | Effect | k | Endpoint | Comparator | Applicable floor | Verdict |
|---|---|---|---|---|---|---|---|
| E1 | CropShift for all 200 epochs beats canonical RSLAD augmentation | **+1.32 / +1.16 pp**, mean **+1.240** | 2 | held-out CE-PGD20, e199, 5,000-image validation split | `BASE` (canonical, all 200 epochs) | between-run difference at e199, **0.307 pp** (INFERRED from the record's own per-run SDs) → MDE 0.61 pp | **Established.** Effect is 2.0× the MDE and positive in both seeds. |
| E2 | CropShift, then CropShift+RandomErasing from epoch 100, beats canonical | **+2.18 / +1.30 / +2.10 pp**, mean **+1.860** | 3 | same | `BASE` | same, 0.307 pp → MDE 0.50 pp | **Established.** Effect is 3.7× the MDE and positive in all three seeds. |
| E3 | Switching to IDBH_WEAK at epoch 100 beats CropShift-throughout | **+1.20 / +1.04 pp**, mean **+1.12** | 2 | same | `CROPSHIFT`, shared prefix to e99 | post-decay fork at e199, **never measured**; nearest is the 0.50 pp placebo fork on a different teacher → MDE 0.99 pp | **Direction established, margin marginal.** Effect is 1.13× a borrowed MDE. |
| E4 | Switching to IDBH_WEAK at epoch 100 beats CropShift+RandomErasing-from-100 | **+0.78 / +0.68 / +0.62 pp**, mean **+0.693** | 3 | same | `CROP_SUFFIX`, shared prefix to e99 | same, borrowed 0.50 pp → MDE 0.81 pp | **Direction established, margin not.** Effect is 0.86× the borrowed MDE — below it. |
| E5 | The epoch-100 switch survives on the official CIFAR-10 test set under AutoAttack | last checkpoint **+0.41 / +0.06 / +0.46 pp** (mean +0.31); best checkpoint **+0.30 / +0.45 / +0.42 pp** (mean +0.39) | 3 | official test split, 10,000 images, standard AutoAttack, e199 | matched `CROP_SUFFIX` | **no floor of this kind exists** at any horizon | **Established as a preregistered directional result.** 6 of 6 seed×checkpoint measurements positive; the preregistered rule was a sign rule and it passed. The magnitude is not resolved. |

VERIFIED for all five rows. E1 from `docs/experiments/ert_rslad_static_trajectory_stabilization_results_v1.json`
→ `paired_crop_minus_base_endpoint_delta_pp.seed{1,2}.199.robust` = 1.32 / 1.16, `mean.199.robust` = 1.24.
E2 and E4 from `docs/experiments/ert_rslad_unseen_confirmation_results_v1.json` → `endpoint_rows`
at `scientific_epoch: 199` (BASE 58.22 / 58.58 / 58.18; CROP_SUFFIX 60.40 / 59.88 / 60.28;
I100_SUFFIX 61.18 / 60.56 / 60.90). E3 from `docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json`
→ `promotion.I100.per_seed.{1,2}.final_robust_delta_pp` = 1.20 / 1.04. E5 from
`docs/experiments/ard_i100_official_test_autoattack_v1.json` → `decision.per_seed_pp` and
`decision.secondary_best_checkpoint_pp`, `dataset_scope = {cifar10, test, 10000}`,
`verdict: CONFIRMED`.

Three things to say plainly about this block.

First, rows E1 and E2 are **two different treatments**, not one treatment on five seeds.
The registry defines `CROPSHIFT` as `Aug(CropShift) @all`, seeds 1 and 2, and `CROP_SUFFIX`
as `Aug(CropShift->CROP_RE@e100)`, seeds confirm-a/b/c (VERIFIED, `docs/ARM_REGISTRY.md:94,97`).
The run configs agree: the dev runs read `augmentation_policy: cropshift`, the confirmation
runs read `augmentation_policy: stagewise`, `stagewise_switch_epoch: 100`,
`stagewise_late_policy: crop_re` (VERIFIED, `ert-rslad-static-trajstab-v1/cropshift-s1-r2/resolved_config.yaml:23`
and `runs/i100-official-test-v1/arms/unseen-confirm-a-crop-suffix-r2/outputs/student/resolved_config.yaml:23-26`).
The project's headline `+1.612 pp for CROPSHIFT over five seeds` is the average of these
two treatments under one name. See §3, rank 1.

Second, rows E3 and E4 are the same intervention against two different comparators, and
they do not agree. The gap between them, 0.43 pp, is close to the size of either effect.
The project's headline `+0.864 pp` was the average of these two; that pooling was corrected
on 2026-09-07 in four documents, and the correction stands.

Third, row E5 is the strongest single result the project owns, and the reason is that it
was preregistered as a sign rule on an untouched split with the strongest available attack.
It is also the result about which the least is known: three seeds, one design, no replicate
floor. The comparison currently printed beside it in the status summary — that +0.31 pp
"sits inside the 0.25–0.50 pp post-decay floor bracket" — is a validation-split CE-PGD20
bracket set against an official-test AutoAttack number, and it should be deleted rather
than corrected. The right statement is that no floor of the matching kind has been measured,
and the result's own between-seed spread is 0.218 pp (REPORTED).

**Also settleable from data already on disk, at no GPU cost.** The augmentation-family
gate that rejected `CROP_RE` and `IDBH_WEAK` as always-on policies was decided on
trajectory AUC, and the reclassification calls that AUC floor "never measured anywhere in
this project". It is measured, in the incumbent block of the record the row itself cites:
two untreated full runs of the same arm differ by 0.0137 pp (CROPSHIFT) and 0.0174 pp
(BASE) in normalised AUC, against deficits of 0.509 to 0.794 pp — 30× to 58× (REPORTED).
The gate decision was correct and can be marked resolved. Nothing downstream changes.

### 1.2 Measurement facts

| # | Claim | Value | Evidence | Verdict |
|---|---|---|---|---|
| M1 | Training is bit-deterministic given identical seeds, parent and recipe | exact | REF2−REF1 = 0.0000 in two independent campaigns; `A1` and `G1` bit-identical on all 15 epochs across two git SHAs and two campaigns; `I100_CONTROL` produces identical weights (`977b7e42…` at e104) in three campaigns | **Established**, on five independent lines. |
| M2 | Pre-decay noise floor | **1.14 / 1.20 / 1.25 pp** at e84 / e89 / e94 | 84 pairwise comparisons per horizon, and four floor kinds independently agree at 1.03–1.25 pp | **Established.** VERIFIED, `docs/MEASUREMENT_DESIGN.md:378-397`. |
| M3 | Post-decay floor for untreated forks that differ **only in the attack and global RNG streams** | **0.159 / 0.124 / 0.092 pp** at e104 / e109 / e114, 4 df, e114 CI [0.055, 0.265] | `docs/experiments/ard_post_decay_floor_v1.json`, 3 replicates × 2 parents, held-out CE-PGD20 | **Established, and narrow.** VERIFIED. Scope: this design, these horizons, this split, this attack. |
| M4 | Adversarial evaluation is reproducible to about 1e-3 in per-sample logit margin | — | cross-campaign endpoint rows agree exactly at 4 of 6 control endpoints and differ in one sample's margin by 1.08e-3 at the other 2 | **Established** (REPORTED). Any claim resting on smaller margin differences rests on noise. |

M3 needs one sentence of care, because it is the number most likely to be misused. Its
replicates vary `continuation_seed`, and `src/ard/analysis/ert_stage_a_runtime.py:876-883`
maps that to `attack_seed` and `other_seed` while pinning `data_seed` to
`config.seeds.data_order` (VERIFIED). Data order and augmentation are held fixed. It is
therefore not a floor for a contrast that varies anything else.

### 1.3 Negative and structural results

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| N1 | Sample-level interventions move the cohort they treat and do not move held-out accuracy | direct effects +1.68 to +4.41 pp on the selected cohort; matched held-out effects −0.98 to +1.50 pp, none clearing its floor | **Established, and it is the project's most important scientific finding.** The multiple is about 5× to 15× a properly scaled train floor, not the 13×–33× currently printed (§3, rank 7). |
| N2 | Best-oriented history routing fails at its own preregistered target | PF-TA − C = −0.18 pp, NR-TA − C = −0.17 pp against a preregistered ≥ +0.50 pp gate; floor 0.426 pp from the same campaign's own placebo forks, same statistic, same horizon | **Refuted, correctly.** The only claim in the project measured against a floor of the right kind and found false. |
| N3 | The corrected secant boundary-distance objective is numerically unsupported | non-finite training loss at e106 (dev-1, last finite loss 8.58e10) and no valid checkpoint after e101 (dev-2), on two different hosts | **Established.** The observable is finite-versus-non-finite, so no floor applies. |
| N4 | Student loss history predicts which samples fail later | preregistered Spearman gate positive on all three held-out confirmation seeds (0.156 / 0.155 / 0.159) | **Established as a prediction claim only.** It licenses no intervention. |
| N5 | Majority-3 smoothing cuts routing churn | switches −52.6% / −52.5%, re-entries −66.1% / −65.6% | **Established as event counts**, on 675,000 state rows per arm. Not an accuracy claim; endpoint robust accuracy was below control in both seeds. |
| N6 | `torch.compile` is slower on this workload | 21–23% slower, and every candidate failed numerical parity | **Established.** Baseline throughput 679.1 img/s. |

All six REPORTED from the audit's verification pass, which traced each to its hash-bound
record.

---

## 2. What is not established, and why not

The project has repeatedly written "not supported" where the truthful statement is "not
measured". These are three different states and they call for three different actions.

### 2.1 Measured and refuted — one claim

Only N2 above. One preregistered target, one matching floor, one honest failure.

### 2.2 Measured against a floor of the wrong kind — the largest category

These claims have numbers and have been judged. The judgement is void, in both directions.
Neither "fails" nor "passes" was earned.

**The nine epoch-114 treated-versus-control contrasts.** `docs/POST_DECAY_FLOOR_RECLASSIFICATION.md`
judges rows E1, E2, E4, E5 and E6 of the reclassification against thresholds of 0.243 pp
and 0.181 pp, both derived from the 0.092 pp figure. That figure varies the attack and
global RNG streams. Every judged arm and its control ran with `continuation_seed: null` and
`rng_source_seeds: null` — one shared stream, differing only in the objective (VERIFIED: I
read `arm-summary.json` for `control` and `pmp` under
`runs/ert-i100-online-state-s2-v1-recovery15/arms/dev-1/`, both null, against
`runs/post-decay-floor-v1/arms/dev-1/rep2/arm-summary.json`, which carries
`{attack_seed: 2, data_seed: 1, other_seed: 2}`). The two designs perturb orthogonal things.
The spread of a shared-stream, objective-perturbed contrast has never been measured, and
its size relative to 0.092 pp is unknown **in both directions**. The "0 of 9" result and
the two exploratory "passes" (CLEAN_WRONG_PLAIN_ADVCE +0.270 pp, CLEAN_WRONG_A7_MARGIN_ONLY
+0.260 pp) must both be withdrawn. The affected arms: DPM, OS-PMP, D-BDD, OS-DBDP, SBF,
TPFM@S2T1, and three Stage-A action transfers.

**The same substitution, in the other direction, in the coefficient audit.**
`docs/COEFFICIENT_AUDIT.md:504-507` judges the two SUPPORTED epoch-114 online-state results
(+0.14 and +0.20 pp) against a 0.94–2.06 pp figure measured at epochs 84–94, before the
decay. The same commit that wrote that sentence also wrote
`docs/MEASUREMENT_DESIGN.md:452-456`, which forbids exactly that substitution by name. The
audit's conclusion — that those two results rest on effects at or inside the floor at a
single untested coefficient — survives against the 0.25–0.50 pp post-decay bracket, but the
implied 5×–15× margin does not; the honest margin is roughly 1×–3×, and the honest
statement is that no floor of the matching kind exists at e114.

**The epoch-199 fork contrasts.** Rows E3 and E4 of §1.1 are paired forks from a shared
epoch-99 checkpoint. The floor table records the post-decay untreated-fork floor as
`n = 0, never measured` (VERIFIED, `docs/MEASUREMENT_DESIGN.md:387`). What is used instead
is a placebo fork on a different teacher (Bartoldson2024, WRN-94-16), from an epoch-39
parent, over 160 epochs under a different decay schedule. That substitution is documented
and conservative, but it is a substitution.

**The 0.25 pp and 0.72 pp post-decay floors are not floors.** See §3, ranks 1 and 2. The
0.25 pp figure is the spread of a five-value set that mixes two comparators; the 0.72 pp
figure is the spread of a five-value column that mixes two treated arms. Both are inflated
by a between-group mean difference rather than by noise. Every MDE and block count keyed to
them is wrong, always in the conservative direction.

**The governing standard's σ_d = 0.35 pp has no source at all.** `docs/MEASUREMENT_STANDARD.md`
sizes every future plan from it (§5.1, §5.2, §9). It entered the repository as a
placeholder, is marked provisional in the document itself, and was never revised after
plan 0092 measured the real thing. Three governing artefacts now carry three different
values for the same slot: 0.35 pp, 0.40 pp, and 0.0924 pp.

### 2.3 Never measured — the honest gaps

| Missing quantity | Why it matters | Status |
|---|---|---|
| Shared-stream, post-decay, treated-versus-control floor | The floor for **every screen this project has actually run** at e104–e114 | Plan 0095, milestone M0, not launched |
| Post-decay untreated-fork floor at e199 | The floor for the two augmentation-switch results in §1.1 | Never run; a by-product of plan 0093's twelve control forks would supply it |
| Official-test AutoAttack floor at e199 | The floor for the project's flagship confirmation | Never run; only the 3-seed I100 spread (0.218 pp) exists |
| Evaluation-attack random-start effect on a scored endpoint | Cited as characterised; it is not | Never run anywhere in the repository (§3, rank 10) |
| Post-decay replicate floor that also re-seeds data order | Would bound how much of the 0.092 pp figure is attack-stream-specific | Never run; costed at 2.2 GPU-h |
| Coefficient neighbourhoods for 18 of 23 calibrated scalars | Every DPM, D-BDD, OS-PMP, OS-DBDP, SBF and TPFM result rests on a single untested value | Never run |
| Whether CropShift-throughout and CropShift→CROP_RE differ from each other | The 0.62 pp gap between the §1.1 group means is a difference of group means over an interleaved sample; it is not an established effect | The missing cells were never run |
| Whether the epoch-114 sign pattern is a treatment effect | 18 of 18 arm-seed values are positive at e114 against two control runs; 3 of 12 at e104 and 5 of 12 at e109 | Confounded with one control draw; not separable without more controls |

---

## 3. What changed in this audit, ranked by consequence

117 candidate defects were examined. 28 survived two independent checks; they reduce to
about 24 distinct defects, because the pooled `+1.612 pp` figure was found separately at
four locations. 89 were rejected as technically true but inconsequential, or as wrong.

**No verdict is overturned into its opposite.** Nothing that was called established becomes
refuted. What changes is what the numbers mean, on how many blocks they rest, and — for the
nine epoch-114 contrasts — whether they were tested at all.

### Rank 1 — the augmentation headline is an average of two different treatments

`+1.612 pp for CROPSHIFT over five seeds` is not a CropShift effect. Two of the five seeds
ran CropShift for 200 epochs; three ran CropShift and then switched to
CropShift+RandomErasing at epoch 100. The honest figures are +1.240 pp on 2 seeds and
+1.860 pp on 3 seeds. The pooled number, the "five seeds" count and the SD 0.487 describe
no measured quantity.

This is the surviving half of a sentence whose other half was corrected on 2026-09-07. The
correction note for `+0.864 pp` sits three lines below `+1.612 pp` in the status summary
and repairs only the second figure, which reads to a careful person as an assurance that
the first was checked and survived (VERIFIED, `docs/ERT_RESEARCH_STATUS_SUMMARY.md:14-25`).

The mislabel is in the hash-bound records, not only in prose: `ert_rslad_five_seed_global_stochasticity_v1.json`
labels all five differences `CROPSHIFT-BASE`, including the three confirmation values
+2.18 / +1.30 / +2.10 (VERIFIED, I enumerated the rows), and
`ert_rslad_five_seed_artifact_inventory_v1.json` lists arm `CROPSHIFT` over
`confirm-*/crop-suffix/` paths. A prose correction alone leaves the trap in place.

One caution against over-claiming. Unlike the corrected `+0.864` case, these two groups
overlap: confirm-b's +1.30 sits between the two dev values, and the confirmation group's
own SD is 0.487 pp — the same as the pooled SD. The split is established by run provenance,
not by a clean separation in the numbers. Do not write "the split is real" language here.

Locations: `docs/ERT_RESEARCH_STATUS_SUMMARY.md:14`; `docs/EVIDENCE_RECLASSIFICATION.md:152`
(row A1) and `:267`; `docs/MEASUREMENT_DESIGN.md:302` and `:691` (where it is the calibration
anchor for what a real effect looks like); `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:10,98`;
`docs/NUMERIC_CONSISTENCY_AUDIT.md:31`, which cites it as a validly detected effect; and the
two records above.

### Rank 2 — the 0.72 pp "pessimistic post-decay floor" is a between-arm gap

The same mixed column produces the floor. Its five-value per-run SD is 0.5096 pp, and
0.5096 × √2 = 0.7207 pp, which is the tabulated 0.72 (VERIFIED — I recomputed it from the
campaign CSV). Computed within arm, the pooled per-run SD is 0.2295 pp and the two-run
difference SD is **0.325 pp**; the gap between the two arm means is 0.857 pp. So the floor
is dominated by a real arm effect, not by seed noise. For contrast, the single-arm columns
have group-mean gaps of 0.24 pp (BASE) and 0.43 pp (I100), and their floors are sound.

Consequence: every MDE keyed to 0.72 pp is about 2.2× too large, and every block count
about 5× too large. The bias is conservative everywhere it is used, so no verdict flips —
but the number should be withdrawn or split by arm, not merely footnoted. Locations:
`docs/MEASUREMENT_DESIGN.md:296, 393, 445, 685, 754`; `docs/EVIDENCE_RECLASSIFICATION.md:61,
152, 206`; `docs/METHOD_DIRECTIONS_V2.md:376, 468`; `docs/ERT_RESEARCH_STATUS_SUMMARY.md:67`.

### Rank 3 — the nine epoch-114 verdicts were never tested

Detailed in §2.2. Two operational points. First, the generator is the place to fix it:
the licensing sentence at `docs/POST_DECAY_FLOOR.md:64` is emitted by
`scripts/aggregate_post_decay_floor.py:471`, and its guard at line 70 — which correctly
warns about horizon, endpoint and split — must be extended to the **kind of perturbation**.
Second, plan 0095 is the prerequisite for the whole state-conditional family, not a
refinement of it; nothing in that family can be judged until it reports.

One sub-claim in the audit needs tightening before it is repeated. "A genuinely inert
treatment would return a bit-identical run" holds only for a treatment that consumes
randomness identically. A real treated arm perturbs the weights and then diverges
chaotically from the same stream. That is precisely why the shared-stream floor is
unmeasured rather than zero, and why its size relative to 0.092 pp is unknown.

### Rank 4 — the reclassification's row A2 never received its own correction

The comparator correction of 2026-09-07 was written into the prose of the same file, into
`docs/MEASUREMENT_DESIGN.md` §2.6 and into the status summary, but not into the table row
the prose points at. Row A2 at `docs/EVIDENCE_RECLASSIFICATION.md:153` still reads
`mean +0.864, SD 0.247, t(4) = 7.82` against a floor cell reading
`0.25 (type c, this exact contrast)` — the exact combination the audit ruled out, and a
circular one, since 0.25 pp is the SD of that same mixed set. The table is what a reader
consults row by row.

### Rank 5 — "identical numbers prove artifact reuse" is invalid in a deterministic pipeline

Two places assert it. `docs/ARM_REGISTRY.md:274-276` calls A1 and G1's identical epoch-94
numbers "the strongest possible evidence they are the same trained run reused"; the record
shows three separate trainings, in three campaigns, at three git SHAs, reproducing
bit-identically. `docs/ARM_REGISTRY.md:347-352` calls `I100_CONTROL` "the exact same run,
byte-for-byte" across three reports; the record shows independent executions with different
checkpoint files that hash to identical model state.

The operative conclusion in both places is right and must be kept: deterministic replicates
of one seed carry the information of one observation, never three or four. What is wrong is
the reason, and the reason matters, because the registry offers it as a general rule. In
this repository, provenance is settled by a checkpoint or rows hash; numeric identity is
evidence of determinism only. The corrected fact is the stronger one — see M1.

Related, and the same root cause: row D1's "one trained run per seed, reported four times"
tells a reader the remedy is bookkeeping. Three campaigns each spent GPU-hours retraining
the same recipe at the same seeds and got the same answer to thirteen decimals; only F1 is
documented reuse. The lesson is that a re-run at fixed seeds is not a replicate.

### Rank 6 — the one "exact coefficient match" in Group C is not one

`ST3K05` and `T3LP05CONF` are both `advkd_advce` on the same cohort with the same AdvKD
multiplier, but they carry different live AdvCE coefficients: 0.07095924764871597 versus
0.075, a +5.694% step — the same rounding step the same paragraph attributes only to the
other two pairs. The registry cannot expose this, because `ard_arm_registry_v1.json` records
only `advkd_multiplier` for all four S3×T3 arms, giving two of them byte-identical canonical
strings for different treatments.

Consequence: row C3's argument — "the identical arm is positive on both seeds in one
campaign and negative in the other, therefore control artefact" — is not established as
written, because the arms are not identical and the design cannot separate the coefficient
step from the +0.94 / +1.78 pp control shift. The honest statement is that treatment and
control both moved. Grouping the pair as one line of evidence still holds.

### Rank 7 — the direct-versus-held-out multiple is mis-scaled

Row E8's "13× to 33× the train two-run gap" divides sub-cohort effects (n = 2,162 to 9,889)
by a whole-train-split floor (n = 45,000), while row D6 nineteen lines earlier scales the
identical floor to its sub-cohort by √(45000/n) and says so. Applying D6's own rule gives
5.7× to 15.4×. A second correction applies: 0.134 pp is a mean absolute gap, and the
document's glossary defines the floor as a standard deviation, so dividing by 0.798 shrinks
the range further to about 4.6× to 12.3×. The verdict STANDS at every version — even 4.6×
is decisive — but 33× is the number a summary carries, and it appears in three other
documents.

### Rank 8 — the standard's σ_d has no derivation and was never updated

Covered in §2.2. The document is honest that the value is provisional and names plan 0092
as the settling instrument; plan 0092 completed on 2026-09-06 and the document was never
revised, because plan 0092's own milestone M4 names only `MEASUREMENT_DESIGN.md`. Note that
the measured value is **not** a drop-in replacement: for the e199 official-test AutoAttack
design that §5.2 sizes, there is no measured σ_d of the right kind at all. That is a worse
state than "stale", and it should be written as such.

### Rank 9 — the standard's justification for parent-level inference describes the wrong quantity

`docs/MEASUREMENT_STANDARD.md:82-84` says the 0.25–0.50 pp bracket "measures only
post-resume noise" and "does not include parent-to-parent variation". Its low end is a
between-seed SD of a within-seed treatment contrast, which by construction contains the
parent-by-treatment interaction; its high end is a placebo fork on a different teacher over
160 epochs. The conclusion the sentence supports — that the unit of inference is the parent
— is strengthened, not weakened, by the correction. But §5.2 then divides that bracket by
√(replicates), treating a between-parent variance component as a within-parent replicate
floor that replication shrinks. That is a design error, not only a wording error.

### Rank 10 — the attack-randomness characterisation is about the training attack

Row G5 answers "does attack random-start choice materially move an endpoint" with a campaign
that holds the evaluation attack fixed at seed 0 and varies only `seeds.train_attack`. The
registry lists `evaluation_attack` among its declared invariant streams. The class label
(not an effect claim) survives; the description does not. The evaluation attack's
random-start effect on a scored endpoint is measured nowhere in this repository. Two further
edits follow: the row should not be listed as unaffected by the floor, since it is an
8-replicate-per-parent estimate of the same variance component the post-decay floor
measures on 4 df; and `docs/METHOD_DIRECTIONS.md:522-523` and `docs/README.md:120` carry
the same wrong framing.

### Rank 11 — the coefficient-coverage count is 18 of 23, not 20

The coefficient audit's own §3.3 inventory marks 18 rows unknown and records an alternative
trained value for 5. Its summary sentences say 20 and 3. Row #6 (`advkd_multiplier` 0.5) is
decisively not single-point — 1.0, 0.5 and 0.0 were trained on the same cohort. Row #5 is
design-matched but outcome-confounded, and "quasi-covered" is the defensible label. The
substantive conclusion is untouched: every coefficient behind DPM, D-BDD, OS-PMP, OS-DBDP,
SBF and TPFM is genuinely single-point. Two downstream restatements carry a VERIFIED tag and
are false as written: `docs/EVIDENCE_RECLASSIFICATION.md:124` and `docs/RED_TEAM.md:177`,
whose "never varied" is flatly contradicted by the audit's own row 6.

### Rank 12 — defects in two live plans

These are the only findings that touch work that has not finished.

**Plan 0096 (running).** Its dose argument states that each allocation arm treats 19,459
images, 43% of the training set. The frozen masks treat 15,085 to 15,343 images — 34.0% of
45,000, i.e. 66% fewer than I100, not 57% fewer. The 19,459 is dev-1's full S1 at epoch 100,
from a different campaign under a different late augmentation policy, and dev-1 is not a
parent of this campaign. The plan's own progress log has the right number twice. The
argument the sentence supports is strengthened by the correction. Two more: the design
section still describes 48 forks with two replicates paired on a shared `continuation_seed`
and prices them at 123 GPU-hours, while the runner executes 24 single runs with no
`continuation_seed` key at all, measured at 2 h 16 m each — about 55 GPU-hours; and the
fallback threshold of 0.25 pp is described as "the widest end of the pre-registered
post-decay bracket" when it is the narrowest and therefore the most permissive choice
available.

**Plan 0095 (not launched, and this is fixable before it runs).** The design promises that
the placebo comparison "reproduces the historical single-stream design", then puts the
untreated control on `continuation_seed = 4` while the placebo arms sit on seeds 1 and 2.
Every term of the preregistered statistic is therefore a different-stream contrast — the
exact kind the plan was written to reject. Because all three placebo draws are differenced
against one control run per parent, the control's stream offset is a common additive bias
that does not shrink with the number of draws, and its scale (0.092 pp) is the same order as
the plan's own 0.15 pp decision boundary. The fix is one line: give the control
`continuation_seed = 1` on dev-1 and `2` on dev-2, which also reproduces plan 0092's
existing replicates as a free check. The spread of the three placebo draws about their own
mean is unaffected and still yields the single-stream floor.

### Rank 13 — footnotes worth fixing when the files are next touched

- The status summary's next-step item 1 proposes a 2.2 GPU-h measurement that ran on
  2026-09-06 (plan 0092). Do not simply delete it: a prerequisite genuinely remains, but it
  is plan 0095's different design at a coincidentally identical price. Matching on the price
  would repeat the wrong-kind error.
- `docs/ERT_RESEARCH_STATUS_SUMMARY.md:65` states "1–2 pp" RNG divergence with no split,
  attack, horizon or design label. The figure is correct pre-decay at e84–94. It has already
  been consumed post-decay at e114 by plan 0087, and — more seriously — it is hard-coded as
  a standing instruction at `.claude/skills/experiment-decide/SKILL.md:26`, which would rule
  out every post-decay effect on record, including the confirmed official-test result.
- `docs/README.md:59` describes the disagreement between the rescue-subtypes and
  reliability-stratified reports as "at the third decimal". The largest tabulated gap is
  0.0894 absolute, 28% relative. The cause is not attack randomness: the two records use
  different feature epochs (84 versus 79) on identical sample sets, which
  `docs/plans/0045-...:69-71` already resolves.
- The arm registry lists per-seed counts for `PF_R` / `NR_R` as a coverage gap; the counts
  are in the document its own rows cite. The real residual gap is that the registry's sweep
  covered `docs/*.md` reports but not `docs/plans/*.md` or run artifacts, which hides two
  completed negative screens (PrescriptiveV3, legacy H3) that the reclassification has no
  row for.

### What did not change

89 candidate defects were examined and dismissed. The great majority were technically true
and consequence-free: rounding conventions in blocks-needed cells, stale line-number
citations, an arm alias missing from one column, a bucket count off by one, a percentage
interval that excluded one seed by 0.1 pp. Several were wrong outright — most importantly,
the claim that the pre-decay floor in Group C is of the wrong kind (it is not; the
placebo-fork family independently reproduces 1.03–1.20 pp on the same arms), and the claim
that an order-only floor of 0.300 pp should replace 0.092 pp for within-campaign screens
(it should not; the screens vary neither stream). Three were rejected because their proposed
replacement number was itself a wrong-kind substitution. The pattern is worth recording: in
this corpus, the arithmetic is reliable and the labels are not.

---

## 4. What cannot be settled without new experiments

Ranked by what each unblocks per GPU-hour.

| # | Question | Cheapest experiment that settles it | Cost | What it unblocks |
|---|---|---|---|---|
| 1 | What is the floor for a shared-stream, post-decay, treated-versus-control contrast? | Plan 0095 as designed, with the control moved onto the placebo arms' `continuation_seed` | 2.9 GPU-h | All nine epoch-114 contrasts; the entire state-conditional family; the two withdrawn "passes" |
| 2 | What is the untreated-fork floor at e199? | Evaluate plan 0093's twelve control forks at six horizons — evaluation only, no new training | ~0 GPU-h beyond a campaign already scoped | The margin on the two augmentation-switch results in §1.1, which currently borrow a floor from a different teacher |
| 3 | How much of the 0.092 pp figure is specific to the attack stream? | Six further forks re-seeding `data_order` as well as the attack stream | 2.2 GPU-h | Whether the measured post-decay floor generalises at all |
| 4 | Does the evaluation attack's random start move a scored endpoint? | Replay one saved checkpoint under several evaluation seeds | minutes, evaluation only | Row G5's actual question, and the reproducibility premise every endpoint rests on |
| 5 | Is there any floor for official-test AutoAttack at e199? | AutoAttack on plan 0093's e199 control forks, plus two more seeds | ~2 h of AutoAttack per cell | The margin on the project's flagship confirmation |
| 6 | Are any of the 18 single-point coefficients at a special value? | One neighbour arm at each edge of the bootstrap band, post-decay, for the arms that matter | ~1 day per arm | Removes COEFFICIENT-UNTESTED from the rows that carry it |
| 7 | Do CropShift-throughout and CropShift→CROP_RE differ from each other? | Run the missing cells: CropShift-throughout on confirm-a/b/c, or CROP_SUFFIX on dev-1/dev-2 | 3 × ~4.6 GPU-h | Turns the 0.62 pp group-mean gap into a measured contrast; currently it is an artefact of which seeds ran which arm |
| 8 | Is the epoch-114 sign pattern real? | More control runs per parent at e114, not more treated arms | included in #1 | 18 of 18 positive against two control runs is currently confounded with one control draw |

Items 1 and 2 are prerequisites, not refinements. Nothing in the sample-level family can be
judged before item 1, and the margins in §1.1 rows E3 and E4 cannot be stated honestly
before item 2.

---

## 5. Correction manifest

Grouped by defect, not by file, so that each is fixed once and everywhere.

**Pooled `+1.612 pp` (rank 1)** — `ERT_RESEARCH_STATUS_SUMMARY.md:14`;
`EVIDENCE_RECLASSIFICATION.md:152, 230, 267`; `MEASUREMENT_DESIGN.md:302, 691`;
`ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md:10, 98`; `NUMERIC_CONSISTENCY_AUDIT.md:31`;
records `ert_rslad_five_seed_global_stochasticity_v1.json` / `_v2.json` (comparison label)
and `ert_rslad_five_seed_artifact_inventory_v1.json` (arm label).

**0.72 pp floor (rank 2)** — `MEASUREMENT_DESIGN.md:296, 393, 445, 685, 754`;
`EVIDENCE_RECLASSIFICATION.md:61, 152, 206`; `METHOD_DIRECTIONS_V2.md:376, 468`;
`ERT_RESEARCH_STATUS_SUMMARY.md:67`.

**Wrong-kind post-decay threshold (rank 3)** — `POST_DECAY_FLOOR.md:64` via
`scripts/aggregate_post_decay_floor.py:471-474`; `POST_DECAY_FLOOR_RECLASSIFICATION.md:29-43`;
`docs/decisions/0002-post-decay-floor-consequences.md:95-100`;
`METHOD_DIRECTIONS.md:160, 211, 240-242, 353`; `METHOD_DIRECTIONS_V2.md:122`; `RED_TEAM.md:101`.

**Row A2's stale cell (rank 4)** — `EVIDENCE_RECLASSIFICATION.md:153` and the sentence at
`:288`; the 0.25 pp value also stands uncaveated at `EVIDENCE_RECLASSIFICATION.md:58`,
`MEASUREMENT_DESIGN.md:398`, and as a live threshold at `docs/plans/0096-...:139` and
`docs/decisions/0006-...:10, 100`.

**Identical numbers ≠ reuse (rank 5)** — `ARM_REGISTRY.md:274-278` and `:347-352`;
`EVIDENCE_RECLASSIFICATION.md:189`.

**S3×T3 coefficient (rank 6)** — `EVIDENCE_RECLASSIFICATION.md:173-174, 180`;
`ARM_REGISTRY.md:147-150, 250`; `ard_arm_registry_v1.json` arms[8] and arms[16].

**E8 scaling (rank 7)** — `EVIDENCE_RECLASSIFICATION.md:213, 267`;
`ERT_RESEARCH_STATUS_SUMMARY.md:30`; `RED_TEAM.md:157` (whose VERIFIED tag does not hold).

**σ_d = 0.35 and the bracket's description (ranks 8, 9)** —
`MEASUREMENT_STANDARD.md:82-84, 158-167, 171-174, 261-266`.

**Training-versus-evaluation attack (rank 10)** — `EVIDENCE_RECLASSIFICATION.md:233`;
`METHOD_DIRECTIONS.md:522-523`; `README.md:120`.

**18 of 23 (rank 11)** — `COEFFICIENT_AUDIT.md:52-53, 499, 764-765`;
`EVIDENCE_RECLASSIFICATION.md:124`; `RED_TEAM.md:177`.

**Live plans (rank 12)** — `docs/plans/0096-hardness-allocation-direction.md:83, 88-89,
103-105, 136-140, 159-163, 180`; `docs/plans/0095-cohort-placebo-floor.md:65-74` and the
duplicate specification at `METHOD_DIRECTIONS.md:285-291`.

**Regime labels and stale next steps (rank 13)** — `ERT_RESEARCH_STATUS_SUMMARY.md:43-47,
65, 67-68, 113, 117`; `.claude/skills/experiment-decide/SKILL.md:26`;
`ARM_REGISTRY.md:439-443`; `README.md:59`; `CONSOLIDATION_LOG.md:43-52`;
`ERT_CLEAN_WRONG_RESCUE_SUBTYPES.md:5-23`.

---

## 6. Limits of this audit

Stated so the next reader knows what was not checked.

- Several records have no `.sha256` sidecar, including `ard_arm_registry_v1.json` and the
  four augmentation-family result records. "Hash-bound" does not literally hold for them.
- The augmentation-family run directory (`CROP_RE`, `IDBH_WEAK`) is not present on this
  host, so those arms rest on their record alone.
- The five-seed per-seed endpoint values live in a working cache under `.cache/analysis/`,
  which is not tracked and not hash-bound. It reproduces the published arm means and its
  provenance is pinned by per-cell parquet hashes in the inventory record, but no checkpoint
  was re-evaluated.
- Rows B2 and B3 have no level-1 record and no reachable run artifacts; their values are
  corroborated only by two prose documents that agree with each other.
- The pre-decay floor's 84 pairwise comparisons were not re-derived from primary artifacts
  in this pass.
- Three operational figures remain untraceable to any artifact, as the earlier numeric audit
  also found: the 2.08 min/epoch Ferret rate and everything derived from it, "about 60
  minutes per AutoAttack job", and "8 × 5 hours = 40 GPU-h per seed".

---

## 7. The one-paragraph version

This project has four established results and one established negative. The four are:
CropShift beats canonical augmentation by about 1.2 pp; CropShift followed by
RandomErasing beats it by about 1.9 pp; switching to IDBH_WEAK at epoch 100 adds about
0.7 to 1.1 pp on top, depending on which comparator you use; and that last effect survives
on the official CIFAR-10 test set under AutoAttack at +0.31 pp on three seeds, all positive,
under a preregistered rule. The negative is larger and more useful than any of them: every
sample-level intervention this project tried produced a big, reproducible effect on the
samples it treated and no held-out effect the apparatus can resolve. Everything else in the
sample-level family — nine contrasts, six mechanisms, four months — has not been refuted and
has not been supported, because the floor that would judge it has never been measured. That
measurement costs 2.9 GPU-hours and is the only thing standing between this project and an
answer.
