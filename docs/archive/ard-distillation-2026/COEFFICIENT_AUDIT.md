# Coefficient audit

Status: records-only audit. No GPU work, no training, no re-calibration, no
change to any existing file. Every number below is either read directly from a
committed artifact or recomputed from the numbers stored inside one.

## 0. How to read this document

### 0.1 Evidence labels

| label | meaning |
|---|---|
| **VERIFIED** | The statement is read directly from the cited file, or arithmetically recomputed from numbers stored in the cited file. Anyone can re-derive it without running a model. |
| **DERIVED** | A new statistic computed in this audit from numbers already stored in the cited files (for example a bootstrap standard error, or an analysis-of-variance over a table that is printed in a report). The inputs are VERIFIED; the statistic itself is new. |
| **INFERRED** | A reading of the records that is not written down anywhere and could be wrong. Always flagged. |
| **UNKNOWN** | The records do not answer the question. This is reported as a result, not filled in by guessing. |

### 0.2 Terms, defined on first use

- **Coefficient.** A single fixed number multiplying an extra loss term that is
  added to the ordinary training loss for a selected subset of samples.
- **Baseline loss.** The ordinary RSLAD distillation loss. Its adversarial part
  is `L_advKD` and carries a fixed weight of 5/6
  (`src/ard/objectives/rslad.py:15`). VERIFIED.
- **Intervention.** The extra loss term that a treatment adds. Examples in this
  project: adversarial cross-entropy (AdvCE), clean cross-entropy (CleanCE),
  pair margin (PM/DPM), detached boundary distance (D-BDD), secant boundary
  distance (S-BDD), student-boundary-floor margin (SBF), teacher-positive-floor
  margin (TPFM).
- **Gradient norm.** The length (L2 norm) of the vector of derivatives of a loss
  with respect to *all* student parameters, for one mini-batch. Implementation:
  `scripts/prepare_ert_i100_s2_rbp.py:198-201` (used by the boundary-distance
  family via the import at `scripts/prepare_ert_i100_s2_dynamic_bdd.py:25`) and
  `src/ard/analysis/ert_stage_a_calibration.py:144-162`. VERIFIED.
- **Gradient ratio.** (gradient norm of the coefficient-scaled intervention) ÷
  (gradient norm of the baseline adversarial term), for one probe batch.
- **Achieved ratio distribution.** The set of those per-batch ratios across all
  probe batches, after the chosen coefficient is applied.
- **Probe batch.** One mini-batch used only for measurement. No optimizer step,
  no scheduler step, no checkpoint write.
- **Neighbour.** A different value of the *same* coefficient, run under an
  otherwise matched design, so that the two results can be compared.
- **Block.** One matched training lineage (teacher seed × replicate), for
  example `L2-R1`.
- **pp.** Percentage points of accuracy.

### 0.3 The one-paragraph answer

The project has a single, clearly stated calibration rule, and it is genuinely
implemented — but it is a *median-pinning* rule, it was applied with three
different targets depending on the family, and it was almost never followed by a
neighbourhood check. Of the 23 calibrated scalars that entered training, 3 have
any alternative value on record. For the one coefficient that was properly swept
(the Clean-Wrong margin lambda), the sweep shows that the coefficient explains
essentially none of the outcome variance: the between-block noise is larger than
the coefficient effect, and the much-quoted "+5% flips the sign" happens in one
of four blocks and reverses direction in another. The correct status for the
other 20 coefficients is therefore **robustness unknown**, which is a different
statement from **robust**.

---

## 1. The calibration rule

### 1.1 The stated rule

`docs/plans/0079-ert-i100-s2-dynamic-boundary-distance.md:89-90` reads, exactly:

> - One pooled calibration coefficient per treatment targets median intervention
>   gradient ratio 0.25; no outcome-driven retuning or threshold sweep.

VERIFIED (the two lines exist and say this).

### 1.2 What the ratio is a ratio of, and over what sample

For the boundary-distance family the implementation is one line,
`scripts/prepare_ert_i100_s2_dynamic_bdd.py:263`:

```python
coefficients[mode] = float(0.25 * torch.median(base / norms).item())
```

Reading that line together with the measurement loop
(`scripts/prepare_ert_i100_s2_dynamic_bdd.py:212-220`):

| element | what it actually is | evidence |
|---|---|---|
| numerator of the ratio | L2 norm over all student parameters of the gradient of the batch-mean intervention loss | `prepare_ert_i100_s2_dynamic_bdd.py:217`, `prepare_ert_i100_s2_rbp.py:198-201` |
| denominator | L2 norm of the gradient of the batch-mean `5/6 × L_advKD` baseline adversarial term | `prepare_ert_i100_s2_dynamic_bdd.py:201, 214`; `rslad.py:15` |
| unit of observation | one mini-batch of 64 samples | `prepare_ert_i100_s2_dynamic_bdd.py:213` (`"n": int(batch.labels.numel())`); stored `n=64` in the artifact |
| sample | 8 deterministic class-stratified batches per development seed, 2 seeds pooled = 16 probe batches, from 512 selected samples per seed | artifact `inputs.dev-1.sample_count = 512`; `recalibrate_...py:29-30` asserts exactly 8 per seed and 16 pooled |
| checkpoint | the epoch-99 parent, viewed under the epoch-100 augmentation view | artifact `parent_epoch = 99`, `calibration_view_epoch = 100` |
| statistic pinned | the *median* over those 16 batch ratios | `prepare_ert_i100_s2_dynamic_bdd.py:263` |

VERIFIED.

The algebra is sound: the median is preserved by the map `x → 1/x` on positive
values, so `c = 0.25 × median(base/norm)` does make
`median(c × norm/base) = 0.25`. The script guards positivity at
`prepare_ert_i100_s2_dynamic_bdd.py:261-262`. VERIFIED.

Two implementation details that matter:

1. **The v1 median convention is the lower-middle order statistic.**
   `torch.median` on an even-length tensor returns the lower of the two middle
   values, so what v1 actually pinned to exactly 0.25 was the 8th of 16 sorted
   ratios, not the interpolated median. The interpolated median of the stored
   `achieved_ratios.pair_margin` is 0.24977, not 0.25. DERIVED from the stored
   list. The recalibration script fixes this deliberately and documents it:
   `scripts/recalibrate_ert_i100_s2_secant_boundary_distance.py:32-41` and the
   artifact field `median_estimator = "torch.quantile(q=0.5, interpolation=linear)"`.
   VERIFIED.
2. **The interventions are hinged, so they are inactive on many samples.** In
   the v1 probe, the number of active samples per 64-sample batch was 34–49
   (mean 40.5) for pair margin and D-BDD, and 39–50 (mean 44.0) for S-BDD.
   DERIVED from `achieved`/`*_active_count` fields in
   `docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json`. The
   ratio is therefore a whole-batch quantity that averages over inactive
   samples, not a per-active-sample quantity.

### 1.3 Was the same rule used for the other families?

No. Each family has its own entry point, its own probe, its own denominator, and
in two cases its own target. Summary table, all VERIFIED against the cited lines:

| family | entry point | target ratio | denominator of the ratio | probe (batches × size) | pooled over |
|---|---|---|---|---|---|
| Dynamic boundary distance (PM/DPM, D-BDD, S-BDD v1) | `scripts/prepare_ert_i100_s2_dynamic_bdd.py:263` | 0.25 | `5/6 × L_advKD` | 16 × 64 | dev-1, dev-2 |
| S-BDD v2 (recovery re-freeze) | `scripts/recalibrate_ert_i100_s2_secant_boundary_distance.py:44-47` | 0.25 | `5/6 × L_advKD` | 16 × 64 | dev-1, dev-2 |
| Robust-boundary preservation (SBF, TPFM) | `scripts/prepare_ert_i100_s2_rbp.py:424-425` | 0.25 | `5/6 × L_advKD` | 16 × 64 | dev-1, dev-2 |
| Clean-Wrong margin (beta_advce, lambda) | `src/ard/analysis/ert_cw_margin_calibration.py:218-227` | 0.25 | `5/6 × L_advKD` | 8 × 64 | L2, L4 |
| Action transfer (beta_advce, margin coefficient) | artifact field `calibration_rule` | 0.25 | `5/6 × L_advKD` | 16 × 64 | dev-1, dev-2 |
| Clean-Wrong broad screen (beta_bce) | artifact `ert_clean_wrong_bce_calibration_v1.json` | 0.25 | `5/6 × L_advKD` | **4** batches | L2, L4 |
| Stage A — weak AdvCE | `src/ard/analysis/ert_stage_a_calibration.py:378` | 0.25 | `base_adv_norm` (adversarial only) | 64 × 64 | L2, L4 × 4 cohorts |
| Stage A — **moderate** AdvCE | `src/ard/analysis/ert_stage_a_calibration.py:379` | **0.50** | `base_adv_norm` | 64 × 64 | as above |
| Stage A — weak CleanCE | `src/ard/analysis/ert_stage_a_calibration.py:380` | 0.25 | **`base_total_norm`** (adversarial + clean) | 64 × 64 | as above |
| Stage A — `alpha_soft` | `src/ard/analysis/ert_stage_a_calibration.py:369` | **1.00** (implicit; it is the raw median of the reciprocal ratio) | `base_adv_norm` | 64 × 64 | as above |
| Confirmatory T1/T2/T3 | `src/ard/analysis/ert_confirmatory_calibration.py:30-34` | **none — the value 0.075 is preregistered and only sanity-checked** | `base_adv_norm` | reuses the 64 Stage A measurements | L2, L4 |

Consequences worth stating plainly:

- **"Ratio 0.25" does not mean the same thing across families.** Stage A's
  CleanCE ratio uses a *different denominator* (the full baseline, adversarial
  plus clean) from every other coefficient in the project. VERIFIED at
  `ert_stage_a_calibration.py:366, 380`.
- **Stage A's "moderate" arm is a deliberate 2× design point, not a second
  calibration.** `beta_advce_moderate` is literally `0.50 * advce_scale` on the
  same measured scale, i.e. exactly twice `beta_advce_weak`
  (`ert_stage_a_calibration.py:378-379`). VERIFIED.
- **The Confirmatory screen did not calibrate at all.** It froze the rounded
  value 0.075 and refuses any other value
  (`src/ard/analysis/ert_confirmatory_calibration.py:33-34`), then measured what
  ratio that produced. The plan says this in words:
  `docs/plans/0039-ert-confirmatory-t123.md:33-34` — "Prior calibration measured
  precise coefficients, but this experiment explicitly freezes rounded
  `beta_advce=0.075`". VERIFIED. This is the project's own precedent for
  rounding, and section 5 uses it.
- **Three coefficients are not gradient-calibrated at all — they are quantiles
  of a margin distribution.** The Clean-Wrong margin target floor/cap are the
  q25/q75 of the pooled positive teacher margins
  (`ert_cw_margin_calibration.py:63-66, 147, 176`), the TPFM floor/cap likewise
  (`scripts/prepare_ert_i100_s2_rbp.py:401-405`), and the SBF floor is the
  student q10 boundary taken from the mask file
  (`scripts/prepare_ert_i100_s2_rbp.py:304-306`, definition at
  `scripts/prepare_ert_i100_s2_rbp.py:68-79`). VERIFIED.
- **SBF is not actually a single pooled coefficient.** Its coefficient is pooled
  but its floor is per-seed: 0.04177670180797577 for dev-1 and
  0.03347739577293396 for dev-2, a 25% difference in the target the two seeds
  were trained against. VERIFIED from
  `docs/experiments/ert_rslad_i100_s2_rbp_calibration_v1.json`
  (`sbf.floor_by_run`).

### 1.4 Does the calibration control what it claims to control?

Partly. It controls one thing well and three things not at all.

| claim | status | evidence |
|---|---|---|
| "The intervention's gradient magnitude is a quarter of the baseline's, on the median probe batch, at the parent checkpoint." | **Controlled.** This is exactly what the solve does. | section 1.2 |
| "...on a typical batch." | **Not controlled.** Only the median is pinned. See section 2 for the spread. | section 2 |
| "...during training." | **Not controlled, and not observed.** The ratio is measured once, at the parent epoch, and frozen for the whole 5–15 epoch continuation. No in-training re-measurement exists. The Clean-Wrong RNG diagnostic states the point directly: "Existing epoch logs contain loss/accuracy/LR only; margin target, hinge, and regime quantities were not logged during training" (`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:12`). | VERIFIED |
| "...therefore the intervention has a quarter of the baseline's influence." | **Not controlled.** A ratio of gradient *norms* says nothing about direction. If the intervention gradient points partly along the baseline gradient, some of it is a rescaling of the baseline rather than a new force. Only Stage A measured this: the median cosine between the AdvCE gradient and the baseline adversarial gradient is 0.7658, with a range of 0.4186 to 0.9248 (`.cache/analysis/ert-stage-a-calibration-v1.json`, `summaries.adv_ce_cosine`). No cosine was recorded for the Clean-Wrong, action-transfer, RBP, or boundary-distance families. | VERIFIED / UNKNOWN for the other families |

---

## 2. What the calibration actually achieved

### 2.1 Full achieved-ratio distributions

The rule pins a median. Below is the whole distribution for every calibrated
coefficient the project trained with. Where the artifact stores the achieved
ratios or the per-batch measurements, the quartiles are exact; the source column
says which.

All values VERIFIED (stored) or DERIVED (recomputed from stored per-batch
measurements using the artifact's own formula).

| coefficient | frozen value | target | n | min | Q1 | median | Q3 | max | IQR | max/min | source of distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage A `alpha_soft` | 1.2522921562194824 | 1.00 | 64 | 0.6749 | 0.8622 | 0.9955 | 1.0673 | 1.2096 | 0.2050 | 1.79× | stored `summaries.achieved_soft_ratio` |
| Stage A `beta_advce_weak` | 0.07095924764871597 | 0.25 | 64 | 0.1968 | 0.2253 | 0.2494 | 0.2658 | 0.2914 | 0.0405 | 1.48× | stored `summaries.achieved_weak_advce_ratio` |
| Stage A `beta_advce_moderate` | 0.14191849529743195 | 0.50 | 64 | 0.3935 | 0.4506 | 0.4988 | 0.5315 | 0.5828 | 0.0809 | 1.48× | stored `summaries.achieved_moderate_advce_ratio` |
| Stage A `beta_cleance_weak` | 0.07825280725955963 | 0.25 | 64 | 0.1354 | 0.2010 | 0.2490 | 0.2729 | 0.3242 | 0.0719 | 2.40× | stored `summaries.achieved_cleance_ratio` |
| Confirmatory `beta_advce` | 0.075 (preregistered) | none | 64 | 0.2080 | 0.2381 | **0.2636** | 0.2809 | 0.3080 | 0.0428 | 1.48× | min/median/max stored; Q1/Q3 DERIVED from the Stage A measurements the sanity check reuses |
| Clean-Wrong broad `beta_bce` | 0.08891977369785309 | 0.25 | **4** | 0.2268 | 0.2334 | 0.2509 | 0.2686 | 0.2754 | 0.0352 | 1.21× | DERIVED from stored `batch_ratios` |
| Clean-Wrong `beta_advce` | 0.07726448029279709 | 0.25 | 8 | 0.2421 | 0.2450 | 0.2492 | 0.2583 | 0.2649 | 0.0133 | 1.09× | DERIVED from stored `measurements` |
| **Clean-Wrong margin lambda** | 0.2388051152229309 | 0.25 | 8 | 0.2302 | 0.2457 | 0.2489 | 0.2675 | 0.2699 | 0.0218 | 1.17× | DERIVED from stored `measurements` |
| Action-transfer `beta_advce` | 0.11834514302628477 | 0.25 | 16 | 0.2069 | 0.2276 | 0.2506 | 0.2728 | 0.3028 | 0.0452 | 1.46× | DERIVED from stored `measurements` |
| Action-transfer margin coefficient | 0.316427398202933 | 0.25 | 16 | 0.1991 | 0.2274 | 0.2500 | 0.2707 | 0.2928 | 0.0433 | 1.47× | DERIVED from stored `measurements` |
| RBP `sbf` coefficient | 0.23594490117507805 | 0.25 | 16 | 0.2238 | 0.2400 | 0.2487 | 0.2704 | 0.2880 | 0.0304 | 1.29× | stored `achieved_ratios.sbf` |
| RBP `tpfm` coefficient | 0.16676844691071563 | 0.25 | 16 | 0.2050 | 0.2272 | 0.2467 | 0.2569 | 0.2718 | 0.0297 | 1.33× | stored `achieved_ratios.tpfm` |
| **Pair margin (PM / DPM / OS-PMP)** | **0.05380932585058825** | 0.25 | 16 | **0.1508** | **0.2175** | **0.2498** | **0.2591** | **0.3249** | **0.0416** | **2.15×** | stored `achieved_ratios.pair_margin` |
| **Detached boundary distance (D-BDD / OS-DBDP)** | **31.649566509850324** | 0.25 | 16 | **0.1825** | **0.1980** | **0.2468** | **0.2700** | **0.3817** | **0.0721** | **2.09×** | stored `achieved_ratios.detached_boundary_distance` |
| Secant boundary distance v1 | 3.487387518680544 | 0.25 | 16 | 0.0979 | 0.1742 | 0.2276 | 0.7464 | **12.373** | 0.5723 | **126×** | stored `achieved_ratios.secant_boundary_distance` |
| Secant boundary distance v2 | 1.5219638832872224 | 0.25 | 16 | 0.0519 | 0.1498 | 0.2500 | 1.7120 | **20.860** | 1.5622 | **402×** | stored `achieved_ratio_summary` + `achieved_ratios` |

Artifact paths for the table above:
`docs/experiments/ert_stage_a_calibration_v1.json` (summary only) with the full
distribution in `.cache/analysis/ert-stage-a-calibration-v1.json`;
`docs/experiments/ert_confirmatory_t123_calibration_sanity_v1.json`;
`docs/experiments/ert_clean_wrong_bce_calibration_v1.json`;
`docs/experiments/ert_cw_margin_calibration_v1.json`;
`docs/experiments/ert_rslad_i100_action_transfer_calibration_v1.json`;
`docs/experiments/ert_rslad_i100_s2_rbp_calibration_v1.json`;
`docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json`;
`docs/experiments/ert_rslad_i100_s2_secant_boundary_distance_calibration_v2.json`.

### 2.2 What is stored and what is not

| artifact | stores per-batch measurements? | stores achieved-ratio list? | stores a summary? |
|---|---|---|---|
| dynamic BDD v1 | yes (16 rows) | yes, all three modes | no |
| S-BDD v2 | yes (16 rows) | yes | yes (min/median/max/IQR only — no Q1/Q3) |
| RBP v1 | yes (16 rows) | yes, both modes | no |
| action transfer v1 | yes (16 rows, `*_ratio_at_*` fields) | only the two medians | no |
| CW margin v1 | yes (8 rows of raw norms) | no | no; stores margin quantiles instead |
| CW broad BCE v1 | only 4 pooled `batch_ratios` scalars | no | median only |
| Stage A v1 (tracked manifest) | **no** | no | only the four achieved medians |
| Stage A v1 (`.cache` full artifact) | yes (64 rows) | no | yes — full min/Q1/median/Q3/max/IQR for all four |
| Confirmatory sanity v1 | no | no | median, min, max only — **no Q1/Q3, no IQR** |

Two records gaps are stated rather than filled:

- The **tracked** Stage A manifest (`docs/experiments/ert_stage_a_calibration_v1.json`)
  carries only medians. The full distribution exists only in the untracked
  `.cache/analysis/ert-stage-a-calibration-v1.json`, whose hash is pinned in the
  manifest (`full_artifact.sha256`). If that cache is lost, the Stage A spread is
  unrecoverable from the repository. VERIFIED.
- The Confirmatory sanity artifact stores no quartiles
  (`ert_confirmatory_t123_calibration_sanity_v1.json` has `ratio_min` and
  `ratio_max` only). The Q1/Q3 in the table above are DERIVED by re-running the
  artifact's own formula (`ert_confirmatory_calibration.py:51`) over the Stage A
  measurement rows it cites by hash. VERIFIED as reproducible, but not stored.

### 2.3 The key question: is the pair-margin coefficient heavy-tailed?

**No. It is tight.** This is the headline finding of section 2.

| coefficient | max/min of achieved ratios | IQR | verdict |
|---|---:|---:|---|
| Pair margin `0.05380932585058825` | **2.15×** | **0.0416** | tight |
| Detached boundary distance `31.649566509850324` | **2.09×** | **0.0721** | tight |
| Clean-Wrong margin lambda `0.2388051152229309` | 1.17× | 0.0218 | tight |
| Stage A weak AdvCE `0.07095924764871597` | 1.48× | 0.0405 | tight |
| Stage A weak CleanCE `0.07825280725955963` | 2.40× | 0.0719 | tight (widest of the tight group) |
| RBP SBF / TPFM | 1.29× / 1.33× | 0.030 / 0.030 | tight |
| Secant boundary distance **v1** | **126×** | 0.572 | heavy-tailed |
| Secant boundary distance **v2** | **402×** | 1.562 | heavy-tailed |

VERIFIED from the stored achieved-ratio lists.

The pathology is confined to the secant formulation. Every pair-margin ratio in
the probe lies between 0.15 and 0.33; every detached-boundary-distance ratio lies
between 0.18 and 0.39. Neither has a tail. The two secant variants are the only
ones where a single probe batch sits two orders of magnitude above the target,
and both of them are the ones that later blew up in training: the corrected v2
runs went non-finite at epoch 106 (dev-1) and stopped after epoch 101 (dev-2),
recorded at `docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:86-87` and
`docs/ERT_RSLAD_I100_SECANT_BOUNDARY_DISTANCE_FORENSIC.md:73-74`. VERIFIED.

`docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:89` records the v2 spread
0.05195–20.86 with IQR 1.562 and calls it "a pre-training warning sign". That
reading is correct and is confirmed here. The important extension is the
*contrast*: the spread was a warning sign specific to S-BDD, and the records
contain everything needed to have seen it before launch, because the v1 artifact
already showed S-BDD at 126× while its two siblings were at ~2×
(`ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json`, all three lists in the same
file). VERIFIED.

So: the DPM and D-BDD results are not the product of a wild coefficient. They are
the product of a well-behaved coefficient whose *neighbourhood was never tested*
— which is section 3.

---

## 3. Neighbourhood coverage

### 3.1 Definition

A coefficient has "neighbour coverage" if some **other** value of that same
coefficient was actually trained under an otherwise matched design. Running the
same value again in a different experiment is reuse, not coverage. Running a
different *mechanism* is a different treatment, not a neighbour.

### 3.2 The one properly swept coefficient: Clean-Wrong margin lambda

Two sweeps exist.

**Coarse sweep** — `docs/ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md:7-36`, lambda ∈ {0,
0.10, 0.238805 (calibrated), 0.25, 0.50}, two teacher seeds, three horizons.
VERIFIED.

**Local sweep** — `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`, lambda at
the calibrated value and ±5% and ±10%, four matched blocks. VERIFIED. The floor
and cap of the margin target were held fixed across every arm
(`docs/plans/0053-ert-cw-margin-lambda-sensitivity.md:11-12`,
`docs/plans/0054-ert-cw-margin-local-lambda-stability.md:12-13`), so lambda is
the only thing that varies. VERIFIED.

#### 3.2.1 Confirming the L2-R1 sign flip

The claimed epoch-94 held-out robust deltas for block L2-R1 are confirmed exactly
against `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:14-18`:

| arm | lambda | offset | L2-R1 epoch-94 held-out robust Δ |
|---|---:|---:|---:|
| N90 | 0.214924604 | −10% | **+1.880 pp** |
| N95 | 0.226864859 | −5% | **+1.600 pp** |
| A100 | 0.238805115 | calibrated | **+1.340 pp** |
| N105 | 0.250745371 | +5% | **−0.560 pp** |
| N110 | 0.262685627 | +10% | **−1.160 pp** |

VERIFIED. Within L2-R1 the response is monotone decreasing across all five arms
and the sign flips between the calibrated value and +5%.

#### 3.2.2 Does the flip replicate? No.

All four blocks, epoch-94 held-out robust delta, from
`docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`. VERIFIED.

| block | N90 (−10%) | N95 (−5%) | A100 (cal.) | N105 (+5%) | N110 (+10%) | monotone decreasing? | sign flips at +5%? |
|---|---:|---:|---:|---:|---:|---|---|
| L2-R1 | +1.880 | +1.600 | +1.340 | −0.560 | −1.160 | **yes** | **yes** |
| L2-R2 | −0.580 | −0.460 | −0.200 | −1.020 | −0.860 | no | no — already negative at the calibrated value |
| L4-R1 | +1.100 | +0.720 | +0.660 | **+1.760** | +0.780 | no | no — +5% is the **best** arm in this block |
| L4-R2 | −0.080 | −1.340 | −0.060 | −0.120 | −1.620 | no | no — all negative, non-monotone |

The step from the calibrated value to +5% is −1.90 pp in L2-R1, −0.82 pp in
L2-R2, **+1.10 pp** in L4-R1, and −0.06 pp in L4-R2. DERIVED (subtraction of two
printed cells). The step changes sign across blocks. The monotone ladder exists in
exactly one block out of four.

At the two earlier horizons the same step is +0.36, +0.52, −1.54, +1.42 pp at
epoch 89 and −0.10, +0.52, −0.54, +0.02 pp at epoch 84. DERIVED from
`docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36`. No stable direction.

#### 3.2.3 How much of the outcome does lambda explain? Essentially none.

DERIVED in this audit, from the 20 printed epoch-94 cells, with a two-way
(block + arm) decomposition and a permutation test that shuffles arm labels
within each block:

| source | sum of squares | df | mean square |
|---|---:|---:|---:|
| block | 10.824 | 3 | 3.608 |
| arm (lambda) | 4.057 | 4 | 1.014 |
| residual | 7.191 | 12 | 0.599 |

F(4,12) = 1.69; permutation p ≈ 0.19 (200,000 permutations, seed 1). The spread of
one arm across the four blocks (standard deviation 1.077 pp) is the same as the
spread of all twenty cells (1.078 pp). DERIVED.

The block term is more than twice the lambda term. Under label permutation, a
lambda effect this large appears about one time in five by chance. **Lambda is
not distinguishable from block noise at this sample size.**

#### 3.2.4 An independent measurement of the same noise floor

`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:27-34` reports the R1-versus-R2
gap at the *same* lambda — pure continuation-RNG variation, no coefficient change.
VERIFIED:

| teacher | epoch | base gap | N95 gap | A100 gap | N105 gap |
|---|---:|---:|---:|---:|---:|
| L2 | 94 | 1.880 pp | 2.060 pp | 1.540 pp | 0.460 pp |
| L4 | 94 | 1.820 pp | 2.060 pp | 0.720 pp | 1.880 pp |

Changing nothing at all moves the epoch-94 held-out robust effect by up to 2.06
pp. The "+5% flips the sign" step in L2-R1 is 1.90 pp. **The coefficient step and
the do-nothing step are the same size.** This is the cleanest single fact in this
audit.

#### 3.2.5 A second, unplanned neighbour pair — and a second noise measurement

The Confirmatory T1/T2/T3 screen froze `beta_advce = 0.075`, which is +5.7% above
Stage A's `beta_advce_weak = 0.07095924764871597`. Checking whether the two
screens are otherwise matched:

| element | Stage A | Confirmatory | same? |
|---|---|---|---|
| L2 epoch-79 parent | `ad43d72d…` (`ert_stage_a_calibration_v1.json` `inputs.L2.checkpoint_sha256`) | `ad43d72d…` (`docs/ERT_CONFIRMATORY_T123_RESULTS.md:38`) | yes |
| L4 epoch-79 parent | `026a36d3…` | `026a36d3…` (`docs/ERT_CONFIRMATORY_T123_RESULTS.md:39`) | yes |
| S3×T1 selected cohort | n = 9,889 (L2) / 9,368 (L4) | n = 9,889 / 9,368 (`docs/ERT_CONFIRMATORY_T123_RESULTS.md:79, 88`) | yes |
| mask file | `ert-state-overlay-v1-review/anchor79-fixed-masks-L2.json` | `ert-state-overlay-v1/anchor79-fixed-masks-L2.json` | different files, **byte-identical selected-ID sets** in all nine masks; only `analysis_provenance` differs. VERIFIED by direct comparison. |
| validation split identity | `16ec66fb…` | `16ec66fb…` | yes (both artifacts) |
| endpoint attack | CE-PGD20, identical identity block | identical | yes |
| training runtime | `src/ard/analysis/ert_stage_a_runtime.py` | same module reused (`docs/plans/0039-ert-confirmatory-t123.md:29-30, 50`) | yes |

So at epoch 84 the two screens are a matched +5.7% coefficient contrast on the
same parents, cohorts, split and attack.

But the **controls disagree**:

| seed | Stage A `C79` held-out robust at e84 | Confirmatory `C79CONF` held-out robust at e84 | difference |
|---|---:|---:|---:|
| L2 | 46.32% | 47.26% | **+0.94 pp** |
| L4 | 44.34% | 46.12% | **+1.78 pp** |

The Stage A figures are DERIVED by aggregating the per-class held-out counts in
`docs/experiments/ert_stage_a_effect_decomposition_v1.json`; the aggregation is
validated because it reproduces the published deltas exactly (ST1W L2:
0.4782 − 0.4632 = +1.500 pp, matching
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52`; ST2W L2: +1.420 pp, matching line
55). The Confirmatory figures are read directly from
`docs/ERT_CONFIRMATORY_T123_RESULTS.md:59, 62`. The two control checkpoints are
different objects: Stage A L2 `C79` is `48cc1458…`, Confirmatory L2 `C79CONF`
epoch-84 is `39feaa4a…`. VERIFIED.

Two nominally identical control continuations, same parent, same five epochs,
same evaluation, differ by 0.94–1.78 pp on held-out robust accuracy. That is an
independent confirmation of section 3.2.4's noise floor, obtained without any new
compute.

The treatment comparison across the +5.7% coefficient step:

| seed | arm | beta | direct robust Δ | held-out robust Δ |
|---|---|---:|---:|---:|
| L2 | Stage A `ST1W` | 0.070959 | +4.409 | +1.500 [+0.680, +2.320] |
| L2 | Confirmatory `T1WCONF` | 0.075 | +0.72 | −0.14 |
| L4 | Stage A `ST1W` | 0.070959 | +2.210 | +0.040 [−0.760, +0.860] |
| L4 | Confirmatory `T1WCONF` | 0.075 | +0.90 | −0.04 |

Sources: `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52, 69` and
`docs/ERT_CONFIRMATORY_T123_RESULTS.md:79, 88`. VERIFIED.

INFERRED reading: the L2 "ST1W is a promising seed-1 generalization signal"
conclusion (`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:89-90`) does not survive a
+5.7% coefficient change combined with a rerun of the control. On L4, where the
Stage A effect was already ~0, the two agree. This is *consistent* with the L2
effect being run-to-run noise rather than a coefficient property, but it does not
prove it, because the control itself moved by 0.94 pp and the two campaigns are
not bit-identical continuations. This comparison is not made anywhere in the
existing records.

### 3.3 Full neighbourhood-coverage inventory

Every calibrated scalar that entered a training run. "Neighbour tested" means a
different value of *this* scalar was trained under a matched design.

| # | family | coefficient | frozen value | neighbours actually trained | result at each | sign stable? | status |
|---:|---|---|---:|---|---|---|---|
| 1 | Stage A | `alpha_soft` (ST1S/ST2S) | 1.2522921562194824 | none | L2 ST1S held-out −0.240; L4 +2.180 | n/a | **unknown** |
| 2 | Stage A | `beta_advce_weak` (ST1W/ST2W/ST3K*) | 0.07095924764871597 | **yes — ×2 (`beta_advce_moderate`)**; plus the unplanned +5.7% Confirmatory pair (3.2.5) | see §4 | **no** | **covered (coarse)** |
| 3 | Stage A | `beta_advce_moderate` (ST1M/ST2M) | 0.14191849529743195 | **yes — ×0.5 (`beta_advce_weak`)** | see §4 | **no** | **covered (coarse)** |
| 4 | Stage A | `beta_cleance_weak` (CW1/CW2/CW3) | 0.07825280725955963 | none — CW1/2/3 share the identical value and differ only in the Clean-Wrong sub-mode | L2 held-out −1.300 / −1.460 / −1.440; L4 +0.440 / +0.100 / −0.580 | n/a | **unknown** |
| 5 | Confirmatory | `beta_advce` | 0.075 (preregistered, never solved) | see #2 | L2 e84 held-out −0.14; L4 −0.04 | n/a | quasi-covered |
| 6 | Confirmatory | `advkd_multiplier_t3` | 0.5 | Stage A ran 1.0 / 0.5 / 0.0 as a *design* grid (ST3K1/K05/K0), not a calibration | L2 held-out +0.680 / +0.980 / +0.300; L4 +1.820 / +1.280 / +1.700 | mixed | covered by design grid |
| 7 | CW broad screen | `beta_bce` | 0.08891977369785309 | none | see `docs/archive/ard-distillation-2026/ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md` | n/a | **unknown** |
| 8 | CW margin | `beta_advce` | 0.07726448029279709 | none | — | n/a | **unknown** |
| 9 | CW margin | **margin lambda** | 0.2388051152229309 | **yes — 0, 0.10, 0.25, 0.50 and ±5%/±10%** | §3.2 | **no** | **covered (dense)** |
| 10 | CW margin | target floor | 0.03221710026264191 | none — held fixed in every lambda arm | — | n/a | **unknown** |
| 11 | CW margin | target cap | 0.13952550292015076 | none — held fixed in every lambda arm | — | n/a | **unknown** |
| 12 | Action transfer | `beta_advce` | 0.11834514302628477 | none | `docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md:123-128` | n/a | **unknown** |
| 13 | Action transfer | margin coefficient | 0.316427398202933 | none (reused unchanged in the long-horizon and gap-completion screens) | as above | n/a | **unknown** |
| 14 | Action transfer | margin floor | 0.17963354289531708 | none | — | n/a | **unknown** |
| 15 | Action transfer | margin cap | 0.5595575273036957 | none | — | n/a | **unknown** |
| 16 | RBP | SBF coefficient | 0.23594490117507805 | none | e114 held-out +0.160 / +0.040 pp | n/a | **unknown** |
| 17 | RBP | SBF floor (per seed) | 0.04177670180797577 / 0.03347739577293396 | none — but the two seeds already differ by 25% | as above | n/a | **unknown** |
| 18 | RBP | TPFM coefficient | 0.16676844691071563 | none | e114 held-out +0.220 / +0.040 pp | n/a | **unknown** |
| 19 | RBP | TPFM floor | 0.16590790450572968 | none | — | n/a | **unknown** |
| 20 | RBP | TPFM cap | 0.32364362478256226 | none | — | n/a | **unknown** |
| 21 | Dynamic BDD / online state | **pair margin** | 0.05380932585058825 | none | DPM e114 held-out −Control: dev-1 +0.08, dev-2 +0.12 pp; OS-PMP: dev-1 +0.14, dev-2 +0.20 pp | n/a | **unknown** |
| 22 | Dynamic BDD / online state | **detached boundary distance** | 31.649566509850324 | none | D-BDD e114 held-out: dev-1 +0.04, dev-2 +0.20 pp; OS-DBDP: dev-1 +0.14, dev-2 +0.06 pp | n/a | **unknown** |
| 23 | Dynamic BDD | secant boundary distance | 3.487387518680544 (v1) → 1.5219638832872224 (v2) | **no** — the two values belong to two different formulas (`formula_version = student_parameter_graph_v2`), so they are not a coefficient pair | both non-finite in training | n/a | **unknown** |

Result sources: `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52-63, 69-80`;
`docs/ERT_CONFIRMATORY_T123_RESULTS.md:79-96`;
`docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:8`;
`docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:36-37`;
`docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:45-56`. VERIFIED.

### 3.4 The count

**3 of 23 calibrated scalars have any neighbour on record** (#2, #3, #9). Counting
distinct *coefficients* rather than arms: lambda is the only one with a designed
neighbourhood; the Stage A weak/moderate pair is a single 2× step; and #5/#6 are
quasi-neighbours from an unplanned reuse and a design grid respectively.

**20 of 23 are single-point.** Their robustness to the coefficient is **unknown**,
not established. Concretely this includes every coefficient behind the DPM,
D-BDD, OS-PMP, OS-DBDP, SBF and TPFM conclusions — the six treatments the project
currently describes as SUPPORTED, NOT_SUPPORTED, or "not a promotion signal".

Stated bluntly: the two treatments classified `SUPPORTED` in
`docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:6-9` rest on effects of
+0.14 to +0.20 pp, at a single untested coefficient, in a system whose
same-configuration rerun noise is measured at 0.94–2.06 pp (§3.2.4, §3.2.5).

---

## 4. The coarse sweep that looks like a neighbourhood

### 4.1 The Stage A 2× step

`docs/ERT_STAGE_A_TREATMENT_RESULTS.md:35-36` records:

```
beta_advce_weak      = 0.07095924764871597
beta_advce_moderate  = 0.14191849529743195
```

VERIFIED, and exactly 2× by construction
(`src/ard/analysis/ert_stage_a_calibration.py:378-379`). The two arms `ST1W` and
`ST1M` are identical in every other respect: same mask key `s3_t1_q10`, same
parent, same runtime. VERIFIED by comparing
`.cache/analysis/ert-stage-a/L2/ST1W/resolved_config.yaml` and
`.cache/analysis/ert-stage-a/L2/ST1M/resolved_config.yaml`, whose `treatment`
blocks differ only in `beta_advce`.

### 4.2 The coefficient response

| seed | arm | beta | direct robust Δ | held-out robust Δ | held-out 95% CI |
|---|---|---:|---:|---:|---|
| L2 | ST1W | 0.070959 | +4.409 | **+1.500** | [+0.680, +2.320] |
| L2 | ST1M | 0.141918 | −1.618 | **−2.380** | [−3.200, −1.520] |
| L4 | ST1W | 0.070959 | +2.210 | **+0.040** | [−0.760, +0.860] |
| L4 | ST1M | 0.141918 | +0.427 | **+0.440** | [−0.400, +1.300] |
| L2 | ST2W | 0.070959 | −0.046 | +1.420 | [+0.600, +2.280] |
| L2 | ST2M | 0.141918 | +3.654 | −0.980 | [−1.760, −0.160] |
| L4 | ST2W | 0.070959 | −0.281 | +1.320 | [+0.440, +2.100] |
| L4 | ST2M | 0.141918 | +0.327 | +1.420 | [+0.620, +2.300] |

Source: `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:52-53, 55-56, 69-70, 72-73`.
VERIFIED.

**Observation as stated in the task, confirmed:** on L2, doubling the coefficient
takes ST1 from +1.500 pp to −2.380 pp, a 3.88 pp swing with non-overlapping
sample-bootstrap intervals.

**Observation the task did not include, and which changes the reading:** on L4
the same doubling takes ST1 from +0.040 pp to +0.440 pp — no flip, and both
intervals cross zero. On ST2 the doubling flips the sign on L2 (+1.420 → −0.980)
but not on L4 (+1.320 → +1.420). DERIVED from the same table.

So the 2× coefficient response reverses sign in one seed and does nothing in the
other, for both cohorts. This is the same pattern as the lambda sweep in §3.2.2:
a clean, large, monotone-looking coefficient response in one lineage that does
not replicate in the matched second lineage. The reported CIs are explicitly
sample-level, not seed-level — the decomposition document says so at
`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:30-31` — so they do not cover the
seed-to-seed variation that is doing the work here.

### 4.3 The untested lower side

For Stage A AdvCE, **nothing below 0.07095924764871597 was ever trained.** The
tested set for this coefficient is exactly {0.070959, 0.075 (unplanned, +5.7%),
0.141918}. VERIFIED by exhaustion of §3.3.

The calibrated value is the *smallest* value ever run. The response between the
two designed points is negative on L2 (more coefficient, worse), which — if the
response were monotone, which §3.2.2 shows it is not — would point downward, into
territory nobody has visited.

Which other families have an untested lower side? All of the single-point ones,
by definition. The ones where this matters most, because the calibrated value is
the only value and the reported effect is small enough to be noise:

| family | calibrated value | anything below it tested? | anything above it tested? |
|---|---:|---|---|
| Stage A AdvCE | 0.070959 | **no** | yes (2×, and +5.7%) |
| Stage A CleanCE | 0.078253 | **no** | **no** |
| Stage A `alpha_soft` | 1.252292 | **no** | **no** |
| CW margin lambda | 0.238805 | **yes** — 0, 0.10, −5%, −10% | yes |
| CW margin `beta_advce` | 0.077264 | **no** | **no** |
| CW broad `beta_bce` | 0.088920 | **no** | **no** |
| Action transfer (both) | 0.118345 / 0.316427 | **no** | **no** |
| RBP SBF / TPFM | 0.235945 / 0.166768 | **no** | **no** |
| Pair margin | 0.053809 | **no** | **no** |
| Detached boundary distance | 31.649567 | **no** | **no** |

VERIFIED by exhaustion of §3.3. Lambda is the only coefficient in the project
with a tested lower side.

### 4.4 One genuine multi-point grid, for contrast

`ST3K1 / ST3K05 / ST3K0` set the baseline adversarial-KD multiplier to 1.0 / 0.5 /
0.0 with the AdvCE coefficient held at the weak value. VERIFIED from the three
resolved configs. This is a proper three-point response curve — but the swept
quantity is a *preregistered design grid*, not a calibrated coefficient, and the
outcome is again seed-dependent: L2 held-out +0.680 / +0.980 / +0.300, L4 +1.820 /
+1.280 / +1.700 (`docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:58-60, 75-77`). Neither
seed shows a monotone response and the two orderings disagree. VERIFIED.

---

## 5. Precision

### 5.1 What the digits are

Every coefficient is a numerical solve of the form
`c = target / median(measured ratio at coefficient 1)`. The printed 17 digits are
the double-precision representation of that division. They are exactly
reproducible from the artifact — and they carry no information about the world
beyond the first two.

The controlling quantity is the sampling error of the median in the denominator.
The relative error of the coefficient equals the relative error of that median.

### 5.2 Estimated sampling error of the median

DERIVED in this audit: 20,000-resample bootstrap of the median of the stored
per-batch ratios, for each family, using the artifact's own definition of the
ratio. Probe size `n` is the number of pooled probe batches recorded in the
artifact.

| coefficient | probe n | relative bootstrap SE of the median | 95% sampling band on the coefficient | justified significant digits |
|---|---:|---:|---|---:|
| Stage A `beta_advce_weak` | 64 | 1.6% | 0.0696 – 0.0739 (−1.9% / +4.1%) | **2** |
| Stage A `alpha_soft` | 64 | 2.2% | 1.212 – 1.309 | **2** |
| Stage A `beta_cleance_weak` | 64 | 3.5% | 0.0741 – 0.0838 | **2** |
| CW `beta_advce` | 8 | 1.8% | 0.0740 – 0.0789 | **2** |
| CW margin lambda | 8 | 3.1% | 0.2206 – 0.2473 (−7.6% / +3.6%) | **2** |
| RBP TPFM | 16 | 3.1% | 0.1602 – 0.1786 | **2** |
| RBP SBF | 16 | 3.4% | 0.2171 – 0.2442 | **2** |
| Action transfer margin | 16 | 4.7% | 0.2931 – 0.3457 | **2** |
| **Pair margin** | 16 | **4.2%** | **0.0520 – 0.0609 (−3.4% / +13.2%)** | **2** |
| Action transfer `beta_advce` | 16 | 6.1% | 0.1068 – 0.1303 | **2** |
| CW broad `beta_bce` | **4** | 5.9% | 0.0810 – 0.0984 | **2** |
| **Detached boundary distance** | 16 | **8.5%** | **29.16 – 38.99 (−7.9% / +23.2%)** | **2** |
| Secant boundary distance v1 | 16 | 75% | 1.10 – 4.53 | **1** |
| Secant boundary distance v2 | 16 | **188%** | 0.17 – 2.52 | **1, at best** |

**Nothing in this project justifies more than two significant digits.** The
secant coefficients do not justify one — the sampling band on the v2 coefficient
spans a factor of 15, which on its own is grounds to reject the secant
formulation without appealing to the training failure.

A secondary precision defect: Stage A converts its ratio lists with
`torch.tensor(ratios_soft)` (`src/ard/analysis/ert_stage_a_calibration.py:369-371`),
which defaults to float32, whereas the boundary-distance and RBP families use
explicit float64 (`prepare_ert_i100_s2_dynamic_bdd.py:257, 260`). VERIFIED. This
is far below the sampling error and changes nothing, but the printed Stage A
digits below the seventh are float32 artefacts rather than a float64 solve.

### 5.3 Is rounding 0.05380932585058825 → 0.05 defensible?

The arithmetic first. DERIVED:

- The change is −7.09%.
- The probe's median ratio-at-coefficient-1 is 4.6417, so 0.05 gives a median
  achieved ratio of **0.2321**, against 0.2498 at the frozen value. This confirms
  the "median moves from 0.25 to about 0.23" figure in the request.
- The 95% sampling band on the coefficient is 0.0520 – 0.0609. **0.05 falls just
  below it**, by about 4% of the coefficient.

So there are two separate questions with two different answers:

| question | answer |
|---|---|
| Is 0.05 inside the band that the calibration probe itself cannot distinguish from the frozen value? | **No — marginally outside.** The lower 95% bound is 0.0520. A defensible two-digit rounding of this coefficient is **0.054**, not 0.05. |
| Is 0.05 inside the band over which the *result* is known to be stable? | **Unknown until run.** No neighbouring value of the pair-margin coefficient was ever trained (§3.3 #21). There is no demonstrated stable band for this coefficient, at any width. |

The second answer is the important one. The project has exactly one measured
stable band, for lambda, and what it measured is that a ±5% band is *not* stable
in one block and *is* in another (§3.2.2) — while a 0% change is worth up to 2.06
pp on its own (§3.2.4). Extrapolating any stability band from lambda to the pair
margin would be unwarranted: they are different losses on different cohorts at
different epochs.

The honest status line for every single-point coefficient is therefore:

> Rounding to two significant digits is justified by the calibration's own
> sampling error. Whether the *result* survives that rounding is unknown,
> because no neighbouring value was ever run.

For the record, the project has already accepted exactly this kind of rounding
once: `docs/plans/0039-ert-confirmatory-t123.md:33-34` deliberately froze
`beta_advce = 0.075` in place of the solved 0.07095924764871597, a +5.7% change,
enforced in code at `src/ard/analysis/ert_confirmatory_calibration.py:33-34`. Its
measured consequence was a median achieved ratio of 0.2636 instead of 0.2494
(§2.1). That precedent supports two-digit rounding as policy; §3.2.5 is what
happened to the result.

---

## 6. Recommended rules going forward

These are records-and-design rules. None of them requires re-running anything
already completed, and none of them is a GPU recommendation.

1. **Report a working range, never a point.** Freeze a coefficient as
   `value [low, high]` where the interval is the bootstrap 95% band of the
   calibration median (§5.2), rounded to two significant digits. Example:
   pair margin `0.054 [0.052, 0.061]`. Publishing 17 digits implies a precision
   that the probe does not have; publishing the band states the precision it does
   have.

2. **Round to the precision the probe supports.** Two significant digits, always,
   for a median-pinned solve at these probe sizes. Keep the full-precision value
   in the artifact for exact reproduction; put the rounded value in prose, plans
   and configs.

3. **Every screen must include at least one neighbour arm.** Minimum: the
   calibrated value and one value at the edge of its sampling band (roughly
   ±10%). A screen with a single coefficient value cannot distinguish a method
   from a coefficient, and must be labelled **robustness untested** in its own
   results document — not merely left unmentioned.

4. **A conclusion needs the neighbour to agree in sign, in at least two blocks.**
   The lambda sweep (§3.2) and the Stage A 2× step (§4.2) both show a clean
   coefficient response in one lineage that reverses or vanishes in the matched
   second lineage. Require sign agreement across ≥2 blocks *and* across the
   neighbour arm before writing SUPPORTED.

5. **Every screen must carry a same-configuration control replicate.** Two runs
   of the identical control, differing only in continuation RNG. This project has
   measured that quantity twice — 0.94–1.78 pp (§3.2.5) and up to 2.06 pp
   (§3.2.4) — and both measurements were accidental. Make it deliberate, and
   print the treatment effect next to it. Any effect smaller than the control
   replicate gap is not reportable as an effect.

6. **Report the whole achieved-ratio distribution, not the median.** Store
   min/Q1/median/Q3/max/IQR and the full ratio list in the *tracked* artifact.
   Two artifacts currently fail this: the tracked Stage A manifest keeps only
   medians (the distribution lives in an untracked cache), and the Confirmatory
   sanity artifact keeps no quartiles (§2.2).

7. **Add a pre-launch tail gate.** Refuse to launch when the achieved-ratio
   max/min exceeds a preregistered factor — 10× would have caught S-BDD in the
   v1 artifact at 126×, before any training, while passing every other
   coefficient in the project at ≤2.4× (§2.3). Record the gate's threshold and
   the measured value in the launch contract.

8. **Record the gradient cosine alongside the norm ratio.** A norm ratio does not
   establish that the intervention is a *new* force; a gradient largely parallel
   to the baseline is a rescaling. Stage A measures this (median 0.7658); no
   other family does (§1.4). Add it to every calibration probe — it is free, the
   gradient vectors are already materialised.

9. **State the denominator in the coefficient's own name or metadata.** "Target
   ratio 0.25" currently means three different denominators and three different
   targets (0.25, 0.50, 1.00) across families (§1.3). Store
   `target`, `denominator`, `median_estimator`, and `probe_n` as explicit fields
   in every calibration artifact — the S-BDD v2 artifact already does the last
   two and should be the template.

10. **Log the achieved ratio during training, not only at the parent
    checkpoint.** The frozen ratio is a property of one checkpoint; the training
    run drifts away from it and nobody has ever observed by how much
    (`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:12`). A per-epoch scalar of
    the intervention-to-baseline gradient-norm ratio would convert this
    permanent unknown into a logged quantity at negligible cost.

11. **Re-label the existing single-point conclusions.** Twenty of twenty-three
    coefficients are untested neighbourhoods (§3.4). Their results documents
    should carry an explicit line — "coefficient robustness untested" — so that
    "we did not check" is never silently read as "we checked and it held".

---

## Appendix A. Citation index

Every file:line cited above, confirmed present at the time of this audit.

| citation | content |
|---|---|
| `docs/plans/0079-ert-i100-s2-dynamic-boundary-distance.md:89-90` | the calibration rule |
| `docs/plans/0039-ert-confirmatory-t123.md:33-34` | deliberate rounding to 0.075 |
| `docs/plans/0053-ert-cw-margin-lambda-sensitivity.md:11-12` | lambda sweep candidates, fixed floor/cap |
| `docs/plans/0054-ert-cw-margin-local-lambda-stability.md:12-13` | local sweep target definition |
| `scripts/prepare_ert_i100_s2_dynamic_bdd.py:201, 212-220, 257-263, 274-277` | BDD probe and solve |
| `scripts/recalibrate_ert_i100_s2_secant_boundary_distance.py:29-30, 32-41, 44-47` | v2 median convention and solve |
| `scripts/prepare_ert_i100_s2_rbp.py:68-79, 198-201, 304-306, 401-405, 424-425` | q10 floor, gradient norm, TPFM q25/q75, SBF/TPFM solve |
| `scripts/run_ert_i100_online_state_s2.py:52-60` | OS-PMP / OS-DBDP reuse the dynamic-BDD coefficients verbatim |
| `src/ard/analysis/ert_cw_margin_calibration.py:63-66, 147, 176, 218-227` | CW margin quantiles and solve |
| `src/ard/analysis/ert_stage_a_calibration.py:144-162, 364-371, 378-380, 384-402` | Stage A probe, solve, summaries |
| `src/ard/analysis/ert_confirmatory_calibration.py:30-34, 51` | preregistered 0.075 and its sanity ratio |
| `src/ard/objectives/rslad.py:15` | `ADVERSARIAL_COEFFICIENT = 5/6` |
| `docs/ERT_STAGE_A_CALIBRATION.md:14-25` | frozen Stage A values and achieved medians |
| `docs/ERT_STAGE_A_TREATMENT_RESULTS.md:35-36` | weak / moderate values, exactly 2× |
| `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md:30-31, 52-63, 69-80, 89-90` | held-out deltas, CIs, ST1W interpretation |
| `docs/ERT_CONFIRMATORY_T123_RESULTS.md:38-39, 59, 62, 79, 88` | parents, control endpoints, T1WCONF effects |
| `docs/ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md:7-36` | coarse lambda sweep |
| `docs/ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md:13-36` | ±5%/±10% lambda neighbourhood, four blocks |
| `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:12, 27-34` | no in-training logging; R1/R2 replicate gaps |
| `docs/ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md:36-37, 86-87, 89` | DPM/D-BDD e114 effects; S-BDD failures; the 0.05195–20.86 spread |
| `docs/ERT_RSLAD_I100_SECANT_BOUNDARY_DISTANCE_FORENSIC.md:69, 73-74` | v2 coefficient and tail, terminal evidence |
| `docs/ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md:8, 19` | SBF/TPFM e114 effects and frozen calibration |
| `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md:6-9, 45-56` | OS decisions and e114 endpoints |
| `docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md:123-135` | action-transfer conclusions |

Calibration artifacts read in full:
`docs/experiments/ert_stage_a_calibration_v1.json`,
`.cache/analysis/ert-stage-a-calibration-v1.json`,
`docs/experiments/ert_confirmatory_t123_calibration_sanity_v1.json`,
`docs/experiments/ert_clean_wrong_bce_calibration_v1.json`,
`docs/experiments/ert_cw_margin_calibration_v1.json`,
`docs/experiments/ert_rslad_i100_action_transfer_calibration_v1.json`,
`docs/experiments/ert_rslad_i100_s2_rbp_calibration_v1.json`,
`docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json`,
`docs/experiments/ert_rslad_i100_s2_secant_boundary_distance_calibration_v2.json`,
`docs/experiments/ert_rslad_i100_s2_dynamic_bdd_contract_v1.json`,
`docs/experiments/ert_stage_a_effect_decomposition_v1.json`,
`docs/experiments/ert_confirmatory_t123_results_v1.json`.

## Appendix B. What could not be resolved from the records

1. **Whether the Stage A and Confirmatory continuations are otherwise identical.**
   Parents, cohorts, validation split, attack and runtime module match (§3.2.5),
   but the two controls produced different checkpoints and differ by 0.94–1.78 pp.
   The records do not say what differs — candidate causes are the RNG stream tied
   to the run label, the total-epoch count (5 vs 15) affecting the learning-rate
   schedule, or host/kernel nondeterminism. The comparison is therefore reported
   as a strong quasi-neighbour, not a clean one. UNKNOWN.
2. **The `ST1S`/`ST2S` "strong" arm is not a third point on the AdvCE axis.** Its
   resolved config sets `beta_advce: null` and `tau: 2.0`, i.e. it is a
   teacher-target softening arm governed by `alpha_soft`, not a larger AdvCE
   coefficient. VERIFIED from the resolved configs, but this is not stated in
   `docs/archive/ard-distillation-2026/ERT_STAGE_A_TREATMENT_RESULTS.md`, and reading the arm names alone would
   suggest a weak/moderate/strong coefficient ladder that does not exist.
3. **What distinguishes `CW1`, `CW2` and `CW3`.** All three carry the identical
   `beta_cleance = 0.07825280725955963`; they differ in a `clean_wrong_mode`
   field. The mechanism behind each mode was not traced in this audit.
4. **The in-training achieved ratio, for every treatment.** Never logged
   (`docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md:12`). UNKNOWN and
   unrecoverable from existing artifacts.
5. **The gradient cosine for every family except Stage A.** Not measured.
   UNKNOWN.
6. **Whether any coefficient below the calibrated value would help.** No value
   below the calibrated point was ever trained for any coefficient except lambda.
   UNKNOWN by construction.
