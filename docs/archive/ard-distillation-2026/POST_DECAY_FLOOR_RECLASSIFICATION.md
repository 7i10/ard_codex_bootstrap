# Re-judging the e114 evidence against the measured floor

`docs/EVIDENCE_RECLASSIFICATION.md` judged its post-decay rows against a bracket of 0.25 to 0.50 pp with **0.40 pp** as the working value, because the floor had never been measured. Plan 0092 measured it: **sigma_d = 0.092 pp** at epoch 114 (`docs/experiments/ard_post_decay_floor_v1.json`). This re-judges everything that measurement legitimately covers.

## Scope: five rows, not forty

The table has 38 rows. 21 are pre-decay and were already judged against the separately measured pre-decay floor of 1.14 to 1.25 pp, which this does not touch. Of the 17 post-decay rows, only **5 rows covering 9 arm contrasts** are measured at epoch 114 on the held-out CE-PGD20 endpoint of the I100 design -- the one thing plan 0092 measured.

The other twelve are out of scope, and saying why matters more than the count:

| row | what it measures | why the new floor does not apply |
| --- | --- | --- |
| A1 | held-out CE-PGD20 at e199 | a different horizon, 85 epochs past the ones measured |
| A2 | held-out CE-PGD20 at e199 | same |
| A3 | held-out CE-PGD20 at e199 | same |
| A4 | last robust at e199 and trajectory AUC | different horizon; the AUC floor has never been measured at all |
| B2 | best AutoAttack over a full 200-epoch run | different endpoint and horizon; judged against a floor from its own campaign's random arms |
| B3 | best validation CE-PGD20, e40 to e199 | different horizon; judged against a floor from placebo forks in its own campaign |
| B7 | held-out CE-PGD20 at e199 | a different horizon |
| B8 | descriptor-to-probe-AUC association | not an accuracy claim |
| E3 | finite versus non-finite training loss | not an accuracy claim: the observable is divergence |
| E7 | held-out CE-PGD20 at five long horizons | horizons outside the measured set, not independent of each other |
| E8 | direct train effect versus held-out effect | a within-run contrast, judged against a separate train two-run gap |

The temptation is to extrapolate to e199, especially since the floor shrinks with post-decay epochs (0.159, 0.124, 0.092 pp at e104, e109, e114). That trend makes a smaller e199 floor plausible. Plausible is not measured, and every one of those rows would move in the direction that favours a positive verdict, which is exactly when extrapolation should be refused.

## The eligible contrasts

Two thresholds are shown. The **preregistered rule** in `docs/decisions/0002-...` asks whether both seeds exceed the minimum detectable effect, 0.243 pp. The **paired test** asks whether the two-seed mean exceeds 0.181 pp, which is t(4) = 2.776 times sigma_d / sqrt(2).

| row | arm | per seed (pp) | mean (pp) | preregistered rule | paired test |
| --- | --- | --- | ---: | --- | --- |
| E1 | DPM (campaign 1 of the same arm) | +0.08 / +0.12 | +0.100 | fails | fails |
| E1 | OS-PMP (campaign 2 of the same arm) | +0.14 / +0.20 | +0.170 | fails | fails |
| E2 | D-BDD (campaign 1) | +0.04 / +0.20 | +0.120 | fails | fails |
| E2 | OS-DBDP (campaign 2) | +0.14 / +0.06 | +0.100 | fails | fails |
| E4 | SBF | +0.16 / +0.04 | +0.100 | fails | fails |
| E5 | TPFM @S2T1 | +0.22 / +0.04 | +0.130 | fails | fails |
| E6 | PILOT_S3_T1_WEAK_ADVCE | +0.22 / +0.04 | +0.130 | fails | fails |
| E6 | CLEAN_WRONG_PLAIN_ADVCE | +0.44 / +0.10 | +0.270 | fails | **passes** |
| E6 | CLEAN_WRONG_A7_MARGIN_ONLY | +0.44 / +0.08 | +0.260 | fails | **passes** |

Under the preregistered rule: **0 of 9**. Under the paired test: **2 of 9**.

## The preregistered rule was mis-specified, and this says so rather than quietly switching

The rule written into the decision packet asks each seed to exceed the minimum detectable effect. That is not a test. The minimum detectable effect is a property of the *mean* of k blocks at 80% power; applying it to each block separately is both the wrong statistic and the wrong quantity, and it is strictly more conservative than the test it was meant to stand in for.

The packet also says the rule must not change after the count is known. Both are therefore reported. **The preregistered column is the answer of record. The paired-test column is exploratory**, because the rule behind it was fixed after the numbers were visible, and anything it turns up is a candidate for a future screen rather than a finding.

## What actually changed

**No verdict changes under the preregistered rule.** A floor four times smaller does not rescue these arms, because the effects themselves are small: the largest single-seed value among the eligible contrasts is +0.44 pp and its partner seed is +0.08 pp. The verdicts were UNDERPOWERED and they remain UNDERPOWERED.

Exploratory, from the paired test only:

- **E6 CLEAN_WRONG_PLAIN_ADVCE**: +0.270 pp, t = 4.13 against a critical value of 2.78. Both are transfer arms from the Stage A lineage.
- **E6 CLEAN_WRONG_A7_MARGIN_ONLY**: +0.260 pp, t = 3.98 against a critical value of 2.78. Both are transfer arms from the Stage A lineage.

These are not results. Two seeds and a threshold chosen after the fact is the exact pattern the measurement standard exists to stop. What they are is the best-supported candidates if a screen is ever run on this question with enough blocks.

## The number that matters most here is not in the table

`PM(online)` -- plan 0093's declared primary arm -- is row E1's second campaign at +0.170 pp, against a paired-test threshold of 0.181 pp. It misses by 0.011 pp.

That is the most useful thing this exercise produces. Plan 0093 as designed, at two blocks, sits just under the line even if its effect is exactly what has been observed twice. Its power should be re-derived from the measured floor before it starts, not after.

