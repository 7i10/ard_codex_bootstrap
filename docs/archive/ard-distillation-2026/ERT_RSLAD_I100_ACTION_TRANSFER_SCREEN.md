# ERT/RSLAD I100 historical-action transfer screen

Status: complete. This is a fixed, two-seed, 15-epoch continuation screen; no
automatic promotion or extension was performed.

## Scope and lineage

The exact I100 epoch-99 parents were continued through epochs 100--114
inclusive for four arms on each development seed:

```text
I100_CONTROL
PILOT_S3_T1_WEAK_ADVCE
CLEAN_WRONG_PLAIN_ADVCE
CLEAN_WRONG_A7_MARGIN_ONLY
```

All 8 training jobs and all 8 endpoint jobs completed on their first
production attempt. The endpoint job produced 32 endpoint JSON/Parquet cells
(3 validation horizons and one train endpoint per arm). The controller state
is `completed`; there are no pending or failed production jobs.

The production manifest SHA-256 is
`7d07b0f6e4a1a1e2bafc3324ea60f11e4fd3e8433c11e3c744b0ec06b9d33fe4` and the
production source SHA is `2522bc9a7a58b30135d85dfdeb33fdad0c23a313`. The source
change relative to the frozen scientific implementation only corrected the
exclusive `epochs` endpoint (`115` is required to include epoch 114); the
scientific contract is unchanged.

Parents were the registered epoch-99 checkpoints:

```text
dev-1: 360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835
dev-2: bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7
```

The fixed masks contain 7,898/8,907 pilot-S3×T1 IDs and 9,263/8,709
Clean-Wrong IDs for dev-1/dev-2, respectively. Calibration remained frozen at
`beta_advce=0.11834514302628477` and
`margin_coefficient=0.316427398202933`.

All endpoint cells use the common CE-PGD20 identity
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`:
pixel `[0,1]`, $L_\infty$, epsilon $8/255$, step $2/255$, 20 steps,
random start, eval mode. The machine-readable aggregation is in
`docs/experiments/ert_rslad_i100_action_transfer_results_v1.json`.

## Held-out endpoint results

Values are percentages on the fixed internal validation set (5,000 samples).
The endpoint attack is independent CE-PGD20; these are not official-test
results.

| seed | epoch | arm | clean | robust |
|---|---:|---|---:|---:|
| dev-1 | 104 | I100_CONTROL | 82.94 | 56.06 |
| dev-1 | 104 | PILOT_S3_T1_WEAK_ADVCE | 83.18 | 56.02 |
| dev-1 | 104 | CLEAN_WRONG_PLAIN_ADVCE | 83.50 | 56.20 |
| dev-1 | 104 | CLEAN_WRONG_A7_MARGIN_ONLY | 83.44 | 56.30 |
| dev-2 | 104 | I100_CONTROL | 82.76 | 55.80 |
| dev-2 | 104 | PILOT_S3_T1_WEAK_ADVCE | 82.92 | 55.60 |
| dev-2 | 104 | CLEAN_WRONG_PLAIN_ADVCE | 83.34 | 55.96 |
| dev-2 | 104 | CLEAN_WRONG_A7_MARGIN_ONLY | 83.16 | 56.08 |
| dev-1 | 109 | I100_CONTROL | 83.48 | 56.94 |
| dev-1 | 109 | PILOT_S3_T1_WEAK_ADVCE | 83.88 | 56.70 |
| dev-1 | 109 | CLEAN_WRONG_PLAIN_ADVCE | 84.20 | 56.88 |
| dev-1 | 109 | CLEAN_WRONG_A7_MARGIN_ONLY | 84.08 | 57.32 |
| dev-2 | 109 | I100_CONTROL | 83.58 | 56.40 |
| dev-2 | 109 | PILOT_S3_T1_WEAK_ADVCE | 83.62 | 56.26 |
| dev-2 | 109 | CLEAN_WRONG_PLAIN_ADVCE | 83.90 | 56.20 |
| dev-2 | 109 | CLEAN_WRONG_A7_MARGIN_ONLY | 83.74 | 56.54 |
| dev-1 | 114 | I100_CONTROL | 83.64 | 57.32 |
| dev-1 | 114 | PILOT_S3_T1_WEAK_ADVCE | 83.90 | 57.54 |
| dev-1 | 114 | CLEAN_WRONG_PLAIN_ADVCE | 84.48 | 57.76 |
| dev-1 | 114 | CLEAN_WRONG_A7_MARGIN_ONLY | 84.30 | 57.76 |
| dev-2 | 114 | I100_CONTROL | 83.72 | 56.96 |
| dev-2 | 114 | PILOT_S3_T1_WEAK_ADVCE | 83.82 | 57.00 |
| dev-2 | 114 | CLEAN_WRONG_PLAIN_ADVCE | 83.94 | 57.06 |
| dev-2 | 114 | CLEAN_WRONG_A7_MARGIN_ONLY | 84.04 | 57.04 |

At the primary epoch-114 held-out endpoint, treatment-minus-control robust
deltas were:

| arm | dev-1 | dev-2 |
|---|---:|---:|
| PILOT_S3_T1_WEAK_ADVCE | +0.22 pp | +0.04 pp |
| CLEAN_WRONG_PLAIN_ADVCE | +0.44 pp | +0.10 pp |
| CLEAN_WRONG_A7_MARGIN_ONLY | +0.44 pp | +0.08 pp |

The corresponding clean deltas were +0.26/+0.10 pp, +0.84/+0.22 pp, and
+0.66/+0.32 pp. Thus all three treatments were positive at epoch 114 in both
seeds, but the effects are small and this remains a descriptive two-seed
screen.

## Fixed-train direct and spillover effects at epoch 114

Effects are paired against the same-seed I100 control endpoint. `Direct` is
the fixed treatment mask; `spillover` is the complement in the 45,000-sample
training universe. Accuracy deltas equal rescue rate minus harm rate.

| seed | arm | scope | n | clean Δ (pp) | robust Δ (pp) | clean rescue/harm | robust rescue/harm |
|---|---|---|---:|---:|---:|---:|---:|
| dev-1 | PILOT_S3_T1_WEAK_ADVCE | direct | 7,898 | +0.734 | +3.520 | 66/8 | 355/77 |
| dev-1 | PILOT_S3_T1_WEAK_ADVCE | spillover | 37,102 | +0.288 | -0.164 | 226/119 | 189/250 |
| dev-1 | CLEAN_WRONG_PLAIN_ADVCE | direct | 9,263 | +5.020 | +1.663 | 488/23 | 160/6 |
| dev-1 | CLEAN_WRONG_PLAIN_ADVCE | spillover | 35,737 | -0.022 | -0.218 | 91/99 | 364/442 |
| dev-1 | CLEAN_WRONG_A7_MARGIN_ONLY | direct | 9,263 | +3.984 | +1.684 | 395/26 | 165/9 |
| dev-1 | CLEAN_WRONG_A7_MARGIN_ONLY | spillover | 35,737 | -0.112 | +0.014 | 81/121 | 385/380 |
| dev-2 | PILOT_S3_T1_WEAK_ADVCE | direct | 8,907 | +0.853 | +3.222 | 82/6 | 375/88 |
| dev-2 | PILOT_S3_T1_WEAK_ADVCE | spillover | 36,093 | +0.224 | -0.108 | 245/164 | 220/259 |
| dev-2 | CLEAN_WRONG_PLAIN_ADVCE | direct | 8,709 | +4.226 | +1.458 | 392/24 | 143/16 |
| dev-2 | CLEAN_WRONG_PLAIN_ADVCE | spillover | 36,291 | +0.069 | +0.141 | 125/100 | 389/338 |
| dev-2 | CLEAN_WRONG_A7_MARGIN_ONLY | direct | 8,709 | +2.859 | +1.378 | 292/43 | 130/10 |
| dev-2 | CLEAN_WRONG_A7_MARGIN_ONLY | spillover | 36,291 | +0.083 | +0.350 | 126/96 | 398/271 |

The direct robust signal is substantially larger than the held-out signal,
especially for the pilot arm. Spillover is mixed and generally near zero,
which is consistent with a local action effect rather than a uniformly shifted
training population.

## Interpretation and stop decision

- **Pilot-S3×T1 weak AdvCE:** strongest direct robust rescue, but only a small
  held-out gain and an early negative held-out delta. This is a weak transfer
  signal, not confirmation.
- **Clean-Wrong plain AdvCE:** largest held-out epoch-114 clean gain among the
  two Clean-Wrong actions and positive robust delta in both seeds.
- **A7 margin-only:** positive held-out robust delta in both seeds and similar
  to plain AdvCE, but no clear robust or clean dominance over it.
- **Transfer conclusion:** the fixed I100 trajectory did not eliminate the
  short-horizon action-transfer signal. All three tested actions ended above
  control on held-out robustness in both seeds, with plain AdvCE/A7 having the
  largest point estimates. The small sample of seeds and short continuation
  do not support a promoted winner.
- **No automatic follow-up:** no coefficient, mask, arm, seed, horizon,
  official test, AutoAttack, or epoch-199 extension was started. Bootstrap
  intervals were not part of this aggregation; values above are point
  estimates.

This closes the registered I100 action-transfer screen. Any longer
confirmation or combined treatment requires a new human-reviewed plan.
