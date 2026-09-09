# ERT Stage A treatment results

Status: complete; descriptive two-seed screen only. No automatic promotion.

## Scope and provenance

Stage A continued the exact Chen ERT epoch-79 parents for five epochs
(epochs 80--84) on L2/seed 1 and L4/seed 2. The 13 arms per seed were one
common control (`C79`) and the 12 frozen treatment arms (`ST1W/M/S`,
`ST2W/M/S`, `ST3K1/K05/K0`, and `CW1/2/3`). The endpoint is the last epoch
84 checkpoint; it is not an official test result.

Every arm was independently evaluated on all 45,000 training samples with
the same eval-mode CE-PGD20 attack:

```text
pixel [0,1], Linf, epsilon=8/255, step=2/255, steps=20, random start
```

The endpoint evaluator source was `bd1d448decd596657fbc00313463aff8ee71caf7`.
The paired report was generated from clean source
`48094b2e8746aeafd4d32982c9819f2e58d72fe1` and is bound to all 26 endpoint
JSON/Parquet hashes. The report SHA-256 is recorded in
`docs/experiments/ert_stage_a_results_v1.json.sha256`:

```text
97a5fa9a9fabc2b62cd801a1143774eb223beae0f25d0c23d4b9c357bba77243
```

The calibration artifact was frozen before training:

```text
tau=2.0
alpha_soft=1.2522921562194824
beta_advce_weak=0.07095924764871597
beta_advce_moderate=0.14191849529743195
beta_cleance_weak=0.07825280725955963
```

The exact parent, mask, checkpoint, and calibration hashes are in the two
machine-readable experiment artifacts. The modifier analysis is in
`docs/experiments/ert_stage_a_modifiers_v1.json` (SHA-256
`2928d47daf328f4965f7f5b6dff209c7aecd0cc1f67d6dd917775e8470163edb`).

## How to read the tables

All deltas are treatment minus the same-seed `C79` control on the identical
stable-ID cohort. Values are percentage points. `rescue` is control robust
wrong to treatment robust correct; `harm` is the reverse; `net` is rescue
minus harm. `clean Δ` is the paired clean-accuracy change. `overall R` is the
full 45,000-sample endpoint robust accuracy for that arm. The JSON also stores
non-selected paired spillover; it is not a matched-random comparison because
Stage A did not register random masks.

## L2 / Chen seed 1

Control endpoint: clean 78.653%, robust 49.707%.

| arm | selected n | overall R | robust Δ | rescue | harm | net | clean Δ | adv-margin Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST1W | 9,889 | 50.942 | +4.409 | 10.213 | 5.804 | +4.409 | -0.101 | +1.800 |
| ST1M | 9,889 | 47.367 | -1.618 | 6.977 | 8.595 | -1.618 | -2.629 | -1.426 |
| ST1S | 9,889 | 48.658 | -3.448 | 7.109 | 10.557 | -3.448 | -3.752 | -0.831 |
| ST2W | 2,162 | 50.478 | -0.046 | 3.562 | 3.608 | -0.046 | -0.601 | +0.347 |
| ST2M | 2,162 | 48.542 | +3.654 | 6.938 | 3.284 | +3.654 | -2.128 | -0.258 |
| ST2S | 2,162 | 49.931 | -0.648 | 4.209 | 4.857 | -0.648 | -6.568 | +0.260 |
| ST3K1 | 1,790 | 50.207 | +0.782 | 1.955 | 1.173 | +0.782 | -3.631 | +0.587 |
| ST3K05 | 1,790 | 50.887 | -0.279 | 1.117 | 1.397 | -0.279 | +1.788 | -0.331 |
| ST3K0 | 1,790 | 49.584 | +0.335 | 1.732 | 1.397 | +0.335 | -4.190 | -1.115 |
| CW1 | 8,623 | 48.100 | -0.835 | 0.905 | 1.740 | -0.835 | -1.055 | -3.106 |
| CW2 | 8,623 | 48.618 | +0.104 | 1.461 | 1.357 | +0.104 | +5.868 | -1.924 |
| CW3 | 8,623 | 47.180 | +0.545 | 1.809 | 1.264 | +0.545 | +5.439 | -1.614 |

## L4 / Chen seed 2

Control endpoint: clean 78.582%, robust 49.251%.

| arm | selected n | overall R | robust Δ | rescue | harm | net | clean Δ | adv-margin Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST1W | 9,368 | 49.680 | +2.210 | 9.073 | 6.864 | +2.210 | -0.096 | +0.518 |
| ST1M | 9,368 | 49.682 | +0.427 | 8.412 | 7.985 | +0.427 | +2.455 | +0.259 |
| ST1S | 9,368 | 50.567 | -4.441 | 8.508 | 12.948 | -4.441 | +0.523 | -1.140 |
| ST2W | 2,138 | 50.367 | -0.281 | 3.227 | 3.508 | -0.281 | +5.893 | +0.742 |
| ST2M | 2,138 | 50.318 | +0.327 | 4.397 | 4.069 | +0.327 | +3.368 | +0.773 |
| ST2S | 2,138 | 50.473 | -1.543 | 3.227 | 4.771 | -1.543 | +2.385 | -0.197 |
| ST3K1 | 1,771 | 50.576 | -0.734 | 1.186 | 1.920 | -0.734 | +2.936 | -0.181 |
| ST3K05 | 1,771 | 50.613 | -1.468 | 0.847 | 2.315 | -1.468 | +8.075 | -0.281 |
| ST3K0 | 1,771 | 50.844 | -1.468 | 1.016 | 2.484 | -1.468 | +4.630 | -0.588 |
| CW1 | 8,925 | 48.996 | -1.591 | 0.941 | 2.532 | -1.591 | -3.104 | -4.693 |
| CW2 | 8,925 | 49.498 | -0.986 | 1.423 | 2.409 | -0.986 | +2.790 | -3.454 |
| CW3 | 8,925 | 48.302 | -0.605 | 1.221 | 1.826 | -0.605 | +4.874 | -2.541 |

## Teacher-response modifier analysis

This is a fixed epoch-79 descriptive analysis, not a selector or threshold
optimization. Within each registered state cohort, `mT_clean` and
`DeltaT=mT_clean-mT_adv` were split into deterministic rank tertiles with
sample ID as the tie-break. Effects are the same paired endpoint metrics as
above. Full values are in the modifier JSON.

The main qualitative pattern is seed-dependent rather than a stable monotone
effect. For example, ST1W is positive in both seeds (+4.409 pp L2,
+2.210 pp L4), but its tertile effects vary (L2 `mT_clean` low/middle/high
+3.550/+3.823/+5.854 pp; L4 +1.762/+2.305/+2.562 pp). ST1S is harmful in
both seeds (−3.448/−4.441 pp), while ST2M is positive but small in both
(+3.654/+0.327 pp). The T3 KD ablations do not show a consistent benefit:
L2 is near zero to mildly positive, whereas all three L4 T3 arms are
negative.

For T3, the teacher-clean-correct subset is 1,607/1,790 (L2) and
1,599/1,771 (L4). The teacher-clean-wrong subset is therefore 183 and 172,
respectively. Its effects are not a reliable promotion rule: for example,
ST3K0 is −0.546 pp on L2 teacher-clean-wrong but +0.581 pp on L4, while
ST3K1 is 0.000 pp and +1.163 pp. These strata are evidence modifiers only.

## Evidence summary and stop decision

- **ST1 weak AdvCE (`ST1W`)**: positive robust paired delta in both seeds,
  with nearly unchanged clean accuracy. This is the most consistent Stage A
  signal, but it is a short-horizon train-split endpoint and is not yet a
  promoted method.
- **ST1 moderate/softening**: moderate AdvCE is mixed; teacher-only
  softening is harmful in both seeds. The frozen target-softening mechanism is
  not supported by this screen.
- **ST2**: moderate AdvCE is positive in both seeds but much smaller in L4;
  weak AdvCE is approximately neutral and softening is negative. This is
  suggestive, not confirmatory.
- **ST3**: no consistent advantage for retaining, halving, or removing
  adversarial KD. L4 is negative for all three; do not infer a KD route from
  this screen.
- **Clean-Wrong**: CW2/CW3 improve clean accuracy in both seeds, but robust
  effect is positive only for L2 and negative for L4. Clean recovery and
  robust preservation are therefore in tension here.

No automatic `SUPPORTED` winner, Stage B extension, dynamic router, new seed,
official test, or AutoAttack was started. These results justify a human
decision about whether to follow up ST1W/ST2M with a separately preregistered
experiment; they do not justify changing coefficients or thresholds after
observing this endpoint.
