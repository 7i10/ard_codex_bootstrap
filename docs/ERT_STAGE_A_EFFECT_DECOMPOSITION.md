# ERT Stage A: direct, spillover, and held-out effects

Status: complete, descriptive decomposition only. No new training, route,
threshold, coefficient, official-test, or AutoAttack decision was made.

## Contract and provenance

This analysis reuses the 26 completed Stage A epoch-84 checkpoints and the
existing train-split endpoint rows. It adds an independent validation endpoint
for the same 26 checkpoints. The validation is the deterministic stratified
10% split from the parent config: 5,000 samples, exactly 500 per CIFAR-10
class, with no sample-ID overlap with the 45,000-sample train split.

Every validation arm received its own eval-mode white-box CE-PGD20 attack:

```text
pixel [0,1], Linf, epsilon=8/255, step=2/255, 20 steps, random start
```

The report was generated at source SHA `54b21dd2c5d6780f7513dfcc47e642d9fe80b636`.
The machine-readable report is
`docs/experiments/ert_stage_a_effect_decomposition_v1.json`; its sidecar
SHA-256 is:

```text
5d857e908f74efd4f24b59b5c68dc5a477d061841bf5d0d8f249451a3a03229b
```

Bootstrap uses 2,000 deterministic class-stratified paired replicates with
seed `20260813`. These intervals describe sample uncertainty within each
fixed run; they are not training-seed uncertainty.

`Direct` is the selected-cohort treatment-minus-C79 effect. `Spillover` is
the same paired difference on all non-selected train IDs. `Held-out` is the
whole validation-set treatment-minus-C79 difference. All values below are
percentage points. Rescue/harm are only defined for the direct and spillover
paired train cohorts; held-out CI is shown for robust accuracy.

The weighted identity

$$
\Delta_{global}=\frac{|M|}{N}\Delta_{direct}+\frac{|U|}{N}\Delta_{spill}
$$

passed for clean and robust accuracy for every treatment arm in both seeds.
The direct values reproduce the prior Stage A report.

## L2 / Chen seed 1

| arm | n selected | direct R | spillover R | held-out R | held-out R 95% CI | direct clean | spillover clean | held-out clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST1W | 9,889 | +4.409 | +0.342 | +1.500 | [+0.680,+2.320] | -0.101 | -0.812 | -0.300 |
| ST1M | 9,889 | -1.618 | -2.543 | -2.380 | [-3.200,-1.520] | -2.629 | -0.447 | -0.780 |
| ST1S | 9,889 | -3.448 | -0.373 | -0.240 | [-1.180,+0.680] | -3.752 | -1.435 | -1.260 |
| ST2W | 2,162 | -0.046 | +0.812 | +1.420 | [+0.600,+2.280] | -0.601 | +0.310 | +0.960 |
| ST2M | 2,162 | +3.654 | -1.408 | -0.980 | [-1.760,-0.160] | -2.128 | -0.693 | -0.740 |
| ST2S | 2,162 | -0.648 | +0.268 | -0.040 | [-0.960,+0.840] | -6.568 | -0.854 | -1.180 |
| ST3K1 | 1,790 | +0.782 | +0.488 | +0.680 | [-0.180,+1.560] | -3.631 | -0.477 | -0.880 |
| ST3K05 | 1,790 | -0.279 | +1.240 | +0.980 | [+0.120,+1.820] | +1.788 | +0.458 | +0.900 |
| ST3K0 | 1,790 | +0.335 | -0.141 | +0.300 | [-0.600,+1.140] | -4.190 | -0.595 | +0.000 |
| CW1 | 8,623 | -0.835 | -1.790 | -1.300 | [-2.180,-0.460] | -1.055 | -0.627 | -0.200 |
| CW2 | 8,623 | +0.104 | -1.372 | -1.460 | [-2.300,-0.560] | +5.868 | +0.778 | +1.940 |
| CW3 | 8,623 | +0.545 | -3.255 | -1.440 | [-2.360,-0.580] | +5.439 | -1.133 | -0.040 |

## L4 / Chen seed 2

| arm | n selected | direct R | spillover R | held-out R | held-out R 95% CI | direct clean | spillover clean | held-out clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST1W | 9,368 | +2.210 | -0.039 | +0.040 | [-0.760,+0.860] | -0.096 | -0.446 | -0.420 |
| ST1M | 9,368 | +0.427 | +0.432 | +0.440 | [-0.400,+1.300] | +2.455 | +0.878 | +0.940 |
| ST1S | 9,368 | -4.441 | +2.829 | +2.180 | [+1.240,+3.040] | +0.523 | +0.432 | -0.160 |
| ST2W | 2,138 | -0.281 | +1.185 | +1.320 | [+0.440,+2.100] | +5.893 | +0.709 | +0.280 |
| ST2M | 2,138 | +0.327 | +1.104 | +1.420 | [+0.620,+2.300] | +3.368 | +0.548 | +0.840 |
| ST2S | 2,138 | -1.543 | +1.360 | +1.960 | [+1.080,+2.800] | +2.385 | +0.415 | +0.020 |
| ST3K1 | 1,771 | -0.734 | +1.409 | +1.820 | [+0.980,+2.660] | +2.936 | +1.041 | +0.900 |
| ST3K05 | 1,771 | -1.468 | +1.478 | +1.280 | [+0.440,+2.120] | +8.075 | +1.601 | +1.740 |
| ST3K0 | 1,771 | -1.468 | +1.719 | +1.700 | [+0.860,+2.540] | +4.630 | +1.138 | +1.020 |
| CW1 | 8,925 | -1.591 | +0.075 | +0.440 | [-0.500,+1.380] | -3.104 | +1.209 | -0.200 |
| CW2 | 8,925 | -0.986 | +0.552 | +0.100 | [-0.900,+1.080] | +2.790 | +2.267 | +1.460 |
| CW3 | 8,925 | -0.605 | -1.034 | -0.580 | [-1.480,+0.320] | +4.874 | +1.940 | +1.620 |

## Interpretation

### ST1W

ST1W's selected recovery does not uniformly transfer to held-out data. L2
has the desired direct-positive / held-out-positive pattern (+4.409 / +1.500
pp, with a held-out CI above zero). L4 has direct +2.210 pp but held-out
+0.040 pp with a CI spanning zero. Thus ST1W is a promising seed-1
generalization signal, not a two-seed confirmed generalization effect.

### ST2M

The apparent selected improvement is not a stable global effect. L2 is direct
+3.654 pp but spillover -1.408 pp and held-out -0.980 pp. L4 is direct
+0.327 pp, spillover +1.104 pp, and held-out +1.420 pp. This is seed-dependent
redistribution rather than a stable recovery mechanism.

### T3: recovery versus low pressure

No T3 arm is a two-seed recovery winner: direct effects are small/negative in
L4. L4's all-positive spillover and held-out effects for ST3K1/K05/K0 are
consistent with a low-pressure-like pattern, but L2 does not reproduce it
consistently. Therefore it is an exploratory mechanism clue, not evidence to
promote KD=1, 0.5, or 0.

### Clean-Wrong

CW2/CW3 improve clean accuracy on both seeds (held-out clean +1.940/+1.460
pp for CW2 and -0.040/+1.620 pp for CW3), but robust held-out effects are
mixed or negative. Clean recovery is therefore not sufficient evidence for a
robust route. CW1 is harmful or neutral across the decomposition.

## Decision

The decomposition clarifies that Stage A selected improvements can arise
without held-out robust improvement, and that T3 low-pressure behavior is
seed-dependent. No treatment is automatically promoted. The next discussion
should focus on whether to design a new preregistered, longer-horizon ST1W
experiment with held-out selection kept frozen, or to treat these effects as a
negative/mixed result. Stage B, dynamic routing, coefficient retuning, new
seeds, official test, and AutoAttack remain stopped.
