# ERT confirmatory T1/T2/T3 fixed-anchor screen

## Status

Complete as a fixed, descriptive screen. No route was promoted and no Stage B,
new seed, official-test evaluation, or AutoAttack was started.

The experiment continued the exact Chen ERT epoch-79 parents for L2 (seed 1)
and L4 (seed 2) to epochs 84, 89, and 94. The four fixed arms were:

- `C79CONF`: common continuation control;
- `T1WCONF`: S3 × T1 with weak adversarial CE;
- `T2WCONF`: S3 × T2 with weak adversarial CE;
- `T3LP05CONF`: S3 × T3 with adversarial KD multiplier 0.5 and weak CE.

The registered masks were fixed at epoch 79. No mask, coefficient, threshold,
or route was changed after observing an endpoint.

## Reproducibility and attack identity

All 24 endpoints per seed (4 arms × 3 horizons × train/validation) were
evaluated independently with eval-mode CE-PGD20:

```text
pixel [0,1], Linf, epsilon=8/255, step=2/255,
20 steps, random start, hard-label CE
```

The endpoint attack identity SHA-256 is
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
The report was generated from clean source SHA
`e0714671045cd80a99d4fb09dbfc9a7dbd76723d` and is bound to the endpoint,
checkpoint, mask, configuration, and calibration hashes in the machine report.

| Item | SHA-256 |
|---|---|
| Config | `63a550a1b298e70ec27fc7c537700bc3c9b6a3e9fed0769400a44f6345d806fe` |
| L2 epoch-79 parent | `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c` |
| L4 epoch-79 parent | `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1` |
| L2 fixed mask | `73152e263bfd613d1880c9b27a3d15aa567e61537681c012cea248b375868184` |
| L4 fixed mask | `ad3eb7830ca52ad8ee25c842f53e646eaf49805301b1adea526760ffca9e041b` |
| Result JSON | `5967fa9d6ac03f004b76c89c08eeb719b58e080431256d631570cd9950e4ceae` |

The machine-readable report is
[`docs/experiments/ert_confirmatory_t123_results_v1.json`](experiments/ert_confirmatory_t123_results_v1.json).
It contains rescue, harm, net rescue, clean and robust margin deltas,
non-selected spillover, held-out validation effects, and the fixed
2,000-replicate class-stratified sample bootstrap. Sample bootstrap intervals
describe sample uncertainty only; they are not uncertainty over training
seeds.

## Full validation endpoint accuracy

Values are CE-PGD20 robust accuracy. Each row is a separate saved checkpoint;
the control is not re-used as an adversarial example source for treatment arms.

| Seed | Horizon | C79CONF | T1WCONF | T2WCONF | T3LP05CONF |
|---|---:|---:|---:|---:|---:|
| L2 | 84 | 47.26% | 47.12% | 47.58% | 47.04% |
| L2 | 89 | 46.04% | 46.94% | 45.64% | 44.62% |
| L2 | 94 | 46.20% | 45.86% | 45.66% | 44.40% |
| L4 | 84 | 46.12% | 46.08% | 45.94% | 46.02% |
| L4 | 89 | 47.94% | 47.30% | 45.98% | 45.80% |
| L4 | 94 | 47.58% | 46.66% | 47.26% | 46.62% |

The common control and all treatments therefore show substantial ordinary
trajectory variation across horizons. Full validation accuracy is not a
replacement for the pre-registered paired cohort effects below.

## Paired effect summary

Direct effects are measured on the registered selected training cohort against
the same-seed control. Held-out effects are measured on the fixed validation
split. Percentages below are percentage points; `+` is an increase in robust
accuracy.

| Seed | Horizon | Arm | Selected direct robust Δ | Held-out robust Δ | Selected n |
|---|---:|---|---:|---:|---:|
| L2 | 84 | T1WCONF | +0.72 | -0.14 | 9,889 |
| L2 | 84 | T2WCONF | +1.80 | +0.32 | 2,162 |
| L2 | 84 | T3LP05CONF | +0.06 | -0.22 | 1,790 |
| L2 | 89 | T1WCONF | +2.53 | +0.90 | 9,889 |
| L2 | 89 | T2WCONF | +1.06 | -0.40 | 2,162 |
| L2 | 89 | T3LP05CONF | +0.11 | -1.42 | 1,790 |
| L2 | 94 | T1WCONF | +3.54 | -0.34 | 9,889 |
| L2 | 94 | T2WCONF | -0.74 | -0.54 | 2,162 |
| L2 | 94 | T3LP05CONF | +0.22 | -1.80 | 1,790 |
| L4 | 84 | T1WCONF | +0.90 | -0.04 | 9,368 |
| L4 | 84 | T2WCONF | +1.40 | -0.18 | 2,138 |
| L4 | 84 | T3LP05CONF | -0.23 | -0.10 | 1,771 |
| L4 | 89 | T1WCONF | +0.68 | -0.64 | 9,368 |
| L4 | 89 | T2WCONF | -1.45 | -1.96 | 2,138 |
| L4 | 89 | T3LP05CONF | +0.00 | -2.14 | 1,771 |
| L4 | 94 | T1WCONF | -3.77 | -0.92 | 9,368 |
| L4 | 94 | T2WCONF | -0.89 | -0.32 | 2,138 |
| L4 | 94 | T3LP05CONF | -0.11 | -0.96 | 1,771 |

The report JSON additionally records clean harm, rescue/harm counts, margin
changes, and non-selected spillover for every row. At the last horizon, for
example, T1 weak CE is positive on the selected L2 cohort but negative on L4;
its held-out effect is negative for both seeds. This is not sufficient for a
method claim.

## Interpretation

1. **T1 weak CE is a useful mechanism signal but not a confirmed method.** It
   has positive direct selected-cohort effects at all three L2 horizons and at
   the first two L4 horizons, but the full validation endpoint does not retain
   that advantage consistently.
2. **T2 weak CE is not stable.** It is positive on selected cohorts at horizon
   84 for both seeds, then becomes mixed or negative by horizon 94. The effect
   does not establish a general T2 route.
3. **T3 low-pressure AdvKD is unsupported by this screen.** The full validation
   effects are non-positive at all reported L4 horizons and mostly negative
   for L2 after horizon 84.
4. **Endpoint horizon matters.** Direct selected-cohort rescue can increase
   while held-out accuracy falls, so rescue on the training cohort alone is not
   evidence of improved generalization.
5. **No automatic promotion is allowed.** This is a short, two-seed,
   train/validation screen from epoch 79. It does not establish official-test
   or AutoAttack performance, and it does not justify changing the masks,
   coefficients, or routing rules.

## Calibration and execution checks

The no-update coefficient sanity check completed before continuation. It used
the frozen rounded `beta_advce=0.075` and T3 multiplier `0.5`; it did not step
the optimizer, scheduler, or sample state. The scaled AdvCE/base-AdvKD median
gradient ratio was `0.2635770718462068`, with cosine `0.7657734453678131`.
The sanity artifact is
[`docs/experiments/ert_confirmatory_t123_calibration_sanity_v1.json`](experiments/ert_confirmatory_t123_calibration_sanity_v1.json)
with SHA-256
`3bcf69110216ce992b6d3e3e25a3894cc8d6a2f66fe266311c1f166c17f57a5c`.

An initial canary invocation was fail-closed because an exclusive epoch bound
would have produced zero continuation epochs. The corrected canary completed
successfully; no scientific artifact from the rejected invocation was used.
All eight scientific continuations and all 48 endpoint jobs (two splits × 24
per seed) completed, and the runs were logged to W&B online under the
confirmatory namespace.

## Stop boundary

The next action is a human scientific decision. This task intentionally does
not start Stage B, dynamic routing, another seed, official test, or
AutoAttack.
