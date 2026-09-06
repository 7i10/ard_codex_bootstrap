# Post-decay noise floor

Record: `docs/experiments/ard_post_decay_floor_v1.json` (contract `ard_post_decay_floor_v1`). The runs were produced at source `ed3b77daa1de`; this report was rendered at `ee2001bf8767`.

## What this measures

Six untreated control runs were forked from two epoch-99 parents, three from each, sharing a common epoch-100 no-action prefix and differing only in the random stream after the fork. They were trained to epoch 114 and scored on the held-out 5000-example validation split with the registered CE-PGD20 endpoint attack. No treatment was applied to any of them, so every difference between two of them is instrument noise. That spread is the floor: an effect smaller than it cannot be distinguished from nothing, no matter how the run is labelled.

There is no pass or fail here. Only the number matters.

## Epoch 104

| parent | rep 1 | rep 2 | rep 3 | spread | replicate SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | 56.060 | 55.940 | 55.920 | 0.140 pp | 0.076 pp |
| dev-2 | 55.540 | 55.800 | 55.760 | 0.260 pp | 0.140 pp |

The six pairwise absolute differences are 0.020, 0.040, 0.120, 0.140, 0.220, 0.260 pp: mean 0.133 pp, largest 0.260 pp.

Pooled across both parents, one run differs from an identical one by a standard deviation of **0.159 pp** (95% interval 0.095 to 0.457 pp, 4 degrees of freedom).

| paired blocks | smallest detectable effect | under the existing sqrt(7.85/k) convention |
| ---: | ---: | ---: |
| 2 | 0.418 pp | 0.315 pp |
| 5 | 0.265 pp | 0.199 pp |
| 10 | 0.187 pp | 0.141 pp |

## Epoch 109

| parent | rep 1 | rep 2 | rep 3 | spread | replicate SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | 56.940 | 57.040 | 56.900 | 0.140 pp | 0.072 pp |
| dev-2 | 56.200 | 56.400 | 56.280 | 0.200 pp | 0.101 pp |

The six pairwise absolute differences are 0.040, 0.080, 0.100, 0.120, 0.140, 0.200 pp: mean 0.113 pp, largest 0.200 pp.

Pooled across both parents, one run differs from an identical one by a standard deviation of **0.124 pp** (95% interval 0.074 to 0.356 pp, 4 degrees of freedom).

| paired blocks | smallest detectable effect | under the existing sqrt(7.85/k) convention |
| ---: | ---: | ---: |
| 2 | 0.325 pp | 0.245 pp |
| 5 | 0.206 pp | 0.155 pp |
| 10 | 0.146 pp | 0.110 pp |

## Epoch 114

| parent | rep 1 | rep 2 | rep 3 | spread | replicate SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | 57.320 | 57.380 | 57.260 | 0.120 pp | 0.060 pp |
| dev-2 | 57.100 | 56.960 | 57.040 | 0.140 pp | 0.070 pp |

The six pairwise absolute differences are 0.060, 0.060, 0.060, 0.080, 0.120, 0.140 pp: mean 0.087 pp, largest 0.140 pp.

Pooled across both parents, one run differs from an identical one by a standard deviation of **0.092 pp** (95% interval 0.055 to 0.265 pp, 4 degrees of freedom).

| paired blocks | smallest detectable effect | under the existing sqrt(7.85/k) convention |
| ---: | ---: | ---: |
| 2 | 0.243 pp | 0.183 pp |
| 5 | 0.154 pp | 0.116 pp |
| 10 | 0.109 pp | 0.082 pp |

## What this floor does and does not cover

These replicates share a parent, share an epoch-100 prefix, share a data order, and share a campaign. They differ in the random stream after the fork and in nothing else. That is exactly the structure of a screen that compares a treated arm against a control inside one campaign, so the number applies there directly.

It does not license comparisons across campaigns, but not for the reason usually given. The evidence cited for a cross-campaign penalty -- two nominally identical controls differing by 0.94 and 1.78 pp -- is measured at epoch 84, sixteen epochs BEFORE the learning-rate decay, and it sits inside the independently measured pre-decay floor of 1.14 to 1.25 pp. It is the pre-decay floor, not an extra term on top of it. Comparing it against the number on this page compares two different regimes.

Whether a cross-campaign penalty exists *after* the decay is simply untested: no two campaigns have ever run untreated controls from the same parent past epoch 100. Until one does, a post-decay comparison across campaigns has no measured floor at all, and the safe course is still not to make one.

It is also a floor for this design only: two parents, fourteen epochs past the decay, the registered CE-PGD20 endpoint on the validation split. Nothing here transfers to a different horizon, a different endpoint, or the official test split.

## How to read the two spread numbers

The plan preregistered the six pairwise absolute differences and their standard deviation, and that is reported above unchanged. But three runs give only two degrees of freedom per parent, and the six differences drawn from them are not independent of each other, so the standard deviation of those six numbers is not the quantity a future screen needs. The pooled figure is: it combines the two within-parent variances into four degrees of freedom and converts to the difference of two runs. Both are reported so that neither the preregistration nor the correct estimator is hidden.

The interval on that figure is wide because four degrees of freedom is very little. It is stated rather than rounded away: the floor is now measured instead of guessed, but it is not measured precisely.
