# ERT Clean-Wrong local lambda stability

Status: completed matched R1/R2 point-estimate analysis on Hamster.

The first `--epochs 94` launch was fail-closed because the runtime's end boundary is exclusive; the valid v2 launch used `--epochs 95` and produced epoch 94. Partial v1 outputs are excluded.

Primary endpoint: same-block paired held-out CE-PGD20 robust accuracy at epoch 94. Effects are sample-level paired effects, not training-seed confidence intervals.

## Endpoint overall held-out effects

| block | arm | lambda | epoch84 robust Δ | epoch89 robust Δ | epoch94 robust Δ | epoch94 clean Δ |
|---|---|---:|---:|---:|---:|---:|
| L2-R1 | B0_BASE | 0.000000000 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R1 | N90 | 0.214924604 | -0.620 pp | +2.580 pp | +1.880 pp | +0.060 pp |
| L2-R1 | N95 | 0.226864859 | -0.100 pp | +0.660 pp | +1.600 pp | -0.400 pp |
| L2-R1 | A100 | 0.238805115 | -0.520 pp | +0.460 pp | +1.340 pp | +0.080 pp |
| L2-R1 | N105 | 0.250745371 | -0.620 pp | +0.820 pp | -0.560 pp | -0.360 pp |
| L2-R1 | N110 | 0.262685627 | -0.060 pp | +1.500 pp | -1.160 pp | -2.080 pp |
| L2-R2 | B0_BASE | 0.000000000 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R2 | N90 | 0.214924604 | +0.240 pp | +0.020 pp | -0.580 pp | +0.400 pp |
| L2-R2 | N95 | 0.226864859 | +1.080 pp | +0.640 pp | -0.460 pp | -0.520 pp |
| L2-R2 | A100 | 0.238805115 | +0.760 pp | -0.740 pp | -0.200 pp | -0.580 pp |
| L2-R2 | N105 | 0.250745371 | +1.280 pp | -0.220 pp | -1.020 pp | -0.680 pp |
| L2-R2 | N110 | 0.262685627 | +0.820 pp | +2.300 pp | -0.860 pp | -0.860 pp |
| L4-R1 | B0_BASE | 0.000000000 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R1 | N90 | 0.214924604 | +0.340 pp | -0.260 pp | +1.100 pp | +0.440 pp |
| L4-R1 | N95 | 0.226864859 | +0.540 pp | -0.620 pp | +0.720 pp | +1.280 pp |
| L4-R1 | A100 | 0.238805115 | +0.200 pp | +1.020 pp | +0.660 pp | +1.000 pp |
| L4-R1 | N105 | 0.250745371 | -0.340 pp | -0.520 pp | +1.760 pp | +0.660 pp |
| L4-R1 | N110 | 0.262685627 | +0.060 pp | -0.700 pp | +0.780 pp | +1.020 pp |
| L4-R2 | B0_BASE | 0.000000000 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R2 | N90 | 0.214924604 | -0.660 pp | -0.640 pp | -0.080 pp | +1.100 pp |
| L4-R2 | N95 | 0.226864859 | -0.360 pp | +0.700 pp | -1.340 pp | +0.760 pp |
| L4-R2 | A100 | 0.238805115 | -0.160 pp | -0.540 pp | -0.060 pp | +1.680 pp |
| L4-R2 | N105 | 0.250745371 | -0.140 pp | +0.880 pp | -0.120 pp | +0.920 pp |
| L4-R2 | N110 | 0.262685627 | -1.900 pp | -0.800 pp | -1.620 pp | -0.080 pp |

## Epoch-94 block aggregate

| arm | lambda | overall robust mean | positive blocks | direct CW robust mean | non-CW spillover robust mean | held-out CW robust mean |
|---|---:|---:|---:|---:|---:|---:|
| B0_BASE | 0.000000000 | +0.000 pp | 0/4 | +0.000 pp | +0.000 pp | +0.000 pp |
| N90 | 0.214924604 | +0.580 pp | 2/4 | +0.148 pp | +0.369 pp | -0.458 pp |
| N95 | 0.226864859 | +0.130 pp | 2/4 | +0.267 pp | +0.221 pp | -0.479 pp |
| A100 | 0.238805115 | +0.435 pp | 2/4 | +0.337 pp | +0.384 pp | -0.069 pp |
| N105 | 0.250745371 | +0.015 pp | 1/4 | +0.217 pp | +0.219 pp | -0.618 pp |
| N110 | 0.262685627 | -0.715 pp | 1/4 | +0.511 pp | -0.884 pp | +0.043 pp |

## ±5% and ±10% neighborhoods

The ±5% neighborhood is N95/A100/N105; the exploratory ±10% neighborhood is N90 through N110. No lambda is selected automatically.

| neighborhood | arms | epoch94 held-out robust mean range |
|---|---|---:|
| ±5% | N95, A100, N105 | +0.015 pp to +0.435 pp |
| ±10% | N90, N95, A100, N105, N110 | -0.715 pp to +0.580 pp |

## Adjacent lambda jumps at epoch 94

Jumps are right-minus-left held-out robust deltas across matched blocks; they are descriptive and are not used for coefficient selection.

| transition | mean jump | min | max | positive blocks |
|---|---:|---:|---:|---:|
| N90→N95 | -0.450 pp | -1.260 pp | +0.120 pp | 1/4 |
| N95→A100 | +0.305 pp | -0.260 pp | +1.280 pp | 2/4 |
| A100→N105 | -0.420 pp | -1.900 pp | +1.100 pp | 1/4 |
| N105→N110 | -0.730 pp | -1.500 pp | +0.160 pp | 1/4 |

## Epoch-94 held-out CE20 and KL10 Teacher-margin Q1-Q5

Bins are pre-treatment and train-derived; validation outcomes never define boundaries.

| block | arm | domain | Q1 robust Δ | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---:|---:|---:|---:|---:|
| L2-R1 | B0_BASE | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R1 | B0_BASE | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R1 | N90 | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +1.818 pp | +2.215 pp |
| L2-R1 | N90 | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +0.777 pp | +2.290 pp |
| L2-R1 | N95 | CE20 | +0.000 pp | +0.000 pp | +0.408 pp | +2.955 pp | +1.700 pp |
| L2-R1 | N95 | KL10 | +0.000 pp | +0.000 pp | +1.261 pp | -0.259 pp | +1.963 pp |
| L2-R1 | A100 | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +2.273 pp | +1.468 pp |
| L2-R1 | A100 | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +1.554 pp | +1.535 pp |
| L2-R1 | N105 | CE20 | +0.000 pp | +0.474 pp | -0.408 pp | +1.818 pp | -0.927 pp |
| L2-R1 | N105 | KL10 | +0.000 pp | +0.000 pp | +0.840 pp | +0.259 pp | -0.780 pp |
| L2-R1 | N110 | CE20 | +0.000 pp | +0.474 pp | +0.816 pp | +3.182 pp | -1.931 pp |
| L2-R1 | N110 | KL10 | +0.000 pp | +0.503 pp | +0.000 pp | +2.591 pp | -1.736 pp |
| L2-R2 | B0_BASE | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R2 | B0_BASE | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L2-R2 | N90 | CE20 | +0.000 pp | +0.000 pp | -0.816 pp | -1.136 pp | -0.567 pp |
| L2-R2 | N90 | KL10 | +0.000 pp | +0.000 pp | -0.420 pp | -2.073 pp | -0.503 pp |
| L2-R2 | N95 | CE20 | +0.000 pp | +0.000 pp | -0.408 pp | +0.000 pp | -0.567 pp |
| L2-R2 | N95 | KL10 | +0.000 pp | +0.000 pp | -0.420 pp | -1.036 pp | -0.453 pp |
| L2-R2 | A100 | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | -1.591 pp | -0.077 pp |
| L2-R2 | A100 | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | -2.073 pp | -0.050 pp |
| L2-R2 | N105 | CE20 | +0.000 pp | +0.000 pp | -0.816 pp | -1.136 pp | -1.133 pp |
| L2-R2 | N105 | KL10 | +0.000 pp | +0.000 pp | -0.420 pp | -2.332 pp | -1.032 pp |
| L2-R2 | N110 | CE20 | +0.000 pp | +0.474 pp | -0.408 pp | -0.227 pp | -1.082 pp |
| L2-R2 | N110 | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | -1.554 pp | -0.931 pp |
| L4-R1 | B0_BASE | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R1 | B0_BASE | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R1 | N90 | CE20 | +0.000 pp | +0.000 pp | +0.373 pp | +0.899 pp | +1.298 pp |
| L4-R1 | N90 | KL10 | +0.483 pp | -0.483 pp | +1.205 pp | +0.726 pp | +1.249 pp |
| L4-R1 | N95 | CE20 | -0.439 pp | +0.000 pp | -0.746 pp | +0.000 pp | +1.013 pp |
| L4-R1 | N95 | KL10 | +0.000 pp | -0.483 pp | -0.402 pp | -0.484 pp | +1.019 pp |
| L4-R1 | A100 | CE20 | +0.000 pp | +0.000 pp | -0.373 pp | -0.225 pp | +0.909 pp |
| L4-R1 | A100 | KL10 | +0.000 pp | -0.483 pp | +0.402 pp | +0.484 pp | +0.790 pp |
| L4-R1 | N105 | CE20 | +0.439 pp | +0.000 pp | +0.000 pp | -0.674 pp | +2.337 pp |
| L4-R1 | N105 | KL10 | +0.483 pp | -0.483 pp | +0.803 pp | +0.000 pp | +2.192 pp |
| L4-R1 | N110 | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +0.225 pp | +0.987 pp |
| L4-R1 | N110 | KL10 | +0.000 pp | -0.483 pp | -0.402 pp | +1.695 pp | +0.866 pp |
| L4-R2 | B0_BASE | CE20 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R2 | B0_BASE | KL10 | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp | +0.000 pp |
| L4-R2 | N90 | CE20 | -0.439 pp | -0.962 pp | -0.373 pp | -1.124 pp | +0.130 pp |
| L4-R2 | N90 | KL10 | -0.966 pp | -0.483 pp | -0.803 pp | -0.726 pp | +0.102 pp |
| L4-R2 | N95 | CE20 | -0.877 pp | -0.962 pp | +0.000 pp | -0.899 pp | -1.532 pp |
| L4-R2 | N95 | KL10 | -0.966 pp | -0.483 pp | -0.803 pp | -0.484 pp | -1.529 pp |
| L4-R2 | A100 | CE20 | -0.439 pp | -0.962 pp | -0.373 pp | -1.124 pp | +0.156 pp |
| L4-R2 | A100 | KL10 | -0.966 pp | +0.000 pp | -1.606 pp | -0.242 pp | +0.102 pp |
| L4-R2 | N105 | CE20 | -0.877 pp | -0.962 pp | -0.373 pp | -2.022 pp | +0.208 pp |
| L4-R2 | N105 | KL10 | -0.966 pp | -0.483 pp | -0.803 pp | -0.969 pp | +0.076 pp |
| L4-R2 | N110 | CE20 | -0.877 pp | -0.962 pp | -0.373 pp | -0.449 pp | -1.922 pp |
| L4-R2 | N110 | KL10 | -0.966 pp | -0.483 pp | -1.205 pp | +0.484 pp | -1.962 pp |

## Replicate variance

R1/R2 differences are descriptive continuation variance within each teacher; they are not population uncertainty.

| teacher | arm | epoch94 overall robust | epoch94 direct robust | epoch94 spillover robust |
|---|---|---:|---:|---:|
| L2 | B0_BASE | +0.000 pp | +0.000 pp | +0.000 pp |
| L2 | N90 | +2.460 pp | +0.464 pp | +2.691 pp |
| L2 | N95 | +2.060 pp | +0.348 pp | +2.029 pp |
| L2 | A100 | +1.540 pp | +0.313 pp | +1.746 pp |
| L2 | N105 | +0.460 pp | +0.371 pp | +1.457 pp |
| L2 | N110 | +0.300 pp | +0.128 pp | +0.808 pp |
| L4 | B0_BASE | +0.000 pp | +0.000 pp | +0.000 pp |
| L4 | N90 | +1.180 pp | +0.762 pp | +0.330 pp |
| L4 | N95 | +2.060 pp | +0.280 pp | +0.618 pp |
| L4 | A100 | +0.720 pp | +1.569 pp | +0.424 pp |
| L4 | N105 | +1.880 pp | +0.415 pp | +0.715 pp |
| L4 | N110 | +2.400 pp | +1.468 pp | +1.502 pp |

## Frozen contract

- Teacher: Chen2021LTD_WRN34_10; training KL-PGD10; endpoint CE-PGD20.
- Margin-only treatment; CleanCE is zero in every arm.
- Fixed epoch-79 Clean-Wrong masks; no post-hoc lambda, floor, cap, or threshold selection.
- Four matched blocks: L2-R1, L2-R2, L4-R1, L4-R2; two GPUs on Hamster.
- W&B retention is metrics-only; checkpoints and run bundles remain local.
- The L4 registered mask file retains a legacy 9b51... parent binding; the prior recovery audit establishes component-level equivalence and the downstream causal parent remains byte-exact 026a....
- This report contains point estimates; no training-seed confidence interval is claimed.
