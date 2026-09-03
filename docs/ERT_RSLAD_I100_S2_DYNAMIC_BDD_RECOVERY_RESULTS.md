# I100 S2×T1 Dynamic Boundary-Distance Recovery Results

## Decision

S-BDD: **NUMERICALLY_UNSUPPORTED** — corrected secant formulation became non-finite reproducibly in both dev seeds; excluded from causal utility comparison.

The primary causal comparisons are Control, DPM, and D-BDD only. D-BDD vs DPM is mixed across the two development seeds at e114 held-out CE-PGD20, so this screen does not support a D-BDD promotion or an e199 extension.

## Held-out CE-PGD20

| seed | epoch | arm | clean | robust | Δ robust vs Control |
| --- | ---: | --- | ---: | ---: | ---: |
| dev-1 | 104 | CONTROL | 82.94% | 56.06% | +0.00 pp |
| dev-1 | 104 | DPM | 82.92% | 56.02% | -0.04 pp |
| dev-1 | 104 | DBDD | 82.96% | 56.08% | +0.02 pp |
| dev-1 | 109 | CONTROL | 83.48% | 56.94% | +0.00 pp |
| dev-1 | 109 | DPM | 83.34% | 56.84% | -0.10 pp |
| dev-1 | 109 | DBDD | 83.32% | 56.78% | -0.16 pp |
| dev-1 | 114 | CONTROL | 83.64% | 57.32% | +0.00 pp |
| dev-1 | 114 | DPM | 83.76% | 57.40% | +0.08 pp |
| dev-1 | 114 | DBDD | 83.68% | 57.36% | +0.04 pp |
| dev-2 | 104 | CONTROL | 82.76% | 55.80% | +0.00 pp |
| dev-2 | 104 | DPM | 82.74% | 55.80% | +0.00 pp |
| dev-2 | 104 | DBDD | 82.74% | 55.78% | -0.02 pp |
| dev-2 | 109 | CONTROL | 83.58% | 56.40% | +0.00 pp |
| dev-2 | 109 | DPM | 83.46% | 56.22% | -0.18 pp |
| dev-2 | 109 | DBDD | 83.60% | 56.28% | -0.12 pp |
| dev-2 | 114 | CONTROL | 83.72% | 56.96% | +0.00 pp |
| dev-2 | 114 | DPM | 83.62% | 57.08% | +0.12 pp |
| dev-2 | 114 | DBDD | 83.64% | 57.16% | +0.20 pp |

## Primary e114 held-out comparisons

| seed | DPM − Control | D-BDD − Control | D-BDD − DPM |
| --- | ---: | ---: | ---: |
| dev-1 | +0.08 pp | +0.04 pp | -0.04 pp |
| dev-2 | +0.12 pp | +0.20 pp | +0.08 pp |

## e114 paired train effects

| seed | comparison | scope | clean Δ | robust Δ |
| --- | --- | --- | ---: | ---: |
| dev-1 | DPM − CONTROL | direct | +0.05 pp | +1.13 pp |
| dev-1 | DPM − CONTROL | spillover | -0.05 pp | -0.09 pp |
| dev-1 | DPM − CONTROL | global | -0.05 pp | -0.03 pp |
| dev-1 | DBDD − CONTROL | direct | +0.27 pp | +0.99 pp |
| dev-1 | DBDD − CONTROL | spillover | -0.06 pp | -0.17 pp |
| dev-1 | DBDD − CONTROL | global | -0.04 pp | -0.12 pp |
| dev-2 | DPM − CONTROL | direct | +0.05 pp | +0.51 pp |
| dev-2 | DPM − CONTROL | spillover | +0.03 pp | +0.09 pp |
| dev-2 | DPM − CONTROL | global | +0.03 pp | +0.11 pp |
| dev-2 | DBDD − CONTROL | direct | +0.05 pp | +0.14 pp |
| dev-2 | DBDD − CONTROL | spillover | -0.01 pp | +0.01 pp |
| dev-2 | DBDD − CONTROL | global | -0.01 pp | +0.02 pp |

## e114 paired D-BDD − DPM train contrast

| seed | scope | clean Δ | robust Δ |
| --- | --- | ---: | ---: |
| dev-1 | direct | +0.23 pp | -0.14 pp |
| dev-1 | spillover | -0.00 pp | -0.09 pp |
| dev-1 | global | +0.01 pp | -0.09 pp |
| dev-2 | direct | +0.00 pp | -0.37 pp |
| dev-2 | spillover | -0.04 pp | -0.08 pp |
| dev-2 | global | -0.04 pp | -0.10 pp |

## Fixed-mask state transitions at e114

These are descriptive transitions from the fixed e99 S2×T1 mask, not an online selector.

| seed | arm | fixed e99 n | S2→S1 | S2→S2 | S2→S3 | new current S2×T1 entrants |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 2212 | 1344 | 297 | 571 | 2092 |
| dev-1 | DPM | 2212 | 1356 | 310 | 546 | 2070 |
| dev-1 | DBDD | 2212 | 1359 | 304 | 549 | 2075 |
| dev-2 | CONTROL | 2141 | 1239 | 328 | 574 | 2058 |
| dev-2 | DPM | 2141 | 1264 | 314 | 563 | 2074 |
| dev-2 | DBDD | 2141 | 1255 | 315 | 571 | 2049 |

## S-BDD numerical evidence

Both corrected v2 runs used `student_parameter_graph_v2`, the same frozen v2 coefficient, and the same numerical epsilon. The no-update calibration already showed a heavy-tailed achieved-gradient-ratio distribution; no floor, cap, or reciprocal smoothing was introduced in this screen.

| seed | host | first non-finite / terminal evidence |
| --- | --- | --- |
| dev-1 | Hamster GPU1 | last retained finite e105: loss 8.58034e+10; last checkpoint |w|max 3.2e+27; epoch 106: trainer raised FloatingPointError(non-finite training loss) |
| dev-2 | Ferret GPU0 | last retained finite e101: loss 5.66103e+09; not retained locally; terminal run had no valid checkpoint or endpoint after e101 |

The frozen pooled v2 calibration targeted median 0.25 but had achieved ratios spanning 0.05195–20.86 (IQR 1.562), which is retained as a pre-training warning sign rather than an outcome-tuned basis for changing the coefficient.

The two S-BDD failures occurred on Hamster GPU1 and Ferret GPU0 with the same corrected v2 scientific identity; this is therefore not a host/GPU-specific failure. The exact dev-2 first non-finite worker trace was not retained locally, but no valid checkpoint or endpoint exists after e101.

Control, DPM, and D-BDD each reached e114 with finite retained loss/throughput telemetry and registered CE-PGD20 endpoints. Removing S-BDD therefore leaves the preregistered D-BDD vs DPM held-out comparison fully evaluable.

## Scope boundary

This is a fixed-e99 S2×T1 treatment screen. State-transition diagnostics are longitudinal descriptions; they do not turn the fixed mask into an online router. No new training, stabilization variant, threshold change, fresh seed, e199 extension, official test, or AutoAttack was run.
