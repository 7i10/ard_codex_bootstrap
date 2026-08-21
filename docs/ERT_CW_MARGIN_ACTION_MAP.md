# ERT Clean-Wrong Teacher-Margin × Action Map

Status: completed read-only analysis of the fixed epoch-79 Clean-Wrong cohort and epoch-84 CE-PGD20 endpoint.
No training, threshold tuning, or new replay was performed.

CE20 and KL10 Q1–Q5 are independent pre-treatment quantiles (sort by margin then stable ID).
Accuracy deltas are always paired rescue minus harm; probability-margin deltas are reported separately.

## Primary robust action maps (delta vs C0)

### L2 / CE-PGD20 Teacher margin

| arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| C0 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| C1 | +0.00 pp | +0.12 pp | +0.17 pp | -0.64 pp | +0.00 pp |
| C2 | +0.12 pp | -0.23 pp | -0.12 pp | -1.97 pp | -1.62 pp |
| C3 | +0.06 pp | -0.12 pp | +0.00 pp | -0.12 pp | -1.16 pp |
| C4 | +0.06 pp | -0.12 pp | +0.23 pp | +0.75 pp | +1.10 pp |
| C5 | +0.06 pp | +0.00 pp | +0.17 pp | +0.29 pp | +2.20 pp |
| C6 | +0.00 pp | -0.06 pp | +0.41 pp | +0.00 pp | +0.17 pp |
| C7 | +0.06 pp | -0.17 pp | -0.17 pp | -1.51 pp | -2.20 pp |
| C8 | +0.00 pp | +0.06 pp | -0.35 pp | -1.39 pp | +0.75 pp |
| C9 | +0.17 pp | -0.06 pp | -0.23 pp | -1.39 pp | -2.44 pp |
| C10 | +0.12 pp | +0.12 pp | +0.99 pp | +1.45 pp | +3.89 pp |
| C11 | +0.00 pp | +0.00 pp | +0.29 pp | +0.52 pp | +2.67 pp |
| C12 | +0.17 pp | +0.29 pp | +1.22 pp | +1.04 pp | +2.44 pp |
| C13 | +0.06 pp | +0.00 pp | +0.00 pp | +0.23 pp | +2.61 pp |
| C14 | +0.12 pp | +0.12 pp | +0.17 pp | -0.12 pp | -0.23 pp |
| C15 | +0.06 pp | -0.17 pp | +0.06 pp | -1.33 pp | -1.97 pp |

### L2 / CE-PGD20 clean action map

| arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| C0 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| C1 | +0.23 pp | +0.58 pp | -0.12 pp | +0.06 pp | -0.29 pp |
| C2 | +1.45 pp | +0.06 pp | +0.35 pp | -3.07 pp | -7.42 pp |
| C3 | +0.75 pp | -0.06 pp | +2.20 pp | +0.87 pp | -3.77 pp |
| C4 | +1.16 pp | +1.97 pp | +2.61 pp | +5.28 pp | +1.97 pp |
| C5 | +0.87 pp | +2.72 pp | +4.46 pp | +5.16 pp | +5.16 pp |
| C6 | +0.46 pp | +1.04 pp | +1.45 pp | +1.91 pp | +0.58 pp |
| C7 | +2.38 pp | +3.71 pp | +5.10 pp | +2.49 pp | -2.32 pp |
| C8 | +1.80 pp | +4.17 pp | +3.13 pp | +1.16 pp | -2.09 pp |
| C9 | +0.99 pp | +0.93 pp | -1.04 pp | -3.71 pp | -10.73 pp |
| C10 | +2.38 pp | +4.41 pp | +8.17 pp | +8.06 pp | +4.23 pp |
| C11 | +0.81 pp | +2.38 pp | +2.55 pp | +2.55 pp | -0.75 pp |
| C12 | +2.14 pp | +3.48 pp | +5.57 pp | +4.41 pp | -1.22 pp |
| C13 | +0.58 pp | -0.58 pp | +0.06 pp | +0.35 pp | -0.12 pp |
| C14 | +2.43 pp | +3.36 pp | +2.20 pp | +1.04 pp | -4.99 pp |
| C15 | +1.22 pp | +1.28 pp | +1.45 pp | -3.31 pp | -8.53 pp |

### Pareto arms and CE20/KL10 agreement

- CE20 Pareto by Q: {'Q1': ['C12', 'C14'], 'Q2': ['C10', 'C12'], 'Q3': ['C10', 'C12'], 'Q4': ['C10'], 'Q5': ['C10', 'C5']}
- KL10 Pareto by Q: {'Q1': ['C10', 'C12'], 'Q2': ['C10', 'C12'], 'Q3': ['C10', 'C12'], 'Q4': ['C10'], 'Q5': ['C10', 'C5']}
- flattened robust ranking Spearman (CE20 vs KL10): 0.893214721854724
- per-Q robust ranking Spearman: {'Q1': -0.07596947896071923, 'Q2': 0.5207874311323486, 'Q3': 0.9126016059862982, 'Q4': 0.9749080145717373, 'Q5': 0.9896907216494846}

### L4 / CE-PGD20 Teacher margin

| arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| C0 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| C1 | -0.22 pp | +0.00 pp | -0.45 pp | -0.62 pp | -3.03 pp |
| C2 | -0.17 pp | -0.06 pp | -0.17 pp | -0.28 pp | -3.87 pp |
| C3 | -0.17 pp | -0.06 pp | -0.45 pp | -0.28 pp | -6.27 pp |
| C4 | -0.17 pp | +0.06 pp | +0.28 pp | +0.62 pp | -0.84 pp |
| C5 | -0.22 pp | +0.11 pp | +0.06 pp | +0.34 pp | -2.18 pp |
| C6 | -0.22 pp | +0.06 pp | -0.22 pp | +0.06 pp | -4.54 pp |
| C7 | -0.11 pp | -0.06 pp | -0.22 pp | -0.11 pp | -4.09 pp |
| C8 | -0.22 pp | +0.06 pp | -0.11 pp | -0.28 pp | -2.30 pp |
| C9 | -0.17 pp | -0.11 pp | -0.28 pp | -0.50 pp | -5.71 pp |
| C10 | -0.17 pp | -0.11 pp | +0.39 pp | +0.78 pp | +2.07 pp |
| C11 | -0.11 pp | +0.11 pp | -0.06 pp | +1.23 pp | +1.96 pp |
| C12 | -0.06 pp | +0.17 pp | +0.39 pp | +2.75 pp | +1.62 pp |
| C13 | -0.17 pp | -0.06 pp | -0.28 pp | +0.22 pp | -0.50 pp |
| C14 | -0.22 pp | +0.06 pp | -0.17 pp | -0.22 pp | -2.18 pp |
| C15 | -0.11 pp | +0.00 pp | -0.28 pp | -0.06 pp | -3.59 pp |

### L4 / CE-PGD20 clean action map

| arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| C0 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| C1 | +0.78 pp | +1.18 pp | +2.52 pp | +3.70 pp | +2.35 pp |
| C2 | -0.28 pp | +0.84 pp | +0.56 pp | -0.56 pp | -5.43 pp |
| C3 | -0.67 pp | +0.90 pp | +1.23 pp | +1.01 pp | -3.47 pp |
| C4 | +1.12 pp | +1.40 pp | +2.86 pp | +5.27 pp | +3.31 pp |
| C5 | +1.79 pp | +3.70 pp | +6.39 pp | +8.24 pp | +7.34 pp |
| C6 | +0.22 pp | +2.07 pp | +4.15 pp | +3.47 pp | +0.17 pp |
| C7 | +0.34 pp | +2.52 pp | +4.76 pp | +9.24 pp | +3.36 pp |
| C8 | +1.68 pp | +2.91 pp | +4.65 pp | +6.44 pp | +3.98 pp |
| C9 | -0.67 pp | +0.50 pp | -1.40 pp | -5.88 pp | -11.32 pp |
| C10 | +1.62 pp | +5.43 pp | +9.02 pp | +12.38 pp | +10.08 pp |
| C11 | +0.50 pp | +1.34 pp | +1.51 pp | +1.18 pp | +1.68 pp |
| C12 | +1.06 pp | +3.70 pp | +3.92 pp | +5.32 pp | +4.82 pp |
| C13 | +0.73 pp | +0.56 pp | +2.52 pp | +2.63 pp | +1.12 pp |
| C14 | -0.56 pp | +0.56 pp | +1.06 pp | -1.29 pp | -3.31 pp |
| C15 | -0.39 pp | +1.23 pp | +1.96 pp | -0.67 pp | -3.92 pp |

### Pareto arms and CE20/KL10 agreement

- CE20 Pareto by Q: {'Q1': ['C0', 'C10', 'C12', 'C5'], 'Q2': ['C10', 'C12'], 'Q3': ['C10'], 'Q4': ['C10', 'C12'], 'Q5': ['C10']}
- KL10 Pareto by Q: {'Q1': ['C10', 'C11', 'C5'], 'Q2': ['C10', 'C12'], 'Q3': ['C10', 'C12'], 'Q4': ['C10', 'C12'], 'Q5': ['C10']}
- flattened robust ranking Spearman (CE20 vs KL10): 0.8947622603405716
- per-Q robust ranking Spearman: {'Q1': 0.5952060032756757, 'Q2': 0.5283314005235704, 'Q3': 0.8561857496165666, 'Q4': 0.8944671369436956, 'Q5': 0.9970501410659874}

## Requested comparisons

- CleanCE dose response is available for C0/C4/C10 in the machine artifact; it is descriptive.
  No new coefficient was selected.
- AdvKD pressure dose response is available for C9/C2/C0/C11.
- Attack-budget comparison is available for C8/C1/C0.
- Robust-side comparison is available for C10/C11/C12/C13/C0. C12 remains MART-inspired BCE, not plain AdvCE.
- C0–C7 factorial main effects and two-way difference-in-differences are recorded per seed,
  margin domain, and Q.
- Held-out subtype transfer is not reported: pre-treatment CE20/KL10 artifacts are train-only.
  No GPU replay was started.

## Interpretation guardrails

This is a subtype/action heterogeneity map, not a validated router. Q5 is not an optimal threshold.
Direct train effects cannot be promoted to held-out effects.
The previous four-arm sign-gate failure is not reclassified as solved.
No new training or automatic winner promotion was performed.
