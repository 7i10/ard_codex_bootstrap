# ERT Clean-Wrong A7 mechanism diagnostic

> Read-only, no-update checkpoint replay. These results are descriptive mechanism evidence, not causal proof.

## Frozen contract

A5 fixed, A6 zero/cap, A7 positive-floor, and A8 abstain were replayed with unchanged Teacher-clean KL-PGD10. R0--R3 are training-time regimes; CE20 Q1--Q5 are a separate pre-treatment axis.

## L2

### Regime and hinge summary

Epoch-wise regime/target/deficit/hinge summaries are stored in the machine artifact.

### No-update gradient probe

The fixed 128-ID probes are diagnostics only; they do not tune coefficients or use endpoint outcomes.

| arm | epoch | base norm | margin/base ratio | margin/base cosine |
|---|---:|---:|---:|---:|
| A5 | 79 | 9.78548 | 0.0641434 | 0.796257 |
| A5 | 94 | 10.3754 | 0.0554049 | 0.803116 |
| A6 | 79 | 9.78548 | 0.0642624 | 0.795918 |
| A6 | 94 | 9.52316 | 0.0622335 | 0.77593 |
| A7 | 79 | 9.78548 | 0.0642624 | 0.795918 |
| A7 | 94 | 8.46339 | 0.0672895 | 0.768373 |
| A8 | 79 | 9.78548 | 0.0363939 | 0.594403 |
| A8 | 94 | 12.1727 | 0.0317379 | 0.55793 |

- epoch 79: A7 R0=5171, R1=893, R2=1711, R3=848; positive-deficit mean=0.279342
- epoch 84: A7 R0=4696, R1=911, R2=1969, R3=1047; positive-deficit mean=0.231058
- epoch 89: A7 R0=4622, R1=947, R2=2088, R3=966; positive-deficit mean=0.207738
- epoch 94: A7 R0=5106, R1=810, R2=1732, R3=975; positive-deficit mean=0.264630

### Endpoint effects vs BASE (epoch 94)

| arm | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |
|---|---:|---:|---:|---:|---:|---:|
| A5 | +0.0155 | -0.0216 | 1864 | 1168 | 1817 | 2791 |
| A6 | -0.0017 | +0.0047 | 1202 | 1277 | 2113 | 1901 |
| A7 | +0.0035 | +0.0040 | 1340 | 1181 | 2051 | 1869 |
| A8 | -0.0099 | -0.0126 | 1128 | 1575 | 1804 | 2373 |

### A7 dominant-regime rescue/harm

| dominant regime | rescue | harm | net |
|---|---:|---:|---:|
| R0 | 7 | 6 | 1 |
| R1 | 4 | 3 | 1 |
| R2 | 43 | 20 | 23 |
| R3 | 54 | 41 | 13 |

## L4

### Regime and hinge summary

Epoch-wise regime/target/deficit/hinge summaries are stored in the machine artifact.

### No-update gradient probe

The fixed 128-ID probes are diagnostics only; they do not tune coefficients or use endpoint outcomes.

| arm | epoch | base norm | margin/base ratio | margin/base cosine |
|---|---:|---:|---:|---:|
| A5 | 79 | 11.4617 | 0.0560417 | 0.783571 |
| A5 | 94 | 8.84544 | 0.0621427 | 0.774109 |
| A6 | 79 | 11.4617 | 0.0562328 | 0.788552 |
| A6 | 94 | 9.0072 | 0.063719 | 0.796515 |
| A7 | 79 | 11.4617 | 0.056041 | 0.786235 |
| A7 | 94 | 8.80861 | 0.0604882 | 0.717156 |
| A8 | 79 | 11.4617 | 0.0315593 | 0.675528 |
| A8 | 94 | 9.54896 | 0.0408274 | 0.635644 |

- epoch 79: A7 R0=5196, R1=902, R2=1880, R3=947; positive-deficit mean=0.282816
- epoch 84: A7 R0=4937, R1=860, R2=2011, R3=1117; positive-deficit mean=0.216899
- epoch 89: A7 R0=5037, R1=858, R2=1963, R3=1067; positive-deficit mean=0.211787
- epoch 94: A7 R0=4984, R1=851, R2=2014, R3=1076; positive-deficit mean=0.225163

### Endpoint effects vs BASE (epoch 94)

| arm | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |
|---|---:|---:|---:|---:|---:|---:|
| A5 | +0.0191 | -0.0108 | 2011 | 1153 | 2149 | 2636 |
| A6 | -0.0011 | -0.0009 | 1316 | 1364 | 2156 | 2195 |
| A7 | +0.0009 | +0.0088 | 1323 | 1281 | 2355 | 1960 |
| A8 | +0.0019 | +0.0042 | 1320 | 1236 | 2132 | 1941 |

### A7 dominant-regime rescue/harm

| dominant regime | rescue | harm | net |
|---|---:|---:|---:|
| R0 | 8 | 1 | 7 |
| R1 | 8 | 4 | 4 |
| R2 | 65 | 31 | 34 |
| R3 | 96 | 53 | 43 |

## Interpretation boundary and next priorities

The endpoint tables are descriptive joins to replayed training-time regimes; they do not identify a causal effect of any target rule.
The fixed 128-ID gradient probes are no-update diagnostics and were not used to select a coefficient or treatment.
A7 follow-up priorities remain: (1) isolate A7 from extra CleanCE, (2) preregister a small lambda sensitivity check, and (3) separately test floor/cap sensitivity.
No follow-up training is started by this report.

