# Bartoldson CE-PGD20 replay (L1/L3)

更新日: 2026-08-09

This is an offline replay of saved complete-state checkpoints, not a new
training run. Both seeds use the same pixel-space CE-PGD20 attack (`epsilon
= 8/255`, step `2/255`, random start), batch size 64, deterministic replay
backend, and epochs 104, 109, 114. Seed 1 is the local
`bart-rslad-logging-only-s1-confirm-v1` run. Seed 2 is the recovered W&B
control run `bart-h3-c-s2-20260802-v1`.

## Future-failure definition

The outcome is `student_robust_correct(epoch 104)=true` and
`student_robust_correct(epoch 114)=false`. It is a short prospective window,
not a claim about the full 104–199 trajectory.

| seed | future failures | denominator | rate |
|---|---:|---:|---:|
| L1 | 1,999 | 45,000 | 4.44% |
| L3 | 1,762 | 45,000 | 3.92% |

For the failure group, teacher adversarial correctness is 99.95% (L1) and
100.00% (L3); the non-failure groups are 99.12% and 99.13%, respectively.
The all-sample exploratory association at epoch 104 is small for both
signals: teacher signed dominance (maximum wrong-class probability minus
true-class probability) has Spearman rho 0.098 (L1) and 0.090 (L3), while
student adversarial probability margin has rho -0.029 and -0.030. The latter
is not a clean prospective comparison: the all-sample non-failure group
includes samples already wrong at epoch 104, and the epoch-114 margin is
contemporaneous with the endpoint correctness used to define the outcome.
When restricted to the epoch-104-correct risk set and predictors available at
that anchor, the corresponding rhos are teacher 0.144/0.132 and student
margin -0.341/-0.320 (L1/L3). These are association diagnostics only; they do
not establish that either signal is causal or that the teacher is
misclassifying the samples.

The exact row-level outputs and attack/runtime lineage are ignored artifacts:

- `.cache/analysis/ffnr-strong-replay/bart-l1-ce-pgd20-104-109-114-v2/`
- `.cache/analysis/ffnr-strong-replay/bart-l3-ce-pgd20-104-109-114-v1/`

No official test or AutoAttack was run as part of this replay.
