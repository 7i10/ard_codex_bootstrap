# ERT Dynamic S3 recovery screen

Date: 2026-08-15
Status: complete, descriptive screen only; no route was promoted.

## Contract and lineage

The experiment used the exact Chen ERT epoch-79 parents for L2 (seed 1) and
L4 (seed 2). The action was computed from the pre-update, same-step state:

```text
student clean correct AND student adversarial wrong AND teacher adversarial correct
```

Active samples received baseline RSLAD plus `0.075 * adversarial CE`; all
other samples received baseline RSLAD. DYNBASE logged the predicate without
applying it, S3FIX075 captured it at epoch 80 and froze it, and S3DYN075
recomputed it at every visit. The fixed and dynamic children were forked from
the same epoch-80 shared-prefix checkpoint; the earlier independent-prefix
attempt was rejected by the runtime because model/RNG state differed.

Training attack: KL-PGD10, pixel `[0,1]`, $\epsilon=8/255$, step `2/255`,
random start, teacher-clean target. Every endpoint used an independent
eval-mode CE-PGD20 attack with the same pixel domain, epsilon, step, and random
start. No official test or AutoAttack was run.

Frozen identities:

| Item | SHA-256 |
|---|---|
| source Git SHA | `95e72cd1a2a32caa21686ad76d318eab33e1807a` |
| dynamic config | `39691b9559d6df20baf09c9c33b7f63b8c37dd3e2937309e2aa1899cd26b8660` |
| calibration artifact | `0e3b98b4e1cfcc7727786fd23da57a82903fa2c8b95a16b6e12ca1425d34da16` |
| endpoint attack identity | `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2` |
| report | `be23d725d6568ed65bacba4c02dd9631e2d509f067d7ce8870b90b9f07600cb4` |

All 36 endpoint jobs completed with exit code 0: 3 arms × 2 seeds × 3
horizons (84, 89, 94) × train/validation. Train and validation contained
45,000 and 5,000 rows respectively, with matching stable-ID/class hashes
across arms and horizons.

## Held-out validation results

The values below are common CE-PGD20 validation robust accuracy. They are not
test-set results.

| seed | horizon | DYNBASE | S3FIX075 | S3DYN075 |
|---|---:|---:|---:|---:|
| L2 | 84 | 0.4760 | 0.4734 | 0.4512 |
| L2 | 89 | 0.4606 | 0.4608 | 0.4592 |
| L2 | 94 | 0.4446 | 0.4628 | 0.4412 |
| L4 | 84 | 0.4588 | 0.4542 | 0.4550 |
| L4 | 89 | 0.4736 | 0.4520 | 0.4734 |
| L4 | 94 | 0.4662 | 0.4724 | 0.4662 |

The fixed action is above its control at L2/94 (+1.82 percentage points) and
L4/94 (+0.62 pp), but not at every horizon. Dynamic routing is below control
at L2/84 (-2.48 pp) and L2/94 (-0.34 pp), essentially tied at L4/94, and does
not show a consistent improvement.

## Paired fixed-cohort effects

These are paired sample effects on the epoch-80 captured cohort, not a claim
about changing dynamic cohorts.

| seed | horizon | arm | robust net rescue | rescue | harm | held-out net rescue |
|---|---:|---|---:|---:|---:|---:|
| L2 | 84 | S3FIX075 | 0.0000 | 0.0780 | 0.0780 | -0.0026 |
| L2 | 89 | S3FIX075 | 0.0127 | 0.0770 | 0.0644 | +0.0002 |
| L2 | 94 | S3FIX075 | 0.0294 | 0.0935 | 0.0641 | +0.0182 |
| L4 | 84 | S3FIX075 | 0.0095 | 0.0737 | 0.0641 | -0.0046 |
| L4 | 89 | S3FIX075 | -0.0199 | 0.0587 | 0.0787 | -0.0216 |
| L4 | 94 | S3FIX075 | 0.0156 | 0.0843 | 0.0687 | +0.0062 |
| L2 | 84 | S3DYN075 | -0.0156 | 0.0695 | 0.0851 | -0.0248 |
| L2 | 89 | S3DYN075 | +0.0045 | 0.0735 | 0.0690 | -0.0014 |
| L2 | 94 | S3DYN075 | -0.0071 | 0.0621 | 0.0692 | -0.0034 |
| L4 | 84 | S3DYN075 | +0.0035 | 0.0775 | 0.0740 | -0.0038 |
| L4 | 89 | S3DYN075 | -0.0007 | 0.0649 | 0.0656 | -0.0002 |
| L4 | 94 | S3DYN075 | +0.0135 | 0.0778 | 0.0643 | +0.0000 |

The positive rescue rates are accompanied by substantial harm rates. The
dynamic-minus-fixed held-out net effects by horizon are L2 `-0.0222, -0.0016,
-0.0216` and L4 `+0.0008, +0.0214, -0.0062` for 84/89/94. This is mixed and
does not justify a dynamic-routing claim.

## Transition observations

The dynamic arm did change its action on later visits; it was not an alias for
the fixed arm. The active fraction stayed near 21--22% (L2: 0.2175 at epoch
80 and 0.2156 at epoch 94; L4: 0.2141 at epoch 80 and 0.2145 at epoch 94).
For L2 DYNAMIC, the state table records 74,920 entries, 75,003 exits, 56,318
re-entries, and 149,923 action switches. The corresponding L4 counts are
available in the machine report. These transitions establish that the
same-step implementation exercised recovery/relapse behavior, but they do not
establish causal benefit.

## Operational note and decision

The first endpoint command used the analysis-only schema-v1 dynamic config and
failed closed before creating output. It was rerun with the schema-v2 parent
config required by the endpoint evaluator; all 36 corrected jobs passed. No
training artifact was overwritten.

Evidence summary: **MIXED / unsupported for automatic promotion**. The fixed
arm shows an isolated late-horizon validation gain, while the dynamic arm does
not consistently improve over DYNBASE or S3FIX075 and both arms incur sample
harm. Do not start Stage B, change the coefficient, add hysteresis, add a new
seed, or run official test/AutoAttack from this report. A human-reviewed next
experiment would need a preregistered decision rule and should use the saved
report rather than retuning on these validation values.

The complete machine report remains in the ignored artifact path
`.cache/analysis/ert-dynamic-s3-recovery-v1/dynamic-s3-report.json`; its hash is
recorded above. W&B runs for all six continuations were online and retain the
parent, config, mask, and source lineage.
