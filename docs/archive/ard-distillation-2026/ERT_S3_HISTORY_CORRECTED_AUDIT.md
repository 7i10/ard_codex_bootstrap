# ERT S3 corrected history audit

Date: 2026-08-16

Status: passed pre-GPU gate; this is a mechanism audit, not a performance
result. No validation/endpoint metric, training outcome, new seed, official
test, or AutoAttack was used.

## Correction

The original offline replay treated every adversarial-wrong visit as S3. That
was inconsistent with the production action predicate because clean-wrong
samples are a separate route. The corrected observation is:

```text
student_clean_correct AND NOT student_adv_correct
```

The current Teacher adversarial-correctness gate remains separate and is not
history-smoothed. Majority-3 requires all three visits; partial windows are
inactive. Exit2 requires two consecutive current clean-and-adversarial-correct
visits; clean-wrong resets the streak.

Inputs are the same L2/L4 S3DYN075 trajectories (45,000 IDs × epochs 80--94):

| run | trajectory SHA-256 |
|---|---|
| L2 | `cf6827c7b3b8b605720152ea6ebaf7532f434f43d97787396679101661d1cfff` |
| L4 | `e3f40b10e4926c09aefc7a381f2f684b028aeb82f59444675d14be17908a0d5a` |

Corrected report SHA-256: `0c937a0f73c9a9b076ff0a0d2a62dc11346cb7530567bf4b4db02d05a5302ff9`.

## Rule audit

Values are per seed; capture is the descriptive next-three-visit reference,
not a training target.

| seed | rule | action fraction | switches | re-entries | 1-visit re-entry | action capture | median entry delay |
|---|---|---:|---:|---:|---:|---:|---:|
| L2 | Instant | .2148 | 149,923 | 49,758 | .3629 | .4800 | 0 |
| L2 | Majority-3 | .1653 | 65,040 | 15,820 | .1483 | .4059 | 2 |
| L2 | Majority-3 + exit2 | .1868 | 54,672 | 9,987 | 0 | .4501 | 2 |
| L4 | Instant | .2140 | 149,838 | 49,934 | .3631 | .4788 | 0 |
| L4 | Majority-3 | .1648 | 65,062 | 15,886 | .1516 | .4063 | 2 |
| L4 | Majority-3 + exit2 | .1867 | 54,583 | 9,944 | 0 | .4500 | 2 |

## Gate decision

The corrected audit preserves the cross-seed mechanism conclusion:

- Majority-3 reduces action switches by about 56.6% versus Instant in both
  seeds, at the cost of a two-visit entry delay and lower near-future capture.
- Exit2 removes a further 15.9--16.0% of Majority-3 switches, eliminates
  one-visit re-entry, and recovers part of the capture loss without increasing
  entry delay.
- L2 and L4 agree closely. The compact persistence candidate remains
  Majority-3 + exit2.

Therefore the GPU gate is **passed** for the preregistered 8-arm screen. This
does not select a performance winner. BASE, INST075, M3_075, and M3E2_075 use
the frozen `beta_advce=0.075` and the exact Chen epoch-79 parents; conditions
must not be changed after launch.

Reproduction command:

```bash
PYTHONPATH=src python -m ard.cli.ert_s3_history_replay \
  --config configs/analysis/ert_s3_history_replay_v1.yaml \
  --output .cache/analysis/ert-s3-history-replay-v1/corrected-report.json
```
