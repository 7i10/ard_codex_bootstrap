# FF/NR causal pilot: epoch-79 to horizon-94 extension

更新日: 2026-08-09

The original five-arm Route A/B pilot was continued from the same complete
epoch-79 parent, with no new selector, mask, q, coefficient, or treatment.
Both Chen seeds were run for C79, RA, RAR, RB, and RBR. The generated configs
use `training.epochs=94`; zero-based metric epochs 83, 88, and 93 correspond
to the requested horizons 84, 89, and 94. All ten arms reached horizon 94,
and each has a final `sample-stats-train.parquet`.

## Validation PGD trajectory

The following are selected-minus-matched-random differences in validation
PGD accuracy (percentage points). They are descriptive and were not used to
change the mask or launch a new treatment.

| seed | route | horizon 84 | horizon 89 | horizon 94 |
|---|---|---:|---:|---:|
| L2 / Chen seed 1 | A (RA−RAR) | +1.56 | +1.68 | +0.42 |
| L2 / Chen seed 1 | B (RB−RBR) | −0.62 | −0.66 | +0.28 |
| L4 / Chen seed 2 | A (RA−RAR) | +0.56 | −1.42 | +0.44 |
| L4 / Chen seed 2 | B (RB−RBR) | −0.78 | +0.78 | +1.74 |

The Route A advantage is therefore not stable across seed or horizon. Route
B becomes positive at the final horizon for both seeds, but the L2 endpoint
is only +0.28 pp and the L4 endpoint is +1.74 pp; this is not by itself a
confirmed performance claim.

## Endpoint training-state diagnostic (not the preregistered causal endpoint)

For each route, the final `sample-stats-train.parquet` robust-correct field
was compared with its same-parent C79 control on the fixed selected/random
mask. This field is an augmented training-batch state under the training
KL-PGD10 path. It is therefore an exploratory training-state diagnostic, not
the preregistered common eval-mode CE-PGD20 sample endpoint and not an
official-test estimate. A fixed-mask CE-PGD20 replay is still required before
calling this a causal treatment effect.

| seed | route | selected training-state delta | random training-state delta | selected−random |
|---|---|---:|---:|---:|
| L2 | A | +2.91% | +2.95% | −0.04 pp |
| L2 | B | +3.20% | +2.49% | +0.71 pp |
| L4 | A | +2.35% | +1.86% | +0.49 pp |
| L4 | B | +1.93% | +2.20% | −0.26 pp |

The mismatch between validation PGD and this training-state diagnostic is
expected:
the former is a validation aggregate, while the latter is a selected-mask
within-training-panel comparison. It is evidence that the small Route B
signal is seed- and estimand-sensitive, not evidence to promote Route B as a
final method.

## Reproducibility and execution

- parent: Chen epoch-79 complete-state checkpoints, one per seed;
- masks: pre-existing registered Route A/B masks and matched-random controls;
- continuation: fixed fork generator at commit `b296b8b`;
- hosts: L2 on Hamster, L4 on Ferret;
- transfer: final analysis files were collected from Ferret with
  `rsync --checksum` (72 MB received in this run); no separate transfer
  attestation was persisted and checkpoints were not copied;
- no official test, AutoAttack, or 200-epoch retraining was run in this
  extension.

The ignored machine-readable summaries are
`.cache/analysis/causal_horizon_metrics.json` and
`.cache/analysis/causal_endpoint_effects.json`.
