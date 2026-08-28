# ERT/RSLAD Static Augmentation Family Screen

Status: complete. This is a two-seed descriptive screen on Hamster; no
official test, AutoAttack, or automatic follow-up training was run.

## Contract and lineage

- Teacher: Chen2021LTD WRN34-10, SHA-256
  `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.
- Student: `saad_resnet18_cifar_v1`, CIFAR-10 controlled train 45,000 /
  validation 5,000, split seed `20260722`.
- Training attack: KL-PGD10, epsilon `8/255`, step `2/255`, random start,
  teacher-clean target.
- Endpoint: independent CE-PGD20, epsilon `8/255`, step `2/255`, random
  start, eval mode; every endpoint has 5,000 validation rows.
- Existing incumbent: CROPSHIFT seeds 1/2 from the trajectory-stabilization
  result record. It was not retrained.
- Candidate policies: `CROP_RE = CROPSHIFT + torchvision 0.26.0 RandomErasing`
  defaults; `IDBH_WEAK = CROPSHIFT + upstream ColorShape('color') + the same
  RandomErasing`.

The seed-1 training manifests are sourced at `63bfe7b`; seed-2 manifests are
sourced at `cabc125`. The only difference between these SHAs is the plan
document update (`git diff --stat` is plan-only); no scientific source or
configuration code changed. The exact per-run SHAs remain recorded in the
machine artifact, and no stronger single-SHA claim is made.

## Complete training trajectories

Values are validation trajectory summaries; AUC values are normalized over
epochs 0--199. `Δ AUC` and endpoint deltas below are candidate minus the
same-seed CROPSHIFT incumbent, in percentage points where marked `pp`.

| arm | seed | best robust (epoch) | last robust | robust AUC | late mean (150--199) | last clean | median img/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CROP_RE | 1 | 0.6040 (198) | 0.6016 | 0.509514 | 0.595248 | 0.8604 | 638.1 |
| CROP_RE | 2 | 0.6012 (196) | 0.5992 | 0.506527 | 0.591160 | 0.8568 | 634.6 |
| IDBH_WEAK | 1 | 0.6054 (193) | 0.6004 | 0.509032 | 0.598272 | 0.8614 | 622.1 |
| IDBH_WEAK | 2 | 0.5996 (195) | 0.5988 | 0.508120 | 0.595184 | 0.8634 | 621.6 |

| candidate | seed | Δ best robust (pp) | Δ last robust (pp) | Δ AUC (pp) | Δ late mean (pp) | Δ last clean (pp) |
|---|---:|---:|---:|---:|---:|---:|
| CROP_RE | 1 | +1.00 | +0.86 | -0.509 | +0.568 | -0.22 |
| CROP_RE | 2 | +0.62 | +0.80 | -0.794 | +0.089 | -0.62 |
| IDBH_WEAK | 1 | +1.14 | +0.74 | -0.557 | +0.871 | -0.12 |
| IDBH_WEAK | 2 | +0.46 | +0.76 | -0.635 | +0.491 | +0.04 |

The candidates start substantially below CROPSHIFT at the early registered
endpoint and catch up late. Thus a positive final gain does not imply a
positive whole-trajectory gain.

## Independent CE-PGD20 endpoint

The following are validation endpoint accuracies. The incumbent rows are the
registered CROPSHIFT values; candidate rows are the 16 newly generated
endpoint reports.

| seed | display epoch | CROPSHIFT robust | CROP_RE robust | IDBH_WEAK robust | CROP_RE clean | IDBH_WEAK clean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 0.4680 | 0.4618 | 0.4456 | 0.7452 | 0.7214 |
| 1 | 99 | 0.4916 | 0.4784 | 0.4604 | 0.7704 | 0.7562 |
| 1 | 149 | 0.5706 | 0.5780 | 0.5786 | 0.8482 | 0.8460 |
| 1 | 199 | 0.5940 | 0.6040 | 0.6052 | 0.8622 | 0.8616 |
| 2 | 49 | 0.4752 | 0.4592 | 0.4424 | 0.7350 | 0.7176 |
| 2 | 99 | 0.4560 | 0.4766 | 0.4606 | 0.7708 | 0.7548 |
| 2 | 149 | 0.5762 | 0.5754 | 0.5712 | 0.8408 | 0.8420 |
| 2 | 199 | 0.5926 | 0.5986 | 0.5972 | 0.8622 | 0.8616 |

Final endpoint robust deltas versus CROPSHIFT are CROP_RE `+1.00 / +0.60
pp` (seeds 1/2) and IDBH_WEAK `+1.12 / +0.46 pp`. On that independent
endpoint, final clean deltas are CROP_RE `+0.10 / 0.00 pp` and IDBH_WEAK
`+0.04 / -0.06 pp`. Separately, the last training-trajectory clean deltas
are CROP_RE `-0.22 / -0.62 pp` and IDBH_WEAK `-0.12 / +0.04 pp`.

## Frozen promotion gate

The preregistered gate required both seeds to improve final robustness and
have non-negative normalized trajectory-AUC increment, with a clean drop over
1 percentage point blocking automatic promotion. Both candidates pass the
final-robustness check and do not trigger the final-clean guardrail, but both
fail the AUC requirement on both seeds:

- CROP_RE: `-0.509` and `-0.794 pp` AUC.
- IDBH_WEAK: `-0.557` and `-0.635 pp` AUC.

Decision: **no candidate is promoted automatically; CROPSHIFT remains the
incumbent static policy**. IDBH_WEAK is not selected over CROP_RE because its
late/final gain does not repair the negative AUC gate, and it adds a slower,
more complex color operation. This is a screen result, not evidence that
either candidate is harmful in all settings.

## Scope and limitations

- Two training seeds only; no population-level seed inference.
- Endpoint results are internal held-out validation, not official CIFAR-10
  test results.
- Candidate endpoint sample rows are stored locally with hashes in the machine
  artifact. Cross-run sample-level treatment effects against CROPSHIFT are not
  claimed because the incumbent's original row artifacts are not part of this
  output bundle.
- W&B tracking was metrics-only; checkpoints and run-bundles were not uploaded.
- No new seed, AutoAttack, post-hoc augmentation tuning, or automatic
  continuation was started.

See the complete hash-bound records in
[`docs/experiments/ert_rslad_static_augmentation_family_v1.json`](experiments/ert_rslad_static_augmentation_family_v1.json).
