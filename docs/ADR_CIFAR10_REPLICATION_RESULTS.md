# ADR CIFAR-10 replication: official test results

## Decision

Verdict: **SIGN_CONFIRMED**.

- ResNet-18 (Nesterov-matched pair, `adr` vs `pgd_at_nesterov`): mean AutoAttack gain +1.22 pp (n=3, sd=0.36, range +0.85..+1.56)
- MobileNetV2 (`mobilenetv2_adr` vs `mobilenetv2_pgd_at`): mean AutoAttack gain +2.79 pp (n=3, sd=0.36, range +2.38..+3.02)

SIGN_CONFIRMED if MobileNetV2's mean gain is positive and exceeds ResNet-18's (Nesterov-matched pair); NOT_CONFIRMED if MobileNetV2's gain is <= ResNet-18's and <= 0; otherwise MIXED

## Per-arm summary (best checkpoint, model weights, AutoAttack, percentage points)

| arm | role | seeds | mean | sd |
|---|---|---|---:|---:|
| `pgd_at` | baseline_r18_plain_sgd | [0, 1, 2] | 47.28 | 0.44 |
| `pgd_at_nesterov` | baseline_r18_nesterov_matched | [0, 1, 2] | 47.32 | 0.12 |
| `adr` | adr_r18 | [0, 1, 2] | 48.54 | 0.28 |
| `trades` | baseline_trades_r18 | [0, 1, 2] | 47.55 | 0.57 |
| `trades_adr` | adr_trades_r18 | [0, 1, 2] | 48.73 | 0.40 |
| `trades_49k_validation` | trades_49k_validation_pilot | [0] | 47.64 | 0.00 |
| `mobilenetv2_pgd_at` | baseline_mobilenetv2 | [0, 1, 2] | 39.92 | 0.23 |
| `mobilenetv2_adr` | adr_mobilenetv2 | [0, 1, 2] | 42.72 | 0.16 |

## Secondary comparisons

- TRADES (`trades_adr` vs `trades`, no Nesterov-matched TRADES baseline exists): mean gain +1.18 pp (n=3, sd=0.56, range +0.56..+1.65)

this contract measures three seeds (one pilot arm at one seed) at one protocol family; it authorizes no promotion, no ImageNet launch, and no method claim beyond the preregistered sign comparison -- see the plan's preregistered decision rule for the full scope

## Provenance

Record: `docs/experiments/adr_cifar10_replication_v1.json`, hash in `docs/experiments/adr_cifar10_replication_v1.json.sha256`. Contract `adr_cifar10_replication_v1`. Aggregation source SHA `836b3aef02e76a788a3f0f09082958ace61ccadf`. Plan: `docs/plans/0097-adr-cifar10-replication.md`.
