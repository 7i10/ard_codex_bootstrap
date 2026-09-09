# I100 official CIFAR-10 test and AutoAttack

## Decision

The preregistered primary endpoint is the epoch-199 last-checkpoint AutoAttack
accuracy on the official 10,000-example CIFAR-10 test set, I100 minus its matched
CROP_SUFFIX control, on the three previously unused confirmation seeds.

Verdict: **CONFIRMED**.  Per seed: confirm-a +0.41 pp, confirm-b +0.06 pp, confirm-c +0.46 pp.

The rule was fixed before the runs: positive in all three seeds confirms the
claim, two or more negative refutes it, anything else is mixed.  For reference,
the internal-validation CE-PGD20 differences these runs were meant to test were
+0.78 / +0.68 / +0.62 pp.
One test example is 0.01 pp.

This screen measures the official test endpoint only; it authorizes no promotion, no new seed, no architecture or dataset extension, and no sample-level intervention.

## Last checkpoint (epoch 199)

| seed | arm | clean | CE-PGD20 | AutoAttack |
| --- | --- | ---: | ---: | ---: |
| confirm-a | I100 | 82.76% | 57.25% | 53.08% |
| confirm-a | CROP | 82.96% | 56.56% | 52.67% |
| confirm-b | I100 | 82.86% | 57.07% | 53.02% |
| confirm-b | CROP | 82.91% | 56.78% | 52.96% |
| confirm-c | I100 | 82.66% | 57.06% | 53.19% |
| confirm-c | CROP | 82.74% | 56.57% | 52.73% |

| seed | clean Δ | CE-PGD20 Δ | AutoAttack Δ |
| --- | ---: | ---: | ---: |
| confirm-a | -0.20 pp | +0.69 pp | +0.41 pp |
| confirm-b | -0.05 pp | +0.29 pp | +0.06 pp |
| confirm-c | -0.08 pp | +0.49 pp | +0.46 pp |

## Best checkpoint

| seed | arm | clean | CE-PGD20 | AutoAttack |
| --- | --- | ---: | ---: | ---: |
| confirm-a | I100 | 82.78% | 57.28% | 53.08% |
| confirm-a | CROP | 82.82% | 56.73% | 52.78% |
| confirm-b | I100 | 82.86% | 57.07% | 53.02% |
| confirm-b | CROP | 83.22% | 56.68% | 52.57% |
| confirm-c | I100 | 82.82% | 57.03% | 52.96% |
| confirm-c | CROP | 82.87% | 56.39% | 52.54% |

| seed | clean Δ | CE-PGD20 Δ | AutoAttack Δ |
| --- | ---: | ---: | ---: |
| confirm-a | -0.04 pp | +0.55 pp | +0.30 pp |
| confirm-b | -0.36 pp | +0.39 pp | +0.45 pp |
| confirm-c | -0.05 pp | +0.64 pp | +0.42 pp |

## Robust overfitting

Best minus last AutoAttack accuracy, per arm.  A larger value means the arm lost
more robustness between its best epoch and the end of training.

| seed | I100 | CROP_SUFFIX |
| --- | ---: | ---: |
| confirm-a | +0.00 pp | +0.11 pp |
| confirm-b | +0.00 pp | -0.39 pp |
| confirm-c | -0.23 pp | -0.19 pp |

## Scope and lineage

Every evaluation read only saved weights in a separate process.  Each was checked
before use: the official test split with 10,000 examples, the
registered CE-PGD20 threat identity `7081101693340e70...`, and
standard AutoAttack from the pinned upstream commit
`a39220048b3c...`.  Aggregation source `7f37aa2a8077...`;
campaign root `/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/i100-official-test-v1`.

| seed | arm | training run | evaluation record SHA-256 |
| --- | --- | --- | --- |
| confirm-a | I100 | `unseen-confirm-a-i100-suffix-training` | `89ae741a114b1b51...` |
| confirm-a | CROP | `unseen-confirm-a-crop-suffix-r2-training` | `e87812f042621fc4...` |
| confirm-b | I100 | `unseen-confirm-b-i100-suffix-training` | `3bc27d6916b92926...` |
| confirm-b | CROP | `unseen-confirm-b-crop-suffix-training` | `eebef0403d924c9e...` |
| confirm-c | I100 | `unseen-confirm-c-i100-suffix-r3-training` | `ba7facee1e50dd31...` |
| confirm-c | CROP | `unseen-confirm-c-crop-suffix-r2-training` | `e9e377e15411c5e4...` |

Record: `docs/experiments/ard_i100_official_test_autoattack_v1.json` (SHA-256 in the sidecar).
Plan: `docs/plans/0091-i100-official-test-autoattack.md`.
