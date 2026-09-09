# ERT / RSLAD baseline RNG-source decomposition results

Status: completed for the preregistered 16 trajectories and independent CE-PGD20 endpoint matrix.

## Scope and integrity

This report is descriptive. It does not treat the two source perturbations as population samples, does not select a winner for a future method, and does not include official test or AutoAttack.

- Source Git SHA: `09e627e95a66a136a0cc7aa15bcb4deab141c719`
- Parent epoch: `79`; L2 parent `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`; L4 parent `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`
- Teacher checkpoint SHA: `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`
- Training attack: `{'norm': 'linf', 'input_domain': 'pixel_0_1', 'epsilon': '8/255', 'epsilon_value': 0.03137254901960784, 'step_size': '2/255', 'step_size_value': 0.00784313725490196, 'steps': 10, 'random_start': True, 'loss': 'kl', 'kl_target': 'teacher_clean', 'temperature': 1.0, 'temperature_squared': True, 'student_mode': 'eval', 'teacher_mode': 'eval'}`
- Endpoint attack: `{'norm': 'linf', 'input_domain': 'pixel_0_1', 'epsilon': '8/255', 'epsilon_value': 0.03137254901960784, 'step_size': '2/255', 'step_size_value': 0.00784313725490196, 'steps': 20, 'random_start': True, 'loss': 'ce', 'kl_target': None, 'temperature': 1.0, 'temperature_squared': True, 'student_mode': 'eval', 'teacher_mode': 'eval'}`; attack identity SHA `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- Endpoint records: `96/96`; trajectory records: `16/16`
- Input inventory SHA-256: `5ef2d9dd88e6ee9de1647b997be73cb5aa24616ee2eb8f1a685b4de909bd8b3c`

All trajectories reached epoch 94. Every endpoint record uses the same independent eval-mode CE-PGD20 contract, with 45,000 train rows or 5,000 fixed validation rows. The evaluation seed is 0.

## Primary endpoint: epoch 94 validation

The primary endpoint is independent validation CE-PGD20 robust accuracy. Values are absolute accuracies; no checkpoint was chosen from this table.

| Teacher | Arm | Clean | Robust | Δ robust vs REF1 |
| --- | --- | --- | --- | --- |
| L2 | REF1 | 0.7856 | 0.4732 | 0.0000 |
| L2 | REF2 | 0.7856 | 0.4732 | 0.0000 |
| L2 | ATTACK1 | 0.7814 | 0.4660 | -0.0072 |
| L2 | ATTACK2 | 0.7774 | 0.4748 | 0.0016 |
| L2 | DATA1 | 0.7760 | 0.4474 | -0.0258 |
| L2 | DATA2 | 0.7894 | 0.4690 | -0.0042 |
| L2 | BOTH1 | 0.7690 | 0.4668 | -0.0064 |
| L2 | BOTH2 | 0.7868 | 0.4694 | -0.0038 |
| L4 | REF1 | 0.7556 | 0.4622 | 0.0000 |
| L4 | REF2 | 0.7556 | 0.4622 | 0.0000 |
| L4 | ATTACK1 | 0.7702 | 0.4706 | 0.0084 |
| L4 | ATTACK2 | 0.7644 | 0.4546 | -0.0076 |
| L4 | DATA1 | 0.7794 | 0.4622 | 0.0000 |
| L4 | DATA2 | 0.7844 | 0.4826 | 0.0204 |
| L4 | BOTH1 | 0.7722 | 0.4682 | 0.0060 |
| L4 | BOTH2 | 0.7924 | 0.4768 | 0.0146 |

## Validation robust trajectory

Absolute CE-PGD20 validation robust accuracy at epochs 84, 89, and 94.

| Teacher | Arm | 84 | 89 | 94 |
| --- | --- | --- | --- | --- |
| L2 | REF1 | 0.4654 | 0.4554 | 0.4732 |
| L2 | REF2 | 0.4654 | 0.4554 | 0.4732 |
| L2 | ATTACK1 | 0.4642 | 0.4654 | 0.4660 |
| L2 | ATTACK2 | 0.4652 | 0.4632 | 0.4748 |
| L2 | DATA1 | 0.4696 | 0.4604 | 0.4474 |
| L2 | DATA2 | 0.4592 | 0.4546 | 0.4690 |
| L2 | BOTH1 | 0.4682 | 0.4408 | 0.4668 |
| L2 | BOTH2 | 0.4732 | 0.4544 | 0.4694 |
| L4 | REF1 | 0.4750 | 0.4698 | 0.4622 |
| L4 | REF2 | 0.4750 | 0.4698 | 0.4622 |
| L4 | ATTACK1 | 0.4740 | 0.4780 | 0.4706 |
| L4 | ATTACK2 | 0.4710 | 0.4736 | 0.4546 |
| L4 | DATA1 | 0.4650 | 0.4618 | 0.4622 |
| L4 | DATA2 | 0.4546 | 0.4562 | 0.4826 |
| L4 | BOTH1 | 0.4722 | 0.4640 | 0.4682 |
| L4 | BOTH2 | 0.4532 | 0.4640 | 0.4768 |

## Source decomposition at the primary endpoint

For each teacher and metric, REF1 is the reference. `attack`, `data`, and `both` effects are the two preregistered perturbations relative to REF1. Interaction is `both - attack - data` using matched replicate indices.

### Robust accuracy (validation, epoch 94)

| Teacher | REF1 | REF2−REF1 | A1 | A2 | D1 | D2 | AD1 | AD2 | I1 | I2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 0.4732 | 0.0000 | -0.0072 | 0.0016 | -0.0258 | -0.0042 | -0.0064 | -0.0038 | 0.0266 | -0.0012 |
| L4 | 0.4622 | 0.0000 | 0.0084 | -0.0076 | 0.0000 | 0.0204 | 0.0060 | 0.0146 | -0.0024 | 0.0018 |

### Clean accuracy (validation, epoch 94)

| Teacher | REF1 | REF2−REF1 | A1 | A2 | D1 | D2 | AD1 | AD2 | I1 | I2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 0.7856 | 0.0000 | -0.0042 | -0.0082 | -0.0096 | 0.0038 | -0.0166 | 0.0012 | -0.0028 | 0.0056 |
| L4 | 0.7556 | 0.0000 | 0.0146 | 0.0088 | 0.0238 | 0.0288 | 0.0166 | 0.0368 | -0.0218 | -0.0008 |

## Training-side epoch-94 summary

These are the training process's validation-PGD metrics, not the independent CE-PGD20 endpoint. They are included to expose any selection/evaluation distinction.

| Teacher | Arm | Best epoch | Best val PGD | Epoch-94 val PGD | Epoch-94 val clean | Epoch-94 train robust |
| --- | --- | --- | --- | --- | --- | --- |
| L2 | REF1 | 91 | 0.4852 | 0.4730 | 0.7856 | 0.5637 |
| L2 | REF2 | 91 | 0.4852 | 0.4730 | 0.7856 | 0.5637 |
| L2 | ATTACK1 | 91 | 0.4780 | 0.4670 | 0.7814 | 0.5647 |
| L2 | ATTACK2 | 94 | 0.4756 | 0.4756 | 0.7774 | 0.5613 |
| L2 | DATA1 | 87 | 0.4750 | 0.4466 | 0.7760 | 0.5623 |
| L2 | DATA2 | 81 | 0.4774 | 0.4702 | 0.7894 | 0.5597 |
| L2 | BOTH1 | 87 | 0.4802 | 0.4662 | 0.7690 | 0.5617 |
| L2 | BOTH2 | 84 | 0.4740 | 0.4704 | 0.7870 | 0.5603 |
| L4 | REF1 | 90 | 0.4808 | 0.4604 | 0.7556 | 0.5650 |
| L4 | REF2 | 90 | 0.4808 | 0.4604 | 0.7556 | 0.5650 |
| L4 | ATTACK1 | 90 | 0.4806 | 0.4702 | 0.7700 | 0.5611 |
| L4 | ATTACK2 | 91 | 0.4748 | 0.4558 | 0.7644 | 0.5616 |
| L4 | DATA1 | 87 | 0.4832 | 0.4620 | 0.7794 | 0.5583 |
| L4 | DATA2 | 94 | 0.4818 | 0.4818 | 0.7846 | 0.5661 |
| L4 | BOTH1 | 92 | 0.4780 | 0.4698 | 0.7724 | 0.5612 |
| L4 | BOTH2 | 85 | 0.4798 | 0.4774 | 0.7924 | 0.5631 |

## Interpretation

### Reference residual

- L2: REF2−REF1 = `0.0000` robust-accuracy points (absolute fraction). This is the observed same-source residual, not a population uncertainty estimate.
- L4: REF2−REF1 = `0.0000` robust-accuracy points (absolute fraction). This is the observed same-source residual, not a population uncertainty estimate.

### Source sensitivity

- L2: mean absolute source effects — data=0.0150, interaction=0.0139, attack=0.0044.
- L4: mean absolute source effects — data=0.0102, attack=0.0080, interaction=0.0021.

The decomposition identifies how these particular post-epoch-79 continuations diverged under the frozen protocol. It does not establish that one RNG source dominates in general, and it does not justify changing the attack, data pipeline, or training objective.

## Registered stop boundary

The preregistered campaign is complete. No stabilization run, new seed, official test, or AutoAttack was started automatically.
