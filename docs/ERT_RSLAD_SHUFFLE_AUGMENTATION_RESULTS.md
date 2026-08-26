# ERT / RSLAD shuffle-vs-augmentation RNG decomposition results

Status: completed for the preregistered 16 trajectories and 96 independent CE-PGD20 endpoints.

## Scope and integrity

This is a descriptive source decomposition. REF2 is reported as a residual control; the two perturbations are not population samples and no arm is promoted from this report.

- Source Git SHA: `e066fd3300cb58b5d2f32c2820cab3ed81ce9f9d`
- Parent epoch: `79`; L2 `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`; L4 `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`
- Teacher checkpoint SHA: `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`
- Training attack: `{"epsilon": "8/255", "input_domain": "pixel_0_1", "kl_target": "teacher_clean", "loss": "kl", "norm": "linf", "random_start": true, "step_size": "2/255", "steps": 10, "student_mode": "eval", "teacher_mode": "eval", "temperature": 1.0, "temperature_squared": true}`
- Endpoint attack: `{"epsilon": "8/255", "input_domain": "pixel_0_1", "kl_target": null, "loss": "ce", "norm": "linf", "random_start": true, "step_size": "2/255", "steps": 20, "student_mode": "eval", "teacher_mode": "eval", "temperature": 1.0, "temperature_squared": true}`
- Endpoint attack identity SHA: `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- Trajectories: `16/16`; endpoints: `96/96`
- Input inventory SHA-256: `b32c14d4e2d60922d4421722585e73e204bdaadb4f328febb2281b7e3a036850`

All trajectory manifests reached epoch 94 and all endpoint rows have the registered train (45,000) or fixed validation (5,000) cardinality. No official test or AutoAttack was run.

## Primary endpoint: epoch-94 validation CE-PGD20

Absolute accuracy is shown; Δ is relative to REF1 within the same teacher and endpoint.

| Teacher | Arm | Clean | Robust | Δ robust vs REF1 |
| --- | --- | --- | --- | --- |
| L2 | REF1 | 0.7838 | 0.4700 | 0.0000 |
| L2 | REF2 | 0.7838 | 0.4700 | 0.0000 |
| L2 | SHUF1 | 0.7806 | 0.4800 | 0.0100 |
| L2 | SHUF2 | 0.7912 | 0.4648 | -0.0052 |
| L2 | AUG1 | 0.7700 | 0.4714 | 0.0014 |
| L2 | AUG2 | 0.7862 | 0.4726 | 0.0026 |
| L2 | BOTH1 | 0.7886 | 0.4562 | -0.0138 |
| L2 | BOTH2 | 0.7818 | 0.4674 | -0.0026 |
| L4 | REF1 | 0.7704 | 0.4690 | 0.0000 |
| L4 | REF2 | 0.7704 | 0.4690 | 0.0000 |
| L4 | SHUF1 | 0.7702 | 0.4666 | -0.0024 |
| L4 | SHUF2 | 0.7932 | 0.4734 | 0.0044 |
| L4 | AUG1 | 0.7724 | 0.4732 | 0.0042 |
| L4 | AUG2 | 0.7796 | 0.4810 | 0.0120 |
| L4 | BOTH1 | 0.7614 | 0.4512 | -0.0178 |
| L4 | BOTH2 | 0.7846 | 0.4728 | 0.0038 |

## Validation robust trajectory

Independent CE-PGD20 robust accuracy at registered horizons.

| Teacher | Arm | 84 | 89 | 94 |
| --- | --- | --- | --- | --- |
| L2 | REF1 | 0.4726 | 0.4580 | 0.4700 |
| L2 | REF2 | 0.4726 | 0.4580 | 0.4700 |
| L2 | SHUF1 | 0.4626 | 0.4756 | 0.4800 |
| L2 | SHUF2 | 0.4546 | 0.4512 | 0.4648 |
| L2 | AUG1 | 0.4678 | 0.4566 | 0.4714 |
| L2 | AUG2 | 0.4602 | 0.4750 | 0.4726 |
| L2 | BOTH1 | 0.4746 | 0.4720 | 0.4562 |
| L2 | BOTH2 | 0.4416 | 0.4576 | 0.4674 |
| L4 | REF1 | 0.4626 | 0.4538 | 0.4690 |
| L4 | REF2 | 0.4626 | 0.4538 | 0.4690 |
| L4 | SHUF1 | 0.4796 | 0.4364 | 0.4666 |
| L4 | SHUF2 | 0.4762 | 0.4578 | 0.4734 |
| L4 | AUG1 | 0.4706 | 0.4608 | 0.4732 |
| L4 | AUG2 | 0.4736 | 0.4588 | 0.4810 |
| L4 | BOTH1 | 0.4680 | 0.4594 | 0.4512 |
| L4 | BOTH2 | 0.4742 | 0.4596 | 0.4728 |

## Source decomposition at epoch 94 validation

Effects use REF1 as the same-teacher reference. `I = both - shuffle - augmentation`.

### Robust accuracy

| Teacher | REF1 | REF2−REF1 | S1 | S2 | U1 | U2 | SU1 | SU2 | I1 | I2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 0.4700 | 0.0000 | 0.0100 | -0.0052 | 0.0014 | 0.0026 | -0.0138 | -0.0026 | -0.0252 | -0.0000 |
| L4 | 0.4690 | 0.0000 | -0.0024 | 0.0044 | 0.0042 | 0.0120 | -0.0178 | 0.0038 | -0.0196 | -0.0126 |

### Clean accuracy

| Teacher | REF1 | REF2−REF1 | S1 | S2 | U1 | U2 | SU1 | SU2 | I1 | I2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 0.7838 | 0.0000 | -0.0032 | 0.0074 | -0.0138 | 0.0024 | 0.0048 | -0.0020 | 0.0218 | -0.0118 |
| L4 | 0.7704 | 0.0000 | -0.0002 | 0.0228 | 0.0020 | 0.0092 | -0.0090 | 0.0142 | -0.0108 | -0.0178 |

## Mean absolute source sensitivity

Mean absolute effects average the two preregistered perturbations and are descriptive only.

| Teacher | Shuffle | Augmentation | Both | Interaction |
| --- | --- | --- | --- | --- |
| L2 | 0.0076 | 0.0020 | 0.0082 | 0.0126 |
| L4 | 0.0034 | 0.0081 | 0.0108 | 0.0161 |

## Training trajectory and best/last diagnostics

These metrics come from the training log's fixed validation-PGD measurement; they are not the independent endpoint.

| Teacher | Arm | Best epoch | Best val PGD | Epoch-94 val PGD | Epoch-94 val clean | Best−last gap |
| --- | --- | --- | --- | --- | --- | --- |
| L2 | REF1 | 87 | 0.4760 | 0.4698 | 0.7838 | 0.0062 |
| L2 | REF2 | 87 | 0.4760 | 0.4698 | 0.7838 | 0.0062 |
| L2 | SHUF1 | 94 | 0.4798 | 0.4798 | 0.7808 | 0.0000 |
| L2 | SHUF2 | 91 | 0.4810 | 0.4650 | 0.7912 | 0.0160 |
| L2 | AUG1 | 85 | 0.4802 | 0.4712 | 0.7702 | 0.0090 |
| L2 | AUG2 | 85 | 0.4782 | 0.4720 | 0.7862 | 0.0062 |
| L2 | BOTH1 | 93 | 0.4792 | 0.4572 | 0.7886 | 0.0220 |
| L2 | BOTH2 | 85 | 0.4750 | 0.4682 | 0.7818 | 0.0068 |
| L4 | REF1 | 86 | 0.4822 | 0.4688 | 0.7704 | 0.0134 |
| L4 | REF2 | 86 | 0.4822 | 0.4688 | 0.7704 | 0.0134 |
| L4 | SHUF1 | 85 | 0.4800 | 0.4666 | 0.7702 | 0.0134 |
| L4 | SHUF2 | 93 | 0.4860 | 0.4742 | 0.7932 | 0.0118 |
| L4 | AUG1 | 86 | 0.4792 | 0.4736 | 0.7724 | 0.0056 |
| L4 | AUG2 | 94 | 0.4808 | 0.4808 | 0.7796 | 0.0000 |
| L4 | BOTH1 | 83 | 0.4820 | 0.4518 | 0.7614 | 0.0302 |
| L4 | BOTH2 | 93 | 0.4872 | 0.4726 | 0.7846 | 0.0146 |

## Interpretation and stop boundary

REF2−REF1 is retained as an observed same-source residual. The source effects describe this exact continuation and do not establish general dominance of shuffle or augmentation. No seed, checkpoint, attack, schedule, or future adaptive method was selected from the endpoint values.

The preregistered campaign is complete. The result is recorded for human review; no stabilization run, new seed, official test, or AutoAttack was started automatically.
