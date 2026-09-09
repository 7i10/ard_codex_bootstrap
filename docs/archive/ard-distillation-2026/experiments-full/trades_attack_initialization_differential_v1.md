# TRADES attack initialization: what the deviation costs on a fixed batch

Contract: `trades-attack-initialization-differential/v1`. Record: `trades_attack_initialization_differential_v1.json`.

**Question.** Does the local uniform-in-[-eps,eps] attack initialization, against official TRADES' 0.001-scale Gaussian initialization, change the inner maximisation enough to explain the 1.2-1.5 pp shortfall of the corrected local TRADES against the published range?

**Terms.** *Inner KL* is the value the ten-step attack drives the Kullback-Leibler divergence to between the model's prediction on the perturbed image and its own prediction on the clean image; a higher value means the attack found a harder example, so a higher value means a stronger attack. *Uniform* is this repository's initialization, a draw from the uniform distribution on the whole epsilon-ball. *Gaussian* is official TRADES', a draw from a normal distribution at scale 0.001, which starts essentially at the clean image. Everything else -- model, weights, batch, epsilon, step size, step count, target, evaluation mode -- is identical between the two, and one step loop serves both.

**Setting.** 256 images, first N images of the configured split, in index order. epsilon 0.031373, step 0.007843, 10 steps, KL to the student's own clean prediction, beta 6.0. CPU. No training, no test split.

## Result

| checkpoint | inner KL (uniform) | inner KL (Gaussian) | uniform is | outer loss diff | parameter gradient diff | samples where uniform is stronger | identical adversarial examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 48 | 0.079378 | 0.077013 | +3.07% stronger | +1.00% | -9.63% | 50.4% | 0.0% |
| epoch 150 | 0.075333 | 0.071868 | +4.82% stronger | +2.32% | -9.79% | 46.5% | 0.0% |

## Reading

The local initialization makes the inner attack **stronger**, not weaker, at both checkpoints. Training against a harder inner problem raises robustness if it does anything, so this deviation points the wrong way to explain a local TRADES that sits *below* the published range. The hypothesis is not supported.

The size is also wrong. The detached-clean-target defect, measured the same way on a fixed batch, moved the gradient by 58 per cent and cost 2.73 percentage points of official-test AutoAttack. This deviation moves the inner objective by 3 to 5 per cent and the parameter gradient by about 10 per cent, an order of magnitude smaller and in the helping direction.

A separate observation, not the question asked. The two initializations reach adversarial examples that differ by the full width of the ball on at least one pixel of **every** image, and no image lands in the same place under both. Yet per image the uniform attack is the stronger one only about half the time. Which maximum the attack finds is close to arbitrary; how high it climbs is not. Any claim that two attack configurations are equivalent because their adversarial examples agree, or different because they disagree, is reading the wrong quantity.

## What this does not show

Both arms attack a model that was itself trained under the uniform initialization, so the loss surface is the one that initialization produced. A model trained under the Gaussian initialization could differ. A fixed-batch differential cannot rule that out; it can only say that on this surface the deviation is small and helps. The same limitation applied to the detached-target measurement, which predicted the sign correctly.

It also says nothing about the other recorded deviations -- the epsilon and step-size defaults, the normalization path, or the learning-rate schedule. Those remain unmeasured.
