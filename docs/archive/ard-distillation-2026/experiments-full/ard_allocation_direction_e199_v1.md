# Which samples should get the richer augmentation? The answer is: it does not matter.

Contract: `ard_allocation_direction_e199_v1`. Record: `ard_allocation_direction_e199_v1.json`.
Plan: `docs/plans/0096-hardness-allocation-direction.md`. Decision: `docs/decisions/0008-alloc-v1-endpoint-and-floor-order.md`.

**The question.** A strong augmentation pipeline is switched on at epoch 100 for a fifth of the training set. Does it matter *which* fifth? Two published accounts disagree. The capacity account says give it to samples with margin to spare; the vulnerability account says give it to the samples that are failing.

**The design.** Six parents, forked at epoch 100 and trained to epoch 199. Four arms per parent, and **every arm treats the same number of images in the same per-class proportions**, so a difference cannot be a dose effect or a class effect. A fifth arm per parent is a second, independent random draw of that same size and shape: it measures what this design returns when nothing but the arbitrary identity of the selected set changes.

**The instrument.** Held-out CE-PGD20 on the validation, 5000 images, from the epoch-199 checkpoint (`last.pt`), with the attack seeded identically across arms so the comparison uses common random numbers.

## The floor, fixed before any effect was read

| parent | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `|RANDOM - RANDOMB|` pp | 0.28 | 0.34 | 0.36 | 0.26 | 0.36 | 0.16 |

Maximum **0.36 pp**, mean 0.29, SD 0.08. The preregistered rule sets the threshold to `max(floor, 0.25 pp)`, so the threshold is **0.36 pp**.

## The result

| contrast | p1 | p2 | p3 | p4 | p5 | p6 | mean | SD | SE | vs threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `SAFE - RANDOM` (primary) | +0.00 | -0.26 | +0.48 | +0.02 | -0.04 | +0.10 | **+0.050** | 0.243 | 0.099 | inside |
| `FRAGILE - RANDOM` (secondary) | +0.04 | -0.58 | +0.18 | +0.34 | -0.08 | -0.36 | **-0.077** | 0.343 | 0.140 | inside |
| `SAFE - FRAGILE` (direction) | -0.04 | +0.32 | +0.30 | -0.32 | +0.04 | +0.46 | **+0.127** | 0.287 | 0.117 | inside |
| `I100 - RANDOM` (dose, reference) | +1.26 | +0.44 | +1.08 | +0.76 | +0.80 | +0.84 | **+0.863** | 0.282 | 0.115 | **outside** |

## Reading

**All three allocation contrasts lie inside the threshold. This is the plan's fourth preregistered branch, and it is one result, not three nulls: per-sample allocation direction is closed in both directions at once.**

The sharpest way to say it does not need the threshold at all. **Two independent random draws of the same size and class proportions differ by up to 0.36 pp, which is more than either named direction differs from random.** Choosing samples by margin, in either direction, moves the endpoint less than choosing them arbitrarily twice.

**And the same six parents, the same instrument and the same day give a resolved effect when the dose changes**: treating all 45,000 images rather than a matched fifth is worth **+0.863 pp**, 2.4 times the threshold and positive on 6 of 6 parents. So the design is not blind -- it resolves an effect of this kind when one is there. How much hardness there is matters. Where it goes does not.

**The clean-accuracy gate fires on one arm.** `ALLOC_FRAGILE` drops more than 0.5 pp of clean accuracy on p4, p5. Under the preregistered rule that disqualifies the arm regardless of its robust accuracy. It does not change the conclusion, which is already a null on that arm; it adds that the vulnerability direction also costs clean accuracy on average (-0.310 pp).

## What this does not settle

This is one switch epoch, one late policy, one dose, one architecture, one dataset and one teacher. It says that at this operating point the direction of allocation is inert. It does not say that no per-sample criterion could ever matter, and it does not speak to allocation of anything other than augmentation hardness.

The endpoint is a 5,000-image validation split under CE-PGD20, not the official test set under AutoAttack. The threshold is measured for exactly this contrast, at this horizon, on this split, under this attack, and transfers to nothing else.

## Disclosure

The preregistration asked for the floor to be fixed before the effects were read. That ordering was broken. Before the endpoint ran, the per-epoch `val_pgd_accuracy` at epoch 199 was tabulated and reported for the four judged arms, and that quantity is the same CE-PGD20 attack on the same held-out split -- a lower-quality version of the judgment metric, not a different one. It previewed the answer.

What limits the damage is that no discretion was left to exercise. The threshold rule was written down as `max(floor, 0.25 pp)` before the second random draw was launched, the floor is a maximum over six mechanical differences, and the six replicate arms were launched before any endpoint existed. The threshold could only move up from the preregistered 0.25 pp, and it did, to 0.36. But the ordering was not kept, and a reader should know it.
