# The in-house TRADES is not TRADES: the clean target is detached

## Failure signature

The controlled TRADES baseline scores **45.14 % AutoAttack** on the official
CIFAR-10 test set at its best checkpoint (`docs/EXPERIMENT_DASHBOARD.md:167`,
`docs/plans/0027-controlled-teacherless-baselines.md:140`).  Every published
value for TRADES on ResNet-18 is between **49.0 and 49.4 %**.

The same engine's PGD-AT baseline scores 47.63 %, inside its own literature range
of 47.7 to 48.8 %.  **Only TRADES is low.**  That asymmetry is what rules out the
explanations that would move both numbers — the 45,000-image training split, the
evaluation, the schedule, the architecture — and points at the TRADES loss
itself.

## Root cause

TRADES minimises `CE(f(x), y) + beta * KL(f(x') || f(x))`, where `x'` is the
adversarial input and `x` the clean one.  **Both branches are the same network,
and the upstream implementation lets gradients flow through both.**

`.external/trades/trades.py:80-83`:

```python
logits = model(x_natural)
loss_natural = F.cross_entropy(logits, y)
loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                F.softmax(model(x_natural), dim=1))
loss = loss_natural + beta * loss_robust
```

`model(x_natural)` inside the KL is not detached.

This engine routes TRADES through a helper shared with distillation,
`src/ard/objectives/kl.py`, which detaches its target:

```python
target = F.softmax(target_logits.detach() / temperature, dim=1)
```

and `src/ard/objectives/trades.py:33-36` passes `clean_student_logits` as that
target.

**Detaching is correct for a teacher and wrong here.**  A teacher is frozen, so
its logits carry no gradient either way.  The TRADES "target" is the student's
own clean output, and the whole point of the KL term is that it pulls the clean
and adversarial predictions towards each other.  Detached, it only pushes the
adversarial output towards a clean output that the KL term no longer shapes; the
clean branch is then trained by the cross-entropy term alone.

The helper's docstring says the detach is deliberate, and names both uses in one
breath: "so teacher logits and the TRADES clean target do not receive
outer-objective gradients."  The intent was right for the first and wrong for the
second.

**Correction to this document's first version.**  It said the difference had gone
unnoticed.  It had not.  `docs/UPSTREAM_BASELINES.md` section 8 listed it among
"documented upstream-vs-local differences" that "are intentional", and
`tests/regression/test_trades_upstream_differential.py` contained two tests whose
job was to assert that the local and official clean-branch gradients **differ**.
The divergence was known, declared deliberate, and pinned by tests.

That makes the failure a different one, and a more instructive one.  **No reason
for the divergence is recorded anywhere — only the fact of it.**  And its
consequence was never connected to it: nothing links "we changed the TRADES
gradient on purpose" to "our TRADES scores four points below every published
value".  A declaration of intent is not a justification, and a difference that is
documented but whose cost is not measured is indistinguishable from a defect.

**Resolved 2026-09-07.**  Asked directly, the author does not remember a reason
and judges it most likely to have been a mistake.  The fix stands, and this note
is the record of how a mistake came to be labelled intentional: it was written
down as a difference, a test was written to hold it in place, and neither step
asked what it cost.  The cost was four points, and it was visible in the
dashboard the whole time.

## Confirmation

Same loss value, different optimisation.  On a synthetic batch through a linear
model with `beta = 6`:

| | |
| --- | ---: |
| gradient norm, upstream formulation | 3.2276 |
| gradient norm, in-house formulation | 1.9106 |
| norm of the difference | 1.8780 |
| relative difference | **58.2 %** |
| cosine similarity | 0.8547 |

A 58 % difference in the gradient, sustained over 200 epochs, is comfortably
large enough to explain four points of AutoAttack accuracy.  VERIFIED by
execution.

## Blast radius

| call site | target | frozen? | affected |
| --- | --- | --- | --- |
| `objectives/trades.py:33` | the student's own clean logits | **no** | **yes — this is the defect** |
| `objectives/rslad.py:36` | teacher logits | yes | no |
| `objectives/rslad.py:49` | teacher logits | yes | no |

**No ARD or RSLAD result is affected.**  The teacher is frozen, so detaching its
logits changes nothing.  Everything this project has concluded about
distillation stands.

What does not stand is the TRADES row, and anything that used it as a comparator.

## Fix

Give `target_to_student_kl` an explicit `detach_target` argument rather than
detaching unconditionally, and have TRADES pass `False`.  Leaving the default at
`True` keeps every distillation call site unchanged and makes the one place that
differs say so.

This changes a baseline's identity, so:

- the existing TRADES record is superseded, not edited, and the new run is a new
  record;
- `docs/EXPERIMENT_DASHBOARD.md` and
  `docs/plans/0027-controlled-teacherless-baselines.md` gain a line saying the
  45.14 % figure came from a defective loss;
- a regression test asserts that the TRADES outer objective produces a non-zero
  gradient through the clean branch, so this cannot come back silently;
- `tests/regression/test_trades_upstream_differential.py` is inverted: it now
  asserts that local and official TRADES agree on the loss and on **both**
  gradients, and that one optimiser step lands on the same weights. It is a
  parity test where it used to be a record of divergence.

## Why it matters beyond one number

`docs/ARD_VERSUS_AT_ASSESSMENT.md` proposes making the missing fair comparison
between adversarial training and adversarial distillation the thesis question.
That comparison rests on teacher-free baselines being correct.  **A TRADES four
points below the literature would have made the teacher look four points better
than it is**, and the error would have run in the direction that flattered the
project's own line of work.  It was found before the comparison was run, which is
the only reason it is a footnote rather than a retraction.
